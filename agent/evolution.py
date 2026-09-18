"""The `evolution` tool: islands x generations x samples from the original code.

The original code (CO) is measured once per run and shared (see
`evaluator.measure_baseline`). This module never re-measures it, so every mu
in a run is comparable, and "threshold reached" is only reported when
`traits.threshold_evidence` says the claim is proven.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from . import evaluator, memory, sandbox
from . import traits as traits_module
from .llm_client import LLMClient, LLMError
from .logic_check import LogicChecker
from .prompts import build_evolution_prompt
from .traits import TraitSet

log = logging.getLogger(__name__)

TOP_N_SAMPLES = 3


@dataclass
class Sample:
    id: str
    island: int
    generation: int
    trait_set: TraitSet

    @property
    def code(self) -> str:
        return self.trait_set.code

    @property
    def trait(self) -> float:
        return self.trait_set.trait


def _clamp(args: dict, key: str, cap: int) -> int:
    """Tool arguments may lower a setting but never raise it above the run config."""
    raw = args.get(key)
    try:
        requested = int(raw) if raw is not None else cap
    except (TypeError, ValueError):
        requested = cap
    value = max(1, min(requested, cap))
    if raw is not None and value != requested:
        log.info("evolution: %s=%s requested, using %d (run config cap)", key, raw, value)
    return value


async def run(args: dict, agent_name: str, algorithm: str, llm_client: LLMClient, shared) -> str:
    cfg = shared.config
    generations = _clamp(args, "generations", cfg.generations)
    island_count = _clamp(args, "population_size", cfg.islands)
    samples_per_island = _clamp(args, "samples_per_population", cfg.samples_per_island)
    mu_threshold = cfg.mu_threshold

    baseline = evaluator.require_baseline(shared)
    original_code = baseline.code
    dataset_csv = await evaluator.fetch_dataset(shared, algorithm)
    dataset_note = getattr(shared, "dataset_note", "") or None
    logic_checker = LogicChecker(llm_client if cfg.logic_check else None)
    p1 = baseline.secondary.task_performance

    log.info(
        "[evolution:%s] %s: %d generation(s) x %d island(s) x %d sample(s); P1=%.6g, mu target > %.4f",
        agent_name, algorithm, generations, island_count, samples_per_island, p1, mu_threshold,
    )

    best_overall: Sample | None = None
    top_samples: list[Sample] = []
    previous_generation_code: str | None = None
    evidence = traits_module.threshold_evidence(baseline, baseline, mu_threshold)
    generations_run = 0
    traits_file: Path | None = None
    generation_failures = 0

    for generation in range(1, generations + 1):
        generations_run = generation
        generation_samples: list[Sample] = []

        for island in range(1, island_count + 1):
            if generation == 1 and island == 1 and not top_samples:
                mode = "annotate"
            elif top_samples:
                mode = "crossover"
            else:
                mode = "first"
            best_traits_md = _render_best_traits(top_samples) if top_samples else None
            log.info("[evolution:%s]   gen %d/%d island %d/%d (%s): %d sample(s)",
                     agent_name, generation, generations, island, island_count, mode, samples_per_island)

            generated, failures = await _generate_samples(
                llm_client, original_code, samples_per_island, mode, best_traits_md, dataset_note
            )
            generation_failures += failures

            for idx, code in enumerate(generated):
                trait_set = await evaluator.measure(
                    shared, code, original_code, previous_generation_code, dataset_csv, logic_checker
                )
                sample = Sample(
                    id=f"{agent_name}_gen{generation}_island{island}_sample{idx}",
                    island=island, generation=generation, trait_set=trait_set,
                )
                generation_samples.append(sample)
                await shared.record_sample(_sample_record(agent_name, algorithm, sample, mode))

                p, s = trait_set.primary, trait_set.secondary
                log.info(
                    "[evolution:%s]     %s: Ma=%.4f L=%.4f T=%.4fs N=%.3f SE=%d RE=%d EL=%d T0=%.6g",
                    agent_name, sample.id, p.model_accuracy, p.loss, s.time, s.novelty,
                    p.syntax_errors, p.runtime_errors, p.logical_errors, trait_set.trait,
                )
                if best_overall is None or sample.trait > best_overall.trait:
                    best_overall = sample

        if not generation_samples:
            log.warning("[evolution:%s]   gen %d produced no samples", agent_name, generation)
            continue

        pool = sorted(generation_samples + top_samples, key=lambda s: s.trait, reverse=True)
        top_samples = pool[:TOP_N_SAMPLES]
        traits_file = _write_traits_file(shared.run_dir, algorithm, agent_name, generation, top_samples, baseline)
        previous_generation_code = top_samples[0].code

        evidence = traits_module.threshold_evidence(baseline, best_overall.trait_set, mu_threshold)
        log.info("[evolution:%s]   gen %d complete: best T0=%.6g mu=%.4f proven=%s",
                 agent_name, generation, top_samples[0].trait, evidence["mu"], evidence["proven"])
        if evidence["proven"]:
            log.info("[evolution:%s] threshold proven after generation %d; stopping early", agent_name, generation)
            break

    if best_overall is None:
        raise RuntimeError(
            f"evolution produced no viable samples ({generation_failures} generation request(s) failed)"
        )

    stb = best_overall
    async with shared.memory_lock:
        try:
            memory.run(shared.memory, {
                "action": "write", "code": stb.code, "functional_accuracy": stb.trait_set.accuracy,
                "rank": stb.trait, "category": "code", "agent": agent_name, "algorithm": algorithm,
                "id": stb.id,
            })
        except memory.MemoryError as e:
            log.warning("[evolution:%s] could not remember the winner: %s", agent_name, e)

    p, s = stb.trait_set.primary, stb.trait_set.secondary
    return json.dumps({
        "status": "ok",
        "agent": agent_name,
        "algorithm": algorithm,
        "id": stb.id,
        "config_applied": {
            "generations": generations, "population_size": island_count,
            "samples_per_population": samples_per_island, "mu_threshold": mu_threshold,
        },
        "generations_run": generations_run,
        "generation_requests_failed": generation_failures,
        "threshold": evidence,
        "reached_threshold": evidence["proven"],
        "primary_traits": _primary_dict(p),
        "secondary_traits": _secondary_dict(s),
        "traits_file": str(traits_file) if traits_file else None,
        "code": stb.code,
    })


def _primary_dict(p) -> dict:
    return {
        "training_time": p.training_time, "inference_time": p.inference_time,
        "syntax_errors": p.syntax_errors, "runtime_errors": p.runtime_errors,
        "logical_errors": p.logical_errors, "loss": p.loss, "model_accuracy": p.model_accuracy,
        "memory_usage": p.memory_usage, "compute_cost": p.compute_cost,
    }


def _secondary_dict(s) -> dict:
    return {
        "time": s.time, "model_performance": s.model_performance, "novelty": s.novelty,
        "accuracy_term": s.accuracy_term, "resource": s.resource, "performance": s.performance,
        "task_performance": s.task_performance, "trait": s.trait,
    }


def _sample_record(agent_name: str, algorithm: str, sample: Sample, mode: str) -> dict:
    return {
        "agent": agent_name, "algorithm": algorithm, "id": sample.id, "generation": sample.generation,
        "island": sample.island, "mode": mode,
        "primary": _primary_dict(sample.trait_set.primary),
        "secondary": _secondary_dict(sample.trait_set.secondary),
        "stderr": sample.trait_set.stderr[-500:] if sample.trait_set.stderr else "",
        "code": sample.code,
    }


def _render_best_traits(top_samples: list[Sample]) -> str:
    blocks = []
    for position, sample in enumerate(top_samples, start=1):
        blocks.append(traits_module.format_traits_markdown(
            f"Top sample {position} (id: {sample.id}, generation {sample.generation})", sample.trait_set
        ))
        blocks.append(f"\n```python\n{sample.code}\n```\n")
    return "\n".join(blocks)


def _write_traits_file(run_dir: Path, algorithm: str, agent_name: str, generation: int,
                       top_samples: list[Sample], original: TraitSet) -> Path:
    directory = run_dir / "traits" / algorithm
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{agent_name}_gen{generation:02d}_best_traits.md"
    body = [
        f"# Best traits: {algorithm}, {agent_name}, generation {generation}\n",
        traits_module.format_traits_markdown("Original code (CO)", original),
        "\n",
        _render_best_traits(top_samples),
    ]
    path.write_text("\n".join(body), encoding="utf-8")
    return path


async def _generate_samples(llm_client: LLMClient, original_code: str, n: int, mode: str,
                            best_traits_markdown: str | None, dataset_note: str | None) -> tuple[list[str], int]:
    """Asks the generation model for `n` candidates. Returns (samples, failures)."""
    prompt = build_evolution_prompt(original_code, mode, best_traits_markdown, dataset_note)
    samples: list[str] = []
    failures = 0
    for _ in range(n):
        try:
            response = await llm_client.generate(system_instruction="", history=[{"role": "user", "text": prompt}])
        except (LLMError, RuntimeError) as e:
            failures += 1
            log.warning("evolution: sample generation failed: %s", e)
            continue
        cleaned = strip_code_fences(response.text)
        if cleaned.strip():
            samples.append(cleaned)
    return samples, failures


def strip_code_fences(text: str) -> str:
    trimmed = text.strip()
    if not trimmed.startswith("```"):
        return trimmed
    without_lang = trimmed[3:]
    newline = without_lang.find("\n")
    first_line = without_lang[:newline] if newline != -1 else ""
    if first_line.isalnum() or first_line == "":
        without_lang = without_lang[newline + 1:] if newline != -1 else without_lang
    end = without_lang.rfind("```")
    return (without_lang[:end] if end != -1 else without_lang).strip()


# Kept for callers that imported the old private name.
_strip_code_fences = strip_code_fences
_ = sandbox  # re-exported for tests that patch sandbox through this module
