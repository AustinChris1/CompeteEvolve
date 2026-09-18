"""Measuring code and ranking agents.

`measure_baseline` runs once per run, before any agent starts, and stores the
original code's traits on `SharedState`. Every other measurement in the run
(evolution samples, the `evaluator` tool, the benchmark) compares against
that one baseline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from . import memory, sandbox
from . import traits as traits_module
from .logic_check import LogicChecker
from .traits import TraitSet

log = logging.getLogger(__name__)


@dataclass
class AgentEvaluation:
    code: str
    trait_set: TraitSet

    @property
    def trait(self) -> float:
        return self.trait_set.trait


async def measure(shared, code: str, original_code: str, previous_code: str | None,
                  dataset_csv: str, logic_checker: LogicChecker, is_reference: bool = False) -> TraitSet:
    """Runs `code` in the sandbox with the run's timeout and repeats, then
    asks the logic reviewer for EL (only when the code ran cleanly), and
    derives the secondary traits."""
    cfg = shared.config
    result = await sandbox.run(code, dataset_csv, cfg.sandbox_timeout, repeats=cfg.sandbox_repeats)
    primary = result.primary
    if result.ok:
        logical_errors, _reasons = await logic_checker.count(code)
        primary.logical_errors = logical_errors
    return traits_module.build(
        primary=primary, sample_code=code, original_code=original_code, previous_code=previous_code,
        stderr=result.stderr, is_reference=is_reference,
    )


async def measure_baseline(shared, algorithm: str, logic_checker: LogicChecker) -> TraitSet:
    """Measures the original code (CO) once and stores it as `shared.baseline`."""
    original_code = await fetch_original_code(shared, algorithm)
    dataset_csv = await fetch_dataset(shared, algorithm)
    baseline = await measure(shared, original_code, original_code, None, dataset_csv, logic_checker, is_reference=True)
    shared.baseline = baseline
    return baseline


async def measure_reference(shared, algorithm: str, reference_code: str, logic_checker: LogicChecker) -> TraitSet:
    """Measures a reference implementation (scikit-learn defaults) and stores it as `shared.reference`."""
    dataset_csv = await fetch_dataset(shared, algorithm)
    reference = await measure(shared, reference_code, reference_code, None, dataset_csv, logic_checker, is_reference=True)
    shared.reference = reference
    return reference


def require_baseline(shared) -> TraitSet:
    if shared.baseline is None:
        raise RuntimeError("the original code has not been measured yet; main() must call measure_baseline first")
    return shared.baseline


async def run(args: dict, shared, agent_name: str, algorithm: str, llm_client=None) -> str:
    """The `evaluator` tool."""
    baseline = require_baseline(shared)
    sample_code = args.get("candidate") or await fetch_agent_sample(shared, agent_name, algorithm)
    dataset_csv = await fetch_dataset(shared, algorithm)
    logic_checker = LogicChecker(llm_client if shared.config.logic_check else None)

    log.info("[evaluator:%s] measuring this agent's candidate", agent_name)
    previous_code = await fetch_agent_previous_sample(shared, agent_name, algorithm, sample_code)
    sample_traits = await measure(shared, sample_code, baseline.code, previous_code, dataset_csv, logic_checker)

    p, s = sample_traits.primary, sample_traits.secondary
    log.info("[evaluator:%s] Ma=%.4f L=%.4f T=%.4fs N=%.3f A=%.3f R=%.3f T0=%.6g",
             agent_name, p.model_accuracy, p.loss, s.time, s.novelty, s.accuracy_term, s.resource, s.trait)

    async with shared.eval_lock:
        shared.agent_evaluations[agent_name] = AgentEvaluation(code=sample_code, trait_set=sample_traits)

    positions = await compute_rank_positions(shared)
    rank_position = next((rp for name, rp in positions if name == agent_name), None)
    if rank_position is None:
        raise RuntimeError(f"internal error: agent '{agent_name}' missing from its own rank computation")

    evidence = traits_module.threshold_evidence(baseline, sample_traits, shared.config.mu_threshold)
    await shared.record_sample({
        "agent": agent_name, "algorithm": algorithm, "id": f"{agent_name}_evaluator", "generation": None,
        "island": None, "mode": "evaluator", "primary": vars(p), "secondary": vars(s),
        "stderr": sample_traits.stderr[-500:], "code": sample_code, "threshold": evidence,
    })

    notes: list[str] = []
    if p.syntax_errors or p.runtime_errors or p.logical_errors:
        notes.append(
            f"Your code scored SE={p.syntax_errors}, RE={p.runtime_errors}, EL={p.logical_errors}. "
            "These reduce A (and therefore your performance) sharply. Fix them and evolve a new sample."
        )
    if not evidence["proven"]:
        why = "; ".join(evidence["reasons"]) or f"mu {evidence['mu']:.4f} is not above {evidence['threshold']:.4f}"
        notes.append(f"The 25 percent improvement is NOT proven yet: {why}. Keep evolving.")
    if sample_traits.stderr:
        notes.append(f"Sandbox stderr: {sample_traits.stderr[-600:]}")

    return json.dumps({
        "status": "ok",
        "agent": agent_name,
        "algorithm": algorithm,
        "rank_position": rank_position,
        "agent_count": len(positions),
        "threshold": evidence,
        "primary_traits": vars(p),
        "secondary_traits": vars(s),
        "original_primary_traits": vars(baseline.primary),
        "notes": notes,
    })


async def compute_rank_positions(shared) -> list[tuple[str, int]]:
    async with shared.eval_lock:
        snapshot = dict(shared.agent_evaluations)
    if not snapshot:
        raise RuntimeError("no agent has been evaluated yet; run the 'evaluator' tool for at least one agent first")
    return traits_module.rank_positions([(name, ev.trait) for name, ev in snapshot.items()])


async def fetch_agent_sample(shared, agent_name: str, algorithm: str) -> str:
    """The agent's best remembered candidate, by trait (not by memory's recall order)."""
    async with shared.memory_lock:
        raw = memory.run(shared.memory, {
            "action": "read", "top_n": 100, "category": "code", "agent": agent_name, "algorithm": algorithm,
        })
    results = json.loads(raw).get("results", [])
    if not results:
        raise RuntimeError(f"no '{algorithm}' sample found in memory for agent '{agent_name}'; run 'evolution' first")
    best = max(results, key=lambda r: (r.get("rank") or 0.0))
    return best["code"]


async def fetch_agent_previous_sample(shared, agent_name: str, algorithm: str, exclude_code: str) -> str | None:
    async with shared.memory_lock:
        raw = memory.run(shared.memory, {
            "action": "read", "top_n": 100, "category": "code", "agent": agent_name, "algorithm": algorithm,
        })
    for r in json.loads(raw).get("results", []):
        code = r.get("code")
        if code and code != exclude_code:
            return code
    return None


async def fetch_original_code(shared, algorithm: str) -> str:
    """CO: the agent-less 'code' entry seeded by main()."""
    async with shared.memory_lock:
        raw = memory.run(shared.memory, {"action": "read", "top_n": 100, "category": "code", "algorithm": algorithm})
    results = json.loads(raw).get("results", [])
    hit = next((r for r in results if r.get("agent") is None), None)
    if hit is None:
        raise RuntimeError(f"no baseline (agent-less) 'code' entry found for algorithm '{algorithm}'")
    return hit["code"]


async def fetch_dataset(shared, algorithm: str) -> str:
    async with shared.memory_lock:
        raw = memory.run(shared.memory, {"action": "read", "top_n": 1, "category": "dataset", "algorithm": algorithm})
    results = json.loads(raw).get("results", [])
    if not results:
        raise RuntimeError(f"no 'dataset' entry found for algorithm '{algorithm}'")
    return results[0]["code"]
