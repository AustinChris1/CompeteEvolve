from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

import benchmark
import sklearn_algorithms
import user_inputs
from agent import Agent, SharedState, evaluator, memory
from agent.config import RunConfig
from agent.llm_client import LLMError, ModelRoles, create_client
from agent.logic_check import LogicChecker
from agent.prompts import DEFAULT_DATASET_NOTE, build_task_prompt
from agent.traits import TraitSet

VALID_PROVIDERS = ("gemini", "openrouter")
log = logging.getLogger("competeevolve")


@dataclass
class RunInputs:
    algorithm: str
    display_name: str
    dataset_name: str
    baseline_code: str
    reference_code: str | None
    dataset_csv: str
    dataset_note: str


async def main() -> int:
    load_dotenv()
    options = _parse_args()
    _configure_logging(options.verbose)

    try:
        config = RunConfig.from_env(
            agents=options.agents, generations=options.generations, islands=options.islands,
            samples_per_island=options.samples, llm_rpm=options.rpm,
            logic_check=False if options.no_logic_check else None,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    roles = _choose_roles(options)
    missing = roles.missing_keys()
    if missing:
        print(f"Error: {', '.join(missing)} not set (add it to your .env file)", file=sys.stderr)
        return 2
    print(f"LLM roles: {roles.describe()}")

    try:
        inputs = await _resolve_inputs(options)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    print(f"Optimizing: {inputs.display_name} on {inputs.dataset_name}")

    if options.agents is None and not options.run_id:
        config = RunConfig.from_env(**{**config.to_dict(), "agents": _read_agent_count(config.agents)})
    run_id = options.run_id or f"{inputs.algorithm}-{time.strftime('%Y%m%d-%H%M%S')}"
    run_dir = Path("runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps({
        "config": config.to_dict(), "roles": roles.describe(), "algorithm": inputs.algorithm,
        "dataset": inputs.dataset_name,
    }, indent=2), encoding="utf-8")
    print(f"Run directory: {run_dir}")
    print(f"Config: {config.describe()}")

    shared = SharedState(config=config, dataset_note=inputs.dataset_note, run_dir=run_dir)
    await _seed_memory(shared, inputs)

    # One client for the measurements that happen outside the agents: the
    # baseline's logic check, the reference, and the report's annotation.
    side_client = create_client(
        roles.evolution_provider, roles.evolution_api_key, roles.evolution_model, shared.rate_limiter
    )
    try:
        logic_checker = LogicChecker(side_client if config.logic_check else None)
        print("\nMeasuring the original code (once, sequentially)...")
        baseline = await evaluator.measure_baseline(shared, inputs.algorithm, logic_checker)
        _print_traits("original", baseline)
        if baseline.primary.syntax_errors or baseline.primary.runtime_errors:
            print(f"Error: the original code failed in the sandbox, so no comparison is possible:\n"
                  f"{baseline.stderr}", file=sys.stderr)
            return 2
        if inputs.reference_code:
            print("Measuring scikit-learn defaults (reference)...")
            try:
                reference = await evaluator.measure_reference(shared, inputs.algorithm, inputs.reference_code, logic_checker)
                _print_traits("reference", reference)
            except RuntimeError as e:
                log.warning("reference measurement failed: %s", e)

        task_prompt = build_task_prompt(inputs.algorithm, inputs.display_name, inputs.dataset_name)
        agents = [
            Agent(f"agent-{i + 1}", inputs.algorithm, inputs.dataset_name, roles, shared)
            for i in range(config.agents)
        ]
        print(f"\nRunning {config.agents} agent(s) concurrently...\n")
        results = await asyncio.gather(*(agent.send(task_prompt) for agent in agents), return_exceptions=True)
        for agent, result in zip(agents, results, strict=True):
            if isinstance(result, Exception):
                print(f"{agent.name}: error: {result}")
            else:
                print(f"{agent.name}: {result}")
        for agent in agents:
            await agent.aclose()

        try:
            report = await benchmark.run_and_print(
                shared, inputs.algorithm, agent_names=[a.name for a in agents], llm_client=side_client
            )
        except RuntimeError as e:
            print(f"\nNo benchmark report: {e}", file=sys.stderr)
            return 1
        summary_path = benchmark.write_summary(run_dir, report)
        print(f"Summary saved to: {summary_path}")
        return 0
    finally:
        await side_client.aclose()


def _configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s" if not verbose else "%(levelname)s %(name)s: %(message)s",
        stream=sys.stdout,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def _print_traits(label: str, t: TraitSet) -> None:
    p, s = t.primary, t.secondary
    print(f"  {label:<10} Ma={p.model_accuracy:.4f} L={p.loss:.4f} TT={p.training_time:.4f}s "
          f"TI={p.inference_time:.4f}s Mu={p.memory_usage:.0f}MB K={p.compute_cost:.2f}s "
          f"SE={p.syntax_errors} RE={p.runtime_errors} EL={p.logical_errors} P*={s.task_performance:.6g}")


async def _resolve_inputs(options) -> RunInputs:
    code_path, data_path = options.code, options.data
    if options.algorithm:
        return _builtin_inputs(options.algorithm)
    if not code_path and not data_path:
        if not _prompt_wants_custom():
            return _builtin_inputs(sklearn_algorithms.choose_algorithm())
        code_path = input("Path to your baseline algorithm (.py): ").strip()
        data_path = input("Path to your dataset (.csv): ").strip()
    if not code_path or not data_path:
        raise RuntimeError(
            "custom input needs BOTH a baseline .py file and a dataset .csv file "
            "(pass --code and --data, or use --algorithm / the menu for a built-in one)"
        )
    print("Loading your algorithm and dataset...")
    custom = await user_inputs.prepare_custom_inputs(code_path, data_path)
    return RunInputs(
        algorithm=custom.algorithm, display_name=custom.display_name, dataset_name=custom.dataset_name,
        baseline_code=custom.baseline_code, reference_code=None, dataset_csv=custom.dataset_csv,
        dataset_note=custom.description.as_prompt_note(),
    )


def _builtin_inputs(algorithm: str) -> RunInputs:
    spec = sklearn_algorithms.get_spec(algorithm)
    return RunInputs(
        algorithm=algorithm, display_name=spec.display_name, dataset_name=spec.dataset_name,
        baseline_code=spec.baseline_code, reference_code=spec.reference_code,
        dataset_csv=sklearn_algorithms.dataset_csv(algorithm), dataset_note=DEFAULT_DATASET_NOTE,
    )


def _prompt_wants_custom() -> bool:
    print("Where should the algorithm and data come from?")
    print("  1) Built-in scikit-learn algorithm on the Iris dataset")
    print("  2) My own algorithm (.py) and my own dataset (.csv)")
    return input("> ").strip() == "2"


async def _seed_memory(shared: SharedState, inputs: RunInputs) -> None:
    async with shared.memory_lock:
        memory.remember_text(shared.memory, inputs.baseline_code, category="code", algorithm=inputs.algorithm)
        memory.remember_text(shared.memory, inputs.dataset_csv, category="dataset", algorithm=inputs.algorithm)


def _parse_args():
    parser = argparse.ArgumentParser(description="CompeteEvolve: competing LLM agents evolving ML code")
    parser.add_argument("agents", nargs="?", type=int, default=None, help="number of agents")
    parser.add_argument("chat_provider", nargs="?", default=None, choices=[*VALID_PROVIDERS, None])
    parser.add_argument("evolution_provider", nargs="?", default=None, choices=[*VALID_PROVIDERS, None])
    parser.add_argument("--algorithm", choices=sklearn_algorithms.ALGORITHM_ORDER, help="built-in algorithm (skips the menu)")
    parser.add_argument("--code", default=None, help="path to your own baseline algorithm (.py)")
    parser.add_argument("--data", default=None, help="path to your own dataset (.csv)")
    parser.add_argument("--run-id", default=None, help="name of the run directory under runs/")
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--islands", type=int, default=None)
    parser.add_argument("--samples", type=int, default=None, help="samples per island")
    parser.add_argument("--rpm", type=int, default=None, help="LLM requests per minute, shared by all agents")
    parser.add_argument("--no-logic-check", action="store_true", help="skip the LLM logical-error review")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def _choose_roles(options) -> ModelRoles:
    chat_provider = options.chat_provider
    evolution_provider = options.evolution_provider
    if chat_provider and not evolution_provider:
        evolution_provider = chat_provider
    if not chat_provider:
        print("Select the LLM provider for the agent conversation loop (high volume):")
        print("  1) Gemini")
        print("  2) OpenRouter")
        chat_provider = "openrouter" if input("> ").strip() == "2" else "gemini"
    if not evolution_provider:
        print("Select the provider for evolution/mutation (code quality matters most here):")
        print("  1) Gemini")
        print("  2) OpenRouter")
        print(f"  3) Same as the conversation loop ({chat_provider})")
        evolution_provider = {"1": "gemini", "2": "openrouter"}.get(input("> ").strip(), chat_provider)
    return ModelRoles.from_env_and_choice(chat_provider, evolution_provider)


def _read_agent_count(default: int) -> int:
    raw = input(f"How many agents? (default {default}) ").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return default


if __name__ == "__main__":
    try:
        sys.exit(asyncio.run(main()))
    except LLMError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
