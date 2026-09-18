"""End to end through evolution, evaluator, reinforcement and the benchmark
report with a fake LLM. No network, no scikit-learn in the candidates."""

import json
from pathlib import Path

import benchmark
from agent import evaluator, evolution, memory, reinforcement
from agent.logic_check import LogicChecker
from tests.conftest import FAST_STRONG, FAST_WEAK, SLOW_WEAK, TINY_CSV, FakeLLM


async def _seed(shared, algorithm="toy"):
    async with shared.memory_lock:
        memory.remember_text(shared.memory, SLOW_WEAK, category="code", algorithm=algorithm)
        memory.remember_text(shared.memory, TINY_CSV, category="dataset", algorithm=algorithm)
    await evaluator.measure_baseline(shared, algorithm, LogicChecker(None))


async def test_evolution_requires_a_measured_baseline(shared):
    async with shared.memory_lock:
        memory.remember_text(shared.memory, SLOW_WEAK, category="code", algorithm="toy")
        memory.remember_text(shared.memory, TINY_CSV, category="dataset", algorithm="toy")
    try:
        await evolution.run({}, "agent-1", "toy", FakeLLM(), shared)
    except RuntimeError as e:
        assert "has not been measured" in str(e)
    else:
        raise AssertionError("evolution must refuse to run without a shared baseline")


async def test_full_pipeline_proves_threshold_and_writes_artifacts(shared, tmp_path):
    await _seed(shared)
    assert shared.baseline is not None and shared.baseline.primary.model_accuracy == 0.5

    llm = FakeLLM(FAST_STRONG)
    out = json.loads(await evolution.run(
        {"generations": 99, "population_size": 99, "samples_per_population": 99}, "agent-1", "toy", llm, shared
    ))
    assert out["status"] == "ok"
    assert out["config_applied"] == {"generations": 1, "population_size": 1, "samples_per_population": 1,
                                     "mu_threshold": shared.config.mu_threshold}
    assert out["threshold"]["proven"] is True and out["reached_threshold"] is True, out["threshold"]
    assert out["primary_traits"]["model_accuracy"] == 0.9
    assert Path(out["traits_file"]).exists()
    assert (shared.run_dir / "samples.jsonl").exists()
    records = [json.loads(line) for line in (shared.run_dir / "samples.jsonl").read_text().splitlines()]
    assert records and records[0]["agent"] == "agent-1" and records[0]["code"].strip().startswith("def run")

    # A second agent submits a faster but worse candidate: it must not be "proven".
    weak = json.loads(await evolution.run({}, "agent-2", "toy", FakeLLM(FAST_WEAK), shared))
    assert weak["threshold"]["proven"] is False
    assert any("below the original" in r for r in weak["threshold"]["reasons"])

    ev1 = json.loads(await evaluator.run({}, shared, "agent-1", "toy", None))
    ev2 = json.loads(await evaluator.run({}, shared, "agent-2", "toy", None))
    assert ev1["rank_position"] == 1 and ev2["rank_position"] == 2 and ev2["agent_count"] == 2
    assert ev1["threshold"]["proven"] and not ev2["threshold"]["proven"]
    assert any("NOT proven" in n for n in ev2["notes"])

    rl = json.loads(await reinforcement.run({}, shared, "agent-2"))
    assert rl["reward"] == 0 and rl["leaderboard"][0]["agent"] == "agent-1"

    report = await benchmark.run(shared, "toy", agent_names=["agent-1", "agent-2", "agent-3"],
                                 llm_client=None, output_dir=tmp_path / "benchmarks")
    assert report.report_path.exists() and report.chart_path.exists()
    assert report.evidence["proven"] and report.not_evaluated == ["agent-3"]
    text = report.report_path.read_text(encoding="utf-8")
    assert "Met and proven" in text and "agent-3" in text
    assert (shared.run_dir / report.report_path.name).exists()
    summary = json.loads(benchmark.write_summary(shared.run_dir, report).read_text())
    assert summary["ranking"][0]["agent"] == "agent-1"


async def test_evaluator_scores_the_best_remembered_sample_not_the_last(shared):
    await _seed(shared)
    async with shared.memory_lock:
        memory.run(shared.memory, {"action": "write", "code": FAST_STRONG, "category": "code", "agent": "a",
                                   "algorithm": "toy", "rank": 5.0, "functional_accuracy": 0.9})
        memory.run(shared.memory, {"action": "write", "code": FAST_WEAK, "category": "code", "agent": "a",
                                   "algorithm": "toy", "rank": 0.1, "functional_accuracy": 0.3})
    assert await evaluator.fetch_agent_sample(shared, "a", "toy") == FAST_STRONG
