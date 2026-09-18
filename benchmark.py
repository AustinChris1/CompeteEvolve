"""End-of-run report: agent ranking, the winning code annotated, and the
original versus evolved comparison. Nothing is re-measured here; the report
uses the baseline measured at start-up and the agents' evaluator results."""

from __future__ import annotations

import json
import logging
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from agent import evaluator  # noqa: E402
from agent import traits as traits_module  # noqa: E402
from agent.llm_client import LLMClient, LLMError  # noqa: E402
from agent.traits import TraitSet  # noqa: E402

log = logging.getLogger(__name__)


@dataclass
class AgentRow:
    agent: str
    rank_position: int
    reward: int
    trait_set: TraitSet | None = None
    evaluated: bool = True


@dataclass
class BenchmarkReport:
    algorithm: str
    version: int
    rows: list
    original: TraitSet
    reference: TraitSet | None
    evidence: dict
    annotated_winner_code: str
    ranking_table_md: str
    comparison_table_md: str
    chart_path: Path
    report_path: Path
    not_evaluated: list = field(default_factory=list)

    def top(self) -> AgentRow:
        return self.rows[0]

    @property
    def mu(self) -> float:
        return self.evidence["mu"]

    def summary(self) -> dict:
        return {
            "algorithm": self.algorithm,
            "version": self.version,
            "report": str(self.report_path),
            "chart": str(self.chart_path),
            "threshold": self.evidence,
            "not_evaluated": self.not_evaluated,
            "ranking": [
                {
                    "rank": r.rank_position, "agent": r.agent, "reward": r.reward, "evaluated": r.evaluated,
                    "performance": r.trait_set.performance if r.trait_set else None,
                    "task_performance": r.trait_set.secondary.task_performance if r.trait_set else None,
                    "accuracy": r.trait_set.accuracy if r.trait_set else None,
                    "time": r.trait_set.secondary.time if r.trait_set else None,
                }
                for r in self.rows
            ],
        }


def _file_stem(algorithm: str) -> str:
    return algorithm.replace("_", "").lower()


def _next_version(output_dir: Path, algorithm: str) -> int:
    pattern = re.compile(rf"^{re.escape(_file_stem(algorithm))}(\d+)\.md$", re.IGNORECASE)
    highest = 0
    if output_dir.is_dir():
        for path in output_dir.iterdir():
            match = pattern.match(path.name)
            if match:
                highest = max(highest, int(match.group(1)))
    return highest + 1


async def run(
    shared,
    algorithm: str,
    agent_names: list | None = None,
    llm_client: LLMClient | None = None,
    output_dir: Path = Path("benchmarks"),
) -> BenchmarkReport:
    original = evaluator.require_baseline(shared)
    reference = shared.reference

    async with shared.eval_lock:
        snapshot = dict(shared.agent_evaluations)
    if not snapshot:
        raise RuntimeError("no agent has been evaluated yet; run the 'evaluator' tool for at least one agent first")

    positions = await evaluator.compute_rank_positions(shared)
    evaluated_names = {name for name, _ in positions}
    rows = [
        AgentRow(agent=name, rank_position=rank, reward=traits_module.reward_for_rank(rank),
                 trait_set=snapshot[name].trait_set)
        for name, rank in positions
    ]
    all_names = list(agent_names) if agent_names is not None else sorted(evaluated_names)
    for name in evaluated_names:
        if name not in all_names:
            all_names.append(name)
    next_rank = len(rows) + 1
    for name in all_names:
        if name not in evaluated_names:
            rows.append(AgentRow(agent=name, rank_position=next_rank, reward=0, evaluated=False))
            next_rank += 1
    rows.sort(key=lambda r: r.rank_position)
    top = rows[0]
    if not top.evaluated or top.trait_set is None:
        raise RuntimeError("no agent was evaluated this run; nothing to benchmark")

    evidence = traits_module.threshold_evidence(original, top.trait_set, shared.config.mu_threshold)
    ranking_table_md = _build_ranking_table(rows)
    comparison_table_md = _build_comparison_table(top, original, reference, evidence)
    annotated_winner_code = await _annotate_winner(llm_client, algorithm, original, top.trait_set)

    output_dir.mkdir(parents=True, exist_ok=True)
    version = _next_version(output_dir, algorithm)
    stem = _file_stem(algorithm)
    chart_path = output_dir / f"{stem}{version:02d}.png"
    report_path = output_dir / f"{stem}{version:02d}.md"
    _render_chart(chart_path, rows, original, reference, top, algorithm)

    not_evaluated = [r.agent for r in rows if not r.evaluated]
    status = _threshold_status(evidence)
    report_md = (
        f"# Benchmark report: {algorithm} (v{version})\n\n"
        f"Threshold mu = P2/(P1+P2) = **{evidence['mu']:.4f}** "
        f"(target > {evidence['threshold']:.4f}, a 25 percent improvement over the original). {status}\n\n"
        f"Run configuration: `{shared.config.describe()}`\n\n"
        f"## 1. Ranking of agents ({len(rows)} agent(s))\n\n{ranking_table_md}\n"
        + (f"\n> **Note:** {', '.join(not_evaluated)} competed but never completed an `evaluator` call.\n"
           if not_evaluated else "")
        + f"\n## 2. Top ranking evolved code overall\n\n"
        f"Agent **{top.agent}** holds the best code overall (rank 1 of {len(rows)}).\n\n"
        f"{traits_module.format_traits_markdown('Winning candidate traits', top.trait_set)}\n"
        f"### Annotated winning code\n\n"
        f"`# IMPROVED:` comments mark what changed relative to the original and why it helped.\n\n"
        f"```python\n{annotated_winner_code}\n```\n\n"
        f"## 3. Original versus evolved code\n\n{comparison_table_md}\n"
        f"{traits_module.format_traits_markdown('Original code (CO) traits', original)}\n"
        + (f"{traits_module.format_traits_markdown('scikit-learn defaults (reference) traits', reference)}\n"
           if reference else "")
        + f"![Benchmark chart]({chart_path.name})\n"
    )
    report_path.write_text(report_md, encoding="utf-8")

    run_dir = getattr(shared, "run_dir", None)
    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(report_path, run_dir / report_path.name)
        shutil.copy2(chart_path, run_dir / chart_path.name)

    return BenchmarkReport(
        algorithm=algorithm, version=version, rows=rows, original=original, reference=reference,
        evidence=evidence, annotated_winner_code=annotated_winner_code, ranking_table_md=ranking_table_md,
        comparison_table_md=comparison_table_md, chart_path=chart_path, report_path=report_path,
        not_evaluated=not_evaluated,
    )


def _threshold_status(evidence: dict) -> str:
    if evidence["proven"]:
        return "**Met and proven.**"
    if evidence["reached"]:
        return "**Met numerically but NOT proven:** " + "; ".join(evidence["reasons"]) + "."
    return "_Not met._" + (" " + "; ".join(evidence["reasons"]) + "." if evidence["reasons"] else "")


async def run_and_print(shared, algorithm, agent_names=None, llm_client=None, output_dir=Path("benchmarks")):
    report = await run(shared, algorithm, agent_names=agent_names, llm_client=llm_client, output_dir=output_dir)
    print_report(report)
    return report


def print_report(report: BenchmarkReport) -> None:
    print(f"\n=== Benchmark: {report.algorithm} (v{report.version}) ===\n")
    print(f"mu = {report.mu:.4f} (target > {report.evidence['threshold']:.4f}) {_threshold_status(report.evidence)}\n")
    print(f"-- 1. Ranking of agents ({len(report.rows)}) --\n")
    print(report.ranking_table_md)
    print(f"-- 2. Best code overall: {report.top().agent} --\n")
    print("-- 3. Original vs evolved --\n")
    print(report.comparison_table_md)
    print(f"Chart saved to:  {report.chart_path}")
    print(f"Report saved to: {report.report_path}")


def _build_ranking_table(rows: list) -> str:
    headers = ["Rank", "Agent", "Performance (P)", "P* (novelty-free)", "Novelty (N)", "Accuracy (Ma)",
               "Loss (L)", "Time (T)", "A", "Resource (R)", "SE", "RE", "EL", "Reward (Q)"]
    table_rows = []
    for r in rows:
        if r.evaluated and r.trait_set is not None:
            p, s = r.trait_set.primary, r.trait_set.secondary
            table_rows.append([
                str(r.rank_position), r.agent, f"{s.performance:.6g}", f"{s.task_performance:.6g}",
                f"{s.novelty:.4f}", f"{p.model_accuracy:.4f}", f"{p.loss:.4f}", f"{s.time:.4f}",
                f"{s.accuracy_term:.4f}", f"{s.resource:.3f}", str(p.syntax_errors), str(p.runtime_errors),
                str(p.logical_errors), str(r.reward),
            ])
        else:
            table_rows.append([str(r.rank_position), r.agent, "not evaluated"] + ["-"] * 10 + ["0"])
    return _markdown_table(headers, table_rows)


def _build_comparison_table(top: AgentRow, original: TraitSet, reference: TraitSet | None, evidence: dict) -> str:
    """P* (novelty factored out) is the only column comparable across rows: the
    original is defined to have novelty 1.0 while candidates score about 0.2."""
    headers = ["Code", "P* (novelty-free)", "Accuracy (Ma)", "Loss (L)", "Time (T)", "Memory (Mu)", "CPU (K)"]

    def row(label: str, t: TraitSet, base_accuracy: float | None = None) -> list[str]:
        acc = f"{t.accuracy:.4f}"
        if base_accuracy is not None and base_accuracy > 1e-9:
            acc += f" ({(t.accuracy - base_accuracy) / base_accuracy * 100:+.1f}%)"
        return [label, f"{t.secondary.task_performance:.6g}", acc, f"{t.primary.loss:.4f}",
                f"{t.secondary.time:.4f}", f"{t.primary.memory_usage:.1f} MB", f"{t.primary.compute_cost:.3f} s"]

    rows = [row("Original code (given to the agents)", original)]
    if reference is not None:
        rows.append(row("scikit-learn defaults (reference)", reference, original.accuracy))
    rows.append(row(f"Evolved code ({top.agent})", top.trait_set, original.accuracy))
    table = _markdown_table(headers, rows)
    table += f"\nmu = P2/(P1+P2) = **{evidence['mu']:.4f}** {_threshold_status(evidence)}\n"
    if reference is not None:
        beats = top.trait_set.secondary.task_performance > reference.secondary.task_performance
        acc_beats = top.trait_set.accuracy >= reference.accuracy - 1e-9
        table += (
            f"\nAgainst scikit-learn defaults: P* {'higher' if beats else 'lower'}, "
            f"accuracy {'at least equal' if acc_beats else 'lower'}.\n"
        )
    return table


def _markdown_table(headers: list, rows: list) -> str:
    out = "| " + " | ".join(headers) + " |\n|" + " --- |" * len(headers) + "\n"
    for r in rows:
        out += "| " + " | ".join(r) + " |\n"
    return out


async def _annotate_winner(llm_client: LLMClient | None, algorithm: str, original: TraitSet, winner: TraitSet) -> str:
    winning_code = winner.code
    if llm_client is None or not winning_code.strip():
        return winning_code
    prompt = (
        f"Here is the ORIGINAL baseline implementation of the '{algorithm}' algorithm:\n\n{original.code}\n\n"
        f"Here is an EVOLVED version. Measured traits, evolved vs original:\n"
        f"  model accuracy (Ma): {winner.accuracy:.4f} vs {original.accuracy:.4f}\n"
        f"  loss (L):            {winner.primary.loss:.4f} vs {original.primary.loss:.4f}\n"
        f"  time (T):            {winner.secondary.time:.4f}s vs {original.secondary.time:.4f}s\n"
        f"  performance (P*):    {winner.secondary.task_performance:.6g} vs {original.secondary.task_performance:.6g}\n\n"
        f"{winning_code}\n\n"
        "Return the EVOLVED code verbatim, but add a `# IMPROVED: ...` comment directly above each line or "
        "block that differs from the original, naming which trait that change improved and why. Do not change "
        "any actual code. Respond with ONLY the annotated Python code, no markdown fences, no extra commentary."
    )
    try:
        response = await llm_client.generate(system_instruction="", history=[{"role": "user", "text": prompt}])
    except (LLMError, RuntimeError) as e:
        log.warning("[benchmark:%s] annotation failed (%s); using the plain code", algorithm, e)
        return winning_code
    annotated = response.text.strip()
    if annotated.startswith("```"):
        lines = annotated.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        annotated = "\n".join(lines).strip()
    return annotated or winning_code


def _render_chart(path: Path, rows: list, original: TraitSet, reference: TraitSet | None, top: AgentRow,
                  algorithm: str) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 5))
    _render_ranking_panel(left, rows, algorithm)
    _render_comparison_panel(right, original, reference, top)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _render_ranking_panel(ax, rows: list, algorithm: str) -> None:
    names = [r.agent for r in rows]
    performances = [r.trait_set.performance if (r.evaluated and r.trait_set) else 0.0 for r in rows]
    colors = ["#1e78dc" if r.evaluated else "#cccccc" for r in rows]
    bars = ax.bar(range(len(rows)), performances, color=colors)
    ax.set_title(f"Agent ranking for {algorithm.replace('_', ' ')}")
    ax.set_xlabel("Agents")
    ax.set_ylabel("Performance (P)")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels(names, rotation=15 if len(rows) > 4 else 0)
    ax.bar_label(bars, labels=[f"{p:.3g}" if r.evaluated else "N/E" for r, p in zip(rows, performances, strict=True)], fontsize=8)
    if any(performances):
        ax.set_ylim(0, max(performances) * 1.25)


def _render_comparison_panel(ax, original: TraitSet, reference: TraitSet | None, top: AgentRow) -> None:
    labels = ["Original"]
    values = [original.secondary.task_performance]
    colors = ["#808080"]
    if reference is not None:
        labels.append("sklearn defaults")
        values.append(reference.secondary.task_performance)
        colors.append("#c0a000")
    labels.append("Evolved")
    values.append(top.trait_set.secondary.task_performance)
    colors.append("#00a550")
    bars = ax.bar(labels, values, color=colors)
    ax.set_title("Novelty-free performance P*")
    ax.set_ylabel("P*")
    ax.set_ylim(0, max(max(values) * 1.25, 1e-9))
    ax.bar_label(bars, labels=[f"{v:.3g}" for v in values], fontsize=9)


def write_summary(run_dir: Path, report: BenchmarkReport) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "summary.json"
    path.write_text(json.dumps(report.summary(), indent=2), encoding="utf-8")
    return path
