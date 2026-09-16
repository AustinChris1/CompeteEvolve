from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  
import matplotlib.pyplot as plt  

from agent import evaluator, memory, sandbox  
from agent import traits as traits_module  
from agent.llm_client import LLMClient  
from agent.logic_check import LogicChecker
from agent.traits import TraitSet


@dataclass
class AgentRow:
    agent: str
    rank_position: int
    reward: int
    trait_set: Optional[TraitSet] = None
    evaluated: bool = True


@dataclass
class BenchmarkReport:
    algorithm: str
    version: int
    rows: list            
    original_ok: bool
    mu: float
    annotated_winner_code: str
    ranking_table_md: str
    comparison_table_md: str
    chart_path: Path
    report_path: Path

    def top(self) -> AgentRow:
        return self.rows[0]


def _file_stem(algorithm: str) -> str:
    return algorithm.replace("_", "").lower()


def _next_version(output_dir: Path, algorithm: str) -> int:
    stem = _file_stem(algorithm)
    pattern = re.compile(rf"^{re.escape(stem)}(\d+)\.md$", re.IGNORECASE)
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
    agent_names: Optional[list] = None,
    llm_client: Optional[LLMClient] = None,
    output_dir: Path = Path("benchmarks"),
) -> BenchmarkReport:
    async with shared.eval_lock:
        snapshot = dict(shared.agent_evaluations)

    if not snapshot:
        raise RuntimeError(
            "no agent has been evaluated yet; run the 'evaluator' tool for at least one agent first"
        )

    positions = await evaluator.compute_rank_positions(shared)
    evaluated_names = {name for name, _ in positions}

    rows = [
        AgentRow(
            agent=name,
            rank_position=rank,
            reward=traits_module.reward_for_rank(rank),
            trait_set=snapshot[name].trait_set,
            evaluated=True,
        )
        for name, rank in positions
    ]

    # Agents that competed but never got scored still get a row.
    all_names = list(agent_names) if agent_names is not None else list(evaluated_names)
    for name in evaluated_names:
        if name not in all_names:
            all_names.append(name)
    next_rank = len(rows) + 1
    for name in all_names:
        if name in evaluated_names:
            continue
        rows.append(AgentRow(agent=name, rank_position=next_rank, reward=0, evaluated=False))
        next_rank += 1

    rows.sort(key=lambda r: r.rank_position)
    top = rows[0]
    if not top.evaluated:
        raise RuntimeError("no agent was evaluated this run; nothing to benchmark")

    # ---- the original, re-measured fresh through the same pipeline ----
    original_code = await _fetch_original_code(shared, algorithm)
    dataset_csv = await _fetch_dataset(shared, algorithm)
    logic_checker = LogicChecker(llm_client)

    sandbox_result = await sandbox.run(original_code, dataset_csv, sandbox.default_time_limit())
    primary = sandbox_result.primary
    if sandbox_result.ok:
        el, _ = await logic_checker.count(original_code)
        primary.logical_errors = el
    original = traits_module.build(
        primary, original_code, original_code, None, sandbox_result.stderr, is_reference=True
    )
    original_ok = sandbox_result.ok

    if not original_ok:
        print(
            f"[benchmark:{algorithm}] WARNING: original code failed in the sandbox "
            f"(SE={primary.syntax_errors}, RE={primary.runtime_errors}) — reporting as unmeasured."
        )

    mu = traits_module.compute_mu(original.secondary.task_performance, top.trait_set.secondary.task_performance)

    # ---- tables, annotated winner, chart, report ----
    ranking_table_md = _build_ranking_table(rows)
    comparison_table_md = _build_comparison_table(top, original, original_ok, mu)
    annotated_winner_code = await _annotate_winner(
        llm_client, algorithm, original_code, top.trait_set, original
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    version = _next_version(output_dir, algorithm)
    stem = _file_stem(algorithm)
    chart_path = output_dir / f"{stem}{version:02d}.png"
    report_path = output_dir / f"{stem}{version:02d}.md"

    _render_chart(chart_path, rows, original, top, algorithm)

    not_evaluated = [r.agent for r in rows if not r.evaluated]
    not_evaluated_note = (
        f"\n> **Note:** {', '.join(not_evaluated)} competed but never completed an `evaluator` "
        "call, so no traits could be measured for them.\n"
        if not_evaluated
        else ""
    )
    baseline_note = (
        f"\n> **Note:** the original code failed in the sandbox — its row is unmeasured. "
        f"Sandbox output: {original.stderr}\n"
        if not original_ok
        else ""
    )

    report_md = (
        f"# Benchmark report: {algorithm} (v{version})\n\n"
        f"Threshold μ = P2/(P1+P2) = **{mu:.4f}** "
        f"(target > {traits_module.MU_THRESHOLD:.4f}, i.e. a 25% improvement over the original). "
        f"{'**Met.**' if mu > traits_module.MU_THRESHOLD else '_Not met._'}\n\n"
        f"## 1. Ranking of agents ({len(rows)} agent(s))\n\n{ranking_table_md}\n"
        f"{not_evaluated_note}\n"
        f"## 2. Top ranking evolved code overall\n\n"
        f"Agent **{top.agent}** holds the best code overall, Cα (rank 1 of {len(rows)}).\n\n"
        f"{traits_module.format_traits_markdown('Winning candidate traits', top.trait_set)}\n"
        f"### Annotated winning code\n\n"
        f"`# IMPROVED:` comments mark what changed relative to the original and why it helped.\n\n"
        f"```python\n{annotated_winner_code}\n```\n\n"
        f"## 3. Original versus evolved code\n\n{comparison_table_md}\n"
        f"{baseline_note}\n"
        f"{traits_module.format_traits_markdown('Original code (CO) traits', original)}\n"
        f"![Benchmark chart]({chart_path.name})\n"
    )
    report_path.write_text(report_md, encoding="utf-8")

    return BenchmarkReport(
        algorithm=algorithm,
        version=version,
        rows=rows,
        original=original,
        original_ok=original_ok,
        mu=mu,
        annotated_winner_code=annotated_winner_code,
        ranking_table_md=ranking_table_md,
        comparison_table_md=comparison_table_md,
        chart_path=chart_path,
        report_path=report_path,
    )


async def run_and_print(shared, algorithm, agent_names=None, llm_client=None) -> BenchmarkReport:
    report = await run(shared, algorithm, agent_names=agent_names, llm_client=llm_client)
    print_report(report)
    return report


def print_report(report: BenchmarkReport) -> None:
    top = report.top()
    print(f"\n=== Benchmark: {report.algorithm} (v{report.version}) ===\n")
    print(f"mu = {report.mu:.4f} (target > {traits_module.MU_THRESHOLD:.4f})\n")
    print(f"-- 1. Ranking of agents ({len(report.rows)}) --\n")
    print(report.ranking_table_md)
    print(f"-- 2. Best code overall: {top.agent} --\n")
    print("-- 3. Original vs evolved --\n")
    print(report.comparison_table_md)
    print(f"Chart saved to:  {report.chart_path}")
    print(f"Report saved to: {report.report_path}")


def _build_ranking_table(rows: list) -> str:
    headers = [
        "Rank", "Agent", "Performance (P)", "Novelty (Nθ)", "Accuracy (Ma)", "Loss (L)",
        "Time (T)", "Aθ", "Resource (R)", "SE", "RE", "EL", "Reward (Q)",
    ]
    table_rows = []
    for r in rows:
        if r.evaluated and r.trait_set is not None:
            p, s = r.trait_set.primary, r.trait_set.secondary
            table_rows.append([
                str(r.rank_position), r.agent,
                f"{s.performance:.6g}", f"{s.novelty:.4f}", f"{p.model_accuracy:.4f}",
                f"{p.loss:.4f}", f"{s.time:.4f}", f"{s.accuracy_term:.4f}", f"{s.resource:.3f}",
                str(p.syntax_errors), str(p.runtime_errors), str(p.logical_errors), str(r.reward),
            ])
        else:
            table_rows.append(
                [str(r.rank_position), r.agent, "not evaluated"] + ["—"] * 9 + ["0"]
            )
    return _markdown_table(headers, table_rows)


def _build_comparison_table(top: AgentRow, original: TraitSet, original_ok: bool, mu: float) -> str:
    headers = ["Code", "Performance (P)", "Accuracy (Ma)", "Loss (L)", "Time (T)", "Novelty (Nθ)"]
    if original_ok:
        original_row = [
            "Original code", f"{original.performance:.6g}", f"{original.accuracy:.4f}",
            f"{original.primary.loss:.4f}", f"{original.secondary.time:.4f}",
            f"{original.novelty:.4f}",
        ]
    else:
        original_row = ["Original code", "N/A*", "N/A*", "N/A*", "N/A*", "N/A*"]

    s, p = top.trait_set.secondary, top.trait_set.primary
    if original_ok and original.accuracy > 1e-9:
        delta = (p.model_accuracy - original.accuracy) / original.accuracy * 100.0
        accuracy_cell = f"{p.model_accuracy:.4f} ({delta:+.1f}%)"
    else:
        accuracy_cell = f"{p.model_accuracy:.4f}"

    evolved_row = [
        f"Evolved code ({top.agent})", f"{s.performance:.6g}", accuracy_cell,
        f"{p.loss:.4f}", f"{s.time:.4f}", f"{s.novelty:.4f}",
    ]

    table = _markdown_table(headers, [original_row, evolved_row])
    table += f"\nμ = P2/(P1+P2) = **{mu:.4f}**\n"
    if not original_ok:
        table += "\n\\* the original code failed in the sandbox — see the note below.\n"
    return table


def _markdown_table(headers: list, rows: list) -> str:
    out = "| " + " | ".join(headers) + " |\n|" + " --- |" * len(headers) + "\n"
    for row in rows:
        out += "| " + " | ".join(row) + " |\n"
    return out


async def _annotate_winner(
    llm_client: Optional[LLMClient], algorithm: str, original_code: str,
    winner: TraitSet, original: TraitSet,
) -> str:
    """Returns the winning code with `# IMPROVED:` comments explaining
    which changes produced the measured gain. Falls back to the plain
    code if annotation isn't available — it must never block the report."""
    winning_code = winner.code
    if llm_client is None or not winning_code.strip():
        return winning_code + (
            "\n# NOTE: no LLM client was available to annotate this code.\n" if winning_code else ""
        )

    prompt = (
        f"Here is the ORIGINAL baseline implementation of the '{algorithm}' algorithm:\n\n"
        f"{original_code}\n\n"
        f"Here is an EVOLVED version. Measured traits — evolved vs original:\n"
        f"  model accuracy (Ma): {winner.accuracy:.4f} vs {original.accuracy:.4f}\n"
        f"  loss (L):            {winner.primary.loss:.4f} vs {original.primary.loss:.4f}\n"
        f"  time (T):            {winner.secondary.time:.4f}s vs {original.secondary.time:.4f}s\n"
        f"  novelty (Nθ):        {winner.novelty:.4f} vs {original.novelty:.4f}\n"
        f"  performance (P):     {winner.performance:.6g} vs {original.performance:.6g}\n\n"
        f"{winning_code}\n\n"
        "Return the EVOLVED code verbatim, but add a `# IMPROVED: ...` comment directly above "
        "each line or block that differs from the original, naming which trait that change "
        "improved and why. Do not change any actual code — only add comments. Respond with ONLY "
        "the annotated Python code, no markdown fences, no extra commentary."
    )

    try:
        response = await llm_client.generate(
            system_instruction="", history=[{"role": "user", "text": prompt}]
        )
        annotated = response.text.strip()
        if annotated.startswith("```"):
            lines = annotated.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            annotated = "\n".join(lines).strip()
        return annotated or winning_code
    except Exception as e:  # noqa: BLE001 - annotation must never block the report
        print(f"[benchmark:{algorithm}] WARNING: annotation failed ({e}); using plain code.")
        return winning_code + f"\n# NOTE: automatic annotation failed ({e}).\n"


async def _fetch_original_code(shared, algorithm: str) -> str:
    async with shared.memory_lock:
        raw = memory.run(
            shared.memory, {"action": "read", "top_n": 100, "category": "code", "algorithm": algorithm}
        )
    results = json.loads(raw).get("results", [])
    hit = next((r for r in results if r.get("agent") is None), None)
    if hit is None:
        raise RuntimeError(f"no baseline (agent-less) 'code' entry found for algorithm '{algorithm}'")
    return hit["code"]


async def _fetch_dataset(shared, algorithm: str) -> str:
    async with shared.memory_lock:
        raw = memory.run(
            shared.memory, {"action": "read", "top_n": 1, "category": "dataset", "algorithm": algorithm}
        )
    results = json.loads(raw).get("results", [])
    if not results:
        raise RuntimeError(f"no 'dataset' entry found for algorithm '{algorithm}'")
    return results[0]["code"]


def _render_chart(path: Path, rows: list, original: TraitSet, top: AgentRow, algorithm: str) -> None:
    """Two panels: per-agent performance (the paper's Figure 5-9 style
    ranking chart) and original-vs-evolved performance."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 5))
    _render_ranking_panel(left, rows, algorithm)
    _render_comparison_panel(right, original, top)
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

    labels = [f"{p:.3g}" if r.evaluated else "N/E" for r, p in zip(rows, performances)]
    ax.bar_label(bars, labels=labels, fontsize=8)
    if any(performances):
        ax.set_ylim(0, max(performances) * 1.25)


def _render_comparison_panel(ax, original: TraitSet, top: AgentRow) -> None:
    values = [original.performance, top.trait_set.performance]
    bars = ax.bar(["Original code", "Evolved code"], values, color=["#808080", "#00a550"])
    ax.set_title("Original versus evolved code")
    ax.set_ylabel("Performance (P)")
    ax.set_ylim(0, max(max(values) * 1.25, 1e-9))
    ax.bar_label(bars, labels=[f"{v:.3g}" for v in values], fontsize=9)
