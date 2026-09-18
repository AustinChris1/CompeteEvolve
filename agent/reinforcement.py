from __future__ import annotations

import json

from . import evaluator
from . import traits as traits_module
from .prompts import build_feedback_prompt

TOP_N_IN_FEEDBACK = 3


async def run(args: dict, shared, agent_name: str) -> str:
    """The `reinforcement` tool: rank, reward and the top agents' trait tables."""
    positions = await evaluator.compute_rank_positions(shared)
    n = len(positions)

    async with shared.eval_lock:
        snapshot = dict(shared.agent_evaluations)

    leaderboard = [
        {
            "agent": name,
            "rank_position": rp,
            "reward": traits_module.reward_for_rank(rp),
            "trait": snapshot[name].trait if name in snapshot else 0.0,
        }
        for name, rp in positions
    ]
    leaderboard.sort(key=lambda e: e["rank_position"])

    self_entry = next((e for e in leaderboard if e["agent"] == agent_name), None)
    if self_entry is None:
        raise RuntimeError(f"agent '{agent_name}' has not been evaluated yet")

    leaderboard_markdown = _render_leaderboard(leaderboard[:TOP_N_IN_FEEDBACK], snapshot)
    feedback_prompt = build_feedback_prompt(self_entry["rank_position"], n, self_entry["reward"], leaderboard_markdown)

    return json.dumps({
        "status": "ok",
        "agent": agent_name,
        "rank_position": self_entry["rank_position"],
        "reward": self_entry["reward"],
        "agent_count": n,
        "feedback_prompt": feedback_prompt,
        "leaderboard": [{**e, "trait": round(e["trait"], 9)} for e in leaderboard],
    })


def _render_leaderboard(top_entries: list[dict], snapshot: dict) -> str:
    blocks = []
    for entry in top_entries:
        evaluation = snapshot.get(entry["agent"])
        if evaluation is None:
            continue
        blocks.append(traits_module.format_traits_markdown(
            f"{entry['agent']}: rank {entry['rank_position']}, reward {entry['reward']}", evaluation.trait_set
        ))
    return "\n".join(blocks) if blocks else "(no evaluated agents yet)"
