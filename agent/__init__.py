from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from . import evaluator, evolution, memory, reinforcement
from .config import RunConfig
from .llm_client import LLMClient, LLMError, ModelRoles, create_client
from .prompts import build_system_prompt
from .rate_limiter import RateLimiter
from .traits import TraitSet

log = logging.getLogger(__name__)


@dataclass
class SharedState:
    """Everything the agents of one run share."""

    config: RunConfig = field(default_factory=RunConfig)
    dataset_note: str = ""
    run_dir: Path = field(default_factory=lambda: Path("runs") / "adhoc")

    # The original code (CO), measured once and sequentially before any agent
    # starts. Every mu in the run is computed against this one measurement.
    baseline: TraitSet | None = None
    # scikit-learn defaults for the same algorithm, when known. Reported next
    # to the baseline so a reader can tell "recovered a sane config" from
    # "beat the library".
    reference: TraitSet | None = None

    memory: memory.AgentMemory = field(default_factory=memory.AgentMemory)
    memory_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    agent_evaluations: dict = field(default_factory=dict)
    eval_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    rate_limiter: RateLimiter | None = None
    samples_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        if self.rate_limiter is None:
            self.rate_limiter = RateLimiter(self.config.llm_rpm)

    async def record_sample(self, record: dict) -> None:
        """Appends one evaluated sample to `runs/<id>/samples.jsonl`."""
        async with self.samples_lock:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            with open(self.run_dir / "samples.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")


_TOOLS = [
    {
        "name": "evolution",
        "description": (
            "Generates, scores and mutates candidate implementations across islands and "
            "generations, and remembers the best one. The run configuration caps generations, "
            "population_size and samples_per_population."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "generations": {"type": "integer"},
                "population_size": {"type": "integer"},
                "samples_per_population": {"type": "integer"},
            },
        },
    },
    {
        "name": "evaluator",
        "description": "Runs your evolved candidate for real and scores its traits and rank",
        "parameters": {"type": "object", "properties": {"candidate": {"type": "string"}}},
    },
    {
        "name": "reinforcement",
        "description": "Fetches your reward, rank position, and the top agents' trait breakdowns",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "memory",
        "description": (
            "Reads from the shared memory store. Pass a JSON string such as "
            '{"action": "read", "category": "code", "algorithm": "<name>"} or, to store code, '
            '{"action": "write", "code": "...", "category": "code"}.'
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
]


class Agent:
    def __init__(self, name: str, algorithm: str, dataset_name: str, roles: ModelRoles, shared: SharedState) -> None:
        self.name = name
        self.algorithm = algorithm
        self.dataset_name = dataset_name
        self.shared = shared
        self.roles = roles

        self.chat_client: LLMClient = create_client(
            roles.chat_provider, roles.chat_api_key, roles.chat_model, shared.rate_limiter
        )
        if roles.same_endpoint():
            self.evolution_client: LLMClient = self.chat_client
            self._owns_evolution_client = False
        else:
            self.evolution_client = create_client(
                roles.evolution_provider, roles.evolution_api_key, roles.evolution_model, shared.rate_limiter
            )
            self._owns_evolution_client = True

        self.history: list[dict] = []
        self.tools = _TOOLS
        self.system_instruction = build_system_prompt(name, algorithm, dataset_name)
        self.turns_used = 0
        self.tool_calls: list[str] = []

    async def aclose(self) -> None:
        await self.chat_client.aclose()
        if self._owns_evolution_client:
            await self.evolution_client.aclose()

    async def send(self, message: str) -> str:
        """Runs the tool-calling loop until the model stops calling tools or the
        turn budget (`config.max_agent_turns`) is spent."""
        log.info("[%s] sending task, awaiting the LLM", self.name)
        self.history.append({"role": "user", "text": message})
        budget = self.shared.config.max_agent_turns

        while True:
            if self.turns_used >= budget:
                log.warning("[%s] turn budget of %d spent; ending the loop", self.name, budget)
                return await self._closing_summary()
            self.turns_used += 1

            response = await self.chat_client.generate(self.system_instruction, self.history, self.tools)

            if not response.function_calls:
                self.history.append({"role": "assistant", "text": response.text, "raw": response.raw})
                log.info("[%s] done after %d turn(s), %d tool call(s)", self.name, self.turns_used, len(self.tool_calls))
                return response.text

            self.history.append(
                {"role": "assistant", "text": response.text, "function_calls": response.function_calls, "raw": response.raw}
            )
            for fc in response.function_calls:
                fc_name = fc.get("name", "")
                fc_args = fc.get("args", {})
                log.info("[%s] tool %s %s", self.name, fc_name, json.dumps(fc_args)[:160])
                result = await self._call_tool(fc_name, fc_args)
                self.tool_calls.append(fc_name)
                self.history.append({"role": "tool", "id": fc.get("id", ""), "name": fc_name, "result": result})

    async def _closing_summary(self) -> str:
        self.history.append({
            "role": "user",
            "text": "Your turn budget is spent. Reply in plain text with what you achieved and your final rank.",
        })
        try:
            response = await self.chat_client.generate(self.system_instruction, self.history, tools=None)
            return response.text or "(turn budget spent)"
        except LLMError as e:
            return f"(turn budget spent; closing summary failed: {e})"

    async def _call_tool(self, name: str, args: dict) -> str:
        try:
            if name == "evolution":
                return await evolution.run(args, self.name, self.algorithm, self.evolution_client, self.shared)
            if name == "evaluator":
                return await evaluator.run(args, self.shared, self.name, self.algorithm, self.evolution_client)
            if name == "reinforcement":
                return await reinforcement.run(args, self.shared, self.name)
            if name == "memory":
                async with self.shared.memory_lock:
                    return memory.run(self.shared.memory, args)
            return json.dumps({"status": "error", "message": f"no such tool: {name}"})
        except (memory.MemoryError, RuntimeError, ValueError) as e:
            # A tool failure becomes a result the agent can react to, not the end of its turn.
            log.warning("[%s] tool %s failed: %s", self.name, name, e)
            return json.dumps({"status": "error", "message": f"{type(e).__name__}: {e}"})
