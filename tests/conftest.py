from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import SharedState  # noqa: E402
from agent.config import RunConfig  # noqa: E402
from agent.llm_client import LLMClient, LLMResponse  # noqa: E402

# A tiny dataset and two candidates that need only the standard library, so
# sandbox tests take a fraction of a second each.
TINY_CSV = "a,b,target\n" + "\n".join(f"{i},{i % 3},{i % 2}" for i in range(20)) + "\n"

SLOW_WEAK = '''
def run(data_path: str) -> dict:
    import csv, time
    rows = list(csv.DictReader(open(data_path)))
    t0 = time.perf_counter(); time.sleep(0.10); tt = time.perf_counter() - t0
    t1 = time.perf_counter(); time.sleep(0.02); ti = time.perf_counter() - t1
    return {"accuracy": 0.5, "loss": 0.9, "training_time": tt, "inference_time": ti}
'''

FAST_STRONG = '''
def run(data_path: str) -> dict:
    import csv, time
    rows = list(csv.DictReader(open(data_path)))
    t0 = time.perf_counter(); time.sleep(0.005); tt = time.perf_counter() - t0
    t1 = time.perf_counter(); ti = time.perf_counter() - t1
    return {"accuracy": 0.9, "loss": 0.2, "training_time": tt, "inference_time": ti}
'''

# Worse accuracy and no faster than the baseline. (A worse model that merely
# reports a tiny time would outrank a better one under P = Ma*N*A/(T*R); that
# is a property of the paper's formula, see docs/REVIEW.md section 1.1.)
FAST_WEAK = '''
def run(data_path: str) -> dict:
    return {"accuracy": 0.3, "loss": 1.5, "training_time": 0.12, "inference_time": 0.02}
'''


class FakeLLM(LLMClient):
    """Returns canned code for generation prompts and no tool calls for chat."""

    provider_name = "fake"

    def __init__(self, code: str = FAST_STRONG):
        self.code = code
        self.calls = 0

    async def generate(self, system_instruction, history, tools=None):
        self.calls += 1
        return LLMResponse(text=f"```python\n{self.code}\n```")

    async def aclose(self):
        pass


@pytest.fixture
def config():
    return RunConfig(agents=2, generations=1, islands=1, samples_per_island=1, sandbox_timeout=20,
                     sandbox_repeats=1, llm_rpm=600, logic_check=False)


@pytest.fixture
def shared(tmp_path, config):
    return SharedState(config=config, run_dir=tmp_path / "run")
