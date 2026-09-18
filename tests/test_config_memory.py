import json

import pytest

from agent import memory
from agent.config import RunConfig


def test_config_env_and_overrides(monkeypatch):
    monkeypatch.setenv("CE_GENERATIONS", "2")
    monkeypatch.setenv("LLM_RPM", "7")
    monkeypatch.setenv("LOGIC_CHECK", "0")
    cfg = RunConfig.from_env(islands=1)
    assert cfg.generations == 2 and cfg.llm_rpm == 7 and cfg.islands == 1 and cfg.logic_check is False
    assert "generations=2" in cfg.describe()
    with pytest.raises(ValueError):
        RunConfig.from_env(agents=0)
    with pytest.raises(ValueError):
        RunConfig.from_env(mu_threshold=1.5)


def test_memory_roundtrip_and_filters():
    mem = memory.AgentMemory()
    memory.remember_text(mem, "def run(p): return 0.5", category="code", algorithm="knn")
    memory.remember_text(mem, "a,b,target\n1,2,0\n", category="dataset", algorithm="knn")
    memory.remember_text(mem, "def run(p): return 0.9", category="code", algorithm="knn", agent="agent-1")

    out = json.loads(memory.run(mem, {"action": "read", "category": "code", "algorithm": "knn", "top_n": 10}))
    assert out["status"] == "ok" and len(out["results"]) == 2
    mine = json.loads(memory.run(mem, {"action": "read", "category": "code", "agent": "agent-1"}))
    assert mine["results"][0]["agent"] == "agent-1"


def test_bare_query_is_a_read_not_a_write():
    mem = memory.AgentMemory()
    memory.remember_text(mem, "def run(p): return 0.5", category="code", algorithm="knn")
    before = len(mem.ranked_store.entries)
    out = json.loads(memory.run(mem, {"query": "knn"}))
    assert out["status"] == "ok" and out["results"][0]["algorithm"] == "knn"
    out = json.loads(memory.run(mem, {"query": "logistic_regression"}))
    assert out["status"] == "ok"  # falls back to an unfiltered read
    assert len(mem.ranked_store.entries) == before, "a lookup must never store anything"
    written = json.loads(memory.run(mem, {"query": json.dumps({"action": "write", "code": "x = 1"})}))
    assert written["status"] == "ok" and len(mem.ranked_store.entries) == before + 1


def test_dataset_writes_are_validated():
    mem = memory.AgentMemory()
    with pytest.raises(memory.MemoryError):
        memory.run(mem, {"action": "write", "code": "a,b\n1\n", "category": "dataset"})
