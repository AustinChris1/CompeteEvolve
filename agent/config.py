"""One place for every knob a run depends on.

Values come from defaults, then environment variables, then explicit
overrides (CLI flags). The resolved config is printed at start-up and written
into the run directory so a report can always be traced back to its settings.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields

# mu = P2 / (P1 + P2) for a 25 percent improvement: 1.25 / 2.25.
MU_THRESHOLD_DEFAULT = 1.25 / 2.25


@dataclass(frozen=True)
class RunConfig:
    agents: int = 5
    generations: int = 3
    islands: int = 4
    samples_per_island: int = 5
    mu_threshold: float = MU_THRESHOLD_DEFAULT
    max_agent_turns: int = 12
    sandbox_timeout: float = 60.0
    sandbox_concurrency: int = 1
    sandbox_repeats: int = 1
    llm_rpm: int = 10
    logic_check: bool = True

    _ENV = {
        "agents": ("CE_AGENTS",),
        "generations": ("CE_GENERATIONS",),
        "islands": ("CE_ISLANDS",),
        "samples_per_island": ("CE_SAMPLES_PER_ISLAND",),
        "mu_threshold": ("CE_MU_THRESHOLD",),
        "max_agent_turns": ("CE_MAX_AGENT_TURNS",),
        "sandbox_timeout": ("CE_SANDBOX_TIMEOUT", "SANDBOX_TIMEOUT_SECS"),
        "sandbox_concurrency": ("CE_SANDBOX_CONCURRENCY", "SANDBOX_CONCURRENCY"),
        "sandbox_repeats": ("CE_SANDBOX_REPEATS", "SANDBOX_REPEATS"),
        "llm_rpm": ("CE_LLM_RPM", "LLM_RPM", "GEMINI_RPM"),
        "logic_check": ("CE_LOGIC_CHECK", "LOGIC_CHECK"),
    }

    @classmethod
    def from_env(cls, **overrides) -> RunConfig:
        values = {}
        for f in fields(cls):
            if f.name.startswith("_"):
                continue
            raw = next((os.environ[n] for n in cls._ENV.get(f.name, ()) if os.environ.get(n)), None)
            if raw is not None:
                values[f.name] = _coerce(f.type, raw, f.name)
        values.update({k: v for k, v in overrides.items() if v is not None})
        cfg = cls(**values)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        positive = ["agents", "generations", "islands", "samples_per_island", "max_agent_turns",
                    "sandbox_concurrency", "sandbox_repeats", "llm_rpm"]
        for name in positive:
            if getattr(self, name) < 1:
                raise ValueError(f"config.{name} must be at least 1, got {getattr(self, name)}")
        if self.sandbox_timeout <= 0:
            raise ValueError("config.sandbox_timeout must be positive")
        if not 0.0 < self.mu_threshold < 1.0:
            raise ValueError("config.mu_threshold must be in (0, 1)")

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if not k.startswith("_")}

    def describe(self) -> str:
        return ", ".join(f"{k}={v}" for k, v in self.to_dict().items())


def _coerce(type_name, raw: str, name: str):
    type_name = str(type_name)
    try:
        if "bool" in type_name:
            return raw.strip().lower() not in ("0", "false", "no", "off", "")
        if "int" in type_name:
            return int(raw)
        if "float" in type_name:
            return float(raw)
    except ValueError as e:
        raise ValueError(f"bad value for {name}: {raw!r}") from e
    return raw
