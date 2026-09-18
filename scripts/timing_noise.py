"""Measure one baseline sequentially and under concurrent load.

Shows how much T (and therefore P) depends on what else is running. Run from
the repository root:

    python scripts/timing_noise.py logistic_regression
"""

from __future__ import annotations

import asyncio
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("CE_SANDBOX_CONCURRENCY", "32")  # let the fan-out actually overlap

import sklearn_algorithms  # noqa: E402
from agent import sandbox, traits  # noqa: E402


async def measure(code: str, csv: str):
    result = await sandbox.run(code, csv, 60)
    return traits.build(result.primary, code, code, None, result.stderr, is_reference=True)


def summarise(label: str, results) -> None:
    t = [r.secondary.time for r in results]
    p = [r.secondary.task_performance for r in results]
    errors = sum(1 for r in results if r.has_errors())
    print(
        f"{label:<26} n={len(results):<3} errors={errors:<2} "
        f"T median={statistics.median(t):.4f}s min={min(t):.4f} max={max(t):.4f}   "
        f"P* median={statistics.median(p):.4g} min={min(p):.4g} max={max(p):.4g}"
    )


async def main() -> None:
    algorithm = sys.argv[1] if len(sys.argv) > 1 else "logistic_regression"
    spec = sklearn_algorithms.get_spec(algorithm)
    csv = sklearn_algorithms.dataset_csv(algorithm)

    sequential = []
    for _ in range(5):
        sequential.append(await measure(spec.baseline_code, csv))
    summarise(f"{algorithm} sequential x5", sequential)

    for fan_out in (5, 10, 20):
        started = time.perf_counter()
        concurrent = await asyncio.gather(*(measure(spec.baseline_code, csv) for _ in range(fan_out)))
        summarise(f"{algorithm} concurrent x{fan_out}", concurrent)
        print(f"   wall {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
