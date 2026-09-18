import pytest

from agent import sandbox
from tests.conftest import FAST_STRONG, SLOW_WEAK, TINY_CSV


async def test_ok_run_reports_measured_traits():
    result = await sandbox.run(FAST_STRONG, TINY_CSV, time_limit=20)
    assert result.ok
    p = result.primary
    assert p.model_accuracy == 0.9 and p.loss == 0.2
    assert p.training_time > 0.0
    assert p.memory_usage > 1.0, "peak RSS must be measured inside the child"
    assert p.compute_cost >= sandbox.MIN_CPU_SECONDS, "CPU seconds must be measured inside the child"
    assert result.time_samples and len(result.time_samples) == 1


async def test_repeats_return_the_median_run():
    result = await sandbox.run(SLOW_WEAK, TINY_CSV, time_limit=20, repeats=3)
    assert result.ok and len(result.time_samples) == 3
    assert result.time_samples == sorted(result.time_samples)
    chosen = result.primary.training_time + result.primary.inference_time
    assert chosen == pytest.approx(result.time_samples[1], rel=1e-9)


async def test_syntax_error_is_caught_without_spawning():
    result = await sandbox.run("def run(data_path):\n    return (\n", TINY_CSV, time_limit=20)
    assert result.primary.syntax_errors == 1 and "SyntaxError" in result.stderr
    result = await sandbox.run("x = 1\n", TINY_CSV, time_limit=20)
    assert result.primary.syntax_errors == 1 and "run(data_path)" in result.stderr


async def test_runtime_error_and_timeout():
    result = await sandbox.run("def run(p):\n    raise ValueError('boom')\n", TINY_CSV, time_limit=20)
    assert result.primary.runtime_errors == 1 and "boom" in result.stderr

    result = await sandbox.run("def run(p):\n    import time\n    time.sleep(30)\n", TINY_CSV, time_limit=1.0)
    assert result.primary.runtime_errors == 1 and "timed out" in result.stderr
    assert result.primary.training_time == 1.0


async def test_bare_float_return_is_accepted():
    result = await sandbox.run("def run(p):\n    return 0.75\n", TINY_CSV, time_limit=20)
    assert result.ok and result.primary.model_accuracy == 0.75 and result.primary.loss == pytest.approx(0.25)
