"""Runs one candidate in a fresh subprocess and measures its primary traits.

Measurement rules that keep the numbers comparable between candidates:

- Runs are serialised through a semaphore (`sandbox_concurrency`, default 1),
  so wall-clock times are not distorted by other candidates.
- CPU time (K) and peak RSS (Mu) are reported by the child itself at exit,
  not sampled from outside, so they cannot be lost to a sampler failure.
- With `repeats > 1` the candidate is executed several times and the run with
  the median wall time is returned, with every timing kept in `time_samples`.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import statistics
import tempfile
from dataclasses import dataclass, field

import psutil

from .traits import PrimaryTraits

log = logging.getLogger(__name__)

_MARKER = "__TRAITS__"
MIN_CPU_SECONDS = 1e-3
MIN_PEAK_RSS_MB = 1.0


def default_time_limit() -> float:
    raw = os.environ.get("CE_SANDBOX_TIMEOUT") or os.environ.get("SANDBOX_TIMEOUT_SECS") or "60"
    try:
        return float(raw)
    except ValueError:
        return 60.0


def default_concurrency() -> int:
    raw = os.environ.get("CE_SANDBOX_CONCURRENCY") or os.environ.get("SANDBOX_CONCURRENCY") or "1"
    try:
        return max(1, int(raw))
    except ValueError:
        return 1


def _python_bin() -> str:
    return os.environ.get("SANDBOX_PYTHON") or ("python" if os.name == "nt" else "python3")


@dataclass
class SandboxResult:
    primary: PrimaryTraits
    stderr: str = ""
    time_samples: list[float] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.primary.syntax_errors == 0 and self.primary.runtime_errors == 0


_HARNESS = '''
import sys, os, json, time, traceback

CANDIDATE_CODE

def _peak_rss_mb():
    try:
        import psutil
        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None) or info.rss
        return peak / (1024.0 * 1024.0)
    except Exception:
        pass
    try:
        import resource
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return (maxrss if sys.platform == "darwin" else maxrss * 1024.0) / (1024.0 * 1024.0)
    except Exception:
        return 0.0

def _main():
    data_path = sys.argv[1]
    wall_start = time.perf_counter()
    try:
        result = run(data_path)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
    wall_elapsed = time.perf_counter() - wall_start

    if isinstance(result, dict):
        accuracy = float(result.get("accuracy", 0.0))
        loss = result.get("loss", None)
        training_time = result.get("training_time", None)
        inference_time = result.get("inference_time", None)
    else:
        accuracy = float(result)
        loss = training_time = inference_time = None

    if training_time is None:
        training_time, inference_time = wall_elapsed, 0.0
    if inference_time is None:
        inference_time = 0.0
    if loss is None:
        loss = 1.0 - accuracy

    t = os.times()
    cpu_seconds = time.process_time() + t.children_user + t.children_system
    print(_MARKER_ + json.dumps({
        "accuracy": accuracy,
        "loss": float(loss),
        "training_time": float(training_time),
        "inference_time": float(inference_time),
        "wall_elapsed": wall_elapsed,
        "cpu_seconds": cpu_seconds,
        "peak_rss_mb": _peak_rss_mb(),
    }))

if __name__ == "__main__":
    _main()
'''.replace("_MARKER_", repr(_MARKER))


_semaphore_state: tuple[asyncio.AbstractEventLoop, asyncio.Semaphore] | None = None


def _semaphore() -> asyncio.Semaphore:
    """One semaphore per event loop. The loop is held by reference (not id), so
    a new loop at a recycled address never inherits a stale semaphore."""
    global _semaphore_state
    loop = asyncio.get_running_loop()
    if _semaphore_state is None or _semaphore_state[0] is not loop:
        _semaphore_state = (loop, asyncio.Semaphore(default_concurrency()))
    return _semaphore_state[1]


def check_syntax(candidate_code: str) -> str | None:
    try:
        compile(candidate_code, "candidate.py", "exec")
    except SyntaxError as e:
        return f"SyntaxError: {e.msg} (line {e.lineno}, col {e.offset})"
    if "def run" not in candidate_code:
        return "candidate does not define run(data_path)"
    return None


async def run(
    candidate_code: str,
    dataset_csv: str,
    time_limit: float | None = None,
    repeats: int = 1,
) -> SandboxResult:
    """Measures `candidate_code` against `dataset_csv`.

    Syntax is checked in-process before anything is spawned. The subprocess
    runs under the concurrency semaphore. Any failed repeat is returned
    immediately; otherwise the repeat with the median wall time is returned.
    """
    time_limit = time_limit if time_limit is not None else default_time_limit()
    repeats = max(1, int(repeats))

    syntax_error = check_syntax(candidate_code)
    if syntax_error:
        return SandboxResult(PrimaryTraits(syntax_errors=1, loss=1.0), stderr=syntax_error)

    tmp_dir = tempfile.mkdtemp(prefix="sandbox_")
    try:
        data_path = os.path.join(tmp_dir, "data.csv")
        script_path = os.path.join(tmp_dir, "candidate.py")
        with open(data_path, "w", encoding="utf-8") as f:
            f.write(dataset_csv)
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(_HARNESS.replace("CANDIDATE_CODE", candidate_code))

        async with _semaphore():
            results: list[SandboxResult] = []
            for _ in range(repeats):
                result = await _execute(script_path, data_path, tmp_dir, time_limit)
                if not result.ok:
                    return result
                results.append(result)
        return _median_by_time(results)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


async def _execute(script_path: str, data_path: str, cwd: str, time_limit: float) -> SandboxResult:
    try:
        proc = await asyncio.create_subprocess_exec(
            _python_bin(), script_path, data_path,
            cwd=cwd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as e:
        raise RuntimeError(
            f"could not launch a Python interpreter ({e}); set SANDBOX_PYTHON to the interpreter path"
        ) from e

    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=time_limit)
    except asyncio.TimeoutError:
        _kill_tree(proc.pid)
        await proc.wait()
        return SandboxResult(
            PrimaryTraits(runtime_errors=1, training_time=time_limit, loss=1.0),
            stderr=f"candidate timed out after {time_limit:.1f}s",
        )

    stderr_text = stderr.decode("utf-8", errors="replace")
    if proc.returncode != 0:
        return SandboxResult(PrimaryTraits(runtime_errors=1, loss=1.0), stderr=stderr_text.strip()[-2000:])

    payload = None
    for line in stdout.decode("utf-8", errors="replace").splitlines():
        if line.startswith(_MARKER):
            payload = line[len(_MARKER):]
    if payload is None:
        return SandboxResult(
            PrimaryTraits(runtime_errors=1, loss=1.0),
            stderr="candidate produced no trait output (did it print or exit early?)",
        )
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as e:
        return SandboxResult(PrimaryTraits(runtime_errors=1, loss=1.0), stderr=f"trait output was not JSON: {e}")

    accuracy = max(0.0, min(1.0, float(parsed.get("accuracy", 0.0))))
    loss = float(parsed.get("loss", 1.0 - accuracy))
    if loss != loss or loss < 0.0:
        loss = 1.0 - accuracy
    training_time = max(0.0, float(parsed.get("training_time", 0.0)))
    inference_time = max(0.0, float(parsed.get("inference_time", 0.0)))

    return SandboxResult(
        PrimaryTraits(
            training_time=training_time,
            inference_time=inference_time,
            loss=loss,
            model_accuracy=accuracy,
            # Floors: process CPU time ticks at 15.6 ms on Windows and can read 0
            # for a short child, and no Python process uses under 1 MB. Without
            # them R = K x Mu collapses to its 1e-6 floor and P inflates 10^6-fold.
            memory_usage=max(MIN_PEAK_RSS_MB, float(parsed.get("peak_rss_mb", 0.0))),
            compute_cost=max(MIN_CPU_SECONDS, float(parsed.get("cpu_seconds", 0.0))),
        ),
        time_samples=[training_time + inference_time],
    )


def _median_by_time(results: list[SandboxResult]) -> SandboxResult:
    times = [r.primary.training_time + r.primary.inference_time for r in results]
    median = statistics.median(times)
    chosen = min(results, key=lambda r: abs(r.primary.training_time + r.primary.inference_time - median))
    chosen.time_samples = sorted(times)
    return chosen


def _kill_tree(pid: int) -> None:
    try:
        parent = psutil.Process(pid)
    except psutil.Error:
        return
    for child in parent.children(recursive=True):
        try:
            child.kill()
        except psutil.Error:
            pass
    try:
        parent.kill()
    except psutil.Error:
        pass


import json  # noqa: E402  (kept at the bottom so the harness string above stays readable)
