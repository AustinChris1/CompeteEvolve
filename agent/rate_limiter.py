from __future__ import annotations

import asyncio
import logging
import time
from collections import deque

log = logging.getLogger(__name__)


class RateLimiter:
    """Sliding-window limiter shared by every LLM call in a run."""

    def __init__(self, max_per_minute: int) -> None:
        self.max_per_minute = max(1, int(max_per_minute))
        self._window: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            sleep_for: float | None = None
            async with self._lock:
                now = time.monotonic()
                while self._window and now - self._window[0] >= 60.0:
                    self._window.popleft()
                if len(self._window) < self.max_per_minute:
                    self._window.append(now)
                else:
                    sleep_for = max(0.0, 60.0 - (now - self._window[0])) + 0.1
            if sleep_for is None:
                return
            log.debug("rate limiter at capacity (%d/min); waiting %.1fs", self.max_per_minute, sleep_for)
            await asyncio.sleep(sleep_for)
