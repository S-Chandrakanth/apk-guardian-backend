"""Small in-memory limiter for failed logins.

Limits: it lives in one process's memory, so it resets on restart and is not shared between
several server processes. It slows password guessing in this prototype; it is not a
replacement for a proper gateway or WAF in production.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from typing import Callable, Deque


class LoginRateLimiter:
    def __init__(
        self,
        max_failures: int,
        window_seconds: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_failures = max_failures
        self.window = window_seconds
        self._clock = clock
        self._failures: dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _prune(self, key: str, now: float) -> Deque[float]:
        events = self._failures[key]
        while events and now - events[0] > self.window:
            events.popleft()
        if not events:
            self._failures.pop(key, None)
            return self._failures[key]
        return events

    def retry_after(self, key: str) -> int:
        """Seconds the caller must wait, or 0 if an attempt is allowed."""
        with self._lock:
            now = self._clock()
            events = self._prune(key, now)
            if len(events) >= self.max_failures:
                return max(1, int(self.window - (now - events[0])) + 1)
            return 0

    def record_failure(self, key: str) -> None:
        with self._lock:
            now = self._clock()
            self._prune(key, now).append(now)

    def reset(self, key: str) -> None:
        with self._lock:
            self._failures.pop(key, None)
