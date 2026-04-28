"""Per-user rate limiter for agent tasks."""
from __future__ import annotations
import time
from collections import defaultdict


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self._max = max_requests
        self._window = window_seconds
        self._log: dict[int, list[float]] = defaultdict(list)

    def check(self, user_id: int) -> bool:
        """Returns True if the request is allowed."""
        if self._max <= 0:
            return True
        now = time.monotonic()
        cutoff = now - self._window
        timestamps = [t for t in self._log[user_id] if t > cutoff]
        self._log[user_id] = timestamps
        if len(timestamps) >= self._max:
            return False
        self._log[user_id].append(now)
        return True
