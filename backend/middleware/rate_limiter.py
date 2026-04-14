"""In-memory sliding-window per-IP rate limiter and failure-based cooldown."""
from __future__ import annotations

import threading
import time
from collections import defaultdict


class SlidingWindowLimiter:
    """Thread-safe sliding-window rate limiter keyed by client identifier."""

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self._max = max_requests
        self._window = window_seconds
        self._lock = threading.Lock()
        self._requests: dict[str, list[float]] = defaultdict(list)

    def _prune(self, key: str, now: float) -> None:
        cutoff = now - self._window
        timestamps = self._requests[key]
        self._requests[key] = [t for t in timestamps if t > cutoff]

    def is_allowed(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            self._prune(key, now)
            if len(self._requests[key]) >= self._max:
                return False
            self._requests[key].append(now)
            return True

    def get_retry_after(self, key: str) -> int | None:
        now = time.monotonic()
        with self._lock:
            self._prune(key, now)
            timestamps = self._requests[key]
            if len(timestamps) < self._max:
                return None
            oldest = timestamps[0]
            return max(1, int(self._window - (now - oldest)))


class CooldownLimiter:
    """Lock out a client for *cooldown_seconds* after *max_failures* consecutive failures.

    Call ``record_failure(key)`` on every failed recognition attempt and
    ``record_success(key)`` on successes (resets the counter).
    """

    def __init__(self, max_failures: int, cooldown_seconds: int) -> None:
        self._max_failures = max_failures
        self._cooldown = cooldown_seconds
        self._lock = threading.Lock()
        self._failures: dict[str, int] = defaultdict(int)
        self._locked_until: dict[str, float] = {}

    def is_locked(self, key: str) -> bool:
        """Return True if the client is currently in cooldown."""
        now = time.monotonic()
        with self._lock:
            until = self._locked_until.get(key)
            if until is not None and now < until:
                return True
            if until is not None and now >= until:
                # Cooldown expired — reset
                self._locked_until.pop(key, None)
                self._failures[key] = 0
            return False

    def get_cooldown_remaining(self, key: str) -> int:
        """Seconds remaining in the cooldown, or 0 if not locked."""
        now = time.monotonic()
        with self._lock:
            until = self._locked_until.get(key)
            if until is not None and now < until:
                return max(1, int(until - now))
        return 0

    def record_failure(self, key: str) -> None:
        """Increment failure counter; start cooldown if threshold reached."""
        with self._lock:
            self._failures[key] += 1
            if self._failures[key] >= self._max_failures:
                self._locked_until[key] = time.monotonic() + self._cooldown

    def record_success(self, key: str) -> None:
        """Reset the failure counter and cancel any pending cooldown."""
        with self._lock:
            self._failures.pop(key, None)
            self._locked_until.pop(key, None)
