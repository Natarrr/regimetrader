"""regime_trader/utils/token_bucket.py
Reusable, thread-safe token-bucket rate limiter.

Black-Scholes-Merton (1997 Nobel) — bounded computation time is as important
as bounded risk. Rate-limiting prevents SEC IP bans and FMP daily-quota overruns.

The classic token-bucket algorithm:
  $N_{tokens}(t) = \\min(capacity,\\; N_{tokens}(t-\\Delta t) + rate \\cdot \\Delta t)$

Tokens refill continuously.  acquire() blocks (sleeps) until a token is
available, then atomically consumes it.  Thread-safe via threading.Lock.

Usage:
    from regime_trader.utils.token_bucket import TokenBucket

    # 0.2 calls/sec = 1 call every 5 s  (SEC guidance)
    bucket = TokenBucket(rate_per_sec=0.2)
    bucket.acquire()            # blocks until a token is available
    bucket.acquire(n=3)         # consume 3 tokens atomically

    # Inject from env var:
    import os
    rate = float(os.getenv("EDGAR_RATE_LIMIT", "0.2"))
    bucket = TokenBucket.from_env("EDGAR_RATE_LIMIT", default=0.2)
"""
from __future__ import annotations

import os
import threading
import time


class TokenBucket:
    """Thread-safe token-bucket rate limiter.

    Black-Scholes-Merton (1997 Nobel): time-bounded execution is a risk-
    management invariant as fundamental as position sizing.

    Args:
        rate_per_sec: Tokens added per second (= max requests per second).
        capacity:     Maximum burst size (default = 1.0, no burst).
        clock:        Callable returning current time (default time.monotonic).
                      Inject a fake clock in tests for deterministic behaviour.
    """

    def __init__(
        self,
        rate_per_sec: float,
        capacity: float = 1.0,
        clock: object = None,
    ) -> None:
        if rate_per_sec <= 0:
            raise ValueError(f"rate_per_sec must be > 0, got {rate_per_sec}")
        self._rate     = rate_per_sec
        self._capacity = max(capacity, 1.0)
        self._tokens   = self._capacity
        self._last_ts  = (clock or time.monotonic)()
        self._lock     = threading.Lock()
        self._clock    = clock or time.monotonic

    # ── Public API ─────────────────────────────────────────────────────────────

    def acquire(self, n: int = 1) -> float:
        """Block until n tokens are available, consume them, return wait time.

        Sleeps in small increments so the thread releases the GIL periodically.

        Args:
            n: Number of tokens to consume (default 1).

        Returns:
            Actual wall-clock seconds spent waiting (0.0 if no wait needed).
        """
        if n <= 0:
            raise ValueError(f"n must be >= 1, got {n}")

        t_start = self._clock()
        needed  = float(n)

        while True:
            with self._lock:
                now   = self._clock()
                delta = now - self._last_ts
                self._last_ts = now
                self._tokens  = min(self._capacity, self._tokens + delta * self._rate)

                if self._tokens >= needed:
                    self._tokens -= needed
                    return self._clock() - t_start

            # Release lock and sleep proportionally to the deficit.
            deficit    = needed - self._tokens
            sleep_time = max(deficit / self._rate, 0.001)
            time.sleep(min(sleep_time, 0.5))   # cap single sleep at 0.5 s

    def try_acquire(self, n: int = 1) -> bool:
        """Non-blocking: consume n tokens if available, return True. Else False.

        Args:
            n: Number of tokens to consume.

        Returns:
            True if tokens were consumed, False if insufficient tokens.
        """
        with self._lock:
            now   = self._clock()
            delta = now - self._last_ts
            self._last_ts = now
            self._tokens  = min(self._capacity, self._tokens + delta * self._rate)

            if self._tokens >= float(n):
                self._tokens -= float(n)
                return True
            return False

    @property
    def tokens(self) -> float:
        """Current (approximate) token count — for observability only."""
        with self._lock:
            now   = self._clock()
            delta = now - self._last_ts
            return min(self._capacity, self._tokens + delta * self._rate)

    # ── Factory helpers ────────────────────────────────────────────────────────

    @classmethod
    def from_env(
        cls,
        env_var: str,
        default: float,
        capacity: float = 1.0,
    ) -> "TokenBucket":
        """Create a TokenBucket whose rate is read from an environment variable.

        Args:
            env_var:  Name of the env var (e.g. "EDGAR_RATE_LIMIT").
            default:  Rate to use when the env var is absent or invalid.
            capacity: Max burst size.
        """
        raw  = os.getenv(env_var, "")
        rate = default
        if raw:
            try:
                rate = float(raw)
            except ValueError:
                pass
        return cls(rate_per_sec=rate, capacity=capacity)

    def __repr__(self) -> str:
        return f"TokenBucket(rate_per_sec={self._rate}, capacity={self._capacity})"
