"""tests/test_token_bucket.py
Deterministic tests for regime_trader.utils.token_bucket.TokenBucket.

Black-Scholes-Merton (1997 Nobel) — bounded time is a risk invariant:
rate-limiting must be verifiably deterministic, not probabilistic.
Tests use an injected fake clock so no real wall-clock time is consumed.
"""
from __future__ import annotations

import threading
import time

import pytest

from regime_trader.utils.token_bucket import TokenBucket


class _FakeClock:
    """Injected clock for deterministic tests — no real sleeping."""

    def __init__(self, t: float = 0.0) -> None:
        self._t = t

    def __call__(self) -> float:
        return self._t

    def advance(self, dt: float) -> None:
        self._t += dt


# ── Construction ───────────────────────────────────────────────────────────────

class TestTokenBucketConstruction:
    def test_positive_rate_accepted(self):
        tb = TokenBucket(rate_per_sec=1.0)
        assert tb is not None

    def test_zero_rate_raises(self):
        with pytest.raises(ValueError, match="rate_per_sec"):
            TokenBucket(rate_per_sec=0.0)

    def test_negative_rate_raises(self):
        with pytest.raises(ValueError):
            TokenBucket(rate_per_sec=-1.0)

    def test_repr(self):
        tb = TokenBucket(rate_per_sec=0.2)
        assert "0.2" in repr(tb)


# ── try_acquire (non-blocking) ─────────────────────────────────────────────────

class TestTryAcquire:
    def test_initial_token_available(self):
        clock = _FakeClock(0.0)
        tb    = TokenBucket(rate_per_sec=1.0, capacity=1.0, clock=clock)
        assert tb.try_acquire() is True

    def test_no_second_token_without_refill(self):
        clock = _FakeClock(0.0)
        tb    = TokenBucket(rate_per_sec=1.0, capacity=1.0, clock=clock)
        tb.try_acquire()        # consume the only token
        assert tb.try_acquire() is False

    def test_refill_after_time_passes(self):
        clock = _FakeClock(0.0)
        tb    = TokenBucket(rate_per_sec=1.0, capacity=1.0, clock=clock)
        tb.try_acquire()        # consume
        clock.advance(1.1)     # 1.1 s at 1 req/s → 1.1 tokens refilled, cap=1
        assert tb.try_acquire() is True

    def test_rate_0_2_refills_after_5s(self):
        """0.2 req/s → 1 token every 5 s."""
        clock = _FakeClock(0.0)
        tb    = TokenBucket(rate_per_sec=0.2, capacity=1.0, clock=clock)
        tb.try_acquire()         # consume
        clock.advance(4.9)       # not yet
        assert tb.try_acquire() is False
        clock.advance(0.2)       # now past 5 s threshold
        assert tb.try_acquire() is True


# ── acquire (blocking) — patched sleep ────────────────────────────────────────

class TestAcquire:
    def test_high_rate_no_effective_wait(self):
        tb = TokenBucket(rate_per_sec=10_000.0, capacity=1.0)
        t0 = time.monotonic()
        wait = tb.acquire()
        elapsed = time.monotonic() - t0
        assert elapsed < 0.1
        assert wait < 0.1

    def test_acquire_returns_float(self):
        tb = TokenBucket(rate_per_sec=10_000.0)
        result = tb.acquire()
        assert isinstance(result, float)
        assert result >= 0.0

    def test_acquire_n_zero_raises(self):
        tb = TokenBucket(rate_per_sec=1.0)
        with pytest.raises(ValueError, match="n must be >= 1"):
            tb.acquire(n=0)

    def test_try_acquire_n_multiple_fails_without_capacity(self):
        clock = _FakeClock(0.0)
        tb    = TokenBucket(rate_per_sec=1.0, capacity=1.0, clock=clock)
        # Try to consume 2 tokens from a bucket with capacity=1 and 1 token
        assert tb.try_acquire(n=2) is False


# ── tokens property ───────────────────────────────────────────────────────────

class TestTokensProperty:
    def test_tokens_starts_at_capacity(self):
        clock = _FakeClock(0.0)
        tb    = TokenBucket(rate_per_sec=1.0, capacity=5.0, clock=clock)
        assert tb.tokens == pytest.approx(5.0, abs=0.01)

    def test_tokens_decreases_after_acquire(self):
        clock = _FakeClock(0.0)
        tb    = TokenBucket(rate_per_sec=1.0, capacity=5.0, clock=clock)
        tb.try_acquire()
        # Right after, tokens should be ~4
        tokens = tb.tokens
        assert 3.9 < tokens < 5.0

    def test_tokens_capped_at_capacity(self):
        clock = _FakeClock(0.0)
        tb    = TokenBucket(rate_per_sec=1.0, capacity=2.0, clock=clock)
        tb.try_acquire()       # consume 1
        clock.advance(1000.0)  # huge refill
        assert tb.tokens <= 2.0 + 1e-6   # never exceeds capacity


# ── from_env factory ──────────────────────────────────────────────────────────

class TestFromEnv:
    def test_reads_env_var(self, monkeypatch):
        monkeypatch.setenv("TEST_BUCKET_RATE", "5.0")
        tb = TokenBucket.from_env("TEST_BUCKET_RATE", default=1.0)
        assert tb._rate == pytest.approx(5.0)

    def test_falls_back_to_default_on_missing_var(self, monkeypatch):
        monkeypatch.delenv("MISSING_RATE_VAR", raising=False)
        tb = TokenBucket.from_env("MISSING_RATE_VAR", default=0.5)
        assert tb._rate == pytest.approx(0.5)

    def test_falls_back_to_default_on_invalid_value(self, monkeypatch):
        monkeypatch.setenv("BAD_RATE_VAR", "not_a_number")
        tb = TokenBucket.from_env("BAD_RATE_VAR", default=2.0)
        assert tb._rate == pytest.approx(2.0)


# ── Thread safety ─────────────────────────────────────────────────────────────

class TestThreadSafety:
    def test_concurrent_try_acquire_no_overdraft(self):
        """10 threads each try to consume 1 token; only 1 should succeed initially."""
        tb      = TokenBucket(rate_per_sec=0.001, capacity=1.0)  # almost no refill
        results = []
        lock    = threading.Lock()

        def worker():
            r = tb.try_acquire()
            with lock:
                results.append(r)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # At most 1 success (initial token), never more
        assert sum(results) <= 1
