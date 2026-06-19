"""tests/test_beta_factor.py — CAPM beta producer (P2.1 audit).

compute_beta activates the previously-inert CAPITULATION low-beta gate in
src/risk/regime. Pure math + the survivor-gate integration.
"""
from __future__ import annotations

import pytest

from src.scoring.momentum_signals import compute_beta
from src.risk.regime import apply_capitulation_filter, _is_capitulation_survivor


def _closes_from_returns(rets: list[float], start: float = 100.0) -> list[float]:
    closes = [start]
    for r in rets:
        closes.append(closes[-1] * (1.0 + r))
    return closes


_BENCH_RETS = [0.01, -0.02, 0.015, -0.005, 0.02, -0.01, 0.005, -0.015, 0.01,
               -0.02, 0.012, -0.008, 0.018, -0.011, 0.006, -0.013, 0.009,
               -0.017, 0.011, -0.004, 0.016, -0.009, 0.007, -0.012, 0.014,
               -0.006, 0.008, -0.016, 0.013, -0.003]


class TestComputeBeta:
    def test_identical_series_beta_one(self):
        bench = _closes_from_returns(_BENCH_RETS)
        assert compute_beta(bench, bench, window=30) == pytest.approx(1.0, abs=1e-6)

    def test_double_leverage_beta_two(self):
        bench = _closes_from_returns(_BENCH_RETS)
        asset = _closes_from_returns([2.0 * r for r in _BENCH_RETS])
        assert compute_beta(asset, bench, window=30) == pytest.approx(2.0, abs=1e-3)

    def test_inverse_series_negative_beta(self):
        bench = _closes_from_returns(_BENCH_RETS)
        asset = _closes_from_returns([-r for r in _BENCH_RETS])
        assert compute_beta(asset, bench, window=30) < 0.0

    def test_none_when_too_few_points(self):
        bench = _closes_from_returns(_BENCH_RETS[:5])
        assert compute_beta(bench, bench, window=30) is None

    def test_none_on_flat_benchmark(self):
        flat = [100.0] * 32
        asset = _closes_from_returns(_BENCH_RETS)
        assert compute_beta(asset, flat, window=30) is None


def _iso_dates(n: int, start: str = "2026-01-01") -> list[str]:
    from datetime import date, timedelta
    d0 = date.fromisoformat(start)
    return [(d0 + timedelta(days=i)).isoformat() for i in range(n)]


class TestComputeBetaAligned:
    """Date-aligned beta for INTL (local listing vs US SPY) — only co-traded
    sessions enter the returns."""

    def test_identical_aligned_beta_one(self):
        from src.scoring.momentum_signals import compute_beta_aligned
        closes = _closes_from_returns(_BENCH_RETS)
        dts = _iso_dates(len(closes))
        assert compute_beta_aligned(dts, closes, dts, closes, window=30) == \
            pytest.approx(1.0, abs=1e-6)

    def test_drops_non_cotraded_dates(self):
        from src.scoring.momentum_signals import compute_beta_aligned
        closes = _closes_from_returns(_BENCH_RETS)
        dts = _iso_dates(len(closes))
        # 3 local-holiday sessions the benchmark never traded → dropped, leaving
        # the 31 co-traded sessions → beta 1.0.
        a_dates = ["2025-12-20", "2025-12-21", "2025-12-22"] + dts
        a_closes = [50.0, 51.0, 52.0] + closes
        assert compute_beta_aligned(a_dates, a_closes, dts, closes, window=30) == \
            pytest.approx(1.0, abs=1e-6)

    def test_none_when_too_few_cotraded(self):
        from src.scoring.momentum_signals import compute_beta_aligned
        closes = _closes_from_returns(_BENCH_RETS)
        dts = _iso_dates(len(closes))
        assert compute_beta_aligned(dts, closes, dts[:10], closes[:10],
                                    window=30) is None


class TestCapitulationBetaGate:
    """The beta gate (inert without a producer) now excludes high-beta names."""

    def _entry(self, ticker, beta, piotroski=0.9):
        return {"ticker": ticker, "final_score": 0.5,
                "factors": {"beta_30d": beta, "quality_piotroski": piotroski}}

    def test_high_beta_excluded_in_capitulation(self):
        assert _is_capitulation_survivor(self._entry("HOT", 1.5)) is False
        assert _is_capitulation_survivor(self._entry("CALM", 0.9)) is True

    def test_filter_drops_high_beta_at_panic_vix(self):
        entries = [self._entry("HOT", 1.5), self._entry("CALM", 0.8)]
        survivors = apply_capitulation_filter(entries, vix=35.0)
        tickers = {e["ticker"] for e in survivors}
        assert tickers == {"CALM"}                  # high-beta dropped in crash
        assert all(e["badge"] == "WATCHLIST" for e in survivors)
