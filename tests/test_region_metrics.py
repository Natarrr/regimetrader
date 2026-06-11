"""tests/test_region_metrics.py — per-region v3 monitoring metrics.

IC is Spearman(final_score_v3, T+20 forward excess return vs local-currency
benchmark) when forward returns are supplied; quintile (not decile —
universes ~100–160 names) top-minus-bottom spread; top-20 currency mix.
"""
from __future__ import annotations

import pytest

from monitoring.region_metrics import (
    compute_region_metrics,
    currency_of,
    evaluate_region,
)


def _row(ticker, market, score, coverage=1.0, sector="Tech"):
    return {"ticker": ticker, "market": market, "sector": sector,
            "cap_tier": "large", "final_score_v3": score,
            "weight_coverage_v3": coverage}


def _universe():
    us = [_row(f"T{i}", "USA", 0.9 - i * 0.03) for i in range(20)]
    eu = [_row(f"E{i}.PA", "EUROPE", 0.8 - i * 0.02) for i in range(20)]
    return us + eu


class TestCurrencyOf:
    @pytest.mark.parametrize("ticker,ccy", [
        ("AAPL", "USD"), ("VOD.L", "GBp"), ("ASML.AS", "EUR"),
        ("7203.T", "JPY"), ("0700.HK", "HKD"), ("005930.KS", "KRW"),
    ])
    def test_suffix_mapping(self, ticker, ccy):
        assert currency_of(ticker) == ccy


class TestComputeRegionMetrics:
    def test_groups_by_market(self):
        metrics = compute_region_metrics(_universe())
        assert set(metrics) == {"USA", "EUROPE"}
        assert metrics["USA"]["n"] == 20

    def test_ic_with_forward_returns(self):
        # Forward returns proportional to score → IC = 1.0
        rows = [_row(f"T{i}", "USA", 0.9 - i * 0.03) for i in range(20)]
        fwd = {r["ticker"]: r["final_score_v3"] * 0.1 for r in rows}
        metrics = compute_region_metrics(rows, forward_returns=fwd)
        usa = metrics["USA"]
        assert usa["ic_spearman"] == pytest.approx(1.0)
        assert usa["quintile_spread"] > 0

    def test_ic_none_without_returns(self):
        metrics = compute_region_metrics(_universe())
        assert metrics["USA"]["ic_spearman"] is None

    def test_currency_mix_in_top20(self):
        metrics = compute_region_metrics(_universe())
        assert metrics["EUROPE"]["top20_currency_mix"] == {"EUR": 20}


class TestEvaluateRegion:
    def test_passing_metrics_no_failures(self):
        failures = evaluate_region(
            "USA", {"ic_spearman": 0.05, "coverage_mean": 0.8})
        assert failures == []

    def test_ic_below_target_flagged(self):
        failures = evaluate_region(
            "USA", {"ic_spearman": 0.01, "coverage_mean": 0.8})
        assert any("ic" in f.lower() for f in failures)

    def test_coverage_below_gate_flagged(self):
        failures = evaluate_region(
            "ASIA", {"ic_spearman": 0.05, "coverage_mean": 0.30})
        assert any("coverage" in f.lower() for f in failures)

    def test_missing_ic_not_a_failure(self):
        # Pre-cutover there are no forward returns yet — absence is not red.
        failures = evaluate_region(
            "USA", {"ic_spearman": None, "coverage_mean": 0.8})
        assert failures == []
