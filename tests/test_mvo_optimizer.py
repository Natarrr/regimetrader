"""tests/test_mvo_optimizer.py
Portfolio optimizer hard constraint tests.
"""
from __future__ import annotations

import pytest
from backend.market_intel.portfolio_optimizer import run_optimizer


def _make_tickers(n: int) -> list[str]:
    return [f"T{i:02d}" for i in range(n)]


def _make_scores(n: int) -> list[float]:
    # Spread scores to give MVO something to work with
    return [0.5 + 0.01 * i for i in range(n)]


class TestMVOConstraints:
    def test_weights_sum_at_most_one(self):
        tickers = _make_tickers(10)
        scores = _make_scores(10)
        sectors = ["Tech"] * 5 + ["Healthcare"] * 5
        weights, _ = run_optimizer(tickers, scores, sectors, vix=20.0)
        assert sum(weights.values()) <= 1.0 + 1e-6

    def test_all_weights_nonnegative(self):
        tickers = _make_tickers(10)
        scores = _make_scores(10)
        sectors = ["Tech"] * 10
        weights, _ = run_optimizer(tickers, scores, sectors, vix=20.0)
        for t, w in weights.items():
            assert w >= -1e-8, f"{t} has negative weight {w}"

    def test_no_position_exceeds_max(self):
        tickers = _make_tickers(15)
        scores = _make_scores(15)
        sectors = ["Sector"] * 15
        weights, _ = run_optimizer(tickers, scores, sectors, vix=20.0)
        for t, w in weights.items():
            assert w <= 0.10 + 1e-6, f"{t} weight {w:.4f} exceeds 10% cap"

    def test_sector_weight_capped_when_mvo_converges(self):
        # n=10 with 10 unique sectors: identity-cov fallback makes MVO converge.
        # Each single-ticker sector is capped at 10% by the position bound.
        tickers = _make_tickers(10)
        scores = _make_scores(10)
        # Each ticker in its own sector — sector sum == position weight ≤ 10%
        sectors = [f"S{i}" for i in range(10)]
        weights, method = run_optimizer(tickers, scores, sectors, vix=20.0)
        assert method == "MVO", f"Expected MVO convergence, got {method}"
        for t, s in zip(tickers, sectors):
            sector_total = weights[t]  # one ticker per sector
            assert sector_total <= 0.10 + 1e-5, f"Sector {s} weight {sector_total:.4f} > 10%"

    def test_all_tickers_have_weight_key(self):
        tickers = _make_tickers(5)
        scores = _make_scores(5)
        sectors = ["A", "B", "C", "D", "E"]
        weights, _ = run_optimizer(tickers, scores, sectors, vix=20.0)
        for t in tickers:
            assert t in weights

    def test_method_is_valid_string(self):
        tickers = _make_tickers(5)
        scores = _make_scores(5)
        sectors = ["A"] * 5
        _, method = run_optimizer(tickers, scores, sectors, vix=20.0)
        assert method in {"MVO", "risk_parity", "score_proportional"}
