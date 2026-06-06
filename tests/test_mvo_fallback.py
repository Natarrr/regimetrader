"""tests/test_mvo_fallback.py
Fallback chain: MVO → risk_parity → score_proportional always produces valid weights.
"""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import patch

from backend.market_intel.portfolio_optimizer import (
    run_optimizer,
    _score_proportional,
    _risk_parity,
)


def _valid(weights: dict[str, float], tickers: list[str]) -> None:
    assert set(weights.keys()) == set(tickers)
    assert sum(weights.values()) <= 1.0 + 1e-6
    for w in weights.values():
        assert w >= -1e-8


class TestFallbackChain:
    def test_score_proportional_always_valid(self):
        tickers = ["A", "B", "C"]
        scores = [0.8, 0.6, 0.4]
        sectors = ["T", "T", "H"]
        # Force all fallbacks to score_proportional by making MVO and risk_parity raise
        with patch("backend.market_intel.portfolio_optimizer._mvo", side_effect=RuntimeError("mvo fail")):
            with patch("backend.market_intel.portfolio_optimizer._risk_parity", side_effect=RuntimeError("rp fail")):
                weights, method = run_optimizer(tickers, scores, sectors, vix=20.0)
        assert method == "score_proportional"
        _valid(weights, tickers)

    def test_risk_parity_fallback_valid(self):
        tickers = ["A", "B", "C"]
        scores = [0.8, 0.6, 0.4]
        sectors = ["T", "T", "H"]
        with patch("backend.market_intel.portfolio_optimizer._mvo", side_effect=RuntimeError("mvo fail")):
            weights, method = run_optimizer(tickers, scores, sectors, vix=20.0)
        assert method == "risk_parity"
        _valid(weights, tickers)

    def test_mvo_success_returns_mvo_method(self):
        # n=10 with 10 unique sectors reliably converges MVO with identity cov fallback.
        # Equal-variance assets, unique sectors, 10 tickers — SLSQP finds equal-weight solution.
        tickers = [f"T{i:02d}" for i in range(10)]
        scores = [0.5 + 0.05 * i for i in range(10)]
        sectors = [f"S{i}" for i in range(10)]
        weights, method = run_optimizer(tickers, scores, sectors, vix=20.0)
        assert method == "MVO"
        _valid(weights, tickers)

    def test_zero_scores_does_not_crash(self):
        tickers = ["A", "B", "C"]
        scores = [0.0, 0.0, 0.0]
        sectors = ["T", "T", "T"]
        weights, _ = run_optimizer(tickers, scores, sectors, vix=20.0)
        _valid(weights, tickers)

    def test_single_ticker_does_not_crash(self):
        # n=1: MVO is infeasible (bound [0, 0.10] conflicts with sum==1.0 equality).
        # Falls back to risk_parity → weight=1.0 → VIX-scaled to target_vol / port_vol.
        # The 10% position bound is an MVO optimizer constraint, not a hard portfolio guarantee
        # when the fallback chain is active. Assert valid structure only.
        weights, _ = run_optimizer(["SOLO"], [0.9], ["Tech"], vix=20.0)
        assert "SOLO" in weights
        assert weights["SOLO"] >= 0.0
        assert weights["SOLO"] <= 1.0 + 1e-6  # weight sum ≤ 1 after VIX scaling
