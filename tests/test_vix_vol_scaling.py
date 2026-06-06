"""tests/test_vix_vol_scaling.py
VIX vol-targeting: at VIX >= 30, portfolio vol must not exceed 0.05 (panic target).
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.market_intel.portfolio_optimizer import run_optimizer


_IDENTITY_VOL_PER_ASSET = 0.20  # sqrt(0.04) — the identity cov fallback variance


def _portfolio_vol(weights: dict[str, float], tickers: list[str], n: int) -> float:
    w = np.array([weights[t] for t in tickers])
    cov = np.eye(n) * 0.04  # matches the internal fallback
    return float(np.sqrt(w @ cov @ w + 1e-10))


class TestVIXVolScaling:
    def test_panic_vix_portfolio_vol_within_target(self):
        # n=3: equal-weight vol ≈ 0.115 >> 0.05, forces downscale
        tickers = ["A", "B", "C"]
        scores = [0.8, 0.6, 0.4]
        sectors = ["Tech", "Health", "Finance"]
        weights, _ = run_optimizer(tickers, scores, sectors, vix=35.0)
        vol = _portfolio_vol(weights, tickers, n=3)
        assert vol <= 0.05 + 1e-6, f"panic regime vol {vol:.4f} exceeds 0.05"

    def test_vix_exactly_30_triggers_panic(self):
        tickers = ["A", "B", "C"]
        scores = [0.7, 0.6, 0.5]
        sectors = ["Tech", "Health", "Finance"]
        weights, _ = run_optimizer(tickers, scores, sectors, vix=30.0)
        vol = _portfolio_vol(weights, tickers, n=3)
        assert vol <= 0.05 + 1e-6

    def test_normal_vix_does_not_force_downscale(self):
        # At normal VIX (< 25), if port_vol is already below target, weights unchanged
        tickers = [f"T{i}" for i in range(20)]
        scores = [0.5] * 20
        sectors = ["A"] * 10 + ["B"] * 10
        weights_normal, _ = run_optimizer(tickers, scores, sectors, vix=15.0)
        weights_panic, _ = run_optimizer(tickers, scores, sectors, vix=35.0)
        # Panic weights should sum to less (or equal) than normal weights
        assert sum(weights_panic.values()) <= sum(weights_normal.values()) + 1e-6
