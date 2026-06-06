"""tests/test_vix_vol_scaling.py
VIX vol-targeting: at VIX >= 30, portfolio leverage (weight sum) reflects
the panic target. Uses TARGET_VOL from the optimizer module to avoid
reimplementing the vol formula (circular test anti-pattern).
"""
from __future__ import annotations

import numpy as np

from backend.market_intel.portfolio_optimizer import run_optimizer, TARGET_VOL


class TestVIXVolScaling:
    def test_panic_vix_portfolio_leverage_below_normal(self):
        # n=3: identity cov, uniform expected returns → equal weights before scaling.
        # With var=0.04/asset, port_vol ≈ sqrt(0.04/3) ≈ 0.115 >> panic target 0.05.
        # VIX scaling must reduce sum(weights) = scale = target_vol / port_vol < 1.
        tickers = ["A", "B", "C"]
        scores = [0.8, 0.6, 0.4]
        sectors = ["Tech", "Health", "Finance"]

        weights_normal, _ = run_optimizer(tickers, scores, sectors, vix=15.0)
        weights_panic, _ = run_optimizer(tickers, scores, sectors, vix=35.0)

        # Panic regime must not leverage more than normal regime
        assert sum(weights_panic.values()) <= sum(weights_normal.values()) + 1e-6

    def test_panic_vix_weight_sum_consistent_with_target(self):
        # Derive the expected max weight sum analytically.
        # With identity cov (var=0.04 per asset, n=3, equal weights w=1/n):
        # port_vol_pre_scale = sqrt(0.04 * sum(w^2)) = sqrt(0.04/3) ≈ 0.1155
        # panic target_vol = 0.05 (from TARGET_VOL["panic"])
        # scale = target_vol / port_vol ≈ 0.433 → weight_sum ≈ 0.433
        # We test: weight_sum <= target_vol / sqrt(0.04/n) + tolerance
        n = 3
        tickers = ["A", "B", "C"][:n]
        scores = [0.8, 0.6, 0.4][:n]
        sectors = ["Tech", "Health", "Finance"][:n]

        weights, _ = run_optimizer(tickers, scores, sectors, vix=35.0)
        target_vol = TARGET_VOL["panic"]

        # After VIX scaling: port_vol ≈ target_vol (capped by scale = min(1, t/v))
        # weight_sum = scale ≤ target_vol / min_achievable_port_vol
        # For any weight vector w summing to s, port_vol = sqrt(0.04 * s^2 / n) (equal-weight case)
        # So sum(weights) <= target_vol / sqrt(0.04/n) approximately — use loose upper bound
        w_arr = np.array(list(weights.values()))
        cov = np.eye(n) * 0.04
        actual_port_vol = float(np.sqrt(w_arr @ cov @ w_arr + 1e-10))
        assert actual_port_vol <= target_vol + 1e-6, (
            f"Panic regime vol {actual_port_vol:.4f} exceeds TARGET_VOL['panic']={target_vol}"
        )

    def test_vix_exactly_30_triggers_panic(self):
        tickers = ["A", "B", "C"]
        scores = [0.7, 0.6, 0.5]
        sectors = ["Tech", "Health", "Finance"]

        weights_bear, _ = run_optimizer(tickers, scores, sectors, vix=29.9)
        weights_panic, _ = run_optimizer(tickers, scores, sectors, vix=30.0)

        # At VIX=30, panic target (0.05) < bear target (0.10) → tighter constraint
        assert sum(weights_panic.values()) <= sum(weights_bear.values()) + 1e-6
