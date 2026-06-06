"""tests/test_mvo_determinism.py
Determinism: same inputs must produce identical weights on every call.
"""
from __future__ import annotations

from backend.market_intel.portfolio_optimizer import run_optimizer


class TestMVODeterminism:
    def test_identical_weights_across_runs(self):
        tickers = [f"T{i:02d}" for i in range(10)]
        scores = [0.5 + 0.01 * i for i in range(10)]
        sectors = ["Tech"] * 5 + ["Healthcare"] * 5

        results = [
            run_optimizer(tickers, scores, sectors, vix=20.0)
            for _ in range(10)
        ]

        first_weights, first_method = results[0]
        for weights, method in results[1:]:
            assert method == first_method
            for t in tickers:
                assert abs(weights[t] - first_weights[t]) < 1e-9, (
                    f"Non-deterministic weight for {t}: "
                    f"{weights[t]:.10f} vs {first_weights[t]:.10f}"
                )

    def test_deterministic_with_prev_weights(self):
        tickers = ["A", "B", "C"]
        scores = [0.8, 0.6, 0.4]
        sectors = ["Tech", "Health", "Finance"]
        prev = {"A": 0.4, "B": 0.3, "C": 0.3}

        results = [
            run_optimizer(tickers, scores, sectors, vix=22.0, prev_weights=prev)
            for _ in range(5)
        ]
        w0 = results[0][0]
        for w, _ in results[1:]:
            for t in tickers:
                assert abs(w[t] - w0[t]) < 1e-9
