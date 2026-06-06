"""tests/test_vix_monotonicity.py
VIX monotonicity: increasing VIX must never increase portfolio leverage (weight sum).
"""
from __future__ import annotations

import pytest
from backend.market_intel.portfolio_optimizer import run_optimizer


_TICKERS = ["A", "B", "C"]
_SCORES = [0.8, 0.6, 0.4]
_SECTORS = ["Tech", "Health", "Finance"]


class TestVIXMonotonicity:
    def _leverage(self, vix: float) -> float:
        weights, _ = run_optimizer(_TICKERS, _SCORES, _SECTORS, vix=vix)
        return sum(weights.values())

    def test_panic_leverage_le_bear(self):
        assert self._leverage(35.0) <= self._leverage(27.0) + 1e-6

    def test_bear_leverage_le_normal(self):
        assert self._leverage(27.0) <= self._leverage(20.0) + 1e-6

    def test_panic_leverage_le_normal(self):
        assert self._leverage(40.0) <= self._leverage(15.0) + 1e-6

    def test_increasing_vix_sequence(self):
        vix_levels = [10.0, 20.0, 25.0, 30.0, 40.0]
        leverages = [self._leverage(v) for v in vix_levels]
        for i in range(len(leverages) - 1):
            assert leverages[i] >= leverages[i + 1] - 1e-6, (
                f"Leverage increased from VIX={vix_levels[i]} to VIX={vix_levels[i+1]}: "
                f"{leverages[i]:.6f} < {leverages[i+1]:.6f}"
            )
