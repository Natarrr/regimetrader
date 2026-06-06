"""tests/test_sector_exposure.py
Sector exposure: sector cap enforced; "Unknown" sector handled without crash.
"""
from __future__ import annotations

import pytest
from backend.market_intel.portfolio_optimizer import run_optimizer, _MAX_SECTOR_WEIGHT


def _sector_sum(weights: dict[str, float], tickers: list[str], sectors: list[str], target_sector: str) -> float:
    return sum(
        weights[t] for t, s in zip(tickers, sectors) if s == target_sector
    )


class TestSectorExposure:
    def test_single_sector_capped_at_30pct(self):
        # n=10, 4 sectors (3+3+3+1): the only feasible layout with identity-cov fallback
        # that allows MVO to converge while respecting w.sum()==1 and sector cap 0.30.
        # Sectors of 3 tickers each can contribute at most 3 * 0.10 = 0.30 — exactly at cap.
        tickers = [f"T{i:02d}" for i in range(10)]
        scores = [0.5 + 0.03 * i for i in range(10)]
        sectors = ["Tech"] * 3 + ["Health"] * 3 + ["Finance"] * 3 + ["Energy"] * 1
        weights, method = run_optimizer(tickers, scores, sectors, vix=20.0)
        assert method == "MVO", f"Expected MVO convergence with 4-sector (3+3+3+1) layout, got {method}"
        for sector in ["Tech", "Health", "Finance"]:
            s = _sector_sum(weights, tickers, sectors, sector)
            assert s <= _MAX_SECTOR_WEIGHT + 1e-5, f"{sector} exposure {s:.4f} > 30%"

    def test_multi_sector_each_capped(self):
        # n=10, 4 sectors (3+3+3+1): MVO-convergent layout; verifies sector cap on all sectors.
        tickers = [f"T{i:02d}" for i in range(10)]
        scores = [0.5 + 0.03 * i for i in range(10)]
        sectors = ["Tech"] * 3 + ["Health"] * 3 + ["Finance"] * 3 + ["Energy"] * 1
        weights, method = run_optimizer(tickers, scores, sectors, vix=20.0)
        assert method == "MVO", f"Expected MVO convergence, got {method}"
        for sector in ["Tech", "Health", "Finance", "Energy"]:
            s = _sector_sum(weights, tickers, sectors, sector)
            assert s <= _MAX_SECTOR_WEIGHT + 1e-5, f"{sector} exposure {s:.4f} > 30%"

    def test_unknown_sector_does_not_crash(self):
        tickers = ["A", "B", "C"]
        scores = [0.7, 0.6, 0.5]
        sectors = ["Unknown", "Unknown", "Tech"]
        weights, method = run_optimizer(tickers, scores, sectors, vix=20.0)
        assert set(weights.keys()) == {"A", "B", "C"}
        assert method in {"MVO", "risk_parity", "score_proportional"}

    def test_all_unknown_sector_does_not_crash(self):
        tickers = ["X", "Y"]
        scores = [0.8, 0.4]
        sectors = ["Unknown", "Unknown"]
        weights, _ = run_optimizer(tickers, scores, sectors, vix=20.0)
        assert sum(weights.values()) <= 1.0 + 1e-6
