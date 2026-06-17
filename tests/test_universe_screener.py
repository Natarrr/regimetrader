# Path: tests/test_universe_screener.py
"""Hybrid dynamic universe — core + satellite selection (vectorized).

The satellite sleeve surfaces names with surging trading attention that are NOT
already in the curated core, so capital can rotate toward fresh opportunities
instead of re-scoring the same frozen list. Selection is a cross-sectional
volume-velocity rank [WorldQuant 101 Alphas — ts_rank/rank of volume].
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.ingestion.universe_screener import (
    compute_volume_velocity,
    cross_sectional_rank,
    merge_universe,
    select_satellite,
)


class TestComputeVolumeVelocity:
    def test_surging_volume_scores_above_flat(self):
        panel = pd.DataFrame({
            "SURGE": [100, 100, 100, 100, 300, 300],
            "FLAT":  [100, 100, 100, 100, 100, 100],
        })
        vel = compute_volume_velocity(panel, short=2, long=4)
        assert vel["SURGE"] == pytest.approx(0.5)   # 300/200 - 1
        assert vel["FLAT"] == pytest.approx(0.0)

    def test_zero_long_volume_is_nan_not_inf(self):
        panel = pd.DataFrame({"DEAD": [0, 0, 0, 0]})
        vel = compute_volume_velocity(panel, short=2, long=4)
        assert pd.isna(vel["DEAD"])


class TestCrossSectionalRank:
    def test_pct_rank_max_is_one(self):
        s = pd.Series({"A": 0.5, "B": 0.0, "C": 0.2})
        r = cross_sectional_rank(s)
        assert r["A"] == pytest.approx(1.0)
        assert r["B"] == pytest.approx(1 / 3)


class TestSelectSatellite:
    def test_excludes_core_and_takes_top_k_by_velocity(self):
        vel = pd.Series({"A": 0.5, "B": 0.1, "CORE1": 0.9, "C": 0.3})
        out = select_satellite(vel, core={"CORE1"}, k=2)
        assert out == ["A", "C"]

    def test_drops_nan_velocity(self):
        vel = pd.Series({"A": 0.5, "D": float("nan"), "C": 0.3})
        out = select_satellite(vel, core=set(), k=5)
        assert out == ["A", "C"]


class TestMergeUniverse:
    def test_core_precedence_and_origin_tagging(self):
        core = [{"ticker": "CORE1", "sector": "Tech", "cap_tier": "large"}]
        satellite = [
            {"ticker": "CORE1", "sector": "X", "cap_tier": "mid"},   # dup → dropped
            {"ticker": "A", "sector": "Energy", "cap_tier": "mid"},
        ]
        merged = merge_universe(core, satellite)
        by_ticker = {r["ticker"]: r for r in merged}
        assert set(by_ticker) == {"CORE1", "A"}
        assert by_ticker["CORE1"]["origin"] == "core"
        assert by_ticker["CORE1"]["sector"] == "Tech"   # core row wins
        assert by_ticker["A"]["origin"] == "satellite"
