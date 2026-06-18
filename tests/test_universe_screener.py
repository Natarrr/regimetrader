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
    _screen_smid_candidates,
    _resolve_smid_satellite,
    _leverage_rank,
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


class _FakeScreenerClient:
    """Minimal stand-in for FMPClient.get_company_screener."""

    def __init__(self, rows_by_exchange):
        self._api_key = "test-key"
        self._rows = rows_by_exchange

    def get_company_screener(self, exchange, market_cap_more_than=None,
                             market_cap_lower_than=None, volume_more_than=None,
                             is_actively_trading=None, limit=100):
        return self._rows.get(exchange, [])


class TestSmidSatellite:
    def _client(self):
        return _FakeScreenerClient({
            "NASDAQ": [
                {"symbol": "SMALL1", "sector": "Tech", "marketCap": 500_000_000},
                {"symbol": "MID1",   "sector": "Energy", "marketCap": 5_000_000_000},
                {"symbol": "TOOBIG", "sector": "Tech", "marketCap": 15_000_000_000},
            ],
            "NYSE": [
                {"symbol": "SMALL2", "sector": "Health", "marketCap": 900_000_000},
                {"symbol": "TOOSMALL", "sector": "Misc", "marketCap": 100_000_000},
            ],
        })

    def test_band_filter_and_cap_tier_tagging(self):
        cands = _screen_smid_candidates(self._client(), "US")
        assert set(cands) == {"SMALL1", "MID1", "SMALL2"}   # $10B+ and <$300M dropped
        assert cands["SMALL1"]["cap_tier"] == "small"
        assert cands["MID1"]["cap_tier"] == "mid"

    def test_satellite_rows_tagged_and_dedup(self):
        rows = _resolve_smid_satellite(
            self._client(), "US", existing={"MID1"}, log_dir=None)
        tickers = {r["ticker"] for r in rows}
        assert "MID1" not in tickers                 # already in the book
        assert {"SMALL1", "SMALL2"} <= tickers       # small caps surfaced
        assert all(r["origin"] == "smid_satellite" for r in rows)

    def test_smalls_are_represented(self):
        # Balanced selection must not let the mid end crowd out true small caps.
        rows = _resolve_smid_satellite(
            self._client(), "US", existing=set(), log_dir=None)
        assert any(r["cap_tier"] == "small" for r in rows)

    def test_fieldless_rows_survive_missing_price_volume(self):
        # Rows without price/volume must NOT be dropped by the ADV gate —
        # absent data is never used to reject (server share-floor already applied).
        cands = _screen_smid_candidates(self._client(), "US")
        assert {"SMALL1", "MID1", "SMALL2"} <= set(cands)
        assert all(cands[s]["adv_usd"] is None for s in cands)


class TestSmidLiquidityGate:
    def test_adv_gate_drops_illiquid_dollar_volume(self):
        # $2 stock × 100k shares = $200k/day → below the $3M ADV floor → dropped.
        # $50 stock × 1M shares = $50M/day → liquid → kept.
        client = _FakeScreenerClient({
            "NASDAQ": [
                {"symbol": "THIN", "sector": "Tech", "marketCap": 800_000_000,
                 "price": 2.0, "volume": 100_000},
                {"symbol": "LIQUID", "sector": "Tech", "marketCap": 800_000_000,
                 "price": 50.0, "volume": 1_000_000},
            ],
            "NYSE": [],
        })
        cands = _screen_smid_candidates(client, "US")
        assert "THIN" not in cands          # market-impact trap dropped
        assert "LIQUID" in cands
        assert cands["LIQUID"]["adv_usd"] == pytest.approx(50_000_000)


class _BandAwareScreenerClient:
    """Simulates FMP cap-descending + limit: returns only names WITHIN the
    requested [more_than, lower_than) ceiling. Reproduces the single-band bug —
    a $300M–$10B screen would return only the top of the band — so the test
    proves the small/mid SPLIT actually reaches sub-$2B names."""

    _api_key = "test-key"
    _POOL = [
        ("TINY", 4.0e8), ("SMALLA", 1.2e9), ("SMALLB", 1.8e9),
        ("MIDA", 4.0e9), ("MIDB", 9.5e9), ("TOOBIG", 15e9),
    ]

    def get_company_screener(self, exchange, market_cap_more_than=None,
                             market_cap_lower_than=None, volume_more_than=None,
                             is_actively_trading=None, limit=100):
        if exchange != "NASDAQ":
            return []
        lo = market_cap_more_than or 0
        hi = market_cap_lower_than if market_cap_lower_than is not None else float("inf")
        rows = [{"symbol": s, "sector": "Tech", "marketCap": c,
                 "price": 50.0, "volume": 1_000_000, "beta": 1.5}
                for s, c in self._POOL if lo <= c < hi]
        return rows[:limit]


class TestSmidBandSplit:
    def test_split_reaches_true_small_caps(self):
        cands = _screen_smid_candidates(_BandAwareScreenerClient(), "US")
        tiers = {s: m["cap_tier"] for s, m in cands.items()}
        # The $300M–$2B band must surface true small caps (the bug: it never did).
        assert tiers.get("TINY") == "small"
        assert tiers.get("SMALLA") == "small"
        assert tiers.get("MIDA") == "mid"
        assert "TOOBIG" not in tiers                 # > $10B excluded
        assert "small" in tiers.values() and "mid" in tiers.values()


class TestLeverageRank:
    def test_soft_beta_tilts_toward_higher_beta_at_equal_adv(self):
        group = [
            ("LOWB",  {"adv_usd": 10_000_000, "beta": 0.8, "cap_tier": "small"}),
            ("HIGHB", {"adv_usd": 10_000_000, "beta": 2.0, "cap_tier": "small"}),
        ]
        ordered = [s for s, _ in _leverage_rank(group, alpha=0.15)]
        assert ordered[0] == "HIGHB"   # higher beta wins at equal liquidity

    def test_adv_dominates_ranking(self):
        group = [
            ("BIG",   {"adv_usd": 100_000_000, "beta": 0.8, "cap_tier": "small"}),
            ("SMALL", {"adv_usd": 1_000_000,   "beta": 2.0, "cap_tier": "small"}),
        ]
        ordered = [s for s, _ in _leverage_rank(group, alpha=0.15)]
        assert ordered[0] == "BIG"     # soft beta cannot overturn a 100× ADV gap

    def test_missing_fields_are_neutral_stable(self):
        group = [
            ("A", {"adv_usd": None, "beta": None, "cap_tier": "small"}),
            ("B", {"adv_usd": None, "beta": None, "cap_tier": "small"}),
        ]
        ordered = [s for s, _ in _leverage_rank(group)]
        assert ordered == ["A", "B"]   # neutral key → stable insertion order
