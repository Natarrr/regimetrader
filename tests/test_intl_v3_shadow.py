"""tests/test_intl_v3_shadow.py — v3.0 INTL shadow wiring.

The v2.2 INTL job scores EU+ASIA co-mingled under one profile (region
"INTL") with no neutralization. Under SCORING_V3_SHADOW=1 the engine
additionally splits the pool by get_region(ticker), runs each region
through engine_v3 (sector-bucket z-scoring + pillar math), and attaches
final_score_v3 alongside the untouched composite_score.
"""
from __future__ import annotations

import json

import pytest

from src.engine.engine import StrategyEngine
from src.ingestion.fmp_fetcher import FMPFetcher


# ── StrategyEngine shadow split ───────────────────────────────────────────────

_EU_METRIC_KEYS = {
    "quality_piotroski_score", "fcf_yield_score", "pb_value_up_score",
    "analyst_consensus_score", "analyst_revision_score_v3",
    "price_target_upside_score_v3", "inst_concentration_score",
    "dividend_sustain_score", "amihud_shock_score",
}
_ASIA_METRIC_KEYS = {
    "margin_expansion_score", "roic_quality_score", "quality_piotroski_score",
    "analyst_revision_score_v3", "revision_velocity_score",
    "price_target_upside_score_v3", "inst_concentration_score",
    "dividend_sustain_score", "amihud_shock_score",
}


def _asset(ticker, keys, value=0.7):
    metrics = {k: value for k in keys}
    metrics["quality_piotroski_raw"] = 8
    metrics["_v3_cap_tier"] = "large"
    # v2.2 keys the legacy composite uses:
    metrics["momentum_long_score"] = 0.6
    metrics["quality_piotroski_score"] = value
    return {"ticker": ticker, "sector": "Technology", "metrics": metrics}


@pytest.fixture
def engine(tmp_path):
    profile = {
        "region": "INTL",
        "active_factors": {"momentum_long": 0.6, "quality_piotroski": 0.4},
        "output_filename": "top_lists_intl.json",
    }
    p = tmp_path / "intl.json"
    p.write_text(json.dumps(profile), encoding="utf-8")
    return StrategyEngine(str(p))


def _universe():
    eu = [_asset(f"EU{i}.PA", _EU_METRIC_KEYS) for i in range(6)]
    # ASIA gets a varied factor so its bucket has spread — must NOT leak
    # into the EU pool's statistics.
    asia = []
    for i in range(6):
        a = _asset(f"100{i}.T", _ASIA_METRIC_KEYS, value=0.7)
        a["metrics"]["roic_quality_score"] = 0.1 + 0.15 * i
        asia.append(a)
    return eu + asia


class TestEngineShadowSplit:
    def test_no_shadow_no_v3_fields(self, engine, monkeypatch):
        monkeypatch.delenv("SCORING_V3_SHADOW", raising=False)
        out = engine.score_ticker_pool(_universe())
        assert all("final_score_v3" not in r for r in out)

    def test_shadow_attaches_v3_without_touching_composite(
            self, engine, monkeypatch):
        universe = _universe()
        monkeypatch.delenv("SCORING_V3_SHADOW", raising=False)
        baseline = {r["ticker"]: r["composite_score"]
                    for r in engine.score_ticker_pool(universe)}
        monkeypatch.setenv("SCORING_V3_SHADOW", "1")
        out = engine.score_ticker_pool(universe)
        for r in out:
            assert r["composite_score"] == baseline[r["ticker"]]
            assert "final_score_v3" in r

    def test_eu_and_asia_pools_isolated(self, engine, monkeypatch):
        # EU factors are uniform → every EU neutral is 0.5 → final 0.5,
        # regardless of the spread injected into the ASIA pool.
        monkeypatch.setenv("SCORING_V3_SHADOW", "1")
        out = engine.score_ticker_pool(_universe())
        eu_rows = [r for r in out if r["ticker"].endswith(".PA")]
        assert len(eu_rows) == 6
        for r in eu_rows:
            assert r["final_score_v3"] == pytest.approx(0.5)
            assert r["pillar_fundamental_score"] == pytest.approx(0.5)

    def test_us_ticker_in_intl_pool_skipped_not_fatal(
            self, engine, monkeypatch):
        monkeypatch.setenv("SCORING_V3_SHADOW", "1")
        universe = _universe()
        universe.append(_asset("AAPL", _EU_METRIC_KEYS))
        out = engine.score_ticker_pool(universe)
        aapl = next(r for r in out if r["ticker"] == "AAPL")
        assert aapl.get("final_score_v3") is None
        assert aapl["composite_score"] > 0  # v2.2 path unaffected


# ── FMPFetcher v3 column fetches ─────────────────────────────────────────────

def _q(date, filing, rev, op):
    return {"date": date, "filingDate": filing,
            "revenue": rev, "operatingIncome": op}


class FakeIntlClient:
    def __init__(self):
        self.ratios = {"dividendYieldTTM": 0.04,
                       "dividendPayoutRatioTTM": 0.45}
        self.cf = [{"freeCashFlow": 75.0, "dividendsPaid": -25.0}] * 4
        self.inst = {"ownershipPercent": 40.0, "investorsHolding": 400,
                     "increasedPositions": 300, "reducedPositions": 200}
        self.quarters = [
            _q("2026-03-31", "2026-05-10", 100.0, 12.0),
            _q("2025-12-31", "2026-02-10", 100.0, 12.0),
            _q("2025-09-30", "2025-11-10", 100.0, 12.0),
            _q("2025-06-30", "2025-08-10", 100.0, 12.0),
            _q("2025-03-31", "2025-05-10", 100.0, 10.0),
            _q("2024-12-31", "2025-02-10", 100.0, 10.0),
            _q("2024-09-30", "2024-11-10", 100.0, 10.0),
            _q("2024-06-30", "2024-08-10", 100.0, 10.0),
        ]
        self.estimates = [
            {"estimatedEpsAvg": 1.2, "numberAnalystEstimatedEps": 8},
            {"estimatedEpsAvg": 1.0, "numberAnalystEstimatedEps": 8},
            {"estimatedEpsAvg": 1.0, "numberAnalystEstimatedEps": 8},
            {"estimatedEpsAvg": 1.25, "numberAnalystEstimatedEps": 8},
        ]

    def get_ratios_ttm(self, ticker):
        return self.ratios

    def get_cash_flow_statements(self, ticker, limit=4):
        return self.cf

    def get_institutional_ownership(self, ticker):
        return self.inst

    def get_income_statements(self, ticker, period="quarter", limit=8):
        return self.quarters if period == "quarter" else []

    def get_analyst_estimates(self, ticker, period="quarter", limit=6):
        return self.estimates

    def get_analyst_estimate_revision(self, ticker):
        return 0.30, 10

    def get_upside_to_target(self, ticker, max_age_days=90):
        return 0.62

    def get_quote(self, ticker):
        return {"sector": "Technology", "marketCap": 50e9}


class TestFetcherV3Columns:
    def _fetcher(self):
        return FMPFetcher.__new__(FMPFetcher)  # bypass __init__ (no API key)

    def test_full_columns(self):
        cols = self._fetcher()._v3_intl_columns("ASML.AS", FakeIntlClient())
        assert cols["dividend_sustain_score"] == pytest.approx(0.90, abs=1e-6)
        assert cols["inst_concentration_score"] == pytest.approx(0.66925, abs=1e-3)
        assert cols["margin_expansion_score"] == pytest.approx(0.60)
        assert cols["revision_velocity_score"] == pytest.approx(1.0)
        assert cols["analyst_revision_score_v3"] == pytest.approx(1.0)
        assert cols["price_target_upside_score_v3"] == pytest.approx(0.62)
        assert cols["_v3_sector"] == "Technology"
        assert cols["_v3_cap_tier"] == "large"

    def test_sparse_client_degrades_to_unavailable(self):
        client = FakeIntlClient()
        client.ratios = {}
        client.cf = []
        client.inst = {}
        client.quarters = []
        client.estimates = []
        client.get_analyst_estimate_revision = lambda t: (None, 0)
        client.get_upside_to_target = lambda t, max_age_days=90: None
        client.get_quote = lambda t: {}
        cols = self._fetcher()._v3_intl_columns("ASML.AS", client)
        assert cols["dividend_sustain_score"] is None
        assert cols["inst_concentration_score"] is None
        assert cols["margin_expansion_score"] is None
        assert cols["revision_velocity_score"] is None
        assert cols["analyst_revision_score_v3"] is None
        assert cols["price_target_upside_score_v3"] is None
