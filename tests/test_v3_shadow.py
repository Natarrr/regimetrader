"""tests/test_v3_shadow.py — v3.0 shadow-mode wiring (SCORING_V3_SHADOW=1).

Shadow contract: v2.2 fields stay byte-identical; v3 raw columns are computed
from data already in _score_ticker scope (plus two cached client calls) and
final_score_v3 / pillar columns ride alongside via apply_v3_shadow().
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.ingestion.v3_shadow import (
    apply_v3_shadow,
    compute_v3_raw_columns,
    v3_shadow_enabled,
)


def _days_ago(n):
    return (datetime.now(timezone.utc).date() - timedelta(days=n)).isoformat()


class FakeClient:
    def __init__(self, ratios=None, cf=None, ev=None, inst=None):
        self._ratios = ratios if ratios is not None else {}
        self._cf = cf if cf is not None else []
        self._ev = ev
        self._inst = inst if inst is not None else {}

    def get_ratios_ttm(self, ticker):
        return self._ratios

    def get_cash_flow_statements(self, ticker, limit=4):
        return self._cf

    def get_enterprise_value(self, ticker):
        return self._ev

    def get_institutional_ownership(self, ticker):
        return self._inst


_RATIOS = {"netProfitMarginTTM": 0.15, "assetTurnoverTTM": 1.0,
           "debtToEquityRatioTTM": 0.3}
_CF = [{"freeCashFlow": 75.0}] * 4
_INST = {"investorsHolding": 1000, "investorsHoldingChange": 50,
         "increasedPositions": 300, "reducedPositions": 200,
         "ownershipPercentChange": 2.0}
_P_TXS = [
    {"code": "P", "value": 100_000.0, "date": _days_ago(10), "is_ceo": True},
    {"code": "P", "value": 100_000.0, "date": _days_ago(60), "is_ceo": False},
]


def _compute(client=None, **overrides):
    kwargs = dict(
        ticker="AAPL",
        fmp_client=client or FakeClient(ratios=_RATIOS, cf=_CF, ev=6000.0,
                                        inst=_INST),
        ratios_row=None,
        p_transactions=_P_TXS,
        conviction_score=0.8,
        breadth_score=0.3,
        revision_pct=0.30,
        n_analysts=10,
        eps_surprise_pct=0.10,
        eps_surprise_days=30,
        congress_score=0.42,
    )
    kwargs.update(overrides)
    return compute_v3_raw_columns(**kwargs)


class TestComputeColumns:
    def test_full_inputs(self):
        cols = _compute()
        # DuPont fallback (no ROA field): roa_eff = 0.15·1.0
        assert cols["quality_dupont_score"] == pytest.approx(0.83095, abs=1e-3)
        # 300 FCF / 6000 EV = 5% yield → 0.25
        assert cols["fcf_yield_score"] == pytest.approx(0.25, abs=1e-6)
        assert cols["analyst_revision_score_v3"] == pytest.approx(1.0)
        assert cols["pead_surprise_score"] == pytest.approx(0.60)
        # p30=1, p31_180=1 → velocity log1p(1)/log1p(5); tilt 0.75
        assert cols["insider_alpha_score"] == pytest.approx(0.60982, abs=1e-4)
        assert cols["inst_flow_13f_score"] == pytest.approx(0.68, abs=1e-6)
        # surge multiplier is identity until dated congress trades are plumbed
        assert cols["congress_score_v3"] == pytest.approx(0.42)

    def test_bulk_ratios_row_takes_priority(self):
        # When the bulk snapshot record is in scope, no per-ticker refetch.
        cols = _compute(client=FakeClient(ratios={}, cf=_CF, ev=6000.0,
                                          inst=_INST),
                        ratios_row=_RATIOS)
        assert cols["quality_dupont_score"] == pytest.approx(0.83095, abs=1e-3)

    def test_sparse_client_degrades_cleanly(self):
        cols = _compute(client=FakeClient(ratios={}, cf=[], ev=None, inst={}),
                        revision_pct=None, n_analysts=0,
                        eps_surprise_pct=None, eps_surprise_days=0)
        assert cols["quality_dupont_score"] == 0.0      # broken feed → dead
        assert cols["fcf_yield_score"] == 0.0           # unsigned dead
        assert cols["analyst_revision_score_v3"] is None
        assert cols["pead_surprise_score"] is None
        assert cols["inst_flow_13f_score"] is None      # signed unavailable

    def test_no_purchases_dead_insider(self):
        cols = _compute(p_transactions=[], conviction_score=0.0,
                        breadth_score=0.0)
        assert cols["insider_alpha_score"] == 0.0


def _shadow_row(i, value=0.7):
    row = {
        "ticker": f"T{i}", "sector": "Technology", "cap_tier": "large",
        "market": "USA", "quality_piotroski_raw": 8,
        "final_score": 0.55, "weight_coverage": 1.0,
        # v3 raw columns as compute_v3_raw_columns / _score_ticker emit them:
        "quality_dupont_score": value, "fcf_yield_score": value,
        "quality_piotroski_score": value,
        "analyst_revision_score_v3": value, "pead_surprise_score": value,
        "price_target_upside_score": value,
        "insider_alpha_score": value, "congress_score_v3": value,
        "inst_flow_13f_score": value,
    }
    return row


class TestApplyShadow:
    def test_merges_v3_outputs_without_touching_v22(self):
        rows = [_shadow_row(i) for i in range(6)]
        out = apply_v3_shadow(rows)
        for r in out:
            assert r["final_score"] == 0.55              # v2.2 untouched
            assert r["final_score_v3"] == pytest.approx(0.5)
            assert r["pillar_fundamental_score"] == pytest.approx(0.5)
            assert r["weight_coverage_v3"] == pytest.approx(1.0)
            assert r["_low_coverage_v3"] is False

    def test_rows_missing_v3_columns_score_as_unavailable(self):
        # FMP structural-failure rows never got v3 columns — every factor is
        # None → final_score_v3 None + low-coverage, v2.2 fields untouched.
        rows = [_shadow_row(i) for i in range(5)]
        bare = {"ticker": "T5", "sector": "Technology", "cap_tier": "large",
                "market": "USA", "quality_piotroski_raw": None,
                "final_score": 0.1, "weight_coverage": 0.3}
        out = apply_v3_shadow(rows + [bare])
        assert out[5]["final_score"] == 0.1
        assert out[5]["final_score_v3"] is None
        assert out[5]["_low_coverage_v3"] is True

    def test_empty_results_noop(self):
        assert apply_v3_shadow([]) == []


class TestEnvFlag:
    def test_disabled_by_default(self, monkeypatch):
        monkeypatch.delenv("SCORING_V3_SHADOW", raising=False)
        assert v3_shadow_enabled() is False

    def test_enabled_when_set(self, monkeypatch):
        monkeypatch.setenv("SCORING_V3_SHADOW", "1")
        assert v3_shadow_enabled() is True


class TestRegionGuard:
    """--region universe validation (US message byte-identical to v2.2)."""

    def _rows(self, *tickers):
        return [{"ticker": t} for t in tickers]

    def test_us_clean_universe_passes(self):
        from src.ingestion.run_pipeline import _validate_universe_region
        _validate_universe_region(self._rows("AAPL", "MSFT"), "US")

    def test_us_rejects_intl_with_legacy_message(self):
        from src.ingestion.run_pipeline import _validate_universe_region
        with pytest.raises(ValueError, match="US-only"):
            _validate_universe_region(self._rows("AAPL", "ASML.AS"), "US")

    def test_eu_accepts_eu_only(self):
        from src.ingestion.run_pipeline import _validate_universe_region
        _validate_universe_region(self._rows("ASML.AS", "SAP.DE"), "EU")

    def test_eu_rejects_us_ticker(self):
        from src.ingestion.run_pipeline import _validate_universe_region
        with pytest.raises(ValueError, match="non-EU"):
            _validate_universe_region(self._rows("ASML.AS", "AAPL"), "EU")

    def test_asia_rejects_eu_ticker(self):
        from src.ingestion.run_pipeline import _validate_universe_region
        with pytest.raises(ValueError, match="non-ASIA"):
            _validate_universe_region(self._rows("7203.T", "SAP.DE"), "ASIA")

    def test_unknown_region_rejected(self):
        from src.ingestion.run_pipeline import _validate_universe_region
        with pytest.raises(ValueError, match="region"):
            _validate_universe_region(self._rows("AAPL"), "LATAM")
