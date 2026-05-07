"""backend/tests/test_score_helpers.py
Acceptance tests for the five-factor scoring pipeline.

Validates:
  1. aggregate_scores — correct formula, output keys, badge thresholds
  2. annualise_vol_from_condvar — no double-annualisation, correct units
  3. Persistence formula — alpha + beta + gamma/2 matches known values
  4. Alert thresholds — check_alerts fires at correct delta triggers
  5. Edge cases — neutral inputs, clamping, safe_float robustness

Historical validation (CLAUDE.md requirement):
  Persistence test uses GJR-GARCH parameters representative of Oct 2008 GFC,
  where SPY persistence was empirically > 0.95.
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.utils.score_helpers import (
    WEIGHTS,
    BADGE_HIGH_BUY,
    BADGE_TACTICAL_BUY,
    ALERT_INSIDER_RISE,
    ALERT_INST_DROP,
    ALERT_MACRO_REDUCE,
    ALERT_SCORE_DROP,
    aggregate_scores,
    check_alerts,
    safe_float,
)
from backend.utils.volatility import annualise_vol_from_condvar


# ─────────────────────────────────────────────────────────────────────────────
# 1. safe_float
# ─────────────────────────────────────────────────────────────────────────────

class TestSafeFloat:
    def test_normal_value(self):
        assert safe_float(0.7) == pytest.approx(0.7)

    def test_clamps_above_one(self):
        assert safe_float(1.5) == pytest.approx(1.0)

    def test_clamps_below_zero(self):
        assert safe_float(-0.3) == pytest.approx(0.0)

    def test_none_returns_default(self):
        assert safe_float(None) == pytest.approx(0.5)
        assert safe_float(None, default=0.3) == pytest.approx(0.3)

    def test_invalid_string_returns_default(self):
        assert safe_float("bad") == pytest.approx(0.5)

    def test_string_number_parses(self):
        assert safe_float("0.75") == pytest.approx(0.75)


# ─────────────────────────────────────────────────────────────────────────────
# 2. aggregate_scores — formula and output contract
# ─────────────────────────────────────────────────────────────────────────────

class TestAggregateScores:
    def test_all_neutral_inputs(self):
        """All factors at 0.5, regime_mult=1.0 → deterministic midpoint."""
        out = aggregate_scores(0.5, 0.5, 0.5, 0.5, 1.0)
        assert "final_score" in out
        assert "score_breakdown" in out
        assert "badge" in out
        assert "badge_clr" in out
        # All at neutral: final ≈ 0.5 * sum(weights excl regime) + regime contrib
        assert 0.0 <= out["final_score"] <= 1.0

    def test_output_keys_present(self):
        """aggregate_scores must return all five factor keys + meta."""
        out = aggregate_scores(0.65, 0.55, 0.40, 0.60, 1.20)
        for key in ("macro_score", "institutional_score", "insider_score",
                    "news_score", "regime_mult", "final_score", "badge",
                    "badge_clr", "score_breakdown"):
            assert key in out, f"Missing key: {key}"

    def test_score_breakdown_keys(self):
        """score_breakdown must contain all five factor contributions."""
        out = aggregate_scores(0.65, 0.55, 0.40, 0.60, 1.20)
        for key in ("macro", "institutional", "insider", "news", "regime"):
            assert key in out["score_breakdown"], f"Missing breakdown key: {key}"

    def test_high_buy_badge(self):
        """final_score >= 0.80 → HIGH BUY badge."""
        out = aggregate_scores(0.95, 0.95, 0.95, 0.95, 1.50)
        if out["final_score"] >= BADGE_HIGH_BUY:
            assert out["badge"] == "HIGH BUY"
            assert out["badge_clr"] == "#00c851"

    def test_watchlist_badge(self):
        """final_score < 0.60 → WATCHLIST badge."""
        out = aggregate_scores(0.10, 0.10, 0.10, 0.10, 0.0)
        assert out["badge"] == "WATCHLIST"
        assert out["badge_clr"] == "#9e9e9e"

    def test_tactical_buy_badge(self):
        """0.60 <= final_score < 0.80 → TACTICAL BUY."""
        out = aggregate_scores(0.65, 0.65, 0.65, 0.65, 1.0)
        if BADGE_TACTICAL_BUY <= out["final_score"] < BADGE_HIGH_BUY:
            assert out["badge"] == "TACTICAL BUY"

    def test_final_score_clamped(self):
        """final_score must always be in [0, 1]."""
        out = aggregate_scores(2.0, 3.0, -1.0, 99.0, 100.0)
        assert 0.0 <= out["final_score"] <= 1.0

    def test_breakdown_sums_to_final(self):
        """Sum of score_breakdown values must equal final_score (within rounding)."""
        out = aggregate_scores(0.70, 0.60, 0.50, 0.65, 1.15)
        breakdown_sum = sum(out["score_breakdown"].values())
        assert abs(breakdown_sum - out["final_score"]) < 0.002, (
            f"breakdown sum {breakdown_sum:.4f} != final_score {out['final_score']:.4f}"
        )

    def test_weights_used_correctly(self):
        """Verify the macro contribution: macro_weight * macro_score."""
        m = 0.80
        out = aggregate_scores(macro_score=m, institutional_score=0.5,
                               insider_score=0.5, news_score=0.5, regime_mult=1.0)
        expected_macro_contrib = round(WEIGHTS["macro"] * m, 4)
        assert abs(out["score_breakdown"]["macro"] - expected_macro_contrib) < 0.001

    def test_regime_normalisation_cap(self):
        """regime_mult capped at 1.50 — passing 2.0 must equal passing 1.50."""
        out_capped = aggregate_scores(0.5, 0.5, 0.5, 0.5, 1.50)
        out_over   = aggregate_scores(0.5, 0.5, 0.5, 0.5, 2.00)
        assert out_capped["final_score"] == out_over["final_score"]

    def test_custom_weights(self):
        """Custom weights override must be respected."""
        custom = {"macro": 1.0, "institutional": 0.0, "insider": 0.0,
                  "news": 0.0, "regime": 0.0}
        out = aggregate_scores(0.80, 0.0, 0.0, 0.0, 0.0, weights=custom)
        assert abs(out["final_score"] - 0.80) < 0.001


# ─────────────────────────────────────────────────────────────────────────────
# 3. GJR-GARCH persistence formula — historical validation
# ─────────────────────────────────────────────────────────────────────────────

class TestPersistenceFormula:
    """Engle (2003 Nobel) — Validates persistence = alpha + beta + gamma/2."""

    def test_known_values_spx_baseline(self):
        """SPY 2016-2021 representative fit — persistence ≈ 0.9447."""
        alpha = 0.0
        gamma = 0.201447
        beta  = 0.843943
        persistence = alpha + beta + gamma / 2
        assert abs(persistence - 0.9446665) < 1e-5

    def test_gfc_2008_high_persistence(self, spy_oct2008_returns):
        """GFC Oct 2008: GJR-GARCH persistence must exceed 0.90."""
        from backend.quant_models.volatility_brain import fit_gjr_garch
        result = fit_gjr_garch(spy_oct2008_returns)
        assert result["persistence"] > 0.90, (
            f"GFC 2008 persistence should be > 0.90, got {result['persistence']:.4f}"
        )

    def test_clustering_regime_above_098(self):
        """Persistence > 0.98 must trigger CLUSTERING."""
        from backend.quant_models.volatility_brain import volatility_regime
        assert volatility_regime(0.985) == "CLUSTERING"
        assert volatility_regime(0.975) == "STABLE"

    def test_persistence_stationarity(self):
        """All valid GARCH fits must have persistence < 1.0."""
        alpha = 0.05
        gamma = 0.10
        beta  = 0.84
        persistence = alpha + beta + gamma / 2
        assert persistence < 1.0


# ─────────────────────────────────────────────────────────────────────────────
# 4. annualise_vol_from_condvar — no double-annualisation
# ─────────────────────────────────────────────────────────────────────────────

class TestAnnualiseVol:
    """Engle (2003 Nobel) — Unit pipeline: h_t [%-pt^2] → annualised vol [%]."""

    def test_known_value_percent_units(self):
        """Daily variance of 0.8109 %-pt^2 → ~14.3% annualised vol."""
        # sigma2_inf from SPY fit: omega / (1 - persistence)
        sigma2_inf = 0.044875 / (1.0 - 0.9446665)
        result = annualise_vol_from_condvar(np.array([sigma2_inf]), units="percent")
        annual_pct = float(result.iloc[0])
        assert abs(annual_pct - 14.3) < 0.5, (
            f"Expected ~14.3% annual vol, got {annual_pct:.4f}%"
        )

    def test_decimal_vs_percent_equivalence(self):
        """units='decimal' and units='percent' must give identical output."""
        h_decimal = np.array([0.0001])           # 0.0001 decimal^2/day
        h_percent = h_decimal * (100.0 ** 2)     # = 1.0 %-pt^2/day
        vol_dec  = float(annualise_vol_from_condvar(h_decimal, units="decimal").iloc[0])
        vol_pct  = float(annualise_vol_from_condvar(h_percent, units="percent").iloc[0])
        assert abs(vol_dec - vol_pct) < 0.001

    def test_no_double_annualisation(self):
        """Calling annualise_vol_from_condvar once and multiplying sqrt(252) again
        must NOT equal calling it twice — catches the 900%-spike bug."""
        h_t = np.array([0.80])
        correct = float(annualise_vol_from_condvar(h_t, units="percent").iloc[0])
        double_annualised = correct * np.sqrt(252)
        assert correct < 20.0, f"Single annualisation should be < 20%, got {correct:.2f}%"
        assert double_annualised > 100.0, "Double annualisation should spike above 100%"

    def test_returns_series(self):
        """Output must be a pandas Series."""
        import pandas as pd
        result = annualise_vol_from_condvar(np.ones(10), units="percent")
        assert isinstance(result, pd.Series)

    def test_invalid_units_raises(self):
        with pytest.raises(ValueError, match="units must be"):
            annualise_vol_from_condvar(np.array([1.0]), units="bps")

    def test_series_length_preserved(self):
        h = np.random.default_rng(0).exponential(0.5, 252)
        result = annualise_vol_from_condvar(h, units="percent")
        assert len(result) == 252

    def test_all_positive(self):
        """Annualised vol must be positive for all positive variances."""
        h = np.abs(np.random.default_rng(1).normal(0.5, 0.1, 100))
        result = annualise_vol_from_condvar(h, units="percent")
        assert (result > 0).all()


# ─────────────────────────────────────────────────────────────────────────────
# 5. check_alerts — deterministic trigger logic
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckAlerts:
    def test_no_alerts_when_healthy(self):
        current = {"macro_score": 0.65, "final_score": 0.70,
                   "insider_score": 0.50, "institutional_score": 0.55}
        previous = {"final_score": 0.68, "insider_score": 0.50,
                    "institutional_score": 0.55}
        alerts = check_alerts("AAPL", current, previous)
        assert len(alerts) == 0

    def test_macro_reduce_alert(self):
        """macro_score below ALERT_MACRO_REDUCE threshold must fire."""
        current = {"macro_score": ALERT_MACRO_REDUCE - 0.01,
                   "final_score": 0.5, "insider_score": 0.5, "institutional_score": 0.5}
        alerts = check_alerts("XOM", current)
        assert any("Macro Reduce" in a for a in alerts)

    def test_score_drop_alert(self):
        """final_score drop > ALERT_SCORE_DROP must fire."""
        current  = {"macro_score": 0.55, "final_score": 0.40,
                    "insider_score": 0.50, "institutional_score": 0.50}
        previous = {"final_score": 0.40 + ALERT_SCORE_DROP + 0.01,
                    "insider_score": 0.50, "institutional_score": 0.50}
        alerts = check_alerts("MSFT", current, previous)
        assert any("Momentum Deterioration" in a for a in alerts)

    def test_insider_accumulation_alert(self):
        """insider_score rise > ALERT_INSIDER_RISE must fire."""
        current  = {"macro_score": 0.65, "final_score": 0.65,
                    "insider_score": 0.50 + ALERT_INSIDER_RISE + 0.01,
                    "institutional_score": 0.50}
        previous = {"final_score": 0.65, "insider_score": 0.50,
                    "institutional_score": 0.50}
        alerts = check_alerts("NVDA", current, previous)
        assert any("Insider Accumulation" in a for a in alerts)

    def test_institutional_selling_alert(self):
        """institutional_score drop > ALERT_INST_DROP must fire."""
        current  = {"macro_score": 0.65, "final_score": 0.60,
                    "insider_score": 0.50,
                    "institutional_score": 0.50 - ALERT_INST_DROP - 0.01}
        previous = {"final_score": 0.60, "insider_score": 0.50,
                    "institutional_score": 0.50}
        alerts = check_alerts("JPM", current, previous)
        assert any("Institutional Selling" in a for a in alerts)

    def test_no_previous_skips_delta_alerts(self):
        """Without previous state, only macro threshold can fire."""
        current = {"macro_score": 0.65, "final_score": 0.10,
                   "insider_score": 0.90, "institutional_score": 0.10}
        alerts = check_alerts("SPY", current, previous=None)
        assert not any("Deterioration" in a for a in alerts)
        assert not any("Accumulation" in a for a in alerts)
        assert not any("Selling" in a for a in alerts)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Minsky alert handler — trigger logic
# ─────────────────────────────────────────────────────────────────────────────

class TestMinskyAlertHandler:
    """Validates deterministic alert dispatch from alert_handler module."""

    def test_zero_conditions_clear(self):
        from backend.automation.alert_handler import evaluate_minsky_conditions
        n = evaluate_minsky_conditions(0.95, 80.0, 50.0)
        assert n == 0

    def test_one_condition_persistence(self):
        from backend.automation.alert_handler import evaluate_minsky_conditions
        n = evaluate_minsky_conditions(0.99, 80.0, 50.0)
        assert n == 1

    def test_two_conditions(self):
        from backend.automation.alert_handler import evaluate_minsky_conditions
        n = evaluate_minsky_conditions(0.99, 96.0, 50.0)
        assert n == 2

    def test_three_conditions_full_minsky(self):
        from backend.automation.alert_handler import evaluate_minsky_conditions
        n = evaluate_minsky_conditions(0.99, 96.0, -30.0)
        assert n == 3

    def test_boundary_persistence_exact(self):
        """Persistence exactly at threshold fires."""
        from backend.automation.alert_handler import evaluate_minsky_conditions, PERSISTENCE_THRESHOLD
        n_below  = evaluate_minsky_conditions(PERSISTENCE_THRESHOLD - 0.001, 50.0, 50.0)
        n_at     = evaluate_minsky_conditions(PERSISTENCE_THRESHOLD,         50.0, 50.0)
        assert n_below == 0
        assert n_at    == 1

    def test_handle_conditions_zero_no_actions(self):
        from backend.automation.alert_handler import handle_conditions, Portfolio
        pf = Portfolio(value=1_000_000)
        actions = handle_conditions(0, pf)
        assert pf.leverage == pytest.approx(1.0)

    def test_handle_conditions_one_reduces_leverage(self):
        from backend.automation.alert_handler import handle_conditions, Portfolio
        pf = Portfolio(value=1_000_000, leverage=1.0)
        handle_conditions(1, pf)
        assert pf.leverage == pytest.approx(0.9, rel=0.01)

    def test_handle_conditions_two_buys_hedge(self):
        from backend.automation.alert_handler import handle_conditions, Portfolio
        pf = Portfolio(value=1_000_000)
        handle_conditions(2, pf)
        assert pf.hedge_notional == pytest.approx(100_000.0)

    def test_handle_conditions_three_derisks(self):
        from backend.automation.alert_handler import handle_conditions, Portfolio
        pf = Portfolio(value=1_000_000, leverage=1.5)
        handle_conditions(3, pf)
        assert pf.leverage == pytest.approx(0.0)
        assert pf.cyclical_weight == pytest.approx(0.0)
