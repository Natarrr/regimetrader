"""backend/tests/test_prediction_controller.py
Validates Module E — Prediction Controller (Lucas / Sargent).

Historical validation:
- Sep 2008: all 3 Minsky conditions should fire → CRITICAL
- Mar 2020: CAPE was low (not overvalued) → WARNING/WATCH at most
"""

from backend.quant_models.prediction_controller import classify_regime, minsky_moment
from backend.data.schemas import MinskyStatusOut


class TestClassifyRegime:
    def test_bear_hmm_with_macro_stress_is_crash(self):
        """Lucas (1995 Nobel) — Risk-off HMM + macro confirmation → CRASH.

        Per the docstring: CRASH requires risk-off price action AND at least one
        macro stress (TIGHTENING or CLUSTERING). Bear alone with NEUTRAL + STABLE
        maps to FRAGILE (early deterioration, not full systemic stress).
        """
        for label in ("Bear", "Crash", "Panic"):
            assert classify_regime(label, "TIGHTENING", "CLUSTERING") == "CRASH"
            assert classify_regime(label, "TIGHTENING", "STABLE") == "CRASH"
            assert classify_regime(label, "NEUTRAL", "CLUSTERING") == "CRASH"
            assert classify_regime(label, "NEUTRAL", "STABLE") == "FRAGILE"

    def test_bull_easing_stable_is_bull(self):
        assert classify_regime("Bull", "EASING", "STABLE") == "BULL"

    def test_tightening_is_overheated(self):
        assert classify_regime("Euphoria", "TIGHTENING", "STABLE") == "OVERHEATED"
        assert classify_regime("Bull", "TIGHTENING", "STABLE") == "OVERHEATED"

    def test_clustering_neutral_is_fragile(self):
        assert classify_regime("Neutral", "NEUTRAL", "CLUSTERING") == "FRAGILE"

    def test_unknown_hmm_defaults_to_fragile(self):
        result = classify_regime("Unknown", "NEUTRAL", "STABLE")
        assert result in ("FRAGILE", "BULL")  # neutral group with STABLE vol

    def test_all_valid_outputs(self):
        valid = {"BULL", "OVERHEATED", "FRAGILE", "CRASH"}
        combos = [
            ("Bull", "EASING", "STABLE"),
            ("Bull", "TIGHTENING", "STABLE"),
            ("Bear", "NEUTRAL", "CLUSTERING"),
            ("Neutral", "NEUTRAL", "CLUSTERING"),
            ("Euphoria", "EASING", "STABLE"),
        ]
        for hmm, mon, vol in combos:
            assert classify_regime(hmm, mon, vol) in valid


class TestMinskyMoment:
    # ── Sep 2008: All three conditions met ────────────────────────────────────
    def test_sep2008_critical(self):
        """GFC Minsky test: persistence=0.99, CAPE~94th pct, curve inverted."""
        result = minsky_moment(
            garch_persistence=0.99,
            cape_percentile=96.0,
            yield_spread_bps=-30.0,
        )
        assert result.triggered is True
        assert result.alert_level == "CRITICAL"
        assert result.conditions_met == 3

    # ── Mar 2020: CAPE was low — not all 3 conditions ─────────────────────────
    def test_mar2020_not_critical(self):
        """COVID crash: CAPE was ~26 (not overvalued), curve wasn't inverted.

        GARCH persistence was extreme, but valuation and yield curve conditions
        were NOT simultaneously met. Minsky should NOT fire CRITICAL.
        """
        result = minsky_moment(
            garch_persistence=0.99,  # vol clustering: YES
            cape_percentile=60.0,    # CAPE not extreme: NO
            yield_spread_bps=50.0,   # curve not inverted: NO
        )
        assert result.triggered is False
        assert result.alert_level in ("WATCH", "WARNING")
        assert result.conditions_met == 1

    # ── Alert level gradient ───────────────────────────────────────────────────
    def test_zero_conditions_is_clear(self):
        result = minsky_moment(0.95, 70.0, 100.0)
        assert result.alert_level == "CLEAR"
        assert result.conditions_met == 0

    def test_one_condition_is_watch(self):
        result = minsky_moment(0.99, 70.0, 50.0)  # only GARCH
        assert result.alert_level == "WATCH"
        assert result.conditions_met == 1

    def test_two_conditions_is_warning(self):
        result = minsky_moment(0.99, 96.0, 50.0)  # GARCH + CAPE, no inversion
        assert result.alert_level == "WARNING"
        assert result.conditions_met == 2

    def test_three_conditions_is_critical(self):
        result = minsky_moment(0.99, 96.0, -20.0)
        assert result.alert_level == "CRITICAL"
        assert result.triggered is True

    # ── Output schema ──────────────────────────────────────────────────────────
    def test_returns_pydantic_model(self):
        result = minsky_moment(0.95, 70.0, 100.0)
        assert isinstance(result, MinskyStatusOut)

    def test_narrative_non_empty(self):
        result = minsky_moment(0.99, 96.0, -20.0)
        assert len(result.narrative) > 10

    def test_critical_narrative_mentions_minsky(self):
        result = minsky_moment(0.99, 96.0, -20.0)
        assert "MINSKY" in result.narrative.upper()

    def test_boundary_persistence(self):
        """Exactly 0.98 should NOT trigger GARCH condition."""
        result_at = minsky_moment(0.98, 96.0, -20.0)
        result_above = minsky_moment(0.981, 96.0, -20.0)
        assert result_at.conditions_met < result_above.conditions_met
