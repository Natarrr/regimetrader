"""tests/test_risk_regime.py"""
from __future__ import annotations
import pytest
from src.risk.regime import (
    RiskRegime,
    get_regime,
    is_panic,
    score_multiplier,
    strategy_label,
    apply_capitulation_filter,
    vix_multiplier,
)


class TestRiskRegime:
    def test_normal_below_20(self):
        assert get_regime(19.9) == RiskRegime.NORMAL

    def test_bear_starts_exactly_at_20(self):
        """Spec: Bear regime = VIX 20–29 — boundary is inclusive."""
        assert get_regime(20.0) == RiskRegime.BEAR

    def test_bear_20_to_30(self):
        assert get_regime(22.0) == RiskRegime.BEAR
        assert get_regime(27.5) == RiskRegime.BEAR
        assert get_regime(29.9) == RiskRegime.BEAR

    def test_capitulation_at_30(self):
        assert get_regime(30.0) == RiskRegime.CAPITULATION

    def test_public_threshold_aliases_exported(self):
        from src.risk.regime import BEAR_THRESHOLD, CAPITULATION_THRESHOLD
        assert BEAR_THRESHOLD == 20.0
        assert CAPITULATION_THRESHOLD == 30.0

    def test_capitulation_above_40(self):
        assert get_regime(42.0) == RiskRegime.CAPITULATION

    def test_is_panic_true(self):
        assert is_panic(30.0) is True
        assert is_panic(29.9) is False

    def test_multiplier_normal(self):
        assert score_multiplier(RiskRegime.NORMAL) == 1.00

    def test_multiplier_bear(self):
        assert score_multiplier(RiskRegime.BEAR) == 0.80

    def test_multiplier_capitulation(self):
        assert score_multiplier(RiskRegime.CAPITULATION) == 0.50

    def test_strategy_label_capitulation(self):
        label = strategy_label(RiskRegime.CAPITULATION)
        assert "CAPITULATION" in label.upper() or "DISTRESSED" in label.upper()

    def test_invalid_vix_raises(self):
        with pytest.raises(ValueError):
            get_regime(float("nan"))

    def test_negative_vix_raises(self):
        with pytest.raises(ValueError):
            get_regime(-1.0)

    def test_vix_multiplier_tiers(self):
        """Shared US/INTL overlay multiplier — includes the Crash tier that
        score_multiplier (regime-level) does not carry."""
        assert vix_multiplier(15.0) == 1.00
        assert vix_multiplier(20.0) == 0.80
        assert vix_multiplier(30.0) == 0.50
        assert vix_multiplier(40.0) == 0.20
        assert vix_multiplier(55.0) == 0.20

    def test_vix_multiplier_nan_raises(self):
        # NaN comparisons are all False — silent ×1.00 bypass must be impossible
        with pytest.raises(ValueError):
            vix_multiplier(float("nan"))


class TestCapitulationFilter:
    def _make_entries(self):
        return [
            {
                "ticker": "A",
                "final_score": 0.92,
                "badge": "HIGH BUY",
                "factors": {"beta": 1.5, "quality_piotroski": 0.9, "debt_to_equity": 0.2},
            },
            {
                "ticker": "B",
                "final_score": 0.85,
                "badge": "HIGH BUY",
                "factors": {"beta": 0.7, "quality_piotroski": 0.95, "debt_to_equity": 0.1},
            },
            {
                "ticker": "C",
                "final_score": 0.80,
                "badge": "TACTICAL BUY",
                "factors": {"beta": 0.5, "quality_piotroski": 0.4, "debt_to_equity": 0.8},
            },
        ]

    def test_high_beta_filtered_out(self):
        entries = self._make_entries()
        result = apply_capitulation_filter(entries, vix=31.0)
        assert not any(e["ticker"] == "A" for e in result)  # beta=1.5 > 1.2 → removed

    def test_high_piotroski_kept(self):
        entries = self._make_entries()
        result = apply_capitulation_filter(entries, vix=31.0)
        assert any(e["ticker"] == "B" for e in result)  # beta=0.7, piotroski=0.95 → kept

    def test_no_op_in_normal_regime(self):
        entries = self._make_entries()
        result = apply_capitulation_filter(entries, vix=18.0)
        assert result == entries  # unchanged

    def test_filter_does_not_remultiply_scores(self):
        """Dampening is applied exactly once upstream (US: _apply_vix_overlay,
        INTL: cook._normalize_intl_entry). The filter must leave final_score
        untouched or US entries end up 0.25×/0.10× vs INTL 0.50×."""
        entries = self._make_entries()
        original = {e["ticker"]: e["final_score"] for e in entries}
        result = apply_capitulation_filter(entries, vix=31.0)
        for e in result:
            assert e["final_score"] == original[e["ticker"]]

    def test_capitulation_survivor_flag_set(self):
        entries = self._make_entries()
        result = apply_capitulation_filter(entries, vix=31.0)
        for e in result:
            assert e.get("_capitulation_survivor") is True

    def test_low_de_qualifies_without_piotroski(self):
        # Ticker with low Piotroski but bottom D/E quintile should still qualify
        entries = [
            {
                "ticker": "D",
                "final_score": 0.75,
                "badge": "TACTICAL BUY",
                "factors": {"beta": 0.8, "quality_piotroski": 0.20, "debt_to_equity": 0.10},
            }
        ]
        result = apply_capitulation_filter(entries, vix=35.0)
        assert any(e["ticker"] == "D" for e in result)
