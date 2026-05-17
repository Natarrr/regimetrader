"""tests/test_cross_sectional.py
Unit tests for cross-sectional factor normalization in generate_top_lists.

Markowitz (1990 Nobel) — portfolio construction requires comparable, bounded
signals. Validates that normalization produces peer-relative scores rather
than absolute thresholds, and that uniform factors don't crash or mislead.
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.market_intel.generate_top_lists import (
    _apply_vix_overlay,
    _cross_sectional_normalize,
    _effective_weights,
    FACTOR_FIELDS,
    WEIGHTS,
)
from scripts.run_pipeline import score_congress


def _make_results(n: int, overrides: dict | None = None) -> list:
    """Build n neutral result rows, optionally overriding specific fields."""
    base = {
        "edgar_score":    0.50,
        "insider_score":  0.50,
        "congress_score": 0.50,
        "news_score":     0.50,
        "momentum_score": 0.50,
    }
    rows = [{**base} for _ in range(n)]
    if overrides:
        for key, values in overrides.items():
            for i, v in enumerate(values):
                rows[i][key] = v
    return rows


class TestCrossSectionalNormalize:
    def test_higher_raw_score_gives_higher_normalized_score(self):
        results = _make_results(2, {"edgar_score": [0.30, 0.90]})
        normed  = _cross_sectional_normalize(results)
        assert normed[0]["edgar"] < normed[1]["edgar"]

    def test_normalized_scores_bounded_0_to_1(self):
        results = _make_results(10, {
            "edgar_score": np.random.default_rng(42).uniform(0.3, 0.9, 10).tolist()
        })
        normed = _cross_sectional_normalize(results)
        for row in normed:
            for v in row.values():
                assert 0.0 <= v <= 1.0 + 1e-9

    def test_all_identical_scores_return_half(self):
        """When all tickers have the same raw score, normalized output is 0.5."""
        results = _make_results(5)   # all 0.50 by default
        normed  = _cross_sectional_normalize(results)
        for row in normed:
            for v in row.values():
                assert v == pytest.approx(0.5, abs=1e-4)

    def test_all_five_factors_present_in_output(self):
        results = _make_results(3)
        normed  = _cross_sectional_normalize(results)
        for row in normed:
            assert set(row.keys()) == {"edgar", "insider", "congress", "news", "momentum"}

    def test_output_length_matches_input(self):
        results = _make_results(7)
        normed  = _cross_sectional_normalize(results)
        assert len(normed) == 7

    def test_single_ticker_returns_neutral(self):
        """One ticker — no peer comparison possible — returns 0.5 for all factors."""
        results = _make_results(1, {"edgar_score": [0.90]})
        normed  = _cross_sectional_normalize(results)
        assert normed[0]["edgar"] == pytest.approx(0.5, abs=1e-4)

    def test_congress_factor_key_present(self):
        """FACTOR_FIELDS maps 'congress' → congress_score."""
        assert FACTOR_FIELDS.get("congress") == "congress_score"
        assert "macro" not in FACTOR_FIELDS

    def test_momentum_factor_present(self):
        """FACTOR_FIELDS maps 'momentum' → momentum_score (pipeline field name)."""
        assert FACTOR_FIELDS.get("momentum") == "momentum_score"

    def test_2008_crash_outlier_does_not_collapse_scores(self):
        """2020 COVID analog: one ticker with extreme momentum → others not collapsed to 0."""
        scores = [0.50] * 49 + [9999.0]   # 1 extreme outlier
        results = _make_results(50, {"momentum_score": scores})
        normed  = _cross_sectional_normalize(results)
        # The 49 normal tickers should not all map to near-zero
        normal_scores = [normed[i]["momentum"] for i in range(49)]
        assert max(normal_scores) > 0.30   # not all collapsed

    def test_all_zero_values_penalised_not_neutral(self):
        """A fully dead API feed (all 0.0) must return 0.0, not the neutral 0.5."""
        results = _make_results(5, {"edgar_score": [0.0, 0.0, 0.0, 0.0, 0.0]})
        normed  = _cross_sectional_normalize(results)
        for row in normed:
            assert row["edgar"] == pytest.approx(0.0, abs=1e-9)

    def test_null_values_penalised_not_neutral(self):
        """Explicit JSON null (None) must be treated as 0.0 — same as dead API feed."""
        base = {
            "edgar_score": None, "insider_score": None,
            "congress_score": None, "news_score": None, "momentum_score": None,
        }
        results = [dict(base) for _ in range(4)]
        normed  = _cross_sectional_normalize(results)
        for row in normed:
            for v in row.values():
                assert v == pytest.approx(0.0, abs=1e-9)

    def test_null_mixed_with_real_values_does_not_get_neutral_credit(self):
        """Tickers with None score must rank below tickers with a real positive score."""
        results = _make_results(3, {"edgar_score": [None, None, 0.80]})
        normed  = _cross_sectional_normalize(results)
        # Real score should normalise higher than null-coerced 0.0
        assert normed[2]["edgar"] > normed[0]["edgar"]
        assert normed[2]["edgar"] > normed[1]["edgar"]


class TestScoreCongress:
    """score_congress() must return 0.0 for missing data, not the old neutral 0.5."""

    def test_none_returns_zero_not_neutral(self):
        assert score_congress(None) == pytest.approx(0.0)

    def test_empty_dict_returns_zero_not_neutral(self):
        assert score_congress({}) == pytest.approx(0.0)

    def test_equal_buys_sells_is_neutral(self):
        """When congress has traded but bought = sold, result is 0.5 (genuine neutral)."""
        assert score_congress({"purchases": 2, "sales": 2, "total": 4}) == pytest.approx(0.5, abs=1e-4)

    def test_all_purchases_high_signal(self):
        result = score_congress({"purchases": 5, "sales": 0, "total": 5})
        assert result > 0.5

    def test_all_sales_low_signal(self):
        result = score_congress({"purchases": 0, "sales": 5, "total": 5})
        assert result < 0.5


class TestEffectiveWeights:
    """Dead-factor weight redistribution."""

    def test_no_dead_factors_returns_original(self):
        norm = [{"edgar": 0.3, "insider": 0.5, "congress": 0.7, "news": 0.4, "momentum": 0.6}]
        w = _effective_weights(norm, WEIGHTS)
        assert w == pytest.approx(WEIGHTS, abs=1e-6)

    def test_dead_congress_redistributes_to_live(self):
        """All-zero congress → its 22% weight rolls to edgar/insider/news/momentum."""
        norm = [{"edgar": 0.5, "insider": 0.5, "congress": 0.0, "news": 0.5, "momentum": 0.5}]
        w = _effective_weights(norm, WEIGHTS)
        assert "congress" not in w
        assert abs(sum(w.values()) - 1.0) < 1e-5   # weights still sum to 1

    def test_live_weights_increase_proportionally(self):
        """Each live factor gets an equal relative boost when congress is dead."""
        norm = [{"edgar": 0.5, "insider": 0.5, "congress": 0.0, "news": 0.5, "momentum": 0.5}]
        w = _effective_weights(norm, WEIGHTS)
        # edgar / insider ratio must stay constant
        assert w["edgar"] / w["insider"] == pytest.approx(WEIGHTS["edgar"] / WEIGHTS["insider"], rel=1e-4)

    def test_all_dead_returns_original_fallback(self):
        """If everything is dead, fall back to original weights rather than divide by zero."""
        norm = [{"edgar": 0.0, "insider": 0.0, "congress": 0.0, "news": 0.0, "momentum": 0.0}]
        w = _effective_weights(norm, WEIGHTS)
        assert w == pytest.approx(WEIGHTS, abs=1e-6)

    def test_empty_norm_list_returns_original(self):
        w = _effective_weights([], WEIGHTS)
        assert w == pytest.approx(WEIGHTS, abs=1e-6)


class TestVixOverlay:
    def test_normal_regime_no_dampening(self):
        assert _apply_vix_overlay(0.80, 24.9) == pytest.approx(0.80)

    def test_bear_regime_mild_penalty(self):
        assert _apply_vix_overlay(1.0, 25.0) == pytest.approx(0.80)

    def test_bear_regime_upper_boundary(self):
        assert _apply_vix_overlay(1.0, 29.9) == pytest.approx(0.80)

    def test_panic_regime_half_score(self):
        assert _apply_vix_overlay(1.0, 30.0) == pytest.approx(0.50)

    def test_crash_regime_severe_dampening(self):
        assert _apply_vix_overlay(1.0, 40.0) == pytest.approx(0.20)

    def test_vix_none_no_change(self):
        assert _apply_vix_overlay(0.75, None) == pytest.approx(0.75)

    def test_dampening_preserves_relative_ranking(self):
        """Higher raw score stays higher after dampening (monotone transform)."""
        high = _apply_vix_overlay(0.80, 35.0)
        low  = _apply_vix_overlay(0.40, 35.0)
        assert high > low


class TestQuiverEvidence:
    def _make_row(self, ticker="NVDA", **overrides):
        base = {
            "ticker": ticker, "market_cap": 1e12, "sector": "Technology",
            "edgar_score": 0.7, "insider_score": 0.6, "congress_score": 0.8,
            "news_score": 0.5, "momentum_score": 0.5,
        }
        base.update(overrides)
        return base

    def _make_norm(self):
        return {"edgar": 0.7, "insider": 0.6, "congress": 0.8, "news": 0.5, "momentum": 0.5}

    def test_to_entry_includes_quiver_evidence(self):
        from backend.market_intel.generate_top_lists import _to_entry
        evidence = {"politicians": ["Nancy Pelosi"], "recency_days": 5}
        entry = _to_entry(self._make_row(), self._make_norm(), vix=None, quiver_evidence=evidence)
        assert entry["quiver_evidence"]["politicians"] == ["Nancy Pelosi"]
        assert entry["quiver_evidence"]["recency_days"] == 5

    def test_to_entry_no_evidence_gives_empty_dict(self):
        from backend.market_intel.generate_top_lists import _to_entry
        entry = _to_entry(self._make_row(), self._make_norm(), vix=None)
        assert entry.get("quiver_evidence") == {}

    def test_to_entry_none_evidence_gives_empty_dict(self):
        from backend.market_intel.generate_top_lists import _to_entry
        entry = _to_entry(self._make_row(), self._make_norm(), vix=None, quiver_evidence=None)
        assert entry.get("quiver_evidence") == {}

    def test_to_entry_row_quiver_evidence_passthrough(self):
        """generate() passes row.get('quiver_evidence') — verify full roundtrip via row key."""
        from backend.market_intel.generate_top_lists import _to_entry
        row = self._make_row()
        row["quiver_evidence"] = {"congress": {"purchases": 3, "sales": 0}}
        entry = _to_entry(row, self._make_norm(), vix=None, quiver_evidence=row.get("quiver_evidence"))
        assert entry["quiver_evidence"]["congress"]["purchases"] == 3
