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
    _schema_gate,
    FACTOR_FIELDS,
    PipelineIntegrityError,
    WEIGHTS,
)
from src.ingestion.run_pipeline import score_congress


def _make_results(n: int, overrides: dict | None = None) -> list:
    """Build n neutral result rows, optionally overriding specific fields."""
    base = {
        "insider_conviction_score":  0.50,
        "insider_breadth_score":     0.50,
        "congress_score":            0.50,
        "news_sentiment_score":      0.50,
        "news_buzz_score":           0.50,
        "momentum_long_score":       0.50,
        "volume_attention_score":    0.50,
        "analyst_consensus_score":   0.50,
        "analyst_revision_score":    0.50,
        "price_target_upside_score": 0.50,
        "quality_piotroski_score":   0.50,
        "transcript_tone_score":     0.50,
        "fcf_yield_score":           0.50,
        "amihud_shock_score":        0.50,
        "pb_value_up_score":         0.50,
        "roic_quality_score":        0.50,
    }
    rows = [{**base} for _ in range(n)]
    if overrides:
        for key, values in overrides.items():
            for i, v in enumerate(values):
                rows[i][key] = v
    return rows


class TestCrossSectionalNormalize:
    def test_higher_raw_score_gives_higher_normalized_score(self):
        results = _make_results(2, {"insider_conviction_score": [0.30, 0.90]})
        normed = _cross_sectional_normalize(results)
        assert normed[0]["insider_conviction"] < normed[1]["insider_conviction"]

    def test_normalized_scores_bounded_0_to_1(self):
        results = _make_results(10, {
            "insider_conviction_score": np.random.default_rng(42).uniform(0.3, 0.9, 10).tolist()
        })
        normed = _cross_sectional_normalize(results)
        for row in normed:
            for v in row.values():
                assert 0.0 <= v <= 1.0 + 1e-9

    def test_all_identical_scores_return_half(self):
        """When all tickers have the same raw score, normalized output is 0.5."""
        results = _make_results(5)   # all 0.50 by default
        normed = _cross_sectional_normalize(results)
        for row in normed:
            for v in row.values():
                assert v == pytest.approx(0.5, abs=1e-4)

    def test_all_factors_present_in_output(self):
        results = _make_results(3)
        normed = _cross_sectional_normalize(results)
        for row in normed:
            assert set(row.keys()) == set(FACTOR_FIELDS.keys())

    def test_output_length_matches_input(self):
        results = _make_results(7)
        normed = _cross_sectional_normalize(results)
        assert len(normed) == 7

    def test_single_ticker_returns_neutral(self):
        """One ticker — no peer comparison possible — returns 0.5 for all factors."""
        results = _make_results(1, {"insider_conviction_score": [0.90]})
        normed = _cross_sectional_normalize(results)
        assert normed[0]["insider_conviction"] == pytest.approx(0.5, abs=1e-4)

    def test_congress_factor_key_present(self):
        """FACTOR_FIELDS maps 'congress' → congress_score and 'momentum_long' → momentum_long_score."""
        assert FACTOR_FIELDS.get("congress") == "congress_score"
        assert FACTOR_FIELDS.get("momentum_long") == "momentum_long_score"
        assert "macro" not in FACTOR_FIELDS

    def test_momentum_long_factor_present(self):
        """FACTOR_FIELDS maps 'momentum_long' → momentum_long_score (Jegadeesh-Titman)."""
        assert FACTOR_FIELDS.get("momentum_long") == "momentum_long_score"
        assert "macro" not in FACTOR_FIELDS

    def test_2008_crash_outlier_does_not_collapse_scores(self):
        """2020 COVID analog: one ticker with extreme momentum → others not collapsed to 0."""
        scores = [0.50] * 49 + [9999.0]   # 1 extreme outlier
        results = _make_results(50, {"momentum_long_score": scores})
        normed = _cross_sectional_normalize(results)
        # The 49 normal tickers should not all map to near-zero
        normal_scores = [normed[i]["momentum_long"] for i in range(49)]
        assert max(normal_scores) > 0.30   # not all collapsed

    def test_all_zero_values_penalised_not_neutral(self):
        """A fully dead API feed (all 0.0) must return 0.0, not the neutral 0.5."""
        results = _make_results(
            5, {"insider_conviction_score": [0.0, 0.0, 0.0, 0.0, 0.0]})
        normed = _cross_sectional_normalize(results)
        for row in normed:
            assert row["insider_conviction"] == pytest.approx(0.0, abs=1e-9)

    def test_null_values_penalised_not_neutral(self):
        """Explicit JSON null (None) must be treated as 0.0 — same as dead API feed."""
        base = {field: None for field in FACTOR_FIELDS.values()}
        results = [dict(base) for _ in range(4)]
        normed = _cross_sectional_normalize(results)
        for row in normed:
            for v in row.values():
                assert v == pytest.approx(0.0, abs=1e-9)

    def test_null_mixed_with_real_values_does_not_get_neutral_credit(self):
        """Tickers with None score must rank below tickers with a real positive score."""
        results = _make_results(
            3, {"insider_conviction_score": [None, None, 0.80]})
        normed = _cross_sectional_normalize(results)
        # Real score should normalise higher than null-coerced 0.0
        assert normed[2]["insider_conviction"] > normed[0]["insider_conviction"]
        assert normed[2]["insider_conviction"] > normed[1]["insider_conviction"]


class TestScoreCongress:
    """score_congress() must return 0.0 for missing data, not the old neutral 0.5."""

    def test_none_returns_zero_not_neutral(self):
        assert score_congress(None) == pytest.approx(0.0)

    def test_empty_dict_returns_zero_not_neutral(self):
        assert score_congress({}) == pytest.approx(0.0)

    def test_equal_buys_sells_is_neutral(self):
        """When congress has traded but bought = sold, result is 0.5 (genuine neutral)."""
        assert score_congress(
            {"purchases": 2, "sales": 2, "total": 4}) == pytest.approx(0.5, abs=1e-4)

    def test_all_purchases_high_signal(self):
        result = score_congress({"purchases": 5, "sales": 0, "total": 5})
        assert result > 0.5

    def test_all_sales_low_signal(self):
        result = score_congress({"purchases": 0, "sales": 5, "total": 5})
        assert result < 0.5


class TestEffectiveWeights:
    """Dead-factor weight redistribution."""

    def _all_live_norm(self, congress=0.7):
        """Return a single-row norm dict with all 12 factors live (non-zero)."""
        return [{
            "insider_conviction": 0.3, "insider_breadth": 0.5, "congress": congress,
            "news_sentiment": 0.4, "news_buzz": 0.5, "momentum_long": 0.6,
            "volume_attention": 0.3, "analyst_consensus": 0.5, "analyst_revision": 0.4,
            "price_target_upside": 0.6, "quality_piotroski": 0.5, "transcript_tone": 0.4,
        }]

    def test_no_dead_factors_returns_original(self):
        w = _effective_weights(self._all_live_norm(), WEIGHTS)
        assert w == pytest.approx(WEIGHTS, abs=1e-6)

    def test_dead_congress_redistributes_to_live(self):
        """All-zero congress → its weight rolls to live factors."""
        norm = self._all_live_norm(congress=0.0)
        w = _effective_weights(norm, WEIGHTS)
        assert "congress" not in w
        assert abs(sum(w.values()) - 1.0) < 1e-5   # weights still sum to 1

    def test_live_weights_increase_proportionally(self):
        """Each live factor gets an equal relative boost when congress is dead."""
        norm = self._all_live_norm(congress=0.0)
        w = _effective_weights(norm, WEIGHTS)
        # insider_conviction / insider_breadth ratio must stay constant
        assert (w["insider_conviction"] / w["insider_breadth"] ==
                pytest.approx(WEIGHTS["insider_conviction"] / WEIGHTS["insider_breadth"], rel=1e-4))

    def test_all_dead_returns_original_fallback(self):
        """If everything is dead, fall back to original weights rather than divide by zero."""
        norm = [{f: 0.0 for f in WEIGHTS}]
        w = _effective_weights(norm, WEIGHTS)
        assert w == pytest.approx(WEIGHTS, abs=1e-6)

    def test_empty_norm_list_returns_original(self):
        w = _effective_weights([], WEIGHTS)
        assert w == pytest.approx(WEIGHTS, abs=1e-6)


class TestVixOverlay:
    def test_normal_regime_no_dampening(self):
        assert _apply_vix_overlay(0.80, 19.9) == pytest.approx(0.80)

    def test_bear_regime_mild_penalty(self):
        """Bear starts at VIX 20 (src.risk.regime.BEAR_THRESHOLD)."""
        assert _apply_vix_overlay(1.0, 20.0) == pytest.approx(0.80)

    def test_bear_regime_mid(self):
        assert _apply_vix_overlay(1.0, 22.0) == pytest.approx(0.80)

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
        low = _apply_vix_overlay(0.40, 35.0)
        assert high > low


class TestQuiverEvidence:
    def _make_row(self, ticker="NVDA", **overrides):
        base = {
            "ticker": ticker, "market_cap": 1e12, "sector": "Technology",
            "insider_conviction_score": 0.7, "insider_breadth_score": 0.6,
            "congress_score": 0.8,
            "news_sentiment_score": 0.5, "news_buzz_score": 0.4,
            "momentum_long_score": 0.5, "volume_attention_score": 0.3,
        }
        base.update(overrides)
        return base

    def _make_norm(self):
        return {
            "insider_conviction": 0.7, "insider_breadth": 0.6, "congress": 0.8,
            "news_sentiment": 0.5, "news_buzz": 0.4,
            "momentum_long": 0.5, "volume_attention": 0.3,
        }

    def test_to_entry_includes_quiver_evidence(self):
        from backend.market_intel.generate_top_lists import _to_entry
        evidence = {"politicians": ["Nancy Pelosi"], "recency_days": 5}
        entry = _to_entry(self._make_row(), self._make_norm(),
                          vix=None, quiver_evidence=evidence)
        assert entry["quiver_evidence"]["politicians"] == ["Nancy Pelosi"]
        assert entry["quiver_evidence"]["recency_days"] == 5

    def test_to_entry_no_evidence_gives_empty_dict(self):
        from backend.market_intel.generate_top_lists import _to_entry
        entry = _to_entry(self._make_row(), self._make_norm(), vix=None)
        assert entry.get("quiver_evidence") == {}

    def test_to_entry_none_evidence_gives_empty_dict(self):
        from backend.market_intel.generate_top_lists import _to_entry
        entry = _to_entry(self._make_row(), self._make_norm(),
                          vix=None, quiver_evidence=None)
        assert entry.get("quiver_evidence") == {}

    def test_to_entry_row_quiver_evidence_passthrough(self):
        """generate() passes row.get('quiver_evidence') — verify full roundtrip via row key."""
        from backend.market_intel.generate_top_lists import _to_entry
        row = self._make_row()
        row["quiver_evidence"] = {"congress": {"purchases": 3, "sales": 0}}
        entry = _to_entry(row, self._make_norm(), vix=None,
                          quiver_evidence=row.get("quiver_evidence"))
        assert entry["quiver_evidence"]["congress"]["purchases"] == 3


class TestToEntryEvidencePassthrough:
    def _make_norm(self):
        return {
            "insider_conviction": 0.8, "insider_breadth": 0.7, "congress": 0.6,
            "news_sentiment": 0.5, "news_buzz": 0.4,
            "momentum_long": 0.4, "volume_attention": 0.3,
        }

    def test_evidence_fields_present_in_entry(self):
        from backend.market_intel.generate_top_lists import _to_entry

        row = {
            "ticker": "AAPL", "sector": "Tech", "cap_tier": "large",
            "market_cap": 3e12, "ceo_buy": True, "form4_count": 3,
            "news_source": "finnhub",
            "insider_usd": 2_500_000.0,
            "momentum_spy_relative": 0.042,
            "volume_spike": 2.3,
        }
        entry = _to_entry(row, self._make_norm())

        assert entry["news_source"] == "finnhub"
        assert entry["insider_usd"] == pytest.approx(2_500_000.0)
        assert entry["momentum_spy_relative"] == pytest.approx(0.042)
        assert entry["volume_spike"] == pytest.approx(2.3)

    def test_evidence_fields_default_when_absent(self):
        from backend.market_intel.generate_top_lists import _to_entry

        row = {"ticker": "X", "sector": "?",
               "cap_tier": "large", "market_cap": 0}
        entry = _to_entry(row, self._make_norm())

        assert entry["news_source"] == "none"
        assert entry["insider_usd"] == pytest.approx(0.0)
        assert entry["momentum_spy_relative"] == pytest.approx(0.0)
        assert entry["volume_spike"] == pytest.approx(1.0)

    def test_esg_metadata_propagates_to_entry(self):
        from backend.market_intel.generate_top_lists import _to_entry

        row = {
            "ticker": "X", "sector": "?", "cap_tier": "large", "market_cap": 0,
            "esg_score": 22.5, "environmentalScore": 18.3,
        }
        entry = _to_entry(row, self._make_norm())

        assert entry["esg_score"] == pytest.approx(22.5)
        assert entry["esg_e_score"] == pytest.approx(18.3)
        assert entry["esg_flag"] is True


def _make_schema_row(ticker: str = "AAPL", **scores) -> dict:
    """Build a result row with configurable factor scores (default all 0.5)."""
    base = {"ticker": ticker}
    for field in FACTOR_FIELDS.values():
        base[field] = scores.get(field, 0.5)
    return base


class TestSchemaGate:
    """_schema_gate(): per-ticker validation metadata + circuit-breaker."""

    def test_complete_ticker_gets_is_complete_true(self):
        rows = [_make_schema_row()]  # all factors non-zero
        _schema_gate(rows, universe_size=1)
        assert rows[0]["_validation"]["is_complete"] is True
        assert rows[0]["_validation"]["missing_sources"] == []

    def test_ticker_with_one_zero_is_still_complete(self):
        # threshold is >2 missing, so 1 missing → still complete
        rows = [_make_schema_row(insider_breadth_score=0.0)]
        _schema_gate(rows, universe_size=1)
        v = rows[0]["_validation"]
        assert v["is_complete"] is True
        assert "insider_breadth" in v["missing_sources"]

    def test_ticker_with_two_zeros_is_still_complete(self):
        rows = [_make_schema_row(
            insider_breadth_score=0.0, congress_score=0.0)]
        _schema_gate(rows, universe_size=2)
        v = rows[0]["_validation"]
        assert v["is_complete"] is True
        assert len(v["missing_sources"]) == 2

    def test_ticker_with_seven_zeros_is_now_complete(self):
        # Threshold raised to 8 (4 new INTL factors may be 0/None for US tickers):
        # structurally-zero set for a US ticker can now reach 8 factors legitimately.
        # 7 zeros is within tolerance — is_complete = True.
        rows = [_make_schema_row(
            insider_conviction_score=0.0, insider_breadth_score=0.0,
            congress_score=0.0, news_sentiment_score=0.0, news_buzz_score=0.0,
            volume_attention_score=0.0, analyst_consensus_score=0.0,
        )]
        complete_rows = [_make_schema_row(f"T{i}") for i in range(4)]
        all_rows = rows + complete_rows
        _schema_gate(all_rows, universe_size=len(all_rows))
        v = all_rows[0]["_validation"]
        assert v["is_complete"] is True
        assert len(v["missing_sources"]) == 7

    def test_ticker_with_eleven_zeros_is_incomplete(self):
        # >10 zero factors → is_complete = False: indicates genuine price/EDGAR failure.
        # Threshold is 10 (original 6 + 4 always-absent INTL factors for US tickers).
        rows = [_make_schema_row(
            insider_conviction_score=0.0, insider_breadth_score=0.0,
            congress_score=0.0, news_sentiment_score=0.0, news_buzz_score=0.0,
            volume_attention_score=0.0, analyst_consensus_score=0.0,
            momentum_long_score=0.0, quality_piotroski_score=0.0,
            analyst_revision_score=0.0, price_target_upside_score=0.0,
        )]
        complete_rows = [_make_schema_row(f"T{i}") for i in range(4)]
        all_rows = rows + complete_rows
        _schema_gate(all_rows, universe_size=len(all_rows))
        v = all_rows[0]["_validation"]
        assert v["is_complete"] is False
        assert len(v["missing_sources"]) == 11

    def test_ticker_with_three_zeros_is_now_complete(self):
        # With threshold=4, 3 zero factors is within tolerance (normal pattern).
        rows = [_make_schema_row(
            insider_breadth_score=0.0, congress_score=0.0, news_sentiment_score=0.0)]
        complete_rows = [_make_schema_row(f"T{i}") for i in range(4)]
        all_rows = rows + complete_rows
        _schema_gate(all_rows, universe_size=len(all_rows))
        v = all_rows[0]["_validation"]
        assert v["is_complete"] is True
        assert len(v["missing_sources"]) == 3

    def test_missing_sources_names_correct_factors(self):
        rows = [_make_schema_row(
            insider_conviction_score=0.0, momentum_long_score=0.0, news_sentiment_score=0.0
        )]
        complete_rows = [_make_schema_row(f"T{i}") for i in range(4)]
        all_rows = rows + complete_rows
        _schema_gate(all_rows, universe_size=len(all_rows))
        missing = set(all_rows[0]["_validation"]["missing_sources"])
        assert missing == {"insider_conviction",
                           "momentum_long", "news_sentiment"}

    def test_none_score_counts_as_missing(self):
        row = _make_schema_row()
        row["insider_breadth_score"] = None
        complete_rows = [_make_schema_row(f"T{i}") for i in range(4)]
        all_rows = [row] + complete_rows
        _schema_gate(all_rows, universe_size=len(all_rows))
        assert "insider_breadth" in all_rows[0]["_validation"]["missing_sources"]

    def test_circuit_breaker_fires_when_below_5_percent(self):
        # Threshold=10: tickers need >10 zeros to be "incomplete".
        # 5 tickers, all with 11 missing factors → 0 complete < 40% of 5
        rows = [
            _make_schema_row(
                f"T{i}",
                insider_conviction_score=0.0, insider_breadth_score=0.0,
                congress_score=0.0, news_sentiment_score=0.0, news_buzz_score=0.0,
                volume_attention_score=0.0, analyst_consensus_score=0.0,
                momentum_long_score=0.0, quality_piotroski_score=0.0,
                analyst_revision_score=0.0, price_target_upside_score=0.0,
            )
            for i in range(5)
        ]
        with pytest.raises(PipelineIntegrityError, match="Schema gate"):
            _schema_gate(rows, universe_size=5)

    def test_circuit_breaker_does_not_fire_when_enough_complete(self):
        # 10 tickers: 3 complete, 7 missing 3 factors each (3 zeros ≤ threshold 4 → complete)
        # 10/10 = 100% ≥ 5% → should NOT raise
        complete = [_make_schema_row(f"C{i}") for i in range(3)]
        incomplete = [
            _make_schema_row(f"I{i}", insider_breadth_score=0.0,
                             congress_score=0.0, news_sentiment_score=0.0)
            for i in range(7)
        ]
        rows = complete + incomplete
        # must not raise (all 10 are "complete" with threshold=4)
        _schema_gate(rows, universe_size=10)

    def test_circuit_breaker_raises_pipeline_integrity_error_type(self):
        # Need >10 zeros to be incomplete; use 11 zero factors.
        rows = [_make_schema_row(
            insider_conviction_score=0.0, insider_breadth_score=0.0,
            congress_score=0.0, news_sentiment_score=0.0, news_buzz_score=0.0,
            volume_attention_score=0.0, analyst_consensus_score=0.0,
            momentum_long_score=0.0, quality_piotroski_score=0.0,
            analyst_revision_score=0.0, price_target_upside_score=0.0,
        )]
        with pytest.raises(PipelineIntegrityError):
            _schema_gate(rows, universe_size=1)

    def test_validation_metadata_present_on_every_row(self):
        rows = [_make_schema_row(f"T{i}") for i in range(5)]
        _schema_gate(rows, universe_size=5)
        for row in rows:
            assert "_validation" in row
            assert "is_complete" in row["_validation"]
            assert "missing_sources" in row["_validation"]

    def test_esg_exclusion_candidate_flag_added_when_esg_flag_true(self):
        rows = [_make_schema_row(f"T{i}") for i in range(5)]
        rows[0]["esg_flag"] = True
        _schema_gate(rows, universe_size=5)
        assert rows[0]["_validation"].get("esg_exclusion_candidate") is True

    def test_esg_flag_false_does_not_mark_exclusion_candidate(self):
        rows = [_make_schema_row(f"T{i}") for i in range(5)]
        rows[0]["esg_flag"] = False
        _schema_gate(rows, universe_size=5)
        assert rows[0]["_validation"].get("esg_exclusion_candidate") is None

    def test_returns_same_list_in_place(self):
        rows = [_make_schema_row()]
        returned = _schema_gate(rows, universe_size=1)
        assert returned is rows   # mutated in-place, same object
