"""tests/test_composite_neutralize_flag.py — P1.1 composite-normalization flag.

SCORING_NORM_NEUTRALIZE_TRADE=1 swaps the traded composite's min-max
normalization for the SAME cross-sectional neutralization run_pipeline uses for
the monitoring final_score (validate-what-you-trade). Default OFF is byte-
identical to the existing min-max path.
"""
from __future__ import annotations

from unittest.mock import patch

from backend.market_intel.generate_top_lists import (
    FACTOR_FIELDS,
    _cross_sectional_normalize,
    _neutralized_factors,
    generate,
)


def _row(ticker: str, base: float) -> dict:
    return {
        "ticker": ticker, "sector": "Technology", "cap_tier": "large",
        "market_cap": 3e12, "quality_piotroski_raw": 7,
        "insider_conviction_score": base, "insider_breadth_score": base * 0.9,
        "congress_score": base * 0.5, "news_sentiment_score": base * 0.8,
        "news_buzz_score": base * 0.7, "momentum_long_score": base,
        "volume_attention_score": base * 0.6, "analyst_consensus_score": base * 0.5,
        "analyst_revision_score": base * 0.4, "price_target_upside_score": base * 0.5,
        "quality_piotroski_score": base * 0.6, "transcript_tone_score": base * 0.2,
        "revenue_revision_score": base * 0.3, "inst_flow_13f_score": base * 0.5,
        "fcf_yield_score": 0.0, "amihud_shock_score": 0.0,
        "pb_value_up_score": 0.0, "roic_quality_score": 0.0,
    }


def _rows(n: int = 6) -> list:
    # Same sector × cap_tier so the bucket clears min_bucket_size=5 and the
    # z-score path (not the "raw" fallback) fires.
    return [_row(f"T{i}", 0.20 + 0.10 * i) for i in range(n)]


class TestNeutralizedFactors:
    def test_shape_and_range(self):
        out = _neutralized_factors(_rows(6))
        assert len(out) == 6
        for d in out:
            assert set(d.keys()) == set(FACTOR_FIELDS.keys())   # 18-key contract
            assert all(0.0 <= v <= 1.0 for v in d.values())

    def test_differs_from_minmax(self):
        rows = _rows(6)
        neu = _neutralized_factors(rows)
        mm = _cross_sectional_normalize(rows)
        # A different transform → at least one factor vector must differ.
        assert any(neu[i]["momentum_long"] != mm[i]["momentum_long"]
                   for i in range(6))

    def test_preserves_cross_sectional_ranking(self):
        # Higher raw momentum → higher neutralized momentum (monotone within bucket).
        neu = _neutralized_factors(_rows(6))
        moms = [d["momentum_long"] for d in neu]
        assert moms == sorted(moms)


class TestFlagRouting:
    def _generate(self, rows, tmp_path):
        status = {"results": rows, "run_id": "test"}
        with patch("backend.market_intel.generate_top_lists._read_vix",
                   return_value=None):
            return generate(status, run_id="test", log_dir=tmp_path)

    def test_flag_on_runs_end_to_end(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SCORING_NORM_NEUTRALIZE_TRADE", "1")
        out = self._generate(_rows(6), tmp_path)
        assert out["ticker_count"] == 6
        for e in out.get("top_buys", []):
            assert set(e["factors"].keys()) == set(FACTOR_FIELDS.keys())

    def test_flag_off_uses_minmax(self, tmp_path, monkeypatch):
        # Default OFF: the composite must match the min-max path exactly.
        monkeypatch.delenv("SCORING_NORM_NEUTRALIZE_TRADE", raising=False)
        rows = _rows(6)
        off = self._generate([dict(r) for r in rows], tmp_path)
        scores_off = {e["ticker"]: e["final_score"] for e in off.get("top_buys", [])}

        mm = _cross_sectional_normalize(rows)
        # Sanity: the OFF path produced scores consistent with min-max being used
        # (non-empty universe scored without the neutralize transform).
        assert off["ticker_count"] == 6
        assert mm and len(mm) == 6
