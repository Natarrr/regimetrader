"""tests/test_advisor_consistency.py

Regression guard: the Portfolio Advisor's final_score must equal the
generate_top_lists ranking score (pre-VIX-overlay, pre-congress-boost)
for an identical input row.

This test pins the single source-of-truth contract so the two schemas
(Discord/top_lists vs UI advisor) cannot drift silently again.
"""
from __future__ import annotations

import math

from regime_trader.ui.portfolio_advisor_engine import _compute_final_score
from backend.market_intel.generate_top_lists import WEIGHTS, FACTOR_FIELDS


def _us_row(overrides: dict | None = None) -> dict:
    """A synthetic intel_source_status.json result row — all 12 factors present."""
    row = {
        # 12-factor *_score fields (as written by run_pipeline.py)
        "insider_conviction_score":  0.60,
        "insider_breadth_score":     0.45,
        "congress_score":            0.30,
        "news_sentiment_score":      0.55,
        "news_buzz_score":           0.40,
        "momentum_long_score":       0.70,
        "volume_attention_score":    0.20,
        "analyst_consensus_score":   0.65,
        "analyst_revision_score":    0.50,
        "price_target_upside_score": 0.55,
        "quality_piotroski_score":   0.75,
        "transcript_tone_score":     0.60,
        "market":                    "USA",
        "sector":                    "Information Technology",
        "cap_tier":                  "large",
        "ticker":                    "AAPL",
    }
    if overrides:
        row.update(overrides)
    return row


def _expected_score(row: dict) -> float:
    """Replicate generate_top_lists._to_entry weight application on a pre-normalized row.

    When all factors are present (no dead feeds), effective_weights == WEIGHTS
    and the score is the weighted sum of the *_score fields directly.
    """
    score = 0.0
    for short_name, field_key in FACTOR_FIELDS.items():
        val = row.get(field_key)
        if val is not None:
            score += WEIGHTS[short_name] * float(val)
    return round(score, 4)


def test_advisor_matches_generate_top_lists_all_factors():
    """With all 12 factors present, _compute_final_score == canonical weighted sum."""
    row = _us_row()
    advisor_score, factors = _compute_final_score(row)
    expected = _expected_score(row)
    assert math.isclose(advisor_score, expected, abs_tol=1e-6), (
        f"Advisor score {advisor_score} != expected {expected} — schemas have drifted."
    )


def test_advisor_factors_dict_uses_7_canonical_keys():
    """factors dict must carry exactly the 7 canonical short names from WEIGHTS."""
    row = _us_row()
    _, factors = _compute_final_score(row)
    assert set(factors) == set(WEIGHTS), (
        f"factors keys {set(factors)} != WEIGHTS keys {set(WEIGHTS)}"
    )


def test_advisor_weight_redistribution_for_missing_factor():
    """When a factor is absent (EU/Asia pattern), weight is redistributed pro-rata.

    A row with congress_score=None should still produce a non-zero score that
    equals the sum of the remaining 11 factors with renormalized weights.
    """
    row = _us_row({"congress_score": None})
    advisor_score, factors = _compute_final_score(row)

    # Recompute manually: redistribute congress weight across the 11 live factors
    live_weights = {k: v for k, v in WEIGHTS.items() if k != "congress"}
    live_total = sum(live_weights.values())
    eff = {k: w / live_total for k, w in live_weights.items()}
    expected = round(sum(eff[k] * float(row[FACTOR_FIELDS[k]]) for k in eff), 4)

    assert math.isclose(advisor_score, expected, abs_tol=1e-6), (
        f"Redistribution mismatch: advisor={advisor_score} expected={expected}"
    )
    assert factors["congress"] == 0.0  # absent factor shows as 0.0 in output


def test_advisor_all_factors_absent_returns_zero():
    """A row with no recognized factor fields scores 0.0 without raising."""
    row = {"ticker": "XYZ", "market": "USA"}
    score, factors = _compute_final_score(row)
    assert score == 0.0
    assert all(v == 0.0 for v in factors.values())
