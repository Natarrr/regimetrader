# Path: tests/test_transcript_tone.py
"""TDD tests for score_transcript_tone — pure text scorer in news_signals.py.

The scorer is a pure function: text → float | None.
The FMP fetch wrapper remains in run_pipeline.py (score_transcript_tone(ticker, client)).

RED phase: all tests fail until score_transcript_tone is added to news_signals.py
and WEIGHTS_US is updated to include transcript_tone.
"""
import pytest
from src.scoring.news_signals import score_transcript_tone
from src.config.weights import WEIGHTS_US, WEIGHTS_GLOBAL, WEIGHTS_EU, WEIGHTS_ASIA


# ── Pure scorer contract ──────────────────────────────────────────────────────

def test_none_input_returns_none():
    assert score_transcript_tone(None) is None


def test_empty_string_returns_none():
    assert score_transcript_tone("") is None


def test_no_guidance_phrases_returns_none():
    """Irrelevant text — no raise/lower/maintain phrases → no signal."""
    assert score_transcript_tone(
        "revenue was 5 billion. customers love our product. we shipped on time."
    ) is None


def test_raised_guidance_returns_above_half():
    text = (
        "We are raising guidance for the full year. Our Q3 results exceeded expectations. "
        "We are raising our full-year revenue target by 5%."
    )
    result = score_transcript_tone(text)
    assert result is not None
    assert result > 0.5


def test_lowered_guidance_returns_below_half():
    text = (
        "We are lowering guidance due to macro headwinds. "
        "The headwinds in our key markets are significant."
    )
    result = score_transcript_tone(text)
    assert result is not None
    assert result < 0.5


def test_reaffirmed_guidance_returns_near_half():
    text = (
        "We are reaffirming guidance for the year. We reaffirm our commitment "
        "and remain comfortable with our guidance range."
    )
    result = score_transcript_tone(text)
    assert result is not None
    # reaffirm/maintain is neutral-positive (0.5-0.6 range)
    assert 0.4 <= result <= 0.7


def test_result_always_in_unit_interval():
    texts = [
        "raising guidance raised guidance increase our guidance",
        "lowering guidance lowered guidance headwinds headwinds headwinds",
        "reaffirming guidance reaffirm confident in our comfortable with our guidance",
    ]
    for t in texts:
        r = score_transcript_tone(t)
        if r is not None:
            assert 0.0 <= r <= 1.0, f"Out-of-range score {r} for: {t[:40]}"


def test_raised_beats_lowered_when_both_present_with_more_raises():
    text = (
        "raising guidance raising guidance raising guidance "
        "lowering guidance"  # 3 raises vs 1 lower
    )
    result = score_transcript_tone(text)
    assert result is not None
    assert result > 0.5


def test_lowered_beats_raised_when_more_lowers():
    text = (
        "lowering guidance lowering guidance lowering guidance "
        "raising guidance"  # 3 lowers vs 1 raise
    )
    result = score_transcript_tone(text)
    assert result is not None
    assert result < 0.5


# ── WEIGHTS_US now includes transcript_tone ───────────────────────────────────

def test_weights_us_includes_transcript_tone():
    assert "transcript_tone" in WEIGHTS_US


def test_weights_us_transcript_tone_positive():
    assert WEIGHTS_US["transcript_tone"] > 0.0


def test_weights_us_still_sums_to_one():
    assert abs(sum(WEIGHTS_US.values()) - 1.0) < 1e-6


def test_weights_global_transcript_tone_zero():
    """INTL keeps transcript_tone at 0.0 — FMP transcripts are US-only."""
    assert WEIGHTS_GLOBAL.get("transcript_tone", 0.0) == 0.0


def test_weights_eu_transcript_tone_zero():
    assert WEIGHTS_EU.get("transcript_tone", 0.0) == 0.0


def test_weights_asia_transcript_tone_zero():
    assert WEIGHTS_ASIA.get("transcript_tone", 0.0) == 0.0
