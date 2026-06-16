# Path: tests/test_revenue_revision.py
"""TDD tests for revenue_revision factor.

RED phase: score_revenue_revision doesn't exist yet; WEIGHTS don't include it;
FMPClient.get_revenue_estimates doesn't exist yet.
"""
import pytest
from src.scoring.consensus_signals import score_revenue_revision
from src.config.weights import WEIGHTS_US, WEIGHTS_GLOBAL, WEIGHTS_EU, WEIGHTS_ASIA


# ── Pure scorer contract ──────────────────────────────────────────────────────

def test_none_revision_returns_none():
    assert score_revenue_revision(None, 1.0, 5) is None


def test_low_analyst_count_returns_none():
    """< 3 analysts → None (SIGNED: low coverage ≠ bearish)."""
    assert score_revenue_revision(1.05, 1.0, 2) is None
    assert score_revenue_revision(1.05, 1.0, 0) is None


def test_flat_revision_returns_half():
    """No revision (current == prior) → 0.5 (neutral)."""
    result = score_revenue_revision(1.0, 1.0, 5)
    assert result == pytest.approx(0.5, abs=0.05)


def test_positive_revision_above_half():
    """Revenue estimate raised 10% → score > 0.5."""
    result = score_revenue_revision(1.10, 1.0, 5)
    assert result is not None
    assert result > 0.5


def test_negative_revision_below_half():
    """Revenue estimate cut 10% → score < 0.5."""
    result = score_revenue_revision(0.90, 1.0, 5)
    assert result is not None
    assert result < 0.5


def test_result_clipped_to_unit_interval():
    """Extreme revisions (+100%, -100%) stay in [0, 1]."""
    high = score_revenue_revision(2.0, 1.0, 10)
    low  = score_revenue_revision(0.0, 1.0, 10)
    assert high is not None and 0.0 <= high <= 1.0
    assert low  is not None and 0.0 <= low  <= 1.0


def test_damping_scales_with_analyst_count():
    """More analysts → stronger signal (larger deviation from 0.5)."""
    r_few  = score_revenue_revision(1.10, 1.0, 3)
    r_many = score_revenue_revision(1.10, 1.0, 10)
    assert r_few is not None and r_many is not None
    # Both above 0.5; thin-coverage deviation is smaller
    assert (r_many - 0.5) >= (r_few - 0.5)


def test_zero_prior_returns_none():
    """Prior estimate near zero → degenerate base → None."""
    assert score_revenue_revision(0.01, 0.0, 5) is None


# ── WEIGHTS include revenue_revision ─────────────────────────────────────────

def test_weights_us_includes_revenue_revision():
    assert "revenue_revision" in WEIGHTS_US


def test_weights_us_revenue_revision_positive():
    assert WEIGHTS_US["revenue_revision"] > 0.0


def test_weights_us_still_sums_to_one_after_revenue():
    assert abs(sum(WEIGHTS_US.values()) - 1.0) < 1e-6


def test_weights_global_includes_revenue_revision():
    assert "revenue_revision" in WEIGHTS_GLOBAL
    assert WEIGHTS_GLOBAL["revenue_revision"] > 0.0


def test_weights_eu_includes_revenue_revision():
    assert "revenue_revision" in WEIGHTS_EU
    assert WEIGHTS_EU["revenue_revision"] > 0.0


def test_weights_asia_includes_revenue_revision():
    assert "revenue_revision" in WEIGHTS_ASIA
    assert WEIGHTS_ASIA["revenue_revision"] > 0.0


def test_all_weight_sets_sum_to_one():
    for name, w in [("US", WEIGHTS_US), ("GLOBAL", WEIGHTS_GLOBAL),
                    ("EU", WEIGHTS_EU), ("ASIA", WEIGHTS_ASIA)]:
        assert abs(sum(w.values()) - 1.0) < 1e-6, f"WEIGHTS_{name} sum != 1.0"


# ── FMPClient has get_revenue_estimates ──────────────────────────────────────

def test_fmp_client_has_get_revenue_estimates():
    from src.services.fmp_client import FMPClient
    assert hasattr(FMPClient, "get_revenue_estimates"), (
        "FMPClient.get_revenue_estimates() not found — add stable/revenue-estimates endpoint"
    )
