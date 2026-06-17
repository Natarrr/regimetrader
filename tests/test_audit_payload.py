"""TDD tests for scripts/audit_payload.py — pre-flight Discord pipeline audit."""
import pytest

from src.delivery.audit_payload import (
    audit,
    ScoreDivergenceError,
    BadgeMismatchError,
    SortingError,
    CrossContaminationError,
    GeographicLeakageError,
    StructuralIntegrityError,
    VIXCoherenceError,
    VIXOverlayError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_payload(top_buys=None, mid_caps=None, small_caps=None, vix=17.0):
    """Minimal valid top_lists dict."""
    return {
        "vix": vix,
        "kill_switch": False,
        "top_buys": top_buys if top_buys is not None else [],
        "mid_caps": mid_caps if mid_caps is not None else [],
        "small_caps": small_caps if small_caps is not None else [],
    }


def _entry(ticker="AAPL", score=0.75, badge="TACTICAL BUY", market="USA", congress=0.0):
    """Minimal valid entry dict."""
    return {
        "ticker": ticker,
        "final_score": score,
        "badge": badge,
        "market": market,
        "factors": {"congress": congress},
    }


# ---------------------------------------------------------------------------
# A. Score range
# ---------------------------------------------------------------------------

def test_score_in_range_passes():
    payload = _make_payload(top_buys=[_entry(score=0.75)])
    assert audit(payload) is True


def test_score_above_1_raises():
    payload = _make_payload(top_buys=[_entry(score=1.01, badge="HIGH BUY")])
    with pytest.raises(ScoreDivergenceError):
        audit(payload)


def test_score_below_0_raises():
    payload = _make_payload(top_buys=[_entry(score=-0.01, badge="WATCHLIST")])
    with pytest.raises(ScoreDivergenceError):
        audit(payload)


def test_capitulation_watchlist_bucket_is_audited():
    """Regression: under CAPITULATION the cook empties top_buys_* and moves
    everything into watchlist — the safety gate must still see those entries."""
    payload = _make_payload(vix=35.0)
    payload["watchlist"] = [_entry(ticker="JNJ", score=1.5, badge="WATCHLIST")]
    with pytest.raises(ScoreDivergenceError):
        audit(payload)


def test_intl_mid_small_buckets_are_audited():
    payload = _make_payload()
    payload["eu_mid_small"] = [
        _entry(ticker="KGX.DE", score=0.45, badge="WATCHLIST",
               market="EUROPE", congress=0.8),
    ]
    with pytest.raises(CrossContaminationError):
        audit(payload)


# ---------------------------------------------------------------------------
# B. Badge consistency
# ---------------------------------------------------------------------------

def test_badge_high_buy_correct():
    payload = _make_payload(top_buys=[_entry(score=0.85, badge="HIGH BUY")])
    assert audit(payload) is True


def test_badge_tactical_correct():
    payload = _make_payload(top_buys=[_entry(score=0.70, badge="TACTICAL BUY")])
    assert audit(payload) is True


def test_badge_watchlist_correct():
    payload = _make_payload(top_buys=[_entry(score=0.40, badge="WATCHLIST")])
    assert audit(payload) is True


def test_badge_mismatch_raises():
    # score=0.85 should be HIGH BUY, not WATCHLIST
    payload = _make_payload(top_buys=[_entry(score=0.85, badge="WATCHLIST")])
    with pytest.raises(BadgeMismatchError):
        audit(payload)


# ---------------------------------------------------------------------------
# C. Sort order
# ---------------------------------------------------------------------------

def test_sort_correct():
    entries = [
        _entry(ticker="A", score=0.90, badge="HIGH BUY"),
        _entry(ticker="B", score=0.80, badge="HIGH BUY"),
        _entry(ticker="C", score=0.70, badge="TACTICAL BUY"),
    ]
    payload = _make_payload(top_buys=entries)
    assert audit(payload) is True


def test_sort_wrong_raises():
    entries = [
        _entry(ticker="A", score=0.70, badge="TACTICAL BUY"),
        _entry(ticker="B", score=0.90, badge="HIGH BUY"),
        _entry(ticker="C", score=0.80, badge="HIGH BUY"),
    ]
    payload = _make_payload(top_buys=entries)
    with pytest.raises(SortingError):
        audit(payload)


# ---------------------------------------------------------------------------
# D. Geographic leakage
# ---------------------------------------------------------------------------

def test_geo_us_no_suffix():
    payload = _make_payload(top_buys=[_entry(ticker="AAPL", market="USA")])
    assert audit(payload) is True


def test_geo_eu_suffix():
    payload = _make_payload(top_buys=[
        _entry(ticker="SAP.DE", score=0.75, badge="TACTICAL BUY", market="EUROPE")
    ])
    assert audit(payload) is True


def test_geo_leak_suffix_us_market():
    # Ticker has "." but market is USA → GeographicLeakageError
    payload = _make_payload(top_buys=[
        _entry(ticker="SAP.DE", score=0.75, badge="TACTICAL BUY", market="USA")
    ])
    with pytest.raises(GeographicLeakageError):
        audit(payload)


def test_geo_leak_no_suffix_asia():
    # Ticker has no "." but market is ASIA → GeographicLeakageError
    payload = _make_payload(top_buys=[
        _entry(ticker="AAPL", score=0.75, badge="TACTICAL BUY", market="ASIA")
    ])
    with pytest.raises(GeographicLeakageError):
        audit(payload)


# ---------------------------------------------------------------------------
# E. Cross-contamination
# ---------------------------------------------------------------------------

def test_no_congress_eu():
    payload = _make_payload(top_buys=[
        _entry(ticker="SAP.DE", score=0.75, badge="TACTICAL BUY", market="EUROPE", congress=0.0)
    ])
    assert audit(payload) is True


def test_congress_eu_raises():
    payload = _make_payload(top_buys=[
        _entry(ticker="SAP.DE", score=0.75, badge="TACTICAL BUY", market="EUROPE", congress=0.1)
    ])
    with pytest.raises(CrossContaminationError):
        audit(payload)


def test_congress_usa_ok():
    # USA tickers may have congress signal
    payload = _make_payload(top_buys=[
        _entry(ticker="MSFT", score=0.75, badge="TACTICAL BUY", market="USA", congress=0.5)
    ])
    assert audit(payload) is True


# ---------------------------------------------------------------------------
# F. VIX coherence
# ---------------------------------------------------------------------------

def test_vix_valid():
    payload = _make_payload(vix=17.0)
    assert audit(payload) is True


def test_vix_negative_raises():
    payload = _make_payload(vix=-1.0)
    with pytest.raises(VIXCoherenceError):
        audit(payload)


def test_vix_absurd_raises():
    payload = _make_payload(vix=999.0)
    with pytest.raises(VIXCoherenceError):
        audit(payload)


# ---------------------------------------------------------------------------
# G. Ticker format (StructuralIntegrityError)
# ---------------------------------------------------------------------------

def test_ticker_valid_us():
    payload = _make_payload(top_buys=[_entry(ticker="MSFT", market="USA")])
    assert audit(payload) is True


def test_ticker_valid_eu():
    payload = _make_payload(top_buys=[
        _entry(ticker="SAP.DE", score=0.75, badge="TACTICAL BUY", market="EUROPE")
    ])
    assert audit(payload) is True


def test_ticker_invalid_raises():
    # "toolong123" does not match the allowed pattern
    payload = _make_payload(top_buys=[
        _entry(ticker="toolong123", score=0.75, badge="TACTICAL BUY", market="USA")
    ])
    with pytest.raises(StructuralIntegrityError):
        audit(payload)


# ---------------------------------------------------------------------------
# E2. Dynamic range validation — replaces static InternationalScoreOverflowError
# ---------------------------------------------------------------------------

def test_intl_score_of_0_95_passes_eu():
    """EU score of 0.95 must pass now that the 0.90 ceiling is removed."""
    payload = _make_payload(top_buys=[
        _entry(ticker="ASML.AS", score=0.95, badge="HIGH BUY", market="EUROPE")
    ])
    assert audit(payload) is True


def test_intl_score_of_1_0_passes_eu():
    """EU score of exactly 1.0 is now valid — perfect factors, no dampening."""
    payload = _make_payload(top_buys=[
        _entry(ticker="SAP.DE", score=1.0, badge="HIGH BUY", market="EUROPE")
    ])
    assert audit(payload) is True


def test_intl_score_of_1_0_passes_asia():
    """Asia score of exactly 1.0 is now valid."""
    payload = _make_payload(top_buys=[
        _entry(ticker="7203.T", score=1.0, badge="HIGH BUY", market="ASIA")
    ])
    assert audit(payload) is True


def test_intl_score_above_1_still_raises():
    """Score > 1.0 must still raise ScoreDivergenceError regardless of market."""
    payload = _make_payload(top_buys=[
        _entry(ticker="SAP.DE", score=1.01, badge="HIGH BUY", market="EUROPE")
    ])
    with pytest.raises(ScoreDivergenceError):
        audit(payload)


def test_international_score_overflow_error_not_exported():
    """InternationalScoreOverflowError must no longer exist in audit_payload."""
    import src.delivery.audit_payload as ap_module
    assert not hasattr(ap_module, "InternationalScoreOverflowError"), (
        "InternationalScoreOverflowError was removed in v2.2-global"
    )


# ---------------------------------------------------------------------------
# SMID leverage sleeve bucket (top_buys_smid)
# ---------------------------------------------------------------------------

def test_smid_bucket_score_range_audited():
    """Check A must cover the SMID bucket."""
    payload = _make_payload()
    payload["top_buys_smid"] = [_entry(ticker="AAOI", score=1.5, badge="WATCHLIST")]
    with pytest.raises(ScoreDivergenceError):
        audit(payload)


def test_smid_bucket_badge_audited():
    """Check B must cover the SMID bucket."""
    payload = _make_payload()
    payload["top_buys_smid"] = [_entry(ticker="AAOI", score=0.85, badge="WATCHLIST")]
    with pytest.raises(BadgeMismatchError):
        audit(payload)


# ---------------------------------------------------------------------------
# I. VIX macro-overlay applied (kill-switch dampening)
# ---------------------------------------------------------------------------

def test_vix_overlay_normal_regime_is_noop():
    """Under NORMAL (multiplier 1.0) any in-range score passes — overlay no-op."""
    payload = _make_payload(vix=17.0, top_buys=[_entry(score=0.95, badge="HIGH BUY")])
    assert audit(payload) is True


def test_vix_overlay_panic_score_above_ceiling_raises():
    """VIX≥30 dampens ×0.50 — an un-dampened 0.70 score is a kill-switch bypass."""
    payload = _make_payload(vix=35.0, top_buys=[_entry(score=0.70, badge="TACTICAL BUY")])
    with pytest.raises(VIXOverlayError):
        audit(payload)


def test_vix_overlay_panic_score_within_ceiling_passes():
    """A correctly dampened score (≤0.50 under Panic) passes."""
    payload = _make_payload(vix=35.0, top_buys=[_entry(score=0.45, badge="WATCHLIST")])
    assert audit(payload) is True


def test_vix_overlay_bear_ceiling_enforced():
    """VIX≥20 dampens ×0.80 — a 0.90 score exceeds the bear ceiling."""
    payload = _make_payload(vix=25.0, top_buys=[_entry(score=0.90, badge="HIGH BUY")])
    with pytest.raises(VIXOverlayError):
        audit(payload)


def test_vix_overlay_crash_ceiling_enforced():
    """VIX≥40 dampens ×0.20 — anything above 0.20 is an un-applied overlay."""
    payload = _make_payload(vix=42.0)
    payload["watchlist"] = [_entry(ticker="JNJ", score=0.30, badge="WATCHLIST")]
    with pytest.raises(VIXOverlayError):
        audit(payload)


def test_vix_overlay_applies_to_intl_buckets():
    """INTL entries are dampened by cook_toplists — the ceiling applies there too."""
    payload = _make_payload(vix=35.0)
    payload["eu_mid_small"] = [
        _entry(ticker="SAP.DE", score=0.65, badge="TACTICAL BUY", market="EUROPE")
    ]
    with pytest.raises(VIXOverlayError):
        audit(payload)


def test_smid_bucket_geo_leak_audited():
    """Check D must cover the SMID bucket (US-only pool by construction)."""
    payload = _make_payload()
    payload["top_buys_smid"] = [
        _entry(ticker="BAD.DE", score=0.75, badge="TACTICAL BUY", market="USA")
    ]
    with pytest.raises(GeographicLeakageError):
        audit(payload)


def test_smid_not_sort_checked():
    """top_buys_smid is sorted by leverage_score, not final_score — check C
    deliberately excludes it, so ascending final_score must pass."""
    payload = _make_payload()
    payload["top_buys_smid"] = [
        _entry(ticker="LOW", score=0.70, badge="TACTICAL BUY"),
        _entry(ticker="HIGH", score=0.90, badge="HIGH BUY"),
    ]
    assert audit(payload) is True


def test_smid_leverage_score_above_one_passes():
    """leverage_score is a ranking key (max 1.10) — check A gates final_score only."""
    payload = _make_payload()
    entry = _entry(ticker="AAOI", score=0.85, badge="HIGH BUY")
    entry["leverage_score"] = 1.05
    payload["top_buys_smid"] = [entry]
    assert audit(payload) is True


# ---------------------------------------------------------------------------
# H. On-demand single-ticker block (on_demand_ticker)
# ---------------------------------------------------------------------------

def _on_demand_payload(ticker="TSLA", pipeline="US", market="USA",
                       score=0.64, badge="TACTICAL BUY", congress=0.0,
                       vix=17.0, **block_overrides):
    """Minimal on-demand payload — no top_buys_* buckets by design."""
    payload = {
        "on_demand": True,
        "on_demand_ticker": {
            "ticker": ticker,
            "pipeline": pipeline,
            "scoring_mode": "absolute",
            "entry": _entry(ticker=ticker, score=score, badge=badge,
                            market=market, congress=congress),
        },
        "vix": vix,
        "kill_switch": False,
        "ticker_count": 1,
        "generated_at": "2026-06-12T15:40:00+00:00",
    }
    payload["on_demand_ticker"].update(block_overrides)
    return payload


def test_on_demand_valid_us_passes():
    assert audit(_on_demand_payload()) is True


def test_on_demand_valid_intl_passes():
    payload = _on_demand_payload(
        ticker="SAP.DE", pipeline="INTL", market="EUROPE",
        score=0.71, badge="TACTICAL BUY",
    )
    assert audit(payload) is True


def test_on_demand_score_above_1_raises():
    payload = _on_demand_payload(score=1.01, badge="HIGH BUY")
    with pytest.raises(ScoreDivergenceError):
        audit(payload)


def test_on_demand_badge_mismatch_raises():
    payload = _on_demand_payload(score=0.85, badge="WATCHLIST")
    with pytest.raises(BadgeMismatchError):
        audit(payload)


def test_on_demand_intl_congress_raises():
    payload = _on_demand_payload(
        ticker="SAP.DE", pipeline="INTL", market="EUROPE", congress=0.3,
    )
    with pytest.raises(CrossContaminationError):
        audit(payload)


def test_on_demand_geo_leak_raises():
    """Suffixed ticker tagged USA must trip check D inside the block."""
    payload = _on_demand_payload(ticker="SAP.DE", pipeline="INTL", market="USA")
    with pytest.raises(GeographicLeakageError):
        audit(payload)


def test_on_demand_bad_ticker_format_raises():
    payload = _on_demand_payload(ticker="toolong123")
    with pytest.raises(StructuralIntegrityError):
        audit(payload)


def test_on_demand_missing_entry_raises():
    payload = _on_demand_payload()
    del payload["on_demand_ticker"]["entry"]
    with pytest.raises(StructuralIntegrityError):
        audit(payload)


def test_on_demand_block_entry_ticker_mismatch_raises():
    payload = _on_demand_payload()
    payload["on_demand_ticker"]["entry"]["ticker"] = "MSFT"
    with pytest.raises(StructuralIntegrityError):
        audit(payload)


def test_on_demand_bad_pipeline_raises():
    payload = _on_demand_payload(pipeline="EMEA")
    with pytest.raises(StructuralIntegrityError):
        audit(payload)


def test_on_demand_missing_scoring_mode_raises():
    payload = _on_demand_payload()
    del payload["on_demand_ticker"]["scoring_mode"]
    with pytest.raises(StructuralIntegrityError):
        audit(payload)


def test_on_demand_intl_score_above_1_raises():
    """E2 / check-A semantics hold inside the block for INTL entries."""
    payload = _on_demand_payload(
        ticker="SAP.DE", pipeline="INTL", market="EUROPE",
        score=1.01, badge="HIGH BUY",
    )
    with pytest.raises(ScoreDivergenceError):
        audit(payload)


def test_on_demand_vix_still_gated():
    """Check F runs on the root regardless of payload shape."""
    payload = _on_demand_payload(vix=999.0)
    with pytest.raises(VIXCoherenceError):
        audit(payload)


def test_daily_payload_without_on_demand_key_unaffected():
    """Regression pin: a normal daily payload never enters the on-demand branch."""
    payload = _make_payload(top_buys=[_entry(score=0.75)])
    assert audit(payload) is True
