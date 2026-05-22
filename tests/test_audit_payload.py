"""TDD tests for scripts/audit_payload.py — pre-flight Discord pipeline audit."""
import sys
import os
import pytest

# Allow importing from scripts/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from audit_payload import (
    audit,
    ScoreDivergenceError,
    BadgeMismatchError,
    SortingError,
    CrossContaminationError,
    GeographicLeakageError,
    StructuralIntegrityError,
    VIXCoherenceError,
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
