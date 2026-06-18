# Path: tests/test_inst_flow_13f_isolation.py
#
# pytest — audit correction C1: the 13F "whale" factor must be failure-isolated.
# A *structural* FMP 13F-route failure (FMPEndpointError) must degrade the 0.04
# SIGNED factor to None and be recorded for fmp_health.json — it must NOT
# propagate, because in _score_ticker that propagation reaches the outer handler
# and zeroes the WHOLE ticker (discarding insider/momentum/analyst signals).
#
# Exercises the extracted, dependency-injected _guarded_inst_flow_13f helper so
# the isolation contract is testable without driving the full _score_ticker
# closure (which captures 12+ locals).
#
# Run: pytest tests/test_inst_flow_13f_isolation.py -v

import threading

import pytest

from src.ingestion.run_pipeline import _guarded_inst_flow_13f
from src.services.fmp_client import FMPEndpointError

_F13_PATH = "institutional-ownership/symbol-positions-summary"


class _RaisingClient:
    """13F route is structurally dead (HTTP 404)."""
    def get_institutional_ownership(self, ticker):
        raise FMPEndpointError(_F13_PATH, 404)


class _EmptyClient:
    """Route is live but the ticker has no 13F coverage."""
    def get_institutional_ownership(self, ticker):
        return {}


class _WhaleClient:
    """Route returns a real position-delta summary (net accumulation)."""
    def get_institutional_ownership(self, ticker):
        return {
            "investorsHolding":       400.0,
            "investorsHoldingChange":  40.0,   # +10% of holders QoQ
            "increasedPositions":     120.0,
            "reducedPositions":        30.0,
            "ownershipPercentChange":   2.0,
        }


def _fresh():
    return threading.Lock(), set()


def test_structural_failure_degrades_to_none_and_is_recorded():
    lock, failures = _fresh()
    summary, score = _guarded_inst_flow_13f(_RaisingClient(), "ACME", lock, failures)
    # SIGNED factor → unavailable. Never bearish, never a 0.0 mass point.
    assert score is None
    assert summary == {}
    # Not silent (CLAUDE.md §2): the broken endpoint is recorded for fmp_health.
    assert _F13_PATH in failures


def test_structural_failure_does_not_propagate():
    """The crux of C1: the exception is swallowed-with-record, so the caller
    (_score_ticker) keeps every other factor it already computed for the ticker."""
    lock, failures = _fresh()
    _guarded_inst_flow_13f(_RaisingClient(), "ACME", lock, failures)  # must not raise


def test_empty_feed_is_data_absence_not_a_fault():
    lock, failures = _fresh()
    summary, score = _guarded_inst_flow_13f(_EmptyClient(), "ACME", lock, failures)
    assert score is None
    assert summary == {}
    assert not failures   # an empty feed is data-absence, not a structural failure


def test_live_feed_scores_and_passes_summary_through():
    lock, failures = _fresh()
    summary, score = _guarded_inst_flow_13f(_WhaleClient(), "ACME", lock, failures)
    # Raw summary rides along for the 🐋 WHALE / [NICHE ALPHA] display.
    assert summary["investorsHoldingChange"] == 40.0
    # 0.5 + 0.2(=10·40/400 clipped) + 0.2·(90/150) + 0.1·(2/5) = 0.86
    assert score == pytest.approx(0.86, abs=1e-9)
    assert score > 0.5            # net accumulation → bullish tilt above neutral
    assert not failures           # a healthy route records nothing
