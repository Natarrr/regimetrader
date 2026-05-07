"""tests/test_edgar_circuit_breaker.py — Engle: regime transition under repeated failures.

The CB is a binary regime switch driven by a small recent-failure window:
    - closed → calls allowed, fail_count accumulates;
    - open   → calls short-circuited until cooldown elapses.
This mirrors ARCH-style state machines where the regime depends on a
moving observation of recent shocks.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.market_intel import edgar_ingest as ei


@pytest.fixture
def cb_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Engle: isolate CB state to a per-test file — no cross-test contamination."""
    p = tmp_path / "edgar_cb.json"
    monkeypatch.setattr(ei, "_CB_PATH", p)
    return p


def test_cb_allows_calls_on_clean_state(cb_path: Path) -> None:
    """Engle: with no observations, the regime is calm — calls flow freely."""
    assert ei._cb_allows_calls() is True
    assert not cb_path.exists()  # nothing written when state is the default


def test_cb_trips_open_after_threshold(
    cb_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Engle: regime transition triggers exactly when fail_count crosses threshold."""
    monkeypatch.setenv("EDGAR_CB_FAIL_THRESHOLD", "3")
    for _ in range(3):
        ei._cb_record_failure()
    assert ei._cb_allows_calls() is False
    state = ei.cb_state()
    assert state["state"] == "open"
    assert state["fail_count"] == 3


def test_cb_does_not_trip_below_threshold(
    cb_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Engle: a few isolated failures are insufficient to switch regimes."""
    monkeypatch.setenv("EDGAR_CB_FAIL_THRESHOLD", "5")
    for _ in range(3):
        ei._cb_record_failure()
    assert ei._cb_allows_calls() is True


def test_cb_reopens_after_cooldown(
    cb_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Engle: cooldown elapsed → regime resets to calm, calls resume."""
    monkeypatch.setenv("EDGAR_CB_FAIL_THRESHOLD", "2")
    monkeypatch.setenv("EDGAR_CB_COOLDOWN_MIN", "1")  # 1 minute = 60 s

    base_t = 1_700_000_000.0
    monkeypatch.setattr(ei.time, "time", lambda: base_t)
    ei._cb_record_failure()
    ei._cb_record_failure()
    assert ei._cb_allows_calls() is False

    monkeypatch.setattr(ei.time, "time", lambda: base_t + 70.0)  # past 60 s cooldown
    assert ei._cb_allows_calls() is True
    # State must have been reset
    assert ei.cb_state()["state"] == "closed"
    assert ei.cb_state()["fail_count"] == 0


def test_cb_remains_open_during_cooldown(
    cb_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Engle: within the cooldown window, the regime stays 'open' (no early reset)."""
    monkeypatch.setenv("EDGAR_CB_FAIL_THRESHOLD", "2")
    monkeypatch.setenv("EDGAR_CB_COOLDOWN_MIN", "15")

    base_t = 1_700_000_000.0
    monkeypatch.setattr(ei.time, "time", lambda: base_t)
    ei._cb_record_failure()
    ei._cb_record_failure()

    monkeypatch.setattr(ei.time, "time", lambda: base_t + 60.0)  # only 1 min in
    assert ei._cb_allows_calls() is False


def test_fetch_edgar_short_circuits_when_cb_open(
    cb_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Engle: when 'open', the public fetch returns immediately — no HTTP, no get_cik."""
    monkeypatch.setenv("EDGAR_CB_FAIL_THRESHOLD", "1")
    ei._cb_record_failure()  # trips immediately

    def boom(*args, **kwargs):
        raise AssertionError("network call must not happen when CB is open")

    monkeypatch.setattr(ei, "get_cik", boom)
    monkeypatch.setattr(ei, "_http_get", boom)

    out = ei.fetch_edgar_for_ticker("AAPL")
    assert out["ticker"] == "AAPL"
    assert out.get("edgar_cb_open") is True
    assert "edgar_cb_open" in out["errors"]
    assert out["form4"] == []
    assert out["form13f"] == []


def test_http_get_records_failure_on_final_raise(
    cb_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Engle: every final raise from _http_get must register a failure."""
    monkeypatch.setenv("EDGAR_CB_FAIL_THRESHOLD", "10")  # don't trip during this test
    monkeypatch.setattr(ei, "_rate_wait", lambda: None)
    monkeypatch.setattr(ei.time, "sleep", lambda *_: None)

    class _BadResp:
        status_code = 500
        text = "boom"

        def raise_for_status(self):
            raise RuntimeError("500 server error")

    monkeypatch.setattr(ei.requests, "get", lambda *a, **k: _BadResp())

    with pytest.raises(RuntimeError):
        ei._http_get("https://www.sec.gov/x")
    assert ei.cb_state()["fail_count"] == 1
