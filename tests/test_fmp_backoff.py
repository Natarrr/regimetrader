"""Unit tests for FMPClient 429 backoff jitter (WS1).

The computed exponential fallback must be jittered (equal-jitter) so the US and
INTL runners de-sync instead of retrying a shared 429 wave in lockstep. A
server-supplied Retry-After must still be honoured verbatim (un-jittered).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.services.fmp_client import FMPClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "test-key")
    monkeypatch.setenv("FMP_MAX_RPS", "100000")  # negligible rate-gate delay
    return FMPClient(api_key="test-key", cache_root=tmp_path / "fmp")


def _resp(status_code: int, *, json_data=None, headers=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data if json_data is not None else {}
    resp.headers = headers or {}
    return resp


@pytest.fixture
def captured_sleeps(monkeypatch):
    """Record every time.sleep() call and short-circuit it (no real waiting)."""
    sleeps: list[float] = []
    monkeypatch.setattr("src.services.fmp_client.time.sleep", lambda s: sleeps.append(float(s)))
    return sleeps


def _backoff_sleeps(sleeps: list[float]) -> list[float]:
    """Filter out the negligible rate-gate sleeps; keep real backoff waits."""
    return [s for s in sleeps if s >= 0.4]


class TestComputedBackoffJitter:
    def test_jitter_within_equal_jitter_bounds(self, client, captured_sleeps):
        # 3×429 then 200 → attempts 0,1,2 each sleep on the computed fallback.
        seq = [_resp(429), _resp(429), _resp(429), _resp(200, json_data=[{"ok": 1}])]
        with patch.object(client._session, "get", side_effect=seq):
            client._get("quote", {"symbol": "NVDA"}, bucket="quote")

        waits = _backoff_sleeps(captured_sleeps)
        assert len(waits) == 3
        # Equal jitter: _ra ∈ [2**a / 2, 2**a] for attempt a.
        for attempt, wait in enumerate(waits):
            base = float(2 ** attempt)
            assert base / 2.0 <= wait <= base, (attempt, wait, base)

    def test_jitter_varies_across_runs(self, client, captured_sleeps):
        """Without jitter, attempt-0 would always be exactly 1.0s. With jitter
        the first wait varies run-to-run — proves the random term is wired in."""
        first_waits: list[float] = []
        for _ in range(6):
            captured_sleeps.clear()
            seq = [_resp(429), _resp(200, json_data=[{"ok": 1}])]
            with patch.object(client._session, "get", side_effect=seq):
                client._get("quote", {"symbol": "NVDA"}, bucket="quote")
            first_waits.append(_backoff_sleeps(captured_sleeps)[0])
        assert len({round(w, 6) for w in first_waits}) > 1


class TestRetryAfterHonoured:
    def test_retry_after_header_is_not_jittered(self, client, captured_sleeps):
        seq = [
            _resp(429, headers={"Retry-After": "7"}),
            _resp(200, json_data=[{"ok": 1}]),
        ]
        with patch.object(client._session, "get", side_effect=seq):
            client._get("quote", {"symbol": "NVDA"}, bucket="quote")
        waits = _backoff_sleeps(captured_sleeps)
        assert waits == [7.0]  # server-authoritative, exact

    def test_retry_after_json_field_is_not_jittered(self, client, captured_sleeps):
        seq = [
            _resp(429, json_data={"retry_after": 5}),
            _resp(200, json_data=[{"ok": 1}]),
        ]
        with patch.object(client._session, "get", side_effect=seq):
            client._get("quote", {"symbol": "NVDA"}, bucket="quote")
        waits = _backoff_sleeps(captured_sleeps)
        assert waits == [5.0]
