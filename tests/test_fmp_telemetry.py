"""WS4 — FMPClient.telemetry_snapshot() shape, latency math, cache counters."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.services.fmp_client import FMPClient, FMPEndpointError


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "test-key")
    monkeypatch.setenv("FMP_MAX_RPS", "100000")
    FMPClient._rate_last_call = 0.0  # reset class-level state so fake_monotonic starts clean
    return FMPClient(api_key="test-key", cache_root=tmp_path / "fmp")


def _resp(status_code, *, json_data=None):
    r = MagicMock()
    r.status_code = status_code
    r.json.return_value = json_data if json_data is not None else {}
    r.headers = {}
    return r


@pytest.fixture
def fake_monotonic(monkeypatch):
    """Deterministic clock: each call advances 1.0s. Within one _get the latency
    window spans exactly one tick → every resolving call measures 1000.0 ms."""
    state = {"t": 0.0}

    def _mono():
        state["t"] += 1.0
        return state["t"]

    monkeypatch.setattr("src.services.fmp_client.time.monotonic", _mono)
    return state


class TestTelemetrySnapshot:
    def test_latency_and_calls(self, client, fake_monotonic):
        with patch.object(client._session, "get", return_value=_resp(200, json_data=[{"x": 1}])):
            client._get("quote", {"symbol": "NVDA"}, bucket="quote")
            client._get("quote", {"symbol": "AAPL"}, bucket="quote")
        snap = client.telemetry_snapshot()
        ep = snap["endpoints"]["quote"]
        assert ep["calls"] == 2
        assert ep["failures"] == 0
        assert ep["latency_ms_avg"] == 1000.0
        assert ep["latency_ms_max"] == 1000.0
        assert snap["totals"]["calls"] == 2
        assert snap["totals"]["failures"] == 0

    def test_failure_counted(self, client):
        with patch.object(client._session, "get", return_value=_resp(404)):
            with pytest.raises(FMPEndpointError):
                client._get("dead", {}, bucket="quote")
        snap = client.telemetry_snapshot()
        assert snap["endpoints"]["dead"]["failures"] == 1
        assert snap["totals"]["failures"] == 1

    def test_cache_hit_miss_counters(self, client):
        assert client._cache_read("quote", "k") is None        # miss (absent)
        client._cache_write("quote", "k", [{"a": 1}])
        assert client._cache_read("quote", "k") == [{"a": 1}]  # hit
        totals = client.telemetry_snapshot()["totals"]
        assert totals["cache_hits"] == 1
        assert totals["cache_misses"] == 1

    def test_bypass_cache_is_not_a_miss(self, client):
        client._cache_read("quote", "k", bypass_cache=True)
        totals = client.telemetry_snapshot()["totals"]
        assert totals["cache_hits"] == 0
        assert totals["cache_misses"] == 0

    def test_runtime_quarantine_surfaced(self, client):
        with patch.object(client._session, "get", return_value=_resp(404)):
            for _ in range(3):
                with pytest.raises(FMPEndpointError):
                    client._get("dead", {}, bucket="quote")
        assert client.telemetry_snapshot()["totals"]["runtime_quarantined"] == ["dead"]

    def test_empty_snapshot_is_well_formed(self, client):
        snap = client.telemetry_snapshot()
        assert snap["endpoints"] == {}
        assert snap["totals"] == {
            "calls": 0, "failures": 0, "cache_hits": 0,
            "cache_misses": 0, "runtime_quarantined": [],
        }
