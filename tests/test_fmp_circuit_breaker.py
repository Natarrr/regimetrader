"""WS2 — FMPClient dynamic per-endpoint circuit breaker.

An endpoint that returns HTTP 404 repeatedly is dead for the whole run (route
pulled from the plan), not sparse for one ticker. After _BREAKER_THRESHOLD
consecutive 404s the client must stop hammering it and short-circuit subsequent
calls with FMPEndpointError — the same structural signal a live 404 raises, minus
the wasted round-trip. 401/403 (global auth) must never be breaker-managed.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from src.services.fmp_client import FMPClient, FMPEndpointError


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "test-key")
    monkeypatch.setenv("FMP_MAX_RPS", "100000")  # negligible rate-gate delay
    return FMPClient(api_key="test-key", cache_root=tmp_path / "fmp")


def _resp(status_code: int, *, json_data=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = json_data if json_data is not None else {}
    resp.headers = {}
    return resp


class TestCircuitBreaker:
    def test_trips_after_threshold_and_short_circuits(self, client, caplog):
        r404 = _resp(404)
        with caplog.at_level(logging.ERROR, logger="src.services.fmp_client"):
            with patch.object(client._session, "get", return_value=r404) as mget:
                # First 3 calls each make a real request and raise structurally.
                for i in range(3):
                    with pytest.raises(FMPEndpointError):
                        client._get("dead-endpoint", {"symbol": f"T{i}"}, bucket="quote")
                assert mget.call_count == 3
                # 4th call short-circuits: no HTTP, still raises FMPEndpointError.
                with pytest.raises(FMPEndpointError):
                    client._get("dead-endpoint", {"symbol": "Z"}, bucket="quote")
                assert mget.call_count == 3  # frozen — no further round-trip

        assert "dead-endpoint" in client._runtime_dead
        trips = [r for r in caplog.records if "CIRCUIT BREAKER TRIPPED" in r.getMessage()]
        assert len(trips) == 1  # logged exactly once, not on every later call

    def test_intermittent_404s_do_not_trip(self, client):
        """A 200 between 404s clears the streak — transient blips never trip."""
        seq = [
            _resp(404), _resp(404),
            _resp(200, json_data=[{"ok": 1}]),  # recovery — resets streak
            _resp(404), _resp(404),
        ]
        with patch.object(client._session, "get", side_effect=seq) as mget:
            for _ in range(2):
                with pytest.raises(FMPEndpointError):
                    client._get("flaky", {}, bucket="quote")
            assert client._get("flaky", {}, bucket="quote") == [{"ok": 1}]
            for _ in range(2):
                with pytest.raises(FMPEndpointError):
                    client._get("flaky", {}, bucket="quote")
            assert mget.call_count == 5  # every real call made; never short-circuited
        assert "flaky" not in client._runtime_dead

    def test_auth_failure_does_not_trip_breaker(self, client):
        """401 is global auth (handled by preflight sys.exit) — not the breaker."""
        r401 = _resp(401)
        with patch.object(client._session, "get", return_value=r401) as mget:
            for _ in range(5):
                with pytest.raises(FMPEndpointError):
                    client._get("ratios-ttm", {}, bucket="ratios")
            assert mget.call_count == 5  # 401 never short-circuits
        assert "ratios-ttm" not in client._runtime_dead

    def test_reset_clears_quarantine(self, client):
        with patch.object(client._session, "get", return_value=_resp(404)):
            for _ in range(3):
                with pytest.raises(FMPEndpointError):
                    client._get("dead", {}, bucket="quote")
        assert "dead" in client._runtime_dead
        client.reset_circuit_breaker()
        assert "dead" not in client._runtime_dead
        assert dict(client._consecutive_404) == {}
