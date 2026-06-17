"""WS3 — surface swallowed breadth structural failures.

When the insider-trading/search endpoint dies mid-run, breadth degrades to empty
P/S per ticker. That degradation must be VISIBLE (recorded in breadth_health →
intel_source_status.json), not silent — otherwise insider_breadth quietly zeroes
across the whole universe with no operator signal.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from src.ingestion.run_pipeline import fetch_fmp_breadth_all
from src.services.fmp_client import FMPClient, FMPEndpointError


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "test-key")
    monkeypatch.setenv("FMP_MAX_RPS", "100000")
    return FMPClient(api_key="test-key", cache_root=tmp_path / "fmp")


class TestBreadthStructuralVisibility:
    def test_structural_failure_is_recorded_and_degrades(self, client):
        with patch.object(
            client, "get_insider_transactions",
            side_effect=FMPEndpointError("insider-trading/search", 404),
        ):
            cache, health = fetch_fmp_breadth_all(["NVDA", "AAPL"], client=client)

        # Breadth still degrades gracefully (run does not crash)…
        assert cache["NVDA"] == {"P": [], "S": []}
        assert cache["AAPL"] == {"P": [], "S": []}
        # …but the structural failure is surfaced, not silent.
        assert health["structural_failures"] == ["insider-trading/search"]
        assert health["tickers_degraded"] == 2

    def test_generic_error_degrades_quietly(self, client):
        """Timeouts are sparse, not structural — must NOT flag the endpoint dead."""
        with patch.object(
            client, "get_insider_transactions",
            side_effect=RuntimeError("read timeout"),
        ):
            cache, health = fetch_fmp_breadth_all(["NVDA"], client=client)
        assert cache["NVDA"] == {"P": [], "S": []}
        assert health["structural_failures"] == []
        assert health["tickers_degraded"] == 0

    def test_healthy_run_reports_no_failures(self, client):
        with patch.object(
            client, "get_insider_transactions",
            return_value={"P": [{"x": 1}], "S": []},
        ):
            cache, health = fetch_fmp_breadth_all(["NVDA"], client=client)
        assert cache["NVDA"]["P"] == [{"x": 1}]
        assert health == {"structural_failures": [], "tickers_degraded": 0}

    def test_no_api_key_returns_empty_health(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FMP_API_KEY", raising=False)
        c = FMPClient(api_key="", cache_root=tmp_path / "fmp")
        cache, health = fetch_fmp_breadth_all(["NVDA"], client=c)
        assert cache == {}
        assert health == {"structural_failures": [], "tickers_degraded": 0}

    def test_breaker_short_circuit_counts_per_ticker(self, client):
        """Once the breaker quarantines the route, every later ticker still gets
        a recorded structural degradation (the short-circuit raises the same
        FMPEndpointError), so tickers_degraded reflects the true blast radius."""
        from unittest.mock import MagicMock

        def _resp(code):
            m = MagicMock()
            m.status_code = code
            m.json.return_value = {}
            m.headers = {}
            return m

        # Real 404s drive the breaker; get_insider_transactions calls _get under
        # the hood and re-raises FMPEndpointError, so we exercise the true path.
        with patch.object(client._session, "get", return_value=_resp(404)):
            cache, health = fetch_fmp_breadth_all(
                [f"T{i}" for i in range(6)], client=client, max_workers=1,
            )
        assert health["structural_failures"] == ["insider-trading/search"]
        assert health["tickers_degraded"] == 6
        assert "insider-trading/search" in client._runtime_dead
