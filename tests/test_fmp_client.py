"""tests/test_fmp_client.py
Unit tests for regime_trader.services.fmp_client.

Fama (2013 Nobel) — budget enforcement is a data-quality invariant:
exceeding 200 calls/day causes HTTP 429s that are worse than conservative caching.
All FMP HTTP is mocked; no live network traffic in CI.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from regime_trader.services.fmp_client import (
    FmpClient,
    _BudgetManager,
    _cache_read,
    _cache_write,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_dirs(tmp_path: Path):
    return {
        "cache":  tmp_path / "fmp2",
        "quota":  tmp_path / "fmp2" / "quota.json",
    }


@pytest.fixture()
def budget(tmp_dirs) -> _BudgetManager:
    return _BudgetManager(daily_limit=10, quota_path=tmp_dirs["quota"])


@pytest.fixture()
def client(tmp_dirs, monkeypatch) -> FmpClient:
    monkeypatch.setenv("FMP_API_KEY", "test_key_xyz")
    return FmpClient(
        api_key      = "test_key_xyz",
        daily_budget = 10,
        cache_root   = tmp_dirs["cache"],
        quota_path   = tmp_dirs["quota"],
    )


# ── _BudgetManager ─────────────────────────────────────────────────────────────

class TestBudgetManager:
    def test_initial_remaining_equals_limit(self, budget: _BudgetManager):
        assert budget.remaining() == 10

    def test_reserve_decrements_remaining(self, budget: _BudgetManager):
        assert budget.reserve_calls(3) is True
        assert budget.remaining() == 7

    def test_commit_moves_reserved_to_used(self, budget: _BudgetManager):
        budget.reserve_calls(2)
        budget.commit_calls(2)
        assert budget.remaining() == 8

    def test_release_restores_reserved(self, budget: _BudgetManager):
        budget.reserve_calls(3)
        budget.release_calls(3)
        assert budget.remaining() == 10

    def test_over_budget_reserve_returns_false(self, budget: _BudgetManager):
        assert budget.reserve_calls(10) is True
        assert budget.reserve_calls(1) is False  # already at limit

    def test_remaining_never_negative(self, budget: _BudgetManager):
        budget.reserve_calls(10)
        budget.commit_calls(10)
        assert budget.remaining() == 0

    def test_quota_persists_to_disk(self, tmp_dirs, budget: _BudgetManager):
        budget.reserve_calls(5)
        budget.commit_calls(5)
        assert tmp_dirs["quota"].exists()
        data = json.loads(tmp_dirs["quota"].read_text())
        assert data["used"] == 5

    def test_daily_reset_at_midnight(self, tmp_dirs):
        """Budget resets when the persisted date differs from today."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        stale = {
            "date":     "1970-01-01",   # yesterday
            "used":     200,
            "reserved": 0,
        }
        tmp_dirs["quota"].parent.mkdir(parents=True, exist_ok=True)
        tmp_dirs["quota"].write_text(json.dumps(stale))
        budget = _BudgetManager(daily_limit=200, quota_path=tmp_dirs["quota"])
        assert budget.remaining() == 200


# ── Cache helpers ──────────────────────────────────────────────────────────────

class TestCacheHelpers:
    def test_roundtrip(self, tmp_dirs, monkeypatch):
        import regime_trader.services.fmp_client as mod
        monkeypatch.setattr(mod, "_CACHE_ROOT", tmp_dirs["cache"])
        payload = {"symbol": "AAPL", "marketCap": 3e12}
        _cache_write("profile", "AAPL", payload)
        result = _cache_read("profile", "AAPL")
        assert result == payload

    def test_cache_miss_returns_none(self, tmp_dirs, monkeypatch):
        import regime_trader.services.fmp_client as mod
        monkeypatch.setattr(mod, "_CACHE_ROOT", tmp_dirs["cache"])
        assert _cache_read("profile", "MISSING_TICKER_XYZ") is None

    def test_expired_cache_returns_none(self, tmp_dirs, monkeypatch):
        import regime_trader.services.fmp_client as mod
        monkeypatch.setattr(mod, "_CACHE_ROOT", tmp_dirs["cache"])
        _cache_write("profile", "AAPL", {"symbol": "AAPL"})
        # Backdating _ts so it's expired
        p = tmp_dirs["cache"] / "profile" / "AAPL.json"
        data = json.loads(p.read_text())
        data["_ts"] = time.time() - (25 * 3600)   # 25h ago > 24h TTL
        p.write_text(json.dumps(data))
        assert _cache_read("profile", "AAPL") is None


# ── FmpClient.get_profile ─────────────────────────────────────────────────────

class TestGetProfile:
    def test_cache_hit_uses_no_budget(self, client: FmpClient, tmp_dirs, monkeypatch):
        import regime_trader.services.fmp_client as mod
        monkeypatch.setattr(mod, "_CACHE_ROOT", tmp_dirs["cache"])
        _cache_write("profile", "AAPL", {"symbol": "AAPL", "marketCap": 3e12})
        before = client.budget_remaining()
        result = client.get_profile("AAPL")
        assert client.budget_remaining() == before
        assert result is not None

    def test_network_call_decrements_budget(self, client: FmpClient, tmp_dirs, monkeypatch):
        import regime_trader.services.fmp_client as mod
        monkeypatch.setattr(mod, "_CACHE_ROOT", tmp_dirs["cache"])
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = [{"symbol": "MSFT", "marketCap": 2e12}]
        client._session.get = MagicMock(return_value=mock_resp)
        before = client.budget_remaining()
        client.get_profile("MSFT")
        assert client.budget_remaining() == before - 1

    def test_quota_exhausted_uses_yfinance_fallback(self, tmp_dirs, monkeypatch):
        import regime_trader.services.fmp_client as mod
        monkeypatch.setattr(mod, "_CACHE_ROOT", tmp_dirs["cache"])
        client = FmpClient(
            api_key      = "k",
            daily_budget = 0,   # no budget from the start
            cache_root   = tmp_dirs["cache"],
            quota_path   = tmp_dirs["quota"],
        )
        fake_yf = MagicMock(return_value={"symbol": "AAPL", "source": "yfinance_fallback"})
        with patch("regime_trader.services.fmp_client._yfinance_profile", fake_yf):
            result = client.get_profile("AAPL")
        assert result is not None
        fake_yf.assert_called_once_with("AAPL")


# ── FmpClient.get_profiles (batch) ────────────────────────────────────────────

class TestGetProfiles:
    def test_batch_reduces_calls(self, client: FmpClient, tmp_dirs, monkeypatch):
        """50 tickers in one batch = 1 FMP call, not 50."""
        import regime_trader.services.fmp_client as mod
        monkeypatch.setattr(mod, "_CACHE_ROOT", tmp_dirs["cache"])
        tickers = [f"T{i:03d}" for i in range(50)]
        payload = [{"symbol": t, "marketCap": 1e9} for t in tickers]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = payload
        client._session.get = MagicMock(return_value=mock_resp)
        before = client.budget_remaining()
        result = client.get_profiles(tickers)
        # Should have used exactly 1 call
        assert before - client.budget_remaining() == 1
        assert len(result) == 50

    def test_cached_tickers_skip_network(self, client: FmpClient, tmp_dirs, monkeypatch):
        import regime_trader.services.fmp_client as mod
        monkeypatch.setattr(mod, "_CACHE_ROOT", tmp_dirs["cache"])
        _cache_write("profile", "AAPL", {"symbol": "AAPL", "marketCap": 3e12})
        _cache_write("profile", "MSFT", {"symbol": "MSFT", "marketCap": 2e12})
        client._session.get = MagicMock()
        result = client.get_profiles(["AAPL", "MSFT"])
        client._session.get.assert_not_called()
        assert "AAPL" in result
        assert "MSFT" in result


# ── FmpClient.get_screener ────────────────────────────────────────────────────

class TestGetScreener:
    def test_returns_list_on_success(self, client: FmpClient):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = [{"symbol": "AAPL"}, {"symbol": "MSFT"}]
        client._session.get = MagicMock(return_value=mock_resp)
        result = client.get_screener(limit=10)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_quota_exhausted_returns_empty(self, tmp_dirs, monkeypatch):
        import regime_trader.services.fmp_client as mod
        monkeypatch.setattr(mod, "_CACHE_ROOT", tmp_dirs["cache"])
        c = FmpClient(
            api_key="k", daily_budget=0,
            cache_root=tmp_dirs["cache"], quota_path=tmp_dirs["quota"]
        )
        assert c.get_screener() == []


# ── budget_remaining ──────────────────────────────────────────────────────────

class TestBudgetRemaining:
    def test_starts_at_daily_budget(self, client: FmpClient):
        assert client.budget_remaining() == 10

    def test_decreases_after_commit(self, client: FmpClient):
        client._budget.reserve_calls(3)
        client._budget.commit_calls(3)
        assert client.budget_remaining() == 7
