"""tests/test_discovery_scanner.py
Unit tests for regime_trader.scanners.discovery_scanner.

Coverage:
  - disc_get_json()          : 200 OK, 4xx, network error, unexpected shape
  - fmp_screener()           : valid list, non-list, missing key
  - select_candidates()      : zero-cap, empty insider, empty screener,
                               overlap (both), n-cap
  - enrich_with_momentum()   : deterministic mock values, empty input
  - _smart_money_prescore()  : insider-only, inst-only, momentum-only, all zero
  - get_top_alpha_picks_sync(): cache hit, cache miss → fresh scan stub
  - load_disc_cache / save_disc_cache: TTL expiry, atomic write round-trip
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Dict, List
from unittest.mock import MagicMock, patch

import pytest

# ── Path setup: ensure repo root is on sys.path ───────────────────────────────
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from regime_trader.scanners.discovery_scanner import (
    _smart_money_prescore,
    disc_get_json,
    enrich_with_momentum,
    fmp_screener,
    get_top_alpha_picks_sync,
    load_disc_cache,
    save_disc_cache,
    select_candidates,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def tmp_cache(tmp_path, monkeypatch):
    """Redirect _DISC_CACHE_FILE to a temp directory for isolation."""
    import regime_trader.scanners.discovery_scanner as ds
    cache_path = tmp_path / "discovery_cache.json"
    monkeypatch.setattr(ds, "_DISC_CACHE_FILE", cache_path)
    return cache_path


@pytest.fixture()
def sample_screener() -> List[Dict]:
    return [
        {"sym": "AAPL", "market_cap": 3e12, "price": 185.0, "volume": 5e7,
         "avg_volume": 4e7, "volume_spike": 1.25, "price_change_pct": 1.5, "sector": "Tech"},
        {"sym": "MSFT", "market_cap": 2e12, "price": 420.0, "volume": 3e7,
         "avg_volume": 3e7, "volume_spike": 1.00, "price_change_pct": 0.5, "sector": "Tech"},
        {"sym": "NVDA", "market_cap": 2.5e12, "price": 870.0, "volume": 4e7,
         "avg_volume": 3e7, "volume_spike": 1.33, "price_change_pct": 3.0, "sector": "Tech"},
    ]


@pytest.fixture()
def sample_insider() -> List[Dict]:
    return [
        {"sym": "TSLA", "key_value_usd": 5_000_000.0, "normalized_pct_mcap": 0.06,
         "market_cap": 8e9, "roles": ["CEO"], "tx_count": 2, "most_recent_date": "2026-04-01"},
        {"sym": "AAPL", "key_value_usd": 1_000_000.0, "normalized_pct_mcap": 0.003,
         "market_cap": 3e12, "roles": ["CFO"], "tx_count": 1, "most_recent_date": "2026-03-28"},
    ]


# ── disc_get_json ─────────────────────────────────────────────────────────────

class TestDiscGetJson:
    def test_200_returns_parsed_json(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = [{"symbol": "AAPL"}]
        with patch("regime_trader.scanners.discovery_scanner._get_session") as mock_sess:
            mock_sess.return_value.get.return_value = mock_resp
            result = disc_get_json("https://example.com/api", {"apikey": "test"})
        assert result == [{"symbol": "AAPL"}]

    def test_404_returns_none(self):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        with patch("regime_trader.scanners.discovery_scanner._get_session") as mock_sess:
            mock_sess.return_value.get.return_value = mock_resp
            result = disc_get_json("https://example.com/api")
        assert result is None

    def test_network_error_returns_none(self):
        with patch("regime_trader.scanners.discovery_scanner._get_session") as mock_sess:
            mock_sess.return_value.get.side_effect = ConnectionError("timeout")
            result = disc_get_json("https://example.com/api")
        assert result is None

    def test_unexpected_shape_propagated(self):
        """Non-list/dict JSON is returned as-is (caller validates shape)."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = "unexpected string"
        with patch("regime_trader.scanners.discovery_scanner._get_session") as mock_sess:
            mock_sess.return_value.get.return_value = mock_resp
            result = disc_get_json("https://example.com/api")
        assert result == "unexpected string"


# ── fmp_screener ──────────────────────────────────────────────────────────────

def _fmp_rows(prices: List[float], volumes: List[float]) -> list:
    """Build FMP historical-price-eod/full rows (newest-first)."""
    rows = []
    for i, (p, v) in enumerate(zip(prices, volumes)):
        rows.append({"date": f"2026-04-{(i % 28) + 1:02d}", "close": p, "volume": v})
    return list(reversed(rows))   # newest-first


_PRICES = "regime_trader.services.fmp_client.FMPClient.get_historical_prices"
_BATCH  = "regime_trader.services.fmp_client.FMPClient.get_batch_quotes"


class TestFmpScreener:
    def test_no_key_returns_empty(self, monkeypatch):
        """If FMP_API_KEY is absent, fmp_screener() returns []."""
        monkeypatch.delenv("FMP_API_KEY", raising=False)
        result = fmp_screener()
        assert result == []

    def test_valid_ticker_produces_correct_fields(self, monkeypatch):
        """Volume spike and price_change_pct computed correctly from FMP prices."""
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        prices  = [100.0, 101.0, 102.0, 103.0, 104.0, 110.0, 120.0]
        volumes = [1_000_000.0] * 6 + [2_000_000.0]
        rows = _fmp_rows(prices, volumes)

        with patch(_PRICES, return_value=rows), \
             patch(_BATCH, return_value={"AAPL": {"symbol": "AAPL", "marketCap": 3e12}}), \
             patch("regime_trader.scanners.discovery_scanner._YF_WATCHLIST", ["AAPL"]):
            results = fmp_screener()

        assert len(results) == 1
        r = results[0]
        assert r["sym"] == "AAPL"
        assert r["volume_spike"] == pytest.approx(2.0, rel=0.01)
        assert r["price_change_pct"] == pytest.approx((120.0 / 101.0 - 1) * 100, rel=0.01)

    def test_low_price_ticker_filtered(self, monkeypatch):
        """Tickers priced below $1 must not appear in results."""
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        rows = _fmp_rows([0.50] * 7, [50_000_000.0] * 7)
        with patch(_PRICES, return_value=rows), \
             patch(_BATCH, return_value={}), \
             patch("regime_trader.scanners.discovery_scanner._YF_WATCHLIST", ["PENNYSTOCK"]):
            results = fmp_screener()
        assert results == []

    def test_empty_data_returns_empty(self, monkeypatch):
        """No price data from FMP results in []."""
        monkeypatch.setenv("FMP_API_KEY", "test-key")
        with patch(_PRICES, return_value=[]), \
             patch(_BATCH, return_value={}), \
             patch("regime_trader.scanners.discovery_scanner._YF_WATCHLIST", ["AAPL"]):
            result = fmp_screener()
        assert result == []


# ── select_candidates ─────────────────────────────────────────────────────────

class TestSelectCandidates:
    def test_insider_guaranteed_entry_not_in_screener(self, sample_screener, sample_insider):
        """TSLA is only in insider_buys, not in screener — must appear in selected."""
        selected, source_map = select_candidates(sample_screener, sample_insider, n=10)
        assert "TSLA" in selected
        assert source_map["TSLA"] == "insider"

    def test_overlap_marked_as_both(self, sample_screener, sample_insider):
        """AAPL is in both insider_buys and screener — source_map must be 'both'."""
        selected, source_map = select_candidates(sample_screener, sample_insider, n=10)
        assert "AAPL" in selected
        assert source_map["AAPL"] == "both"

    def test_screener_only_symbol_marked_correctly(self, sample_screener, sample_insider):
        selected, source_map = select_candidates(sample_screener, sample_insider, n=10)
        assert source_map.get("NVDA") == "screener"

    def test_n_cap_respected(self, sample_screener, sample_insider):
        selected, _ = select_candidates(sample_screener, sample_insider, n=2)
        assert len(selected) <= 2

    def test_empty_insider_uses_screener_only(self, sample_screener):
        selected, source_map = select_candidates(sample_screener, [], n=5)
        assert len(selected) == 3
        assert all(v == "screener" for v in source_map.values())

    def test_empty_screener_uses_insider_only(self, sample_insider):
        selected, source_map = select_candidates([], sample_insider, n=5)
        assert set(selected) == {"TSLA", "AAPL"}
        assert source_map["TSLA"] == "insider"

    def test_zero_market_cap_insider_included(self):
        """Insider with zero market cap must still be selected (no division crash)."""
        insider = [{"sym": "XYZ", "key_value_usd": 50_000.0,
                    "normalized_pct_mcap": 0.0, "market_cap": 0.0,
                    "roles": ["CEO"], "tx_count": 1, "most_recent_date": "2026-04-01"}]
        selected, source_map = select_candidates([], insider, n=5)
        assert "XYZ" in selected

    def test_no_duplicates_in_selected(self, sample_screener, sample_insider):
        selected, _ = select_candidates(sample_screener, sample_insider, n=20)
        assert len(selected) == len(set(selected))


# ── enrich_with_momentum ──────────────────────────────────────────────────────

class TestEnrichWithMomentum:
    def test_enrichment_adds_fields(self):
        candidates = [{"sym": "AAPL"}, {"sym": "MSFT"}]
        rows = _fmp_rows([100.0 + i for i in range(25)], [1_000_000.0] * 25)

        with patch(_PRICES, return_value=rows):
            result = enrich_with_momentum(candidates, max_workers=2)

        assert all("volume_spike" in r for r in result)
        assert all("price_change_pct" in r for r in result)

    def test_empty_input_returns_empty(self):
        assert enrich_with_momentum([]) == []

    def test_fmp_failure_defaults_to_safe_values(self):
        candidates = [{"sym": "BROKEN"}]
        with patch(_PRICES, side_effect=RuntimeError("API down")):
            result = enrich_with_momentum(candidates, max_workers=1)
        assert result[0]["volume_spike"] == 1.0
        assert result[0]["price_change_pct"] == 0.0


# ── _smart_money_prescore ─────────────────────────────────────────────────────

class TestSmartMoneyPrescore:
    def test_insider_only_signal(self):
        insider_map = {"AAPL": {"normalized_pct_mcap": 0.5}}
        composite, ins, inst, mom = _smart_money_prescore(
            "AAPL", insider_map, {}, {}
        )
        assert ins == 1.0            # 0.30 + 0.70 * min(1, 0.5/0.5)
        assert inst == 0.0
        assert mom == 0.0
        assert composite == pytest.approx(0.45, abs=1e-3)

    def test_inst_only_signal(self):
        inst_map = {"MSFT": {"accumulation_score": 0.80}}
        composite, ins, inst, mom = _smart_money_prescore(
            "MSFT", {}, inst_map, {}
        )
        assert ins == 0.0
        assert inst == pytest.approx(0.60, abs=1e-3)  # (0.80 - 0.50) * 2
        assert composite == pytest.approx(0.35 * 0.60, abs=1e-3)

    def test_momentum_only_signal(self):
        screener_map = {"NVDA": {"volume_spike": 5.0, "price_change_pct": 5.0}}
        composite, ins, inst, mom = _smart_money_prescore(
            "NVDA", {}, {}, screener_map
        )
        assert ins == 0.0
        assert inst == 0.0
        assert mom > 0.0
        assert composite == pytest.approx(0.20 * mom, abs=1e-3)

    def test_all_zero_no_data(self):
        composite, ins, inst, mom = _smart_money_prescore("UNK", {}, {}, {})
        assert composite == 0.0
        assert ins == inst == mom == 0.0

    def test_composite_bounded_0_1(self):
        insider_map = {"X": {"normalized_pct_mcap": 999.0}}
        inst_map = {"X": {"accumulation_score": 1.0}}
        screener_map = {"X": {"volume_spike": 100.0, "price_change_pct": 50.0}}
        composite, *_ = _smart_money_prescore("X", insider_map, inst_map, screener_map)
        assert 0.0 <= composite <= 1.0


# ── Cache helpers ─────────────────────────────────────────────────────────────

class TestCacheHelpers:
    def test_save_and_load_round_trip(self, tmp_cache):
        payload = {"results": [{"symbol": "AAPL"}], "_expires_at": time.time() + 3600}
        save_disc_cache(payload)
        assert tmp_cache.exists()
        loaded = load_disc_cache()
        assert loaded is not None
        assert loaded["cached"] is True
        assert "results" in loaded

    def test_expired_cache_returns_none(self, tmp_cache):
        payload = {"results": [], "_expires_at": time.time() - 1}
        save_disc_cache(payload)
        assert load_disc_cache() is None

    def test_missing_cache_returns_none(self, tmp_cache):
        assert load_disc_cache() is None

    def test_corrupt_cache_returns_none(self, tmp_cache):
        tmp_cache.write_text("not json!!!", encoding="utf-8")
        assert load_disc_cache() is None

    def test_atomic_write_creates_file(self, tmp_cache):
        save_disc_cache({"_expires_at": time.time() + 100, "ok": True})
        assert tmp_cache.exists()
        data = json.loads(tmp_cache.read_text())
        assert data["ok"] is True


# ── get_top_alpha_picks_sync (integration stub) ───────────────────────────────

class TestGetTopAlphaPicks:
    def test_returns_dict_with_results_key(self, tmp_cache, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "testkey")
        fake_result = [
            {"symbol": "AAPL", "smart_money_score": 0.75,
             "insider_score": 1.0, "institutional_score": 0.0,
             "momentum_score": 0.0, "insider_value_usd": 1e6,
             "insider_value_pct_mcap": 0.5, "key_insider_roles": ["CEO"],
             "institutional_net_shares": 0, "institutional_pct_change": 0,
             "volume_spike": 1.0, "price_change_pct": 0.0,
             "market_cap": 3e12, "source_flags": ["insider"]},
        ]
        with patch("regime_trader.scanners.discovery_scanner.run_scan", return_value=fake_result):
            payload = get_top_alpha_picks_sync(limit=1)
        assert "results" in payload
        assert payload["results"][0]["symbol"] == "AAPL"

    def test_cache_hit_skips_scan(self, tmp_cache):
        """Pre-populate a fresh cache; run_scan must not be called."""
        payload = {
            "results": [{"symbol": "CACHED"}],
            "cached": False,
            "computed_at": "2026-01-01T00:00:00Z",
            "_expires_at": time.time() + 3600,
        }
        save_disc_cache(payload)
        with patch("regime_trader.scanners.discovery_scanner.run_scan") as mock_scan:
            result = get_top_alpha_picks_sync(limit=5)
        mock_scan.assert_not_called()
        assert result["cached"] is True
