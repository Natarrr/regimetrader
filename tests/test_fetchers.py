import pytest
from regime_trader.fetchers.base import BaseMarketFetcher, MarketEnum


def test_market_enum_values():
    assert MarketEnum.USA.value == "USA"
    assert MarketEnum.EUROPE.value == "EUROPE"
    assert MarketEnum.ASIA.value == "ASIA"


def test_base_fetcher_is_abstract():
    with pytest.raises(TypeError):
        BaseMarketFetcher()


def test_concrete_fetcher_must_implement_fetch():
    class BadFetcher(BaseMarketFetcher):
        pass
    with pytest.raises(TypeError):
        BadFetcher()


import json
from pathlib import Path


def test_ticker_registry_exists_and_valid():
    reg_path = Path("config/ticker_registry.json")
    assert reg_path.exists(), "config/ticker_registry.json not found"
    data = json.loads(reg_path.read_text(encoding="utf-8"))
    assert "europe" in data
    assert "asia" in data
    for entry in data["europe"]:
        assert "ticker" in entry
        assert "sector" in entry
        assert "cap_tier" in entry
        assert "exchange" in entry
    for entry in data["asia"]:
        assert "ticker" in entry
        assert "sector" in entry
        assert "cap_tier" in entry
        assert "exchange" in entry


def test_registry_ticker_format():
    reg_path = Path("config/ticker_registry.json")
    data = json.loads(reg_path.read_text(encoding="utf-8"))
    import re
    pattern = re.compile(r"^[A-Z0-9]{1,6}\.[A-Z]{1,2}$")
    for entry in data["europe"] + data["asia"]:
        assert pattern.match(entry["ticker"]), f"Bad ticker format: {entry['ticker']}"


from unittest.mock import patch, MagicMock
from regime_trader.fetchers.fmp_fetcher import FMPFetcher


def test_fmp_fetcher_market():
    f = FMPFetcher(api_key="test", market=MarketEnum.EUROPE)
    assert f.market == MarketEnum.EUROPE


def test_fmp_fetcher_market_asia():
    f = FMPFetcher(api_key="test", market=MarketEnum.ASIA)
    assert f.market == MarketEnum.ASIA


def _fmp_price_rows(n: int, start: float = 100.0, end: float = 115.0,
                    vol: float = 2_000_000, last_vol: float | None = None) -> list:
    """Build FMP historical-price-eod/full rows (newest-first) for n trading days."""
    rows = []
    step = (end - start) / max(n - 1, 1)
    for i in range(n):
        close = start + step * i
        v = vol
        if last_vol is not None and i == n - 1:
            v = last_vol
        rows.append({
            "date":   f"2024-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}",
            "close":  close,
            "volume": v,
        })
    return list(reversed(rows))   # newest-first as FMP returns


def test_fmp_fetcher_source_reliability():
    # FMPFetcher uses FMP price feeds but is reduced for non-US reliability.
    f = FMPFetcher(api_key="test", market=MarketEnum.EUROPE)
    assert f.source_reliability("SAP.DE") == 0.60


def test_fmp_fetcher_prepare_empty_on_fmp_error():
    f = FMPFetcher(api_key="test", market=MarketEnum.EUROPE)
    with patch("regime_trader.services.fmp_client.FMPClient.get_historical_prices",
               side_effect=Exception("network")):
        result = f.prepare(["SAP.DE"])
    assert result == []


def test_fmp_fetcher_prepare_empty_on_no_data():
    f = FMPFetcher(api_key="test", market=MarketEnum.EUROPE)
    with patch("regime_trader.services.fmp_client.FMPClient.get_historical_prices",
               return_value=[]):
        result = f.prepare(["SAP.DE"])
    assert result == []


def test_fmp_fetcher_prepare_returns_entry_with_fmp_data():
    f = FMPFetcher(api_key="test", market=MarketEnum.EUROPE)
    rows = _fmp_price_rows(275, last_vol=4_000_000)   # last-day volume spike
    with patch("regime_trader.services.fmp_client.FMPClient.get_historical_prices",
               return_value=rows):
        result = f.prepare(["SAP.DE"])
    assert len(result) == 1
    assert result[0].ticker == "SAP.DE"
    assert result[0].market == MarketEnum.EUROPE
    assert result[0].source_reliability == 0.60
    assert result[0].raw_factors["return_12_1m"] is not None
    assert result[0].raw_factors["volume_spike"] > 0


def test_fmp_fetcher_prepare_asia_market():
    f = FMPFetcher(api_key="test", market=MarketEnum.ASIA)
    rows = _fmp_price_rows(275, start=2800.0, end=3100.0, vol=16_000_000)
    with patch("regime_trader.services.fmp_client.FMPClient.get_historical_prices",
               return_value=rows):
        result = f.prepare(["7203.T"])
    assert len(result) == 1
    assert result[0].market == MarketEnum.ASIA
    assert result[0].source_reliability == 0.60


def test_fmp_fetcher_no_quota_logic():
    """FMPFetcher uses FMP price feed — empty data is skipped, no quota check."""
    f = FMPFetcher(api_key="test", market=MarketEnum.EUROPE)
    with patch("regime_trader.services.fmp_client.FMPClient.get_historical_prices",
               return_value=[]):
        result = f.prepare(["SAP.DE", "ASML.AS"])
    assert result == []


def test_fmp_fetcher_multiple_tickers_returns_multiple_entries():
    """Each ticker with valid FMP data produces one TickerEntry."""
    f = FMPFetcher(api_key="test", market=MarketEnum.EUROPE)
    rows = _fmp_price_rows(275)

    call_count = [0]
    def fake_prices(ticker, limit=280):
        call_count[0] += 1
        return rows

    with patch("regime_trader.services.fmp_client.FMPClient.get_historical_prices",
               side_effect=fake_prices):
        result = f.prepare(["SAP.DE", "SIE.DE"])
    assert len(result) == 2
    assert call_count[0] == 2


from regime_trader.fetchers.orchestrator import Orchestrator
from regime_trader.fetchers.base import TickerEntry


def _make_entry(ticker: str, market: MarketEnum) -> TickerEntry:
    return TickerEntry(ticker=ticker, market=market, sector="Tech",
                       cap_tier="large", source_reliability=1.0, raw_factors={})


def test_orchestrator_collects_all_results():
    f1 = MagicMock()
    f1.market = MarketEnum.USA
    f1.prepare.return_value = [_make_entry("AAPL.US", MarketEnum.USA)]
    f2 = MagicMock()
    f2.market = MarketEnum.EUROPE
    f2.prepare.return_value = [_make_entry("SAP.DE", MarketEnum.EUROPE)]
    orch = Orchestrator([f1, f2])
    results = orch.run({"USA": ["AAPL"], "EUROPE": ["SAP.DE"]})
    tickers = [e.ticker for e in results]
    assert "AAPL.US" in tickers
    assert "SAP.DE" in tickers


def test_orchestrator_non_blocking_on_fetcher_failure():
    failing = MagicMock()
    failing.market = MarketEnum.EUROPE
    failing.prepare.side_effect = Exception("API down")
    ok = MagicMock()
    ok.market = MarketEnum.USA
    ok.prepare.return_value = [_make_entry("AAPL.US", MarketEnum.USA)]
    orch = Orchestrator([failing, ok])
    results = orch.run({"USA": ["AAPL"], "EUROPE": ["SAP.DE"]})
    assert any(e.ticker == "AAPL.US" for e in results)


def test_orchestrator_empty_when_all_fail():
    f = MagicMock()
    f.market = MarketEnum.USA
    f.prepare.side_effect = RuntimeError("dead")
    orch = Orchestrator([f])
    results = orch.run({"USA": ["AAPL"]})
    assert results == []
