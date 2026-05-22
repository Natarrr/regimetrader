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


from unittest.mock import MagicMock, patch
from regime_trader.fetchers.edgar_fetcher import EDGARFetcher
from regime_trader.fetchers.base import MarketEnum


def test_edgar_fetcher_market():
    f = EDGARFetcher()
    assert f.market == MarketEnum.USA


def test_edgar_fetcher_source_reliability():
    f = EDGARFetcher()
    assert f.source_reliability("AAPL") == 1.0


def test_edgar_fetcher_prepare_returns_list():
    f = EDGARFetcher()
    result = f.prepare([])
    assert isinstance(result, list)


from unittest.mock import patch, MagicMock
from regime_trader.fetchers.fmp_fetcher import FMPFetcher


def test_fmp_fetcher_market():
    f = FMPFetcher(api_key="test")
    assert f.market == MarketEnum.EUROPE


def test_fmp_fetcher_source_reliability():
    f = FMPFetcher(api_key="test")
    assert f.source_reliability("SAP.DE") == 0.75


def test_fmp_fetcher_prepare_empty_on_api_error():
    f = FMPFetcher(api_key="test")
    with patch.object(f, "_fetch_quote", side_effect=Exception("network")):
        result = f.prepare(["SAP.DE"])
    assert result == []


def test_fmp_fetcher_prepare_normalizes_ticker():
    f = FMPFetcher(api_key="test")
    mock_quote = {"price": 120.0, "marketCap": 150e9, "eps": 5.0, "volume": 1e6, "avgVolume": 900000}
    with patch.object(f, "_fetch_quote", return_value=mock_quote):
        result = f.prepare(["SAP.DE"])
    assert len(result) == 1
    assert result[0].ticker == "SAP.DE"
    assert result[0].market == MarketEnum.EUROPE
    assert result[0].source_reliability == 0.75


from regime_trader.fetchers.asian_fetcher import AsianMarketFetcher


def test_asian_fetcher_market():
    f = AsianMarketFetcher()
    assert f.market == MarketEnum.ASIA


def test_asian_fetcher_source_reliability():
    f = AsianMarketFetcher()
    assert f.source_reliability("7203.T") == 0.6


def test_asian_fetcher_prepare_empty_on_error():
    f = AsianMarketFetcher()
    with patch("regime_trader.fetchers.asian_fetcher.yf.Ticker", side_effect=Exception("timeout")):
        result = f.prepare(["7203.T"])
    assert result == []


def test_asian_fetcher_prepare_returns_entry():
    f = AsianMarketFetcher()
    mock_fi = MagicMock()
    mock_fi.last_price = 2800.0
    mock_fi.market_cap = 40e12
    mock_fi.three_month_average_volume = 5e6
    mock_ticker = MagicMock()
    mock_ticker.fast_info = mock_fi
    mock_ticker.info = {"regularMarketVolume": 6e6, "trailingEps": 200.0}
    with patch("regime_trader.fetchers.asian_fetcher.yf.Ticker", return_value=mock_ticker):
        result = f.prepare(["7203.T"])
    assert len(result) == 1
    assert result[0].market == MarketEnum.ASIA
    assert result[0].source_reliability == 0.6
