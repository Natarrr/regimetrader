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
