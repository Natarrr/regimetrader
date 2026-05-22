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
