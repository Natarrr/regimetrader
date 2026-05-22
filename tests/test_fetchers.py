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
    today = __import__("datetime").date.today().isoformat()
    fake_usage = {"date": today, "count": 0}
    with patch("regime_trader.fetchers.fmp_fetcher._load_usage", return_value=fake_usage), \
         patch("regime_trader.fetchers.fmp_fetcher._save_usage"), \
         patch.object(f, "_fetch_quote", return_value=mock_quote):
        result = f.prepare(["SAP.DE"])
    assert len(result) == 1
    assert result[0].ticker == "SAP.DE"
    assert result[0].market == MarketEnum.EUROPE
    assert result[0].source_reliability == 0.75


def test_fmp_fetcher_quota_blocks_requests():
    """When daily count >= 200, prepare() returns empty without calling _fetch_quote."""
    f = FMPFetcher(api_key="test")
    today = __import__("datetime").date.today().isoformat()
    full_usage = {"date": today, "count": 250}
    with patch("regime_trader.fetchers.fmp_fetcher._load_usage", return_value=full_usage), \
         patch("regime_trader.fetchers.fmp_fetcher._save_usage"), \
         patch.object(f, "_fetch_quote") as mock_fetch:
        result = f.prepare(["SAP.DE", "ASML.AS"])
    assert result == []
    mock_fetch.assert_not_called()


def test_fmp_fetcher_quota_increments_on_success():
    """Each successful fetch increments the usage counter."""
    f = FMPFetcher(api_key="test")
    today = __import__("datetime").date.today().isoformat()
    usage = {"date": today, "count": 248}
    mock_quote = {"price": 100.0, "marketCap": 1e11, "eps": 2.0, "volume": 1e6, "avgVolume": 1e6}
    saved = []
    with patch("regime_trader.fetchers.fmp_fetcher._load_usage", return_value=usage), \
         patch("regime_trader.fetchers.fmp_fetcher._save_usage", side_effect=saved.append), \
         patch.object(f, "_fetch_quote", return_value=mock_quote):
        result = f.prepare(["SAP.DE", "SIE.DE", "ASML.AS"])
    # Only 2 fetches allowed (248 → 249 → 250, then quota hit)
    assert len(result) == 2
    assert saved[-1]["count"] == 250


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
