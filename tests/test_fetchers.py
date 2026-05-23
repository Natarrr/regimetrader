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
    f = FMPFetcher(api_key="test", market=MarketEnum.EUROPE)
    assert f.market == MarketEnum.EUROPE


def test_fmp_fetcher_market_asia():
    f = FMPFetcher(api_key="test", market=MarketEnum.ASIA)
    assert f.market == MarketEnum.ASIA


def test_fmp_fetcher_source_reliability():
    # Fix #5: FMPFetcher now uses yfinance (FMP 403 for non-US). Reliability=0.60.
    f = FMPFetcher(api_key="test", market=MarketEnum.EUROPE)
    assert f.source_reliability("SAP.DE") == 0.60


def test_fmp_fetcher_prepare_empty_on_yfinance_error():
    # Fix #5: FMPFetcher uses yfinance.download — test that exceptions are caught.
    import pandas as pd
    f = FMPFetcher(api_key="test", market=MarketEnum.EUROPE)
    with patch("yfinance.download", side_effect=Exception("network")):
        result = f.prepare(["SAP.DE"])
    assert result == []


def test_fmp_fetcher_prepare_empty_on_empty_dataframe():
    import pandas as pd
    f = FMPFetcher(api_key="test", market=MarketEnum.EUROPE)
    with patch("yfinance.download", return_value=pd.DataFrame()):
        result = f.prepare(["SAP.DE"])
    assert result == []


def test_fmp_fetcher_prepare_returns_entry_with_yfinance_data():
    import numpy as np
    import pandas as pd
    f = FMPFetcher(api_key="test", market=MarketEnum.EUROPE)
    # 275 bars — enough for 12-1m return
    n = 275
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    prices = np.linspace(100.0, 115.0, n)
    vols = np.full(n, 2_000_000)
    vols[-1] = 4_000_000  # last day spike
    fake_df = pd.DataFrame({("Close", "SAP.DE"): prices, ("Volume", "SAP.DE"): vols}, index=idx)
    fake_df.columns = pd.MultiIndex.from_tuples(fake_df.columns)
    with patch("yfinance.download", return_value=fake_df):
        result = f.prepare(["SAP.DE"])
    assert len(result) == 1
    assert result[0].ticker == "SAP.DE"
    assert result[0].market == MarketEnum.EUROPE
    assert result[0].source_reliability == 0.60
    assert result[0].raw_factors["return_12_1m"] is not None
    assert result[0].raw_factors["volume_spike"] > 0


def test_fmp_fetcher_prepare_asia_market():
    # Fix #5: FMPFetcher now uses yfinance for all markets.
    import numpy as np
    import pandas as pd
    f = FMPFetcher(api_key="test", market=MarketEnum.ASIA)
    n = 275
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    prices = np.linspace(2800.0, 3100.0, n)
    vols = np.full(n, 16_000_000)
    fake_df = pd.DataFrame({("Close", "7203.T"): prices, ("Volume", "7203.T"): vols}, index=idx)
    fake_df.columns = pd.MultiIndex.from_tuples(fake_df.columns)
    with patch("yfinance.download", return_value=fake_df):
        result = f.prepare(["7203.T"])
    assert len(result) == 1
    assert result[0].market == MarketEnum.ASIA
    assert result[0].source_reliability == 0.60


def test_fmp_fetcher_no_quota_logic():
    """Fix #5: FMPFetcher uses yfinance — no daily quota enforcement."""
    import pandas as pd
    f = FMPFetcher(api_key="test", market=MarketEnum.EUROPE)
    # Empty DataFrame → skipped, but no quota check
    with patch("yfinance.download", return_value=pd.DataFrame()):
        result = f.prepare(["SAP.DE", "ASML.AS"])
    assert result == []


def test_fmp_fetcher_multiple_tickers_returns_multiple_entries():
    """Fix #5: Each ticker with valid yfinance data produces one TickerEntry."""
    import numpy as np
    import pandas as pd
    n = 275
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    prices = np.linspace(100.0, 115.0, n)
    vols = np.full(n, 1_000_000)
    f = FMPFetcher(api_key="test", market=MarketEnum.EUROPE)

    call_count = [0]
    def fake_download(ticker, **kwargs):
        call_count[0] += 1
        fake = pd.DataFrame(
            {("Close", ticker): prices, ("Volume", ticker): vols}, index=idx
        )
        fake.columns = pd.MultiIndex.from_tuples(fake.columns)
        return fake

    with patch("yfinance.download", side_effect=fake_download):
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
