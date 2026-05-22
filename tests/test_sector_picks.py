"""tests/test_sector_picks.py — unit tests for _sector_picks()"""
from __future__ import annotations
from backend.market_intel.generate_top_lists import _sector_picks

_ENTRIES = [
    {"ticker": "XOM",  "sector": "Energy",                   "final_score": 0.80, "cap_tier": "large", "market_cap": 4e11},
    {"ticker": "CVX",  "sector": "Energy",                   "final_score": 0.70, "cap_tier": "large", "market_cap": 3e11},
    {"ticker": "OXY",  "sector": "Energy",                   "final_score": 0.60, "cap_tier": "mid",   "market_cap": 5e10},
    {"ticker": "ENPH", "sector": "Energy",                   "final_score": 0.55, "cap_tier": "small", "market_cap": 1e10},
    {"ticker": "NVDA", "sector": "Information Technology",   "final_score": 0.90, "cap_tier": "large", "market_cap": 2e12},
    {"ticker": "MSFT", "sector": "Information Technology",   "final_score": 0.85, "cap_tier": "large", "market_cap": 3e12},
    {"ticker": "AAPL", "sector": "Information Technology",   "final_score": 0.82, "cap_tier": "large", "market_cap": 3e12},
    {"ticker": "DELL", "sector": "Information Technology",   "final_score": 0.50, "cap_tier": "mid",   "market_cap": 8e10},
    {"ticker": "PFE",  "sector": "Healthcare",               "final_score": 0.65, "cap_tier": "large", "market_cap": 1.5e11},
    {"ticker": "TMO",  "sector": "Healthcare",               "final_score": 0.72, "cap_tier": "large", "market_cap": 2e11},
    {"ticker": "PANW", "sector": "Communication Services",   "final_score": 0.88, "cap_tier": "mid",   "market_cap": 9e10},
    {"ticker": "META", "sector": "Communication Services",   "final_score": 0.78, "cap_tier": "large", "market_cap": 1e12},
    {"ticker": "FCX",  "sector": "Materials",                "final_score": 0.66, "cap_tier": "mid",   "market_cap": 6e10},
    {"ticker": "NEM",  "sector": "Financials",               "final_score": 0.91, "cap_tier": "large", "market_cap": 3e10},
]


def test_returns_dict_with_target_sectors():
    result = _sector_picks(_ENTRIES)
    for sector in ["Energy", "Materials", "Communication Services", "Healthcare", "Information Technology"]:
        assert sector in result


def test_top_n_per_sector_sorted_descending():
    result = _sector_picks(_ENTRIES, n=3)
    energy = result["Energy"]
    assert len(energy) == 3
    scores = [e["final_score"] for e in energy]
    assert scores == sorted(scores, reverse=True)


def test_sector_with_fewer_than_n_tickers():
    result = _sector_picks(_ENTRIES, n=3)
    materials = result["Materials"]
    assert len(materials) == 1  # only FCX in Materials


def test_non_target_sector_excluded():
    result = _sector_picks(_ENTRIES, n=3)
    all_tickers = [e["ticker"] for picks in result.values() for e in picks]
    assert "NEM" not in all_tickers  # Financials is not a target sector


def test_empty_entries_returns_empty_lists():
    result = _sector_picks([], n=3)
    for picks in result.values():
        assert picks == []
