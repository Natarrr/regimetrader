"""tests/test_portfolio_weight_schema.py
Schema validation: portfolio_weight field present in all entries,
non-negative, and top-20 positions sum to at most 1.0.
"""
from __future__ import annotations

import pytest
from backend.market_intel.portfolio_optimizer import run_optimizer


def _make_entries(n: int, in_portfolio: set[str]) -> list[dict]:
    """Simulate the entries list from generate_top_lists.generate()."""
    entries = []
    for i in range(n):
        ticker = f"T{i:02d}"
        entries.append({
            "ticker": ticker,
            "final_score": 0.9 - 0.01 * i,
            "sector": "Tech" if i % 2 == 0 else "Health",
        })
    return entries


def _attach_portfolio_weights(entries: list[dict]) -> tuple[list[dict], str]:
    """Mirror the attachment logic from generate_top_lists.generate()."""
    # Sort top-20 by final_score desc, ticker asc for tie-break
    sorted_candidates = sorted(
        entries, key=lambda e: (-e["final_score"], e["ticker"])
    )[:20]

    tickers = [e["ticker"] for e in sorted_candidates]
    scores = [e["final_score"] for e in sorted_candidates]
    sectors = [e.get("sector", "Unknown") for e in sorted_candidates]

    weights, method = run_optimizer(tickers, scores, sectors, vix=20.0)
    weight_set = set(tickers)

    for entry in entries:
        entry["portfolio_weight"] = round(weights.get(entry["ticker"], 0.0), 6)
        entry["portfolio_weight_method"] = method if entry["ticker"] in weight_set else "n/a"
        entry_sector = entry.get("sector", "Unknown")
        if entry["ticker"] in weight_set:
            entry["sector_weight_contribution"] = round(
                sum(
                    weights.get(t, 0.0)
                    for t, s in zip(tickers, sectors)
                    if s == entry_sector
                ),
                6,
            )
        else:
            entry["sector_weight_contribution"] = 0.0

    return entries, method


class TestPortfolioWeightSchema:
    def test_portfolio_weight_present_in_all_entries(self):
        entries = _make_entries(30, set())
        result, _ = _attach_portfolio_weights(entries)
        for entry in result:
            assert "portfolio_weight" in entry, f"{entry['ticker']} missing portfolio_weight"

    def test_portfolio_weight_nonnegative_for_all(self):
        entries = _make_entries(25, set())
        result, _ = _attach_portfolio_weights(entries)
        for entry in result:
            assert entry["portfolio_weight"] >= 0.0, (
                f"{entry['ticker']} has negative portfolio_weight {entry['portfolio_weight']}"
            )

    def test_top20_weight_sum_at_most_one(self):
        entries = _make_entries(30, set())
        result, _ = _attach_portfolio_weights(entries)
        top20_tickers = {
            e["ticker"]
            for e in sorted(entries, key=lambda e: (-e["final_score"], e["ticker"]))[:20]
        }
        top20_sum = sum(e["portfolio_weight"] for e in result if e["ticker"] in top20_tickers)
        assert top20_sum <= 1.0 + 1e-6, f"Top-20 weights sum to {top20_sum:.6f}"

    def test_non_portfolio_entries_have_zero_weight(self):
        entries = _make_entries(30, set())
        result, _ = _attach_portfolio_weights(entries)
        top20_tickers = {
            e["ticker"]
            for e in sorted(entries, key=lambda e: (-e["final_score"], e["ticker"]))[:20]
        }
        for entry in result:
            if entry["ticker"] not in top20_tickers:
                assert entry["portfolio_weight"] == 0.0

    def test_portfolio_weight_method_field_present(self):
        entries = _make_entries(25, set())
        result, _ = _attach_portfolio_weights(entries)
        for entry in result:
            assert "portfolio_weight_method" in entry

    def test_non_portfolio_method_is_na(self):
        entries = _make_entries(25, set())
        result, _ = _attach_portfolio_weights(entries)
        top20_tickers = {
            e["ticker"]
            for e in sorted(entries, key=lambda e: (-e["final_score"], e["ticker"]))[:20]
        }
        for entry in result:
            if entry["ticker"] not in top20_tickers:
                assert entry["portfolio_weight_method"] == "n/a"

    def test_sector_weight_contribution_present_for_all(self):
        entries = _make_entries(25, set())
        result, _ = _attach_portfolio_weights(entries)
        for entry in result:
            assert "sector_weight_contribution" in entry

    def test_sector_weight_contribution_nonnegative(self):
        entries = _make_entries(25, set())
        result, _ = _attach_portfolio_weights(entries)
        for entry in result:
            assert entry["sector_weight_contribution"] >= 0.0
