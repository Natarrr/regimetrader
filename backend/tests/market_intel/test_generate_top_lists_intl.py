"""backend/tests/market_intel/test_generate_top_lists_intl.py

EU/Asia merge contract for generate_top_lists.generate().

After the 7-factor migration, run_pipeline._score_ticker_international emits
`momentum_long_score` / `volume_attention_score`. The merge step in
generate_top_lists must read those keys (not the legacy `momentum_score` /
`insider_score`) so the international momentum factor survives into top_lists.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.market_intel import generate_top_lists as gtl


@pytest.fixture(autouse=True)
def _no_vix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin VIX to None so the macro overlay never scales scores in these tests."""
    monkeypatch.setattr(gtl, "_read_vix", lambda _log_dir: None)


def _us_row(ticker: str) -> dict:
    """A complete US row so the schema-gate circuit breaker passes.
    All 12 FACTOR_FIELDS must be present and non-zero (threshold=4 allows at most 4 zeros).
    """
    return {
        "ticker":                    ticker,
        "market":                    "USA",
        "sector":                    "Information Technology",
        "cap_tier":                  "large",
        "market_cap":                1.0e11,
        "insider_conviction_score":  0.50,
        "insider_breadth_score":     0.50,
        "congress_score":            0.50,
        "news_sentiment_score":      0.50,
        "news_buzz_score":           0.50,
        "momentum_long_score":       0.50,
        "volume_attention_score":    0.50,
        "analyst_consensus_score":   0.50,
        "analyst_revision_score":    0.50,
        "price_target_upside_score": 0.50,
        "quality_piotroski_score":   0.50,
        "transcript_tone_score":     0.50,
    }


def _eu_row(ticker: str, momentum: float) -> dict:
    """An international row as emitted by _score_ticker_international (Fix #5)."""
    return {
        "ticker":                 ticker,
        "market":                 "EUROPE",
        "sector":                 "Industrials",
        "cap_tier":               "large",
        "market_cap":             1.0,
        "company_name":           "Test EU Co",
        "final_score":            momentum,
        "momentum_long_score":    momentum,
        "volume_attention_score": 0.0,
        # structurally absent factors are None (FMP 403 for non-US)
        "insider_conviction_score": None,
        "insider_breadth_score":    None,
        "congress_score":           None,
        "news_sentiment_score":     None,
        "news_buzz_score":          None,
    }


def test_eu_momentum_long_survives_merge(tmp_path: Path) -> None:
    """The EU entry's momentum_long factor must reflect momentum_long_score,
    not the legacy momentum_score key (which the pipeline no longer emits)."""
    status = {
        "results": [_us_row("AAPL"), _eu_row("SAP.DE", momentum=0.62)],
        "weights": gtl.WEIGHTS,
    }
    gtl.generate(status, run_id="test", log_dir=tmp_path)

    top_lists = json.loads((tmp_path / "top_lists.json").read_text(encoding="utf-8"))
    eu = next(e for e in top_lists["top_buys_europe"] if e["ticker"] == "SAP.DE")
    assert eu["factors"]["momentum_long"] == pytest.approx(0.62)


def test_eu_entry_has_no_congress_contamination(tmp_path: Path) -> None:
    """EU rows must carry congress factor == 0.0 (audit_payload enforces this)."""
    status = {
        "results": [_us_row("AAPL"), _eu_row("SAP.DE", momentum=0.62)],
        "weights": gtl.WEIGHTS,
    }
    gtl.generate(status, run_id="test", log_dir=tmp_path)

    top_lists = json.loads((tmp_path / "top_lists.json").read_text(encoding="utf-8"))
    eu = next(e for e in top_lists["top_buys_europe"] if e["ticker"] == "SAP.DE")
    assert eu["factors"]["congress"] == 0.0
