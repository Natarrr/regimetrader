"""tests/scoring/test_analyst.py
Unit tests for analyst consensus scoring from bulk NDJSON snapshot.

Reference: Givoly & Lakonishok (1979) — analyst estimate revisions
precede price moves.
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path

from src.scoring.analyst import score_analyst_consensus, _score_record


def test_cache_missing_returns_cache_missing_tag(tmp_path):
    score, src = score_analyst_consensus("AAPL", bulk_cache_dir=tmp_path)
    assert score == 0.0
    assert src == "cache_missing"


def test_no_coverage_returns_zero(tmp_path):
    cache = tmp_path / "upgrades-downgrades-consensus-bulk.ndjson"
    cache.write_text(json.dumps({"symbol": "MSFT", "consensus": "Buy", "analystRatingsCount": 5}) + "\n")
    score, src = score_analyst_consensus("AAPL", bulk_cache_dir=tmp_path)
    assert score == 0.0
    assert src == "no_coverage"


def test_strong_buy_consensus_string(tmp_path):
    cache = tmp_path / "upgrades-downgrades-consensus-bulk.ndjson"
    record = {"symbol": "AAPL", "consensus": "Strong Buy", "analystRatingsCount": 10}
    cache.write_text(json.dumps(record) + "\n")
    score, src = score_analyst_consensus("AAPL", bulk_cache_dir=tmp_path)
    assert score == 1.00
    assert src == "consensus:Strong Buy:10"


def test_insufficient_coverage_threshold(tmp_path):
    cache = tmp_path / "upgrades-downgrades-consensus-bulk.ndjson"
    record = {"symbol": "AAPL", "consensus": "Buy", "analystRatingsCount": 1}
    cache.write_text(json.dumps(record) + "\n")
    score, src = score_analyst_consensus("AAPL", bulk_cache_dir=tmp_path)
    assert score == 0.0
    assert "insufficient_coverage" in src


def test_fallback_raw_counts(tmp_path):
    cache = tmp_path / "upgrades-downgrades-consensus-bulk.ndjson"
    # No 'consensus' field — use raw counts
    record = {
        "symbol": "AAPL",
        "analystRatingsStrongBuy": 4,
        "analystRatingsBuy": 4,
        "analystRatingsHold": 2,
        "analystRatingsSell": 0,
        "analystRatingsStrongSell": 0,
    }
    cache.write_text(json.dumps(record) + "\n")
    score, src = score_analyst_consensus("AAPL", bulk_cache_dir=tmp_path)
    # weighted = (4*1.0 + 4*0.75 + 2*0.50) / 10 = (4+3+1)/10 = 0.80
    assert abs(score - 0.8) < 0.001
    assert "consensus_computed" in src


def test_soft_failure_returns_zero_not_half(tmp_path):
    # Corrupt NDJSON should soft-fail with 0.0, not 0.5
    cache = tmp_path / "upgrades-downgrades-consensus-bulk.ndjson"
    cache.write_text("{bad json}\n")
    score, src = score_analyst_consensus("AAPL", bulk_cache_dir=tmp_path)
    assert score == 0.0
    # no_coverage is fine — the corrupt line is skipped, symbol never found
    assert src in ("no_coverage", "soft_failure")


def test_symbol_case_insensitive(tmp_path):
    cache = tmp_path / "upgrades-downgrades-consensus-bulk.ndjson"
    record = {"symbol": "aapl", "consensus": "Hold", "analystRatingsCount": 8}
    cache.write_text(json.dumps(record) + "\n")
    score, src = score_analyst_consensus("AAPL", bulk_cache_dir=tmp_path)
    assert score == 0.50
    assert "Hold" in src


def test_score_record_strong_sell():
    """Test _score_record directly with Strong Sell consensus."""
    record = {"consensus": "Strong Sell", "analystRatingsCount": 5}
    score, src = _score_record("AAPL", record)
    assert score == 0.0
    assert src == "consensus:Strong Sell:5"


def test_score_record_num_analysts_fallback():
    """Test _score_record with numAnalysts key fallback (from older FMP schema)."""
    record = {"consensus": "Buy", "numAnalysts": 3}
    score, src = _score_record("AAPL", record)
    assert score == 0.75
    assert src == "consensus:Buy:3"


# ── Live CSV bulk record shape (2026-06-09: strongBuy/buy/... + consensus) ──

def test_live_csv_shape_consensus_with_derived_count():
    """Live CSV records have no analystRatingsCount — coverage derives from
    the rating-count sum. Regression for the silent insufficient_coverage bug."""
    rec = {"symbol": "AAPL", "strongBuy": 10, "buy": 20, "hold": 5,
           "sell": 1, "strongSell": 0, "consensus": "Buy"}
    score, src = _score_record("AAPL", rec)
    assert score == 0.75
    assert src == "consensus:Buy:36"


def test_live_csv_shape_computed_fallback_without_consensus():
    rec = {"symbol": "MSFT", "strongBuy": 4, "buy": 0, "hold": 0,
           "sell": 0, "strongSell": 4, "consensus": ""}
    score, src = _score_record("MSFT", rec)
    assert score == 0.5
    assert src == "consensus_computed:8"


def test_live_csv_shape_thin_coverage_still_zero():
    rec = {"symbol": "TINY", "strongBuy": 1, "buy": 0, "hold": 0,
           "sell": 0, "strongSell": 0, "consensus": "Strong Buy"}
    score, src = _score_record("TINY", rec)
    assert score == 0.0
    assert src.startswith("insufficient_coverage")


def test_legacy_ndjson_shape_unchanged():
    rec = {"symbol": "NVDA", "consensus": "Strong Buy", "analystRatingsCount": 8}
    score, src = _score_record("NVDA", rec)
    assert score == 1.0
    assert src == "consensus:Strong Buy:8"
