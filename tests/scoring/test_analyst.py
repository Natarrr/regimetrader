"""tests/scoring/test_analyst.py
Unit tests for analyst consensus scoring from bulk NDJSON snapshot.

Reference: Givoly & Lakonishok (1979) — analyst estimate revisions
precede price moves.
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path

from regime_trader.scoring.analyst import score_analyst_consensus


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
