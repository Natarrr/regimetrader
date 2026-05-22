import sys
import os
import json
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.backtest_signals import (
    SignalRecord,
    _parse_snapshot,
    _market_flag,
    _failure_label,
    _alpha_decay_tag,
    _truncate_field,
)


def _snapshot_data_with_market():
    """Minimal snapshot data with market and company_name fields."""
    return {
        "generated_at": "2026-01-10T12:00:00Z",
        "weights": {"edgar": 0.28, "insider": 0.23, "congress": 0.22, "news": 0.15, "momentum": 0.12},
        "top_buys": [
            {
                "ticker": "6758.T",
                "final_score": 0.85,  # Must be >= 0.80 for HIGH BUY
                "badge": "HIGH BUY",
                "market": "ASIA",
                "company_name": "Sony",
                "factors": {"edgar": 0.8}
            }
        ]
    }


def test_signal_record_has_market_field():
    rec = SignalRecord(
        ticker="6758.T",
        signal_date=date(2026, 1, 10),
        badge="HIGH BUY",
        cap_tier="large",
        final_score=0.72,
        factors={"edgar": 0.8},
        weights={"edgar": 0.28},
        strategy_era="v2",
        source_file="test.json",
        entry_next_day=True,
        market="ASIA",
        company_name="Sony",
    )
    assert rec.market == "ASIA"
    assert rec.company_name == "Sony"


def test_parse_snapshot_captures_market_and_company_name():
    snap_data = _snapshot_data_with_market()

    # Write to a temporary file so _parse_snapshot can read it
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(snap_data, f)
        temp_path = f.name

    try:
        records = _parse_snapshot(Path(temp_path))
        assert len(records) == 1, f"Expected 1 record, got {len(records)}"
        r = records[0]
        assert r.market == "ASIA", f"Expected market='ASIA', got '{r.market}'"
        assert r.company_name == "Sony", f"Expected company_name='Sony', got '{r.company_name}'"
    finally:
        Path(temp_path).unlink()


def test_market_flag_usa():
    assert _market_flag("USA") == "🇺🇸"


def test_market_flag_europe():
    assert _market_flag("EUROPE") == "🇪🇺"


def test_market_flag_asia():
    assert _market_flag("ASIA") == "🇯🇵"


def test_market_flag_unknown():
    assert _market_flag("OTHER") == "🌐"


def test_failure_label_stopped_out():
    # Returns STOPPED OUT when T+5 return <= -0.05
    assert _failure_label(-0.06) == "STOPPED OUT"


def test_failure_label_mean_reverted():
    # Returns MEAN REVERTED when return is between -0.05 and -0.015
    assert _failure_label(-0.03) == "MEAN REVERTED"


def test_failure_label_noise():
    # Returns NOISE when return is between -0.015 and 0
    assert _failure_label(-0.01) == "NOISE"


def test_failure_label_winner():
    # Returns empty string for positive returns (not a failure)
    assert _failure_label(0.02) == ""


def test_alpha_decay_tag_decaying():
    # Alpha is decaying when T+20 < T+10 * 0.5
    assert _alpha_decay_tag(0.04, 0.01) == "⚠ ALPHA DECAY"


def test_alpha_decay_tag_negative():
    # Alpha is decaying when T+20 goes negative
    assert _alpha_decay_tag(0.04, -0.01) == "⚠ ALPHA DECAY"


def test_alpha_decay_tag_healthy():
    # No decay when T+20 >= T+10 * 0.5 and T+20 >= 0
    assert _alpha_decay_tag(0.04, 0.03) == ""


def test_truncate_field_within_limit():
    assert _truncate_field("hello", 1024) == "hello"


def test_truncate_field_over_limit():
    long_str = "x" * 1100
    result = _truncate_field(long_str, 1024)
    assert len(result) <= 1024
    assert result.endswith("…")
