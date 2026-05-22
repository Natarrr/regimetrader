import sys
import os
import json
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.backtest_signals import SignalRecord, _parse_snapshot


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
