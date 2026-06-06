# Path: research/tests/test_build_qlib_dataset.py
"""Tests for build_qlib_dataset.py — no file I/O to FMP."""
import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from research.scripts.build_qlib_dataset import (
    load_ndjson,
    build_qlib_dataset,
    validate_roundtrip,
    FEATURE_COLS,
    LABEL_COL,
)

SAMPLE_RECORDS = [
    {
        "ticker": "AAPL", "snapshot_date": "2025-06-06",
        "insider_conviction": 0.72, "insider_breadth": 0.45,
        "congress": 0.30, "news_sentiment": 0.61, "news_buzz": 0.38,
        "momentum_long": 0.84, "volume_attention": 0.22,
        "analyst_consensus": 0.70, "quality_piotroski": 0.80,
        "forward_return_21d": 0.034, "spy_return_21d": 0.018,
    },
    {
        "ticker": "AAPL", "snapshot_date": "2025-06-13",
        "insider_conviction": 0.65, "insider_breadth": 0.50,
        "congress": 0.10, "news_sentiment": 0.55, "news_buzz": 0.42,
        "momentum_long": 0.80, "volume_attention": 0.30,
        "analyst_consensus": 0.72, "quality_piotroski": 0.82,
        "forward_return_21d": -0.012, "spy_return_21d": 0.005,
    },
]


def _write_ndjson(records: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_load_ndjson_shape(tmp_path):
    p = tmp_path / "scores.ndjson"
    _write_ndjson(SAMPLE_RECORDS, p)
    df = load_ndjson(p)
    assert len(df) == 2
    assert "ticker" in df.columns
    assert "snapshot_date" in df.columns
    for col in FEATURE_COLS + [LABEL_COL]:
        assert col in df.columns


def test_load_ndjson_dates_parsed(tmp_path):
    p = tmp_path / "scores.ndjson"
    _write_ndjson(SAMPLE_RECORDS, p)
    df = load_ndjson(p)
    assert pd.api.types.is_datetime64_any_dtype(df["snapshot_date"])


def test_build_and_roundtrip(tmp_path):
    ndjson_path = tmp_path / "scores.ndjson"
    _write_ndjson(SAMPLE_RECORDS, ndjson_path)
    df = load_ndjson(ndjson_path)
    out_dir = tmp_path / "qlib_data"
    build_qlib_dataset(df, out_dir)
    # CSV should exist for AAPL
    assert (out_dir / "AAPL.csv").exists()
    # Combined parquet should exist
    assert (out_dir / "all_factors.parquet").exists()
    # Round-trip validation passes
    validate_roundtrip(df, out_dir)


def test_all_factors_in_csv(tmp_path):
    ndjson_path = tmp_path / "scores.ndjson"
    _write_ndjson(SAMPLE_RECORDS, ndjson_path)
    df = load_ndjson(ndjson_path)
    out_dir = tmp_path / "qlib_data"
    build_qlib_dataset(df, out_dir)
    loaded = pd.read_csv(out_dir / "AAPL.csv")
    for col in FEATURE_COLS + [LABEL_COL]:
        assert col in loaded.columns, f"Missing column: {col}"
