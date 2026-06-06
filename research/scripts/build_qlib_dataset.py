# Path: research/scripts/build_qlib_dataset.py
"""Convert research/data/backfill/factor_scores.ndjson → qlib binary dataset.

Run from repo root after backfill_factors.py completes:
    python research/scripts/build_qlib_dataset.py

Output: research/data/qlib_data/ (qlib binary format)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("build_qlib_dataset")

_IN = Path("research/data/backfill/factor_scores.ndjson")
_OUT_DIR = Path("research/data/qlib_data")

FEATURE_COLS = [
    "insider_conviction", "insider_breadth", "congress",
    "news_sentiment", "news_buzz", "momentum_long",
    "volume_attention", "analyst_consensus", "quality_piotroski",
]
LABEL_COL = "forward_return_21d"


def load_ndjson(path: Path) -> pd.DataFrame:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame(records)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df = df.sort_values(["ticker", "snapshot_date"]).reset_index(drop=True)
    return df


def build_qlib_dataset(df: pd.DataFrame, out_dir: Path) -> None:
    """Write qlib-compatible CSV files per ticker under out_dir/."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for ticker, group in df.groupby("ticker"):
        g = group.set_index("snapshot_date").sort_index()
        ticker_df = g[FEATURE_COLS + [LABEL_COL]].copy()
        ticker_df.index.name = "date"
        out_path = out_dir / f"{ticker}.csv"
        ticker_df.to_csv(out_path)
    log.info("Wrote %d ticker CSVs to %s", df["ticker"].nunique(), out_dir)

    # Also write a combined parquet for convenience in notebooks
    combined_path = out_dir / "all_factors.parquet"
    df.to_parquet(combined_path, index=False)
    log.info("Combined parquet: %s (%d rows)", combined_path, len(df))


def validate_roundtrip(df: pd.DataFrame, out_dir: Path) -> None:
    """Verify NDJSON → CSV round-trip is lossless for a sample ticker."""
    sample_ticker = df["ticker"].iloc[0]
    csv_path = out_dir / f"{sample_ticker}.csv"
    loaded = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    original = df[df["ticker"] == sample_ticker].set_index("snapshot_date")[FEATURE_COLS + [LABEL_COL]]
    original.index.name = "date"
    # Check shape
    assert loaded.shape == original.shape, (
        f"Round-trip shape mismatch: {loaded.shape} vs {original.shape}"
    )
    # Check values within float tolerance
    diff = (loaded - original).abs().max().max()
    assert diff < 1e-5, f"Round-trip max diff {diff} exceeds tolerance"
    log.info("Round-trip validation passed for %s (max diff: %.2e)", sample_ticker, diff)


def main() -> None:
    if not _IN.exists():
        raise FileNotFoundError(f"{_IN} not found — run backfill_factors.py first")
    log.info("Loading %s...", _IN)
    df = load_ndjson(_IN)
    log.info("Loaded %d records, %d tickers, %d dates",
             len(df), df["ticker"].nunique(), df["snapshot_date"].nunique())
    build_qlib_dataset(df, _OUT_DIR)
    validate_roundtrip(df, _OUT_DIR)
    log.info("Done.")


if __name__ == "__main__":
    main()
