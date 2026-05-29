"""regime_trader.research.historical_loader — snapshot I/O and schema normalization.

Responsibilities:
  1. archive_current_run()   — copies today's intel_source_status.json to
                               logs/historical/YYYY-MM-DD/intel_source_status.json (idempotent)
  2. detect_schema_version() — identifies v1 (legacy 5-factor) vs v2 (7-factor)
  3. normalize_snapshot_schema() — maps cross-version fields with warnings
  4. load_historical_snapshots() — iterates dated subdirs with quality gates
  5. backfill_from_artifacts() — one-time helper to seed historical/ from archive/
"""
from __future__ import annotations

import json
import logging
import shutil
import warnings
from datetime import date
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

# ── Schema constants ─────────────────────────────────────────────────────────

_V2_SENTINEL = "insider_conviction_score"

V1_SCHEMA = "v1_legacy_5factor"
V2_SCHEMA = "v2_orthogonal_7factor"

# Congress: semantically identical across both schema versions — safe direct map.
# news_score → news_sentiment_score: WARNING issued (v1 contaminated by ~40% buzz).
# All other v1 factors have no safe v2 counterpart — marked MISSING (None).

_V1_TO_V2_MAP: dict[str, str | None] = {
    "congress_score":  "congress_score",       # identical semantics
    "news_score":      "news_sentiment_score",  # WARNING: ~40% buzz contamination
    "edgar_score":     None,                    # MISSING — insider_conviction is NOT edgar
    "insider_score":   None,                    # MISSING — breadth != old insider_score
    "momentum_score":  None,                    # MISSING — 20d reversal ≠ Jegadeesh 12-1m
}

_WARN_ISSUED: set[str] = set()  # deduplicate cross-call warnings


# ── Schema detection ─────────────────────────────────────────────────────────

def detect_schema_version(row: dict) -> str:
    """Return V2_SCHEMA if row has v2 sentinel field, else V1_SCHEMA."""
    return V2_SCHEMA if _V2_SENTINEL in row else V1_SCHEMA


# ── Schema normalization ─────────────────────────────────────────────────────

def normalize_snapshot_schema(snapshot: list[dict]) -> tuple[list[dict], str]:
    """Normalize a snapshot list to v2 schema.

    Returns (normalized_rows, schema_version_detected).

    Rules:
    - v2 rows: returned unchanged.
    - v1 rows: congress_score copied; news_score → news_sentiment_score (WARNING);
      all other v1 factors → None (MISSING, not 0.0 — dead signal ≠ absent signal).
    """
    if not snapshot:
        return [], V2_SCHEMA

    schema = detect_schema_version(snapshot[0])

    if schema == V2_SCHEMA:
        return snapshot, schema

    # v1 → v2 migration
    if "v1_news_buzz_contamination" not in _WARN_ISSUED:
        warnings.warn(
            "IC backtest: v1 news_score mapped to news_sentiment_score. "
            "V1 news_score contains ~40% buzz contamination — IC estimates for "
            "news_sentiment_score will be upward-biased. See ## Schema migration impact "
            "in the report.",
            UserWarning,
            stacklevel=2,
        )
        _WARN_ISSUED.add("v1_news_buzz_contamination")

    normalized: list[dict] = []
    for row in snapshot:
        out = {k: v for k, v in row.items() if k not in _V1_TO_V2_MAP}
        for v1_field, v2_field in _V1_TO_V2_MAP.items():
            if v2_field is not None:
                out[v2_field] = row.get(v1_field)
            # None mapping → field absent from output (treated as MISSING by ic_metrics)
        normalized.append(out)

    return normalized, schema


# ── Archiving ────────────────────────────────────────────────────────────────

def archive_current_run(log_dir: Path) -> Path | None:
    """Copy today's intel_source_status.json to log_dir/historical/YYYY-MM-DD/.

    Idempotent: skips if destination already exists.
    Returns the destination path, or None if source is absent.
    """
    src = log_dir / "intel_source_status.json"
    if not src.exists():
        logger.warning(
            "archive_current_run: %s not found — nothing to archive. "
            "Run the pipeline first.",
            src,
        )
        return None

    today = date.today().isoformat()
    dest_dir = log_dir / "historical" / today
    dest = dest_dir / "intel_source_status.json"

    if dest.exists():
        logger.debug("archive_current_run: %s already exists — skipping.", dest)
        return dest

    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)
    logger.info("archive_current_run: archived → %s", dest)
    return dest


# ── Historical snapshot loading ──────────────────────────────────────────────

def load_historical_snapshots(
    historical_dir: Path,
    start_date: date | None = None,
    end_date: date | None = None,
    min_tickers_per_snapshot: int = 80,
    min_snapshots: int = 60,
) -> Iterator[tuple[date, list[dict]]]:
    """Yield (snapshot_date, normalized_rows) for each qualifying dated subdir.

    Aborts with RuntimeError if < min_snapshots qualify after filtering.

    Args:
        historical_dir: Path to logs/historical/ directory.
        start_date: Inclusive lower bound (None → no lower bound).
        end_date: Inclusive upper bound (None → no upper bound).
        min_tickers_per_snapshot: Snapshots with fewer tickers are skipped.
        min_snapshots: Abort threshold — IC needs sufficient cross-sectional history.

    Raises:
        RuntimeError: If < min_snapshots qualify. Message is actionable:
            instructs user to run archive_current_run() more days.
    """
    if not historical_dir.exists():
        raise RuntimeError(
            f"IC backtest aborted: historical directory not found at {historical_dir}.\n"
            f"Action required:\n"
            f"  1. Wire archive_current_run() into run_pipeline.py (if not done).\n"
            f"  2. Run the pipeline daily for at least {min_snapshots} trading days "
            f"(~3 calendar months).\n"
            f"  3. To backfill from existing archive/ artifacts, call "
            f"backfill_from_artifacts(archive_dir, historical_dir)."
        )

    dated_dirs: list[tuple[date, Path]] = []
    for subdir in sorted(historical_dir.iterdir()):
        if not subdir.is_dir():
            continue
        try:
            snap_date = date.fromisoformat(subdir.name)
        except ValueError:
            logger.debug("load_historical_snapshots: skipping non-date dir %s", subdir.name)
            continue
        if start_date and snap_date < start_date:
            continue
        if end_date and snap_date > end_date:
            continue
        dated_dirs.append((snap_date, subdir))

    qualifying: list[tuple[date, list[dict]]] = []
    for snap_date, subdir in dated_dirs:
        json_file = subdir / "intel_source_status.json"
        if not json_file.exists():
            logger.debug("load_historical_snapshots: no json in %s — skipping", subdir)
            continue
        try:
            rows: list[dict] = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("load_historical_snapshots: parse error %s — %s", json_file, exc)
            continue
        if len(rows) < min_tickers_per_snapshot:
            logger.debug(
                "load_historical_snapshots: %s has %d tickers < min %d — skipping",
                snap_date, len(rows), min_tickers_per_snapshot,
            )
            continue
        normalized, _ = normalize_snapshot_schema(rows)
        qualifying.append((snap_date, normalized))

    if len(qualifying) < min_snapshots:
        raise RuntimeError(
            f"IC backtest aborted: only {len(qualifying)} qualifying snapshots found "
            f"(minimum required: {min_snapshots}).\n"
            f"Action required:\n"
            f"  • Each pipeline run archives one snapshot. You need at least "
            f"{min_snapshots} trading-day runs.\n"
            f"  • Expected time to accumulate: ~{min_snapshots // 5} calendar weeks "
            f"of daily runs.\n"
            f"  • To backfill from existing archive/ artifacts, call "
            f"backfill_from_artifacts(archive_dir, historical_dir)."
        )

    yield from qualifying


# ── Backfill helper (≤30 lines) ───────────────────────────────────────────────

def backfill_from_artifacts(
    source_dir: Path,
    target_historical_dir: Path,
) -> int:
    """Seed historical/ from top_lists JSON artifacts in source_dir (one-time use).

    Looks for files named intel_source_status_YYYY-MM-DD.json or dated subdirs.
    Returns the count of snapshots copied.
    """
    copied = 0
    for f in sorted(source_dir.glob("intel_source_status_*.json")):
        stem = f.stem.replace("intel_source_status_", "")
        try:
            date.fromisoformat(stem)
        except ValueError:
            continue
        dest_dir = target_historical_dir / stem
        dest = dest_dir / "intel_source_status.json"
        if dest.exists():
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)
        copied += 1
    if copied:
        logger.info("backfill_from_artifacts: copied %d snapshots → %s", copied, target_historical_dir)
    return copied
