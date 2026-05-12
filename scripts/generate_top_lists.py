"""scripts/generate_top_lists.py
Five-factor weighted scoring → top_lists.json + top5.csv

Reads  logs/intel_source_status.json  (written by scripts/run_pipeline.py)
and applies Markowitz (1990 Nobel) portfolio ranking:

  final_score = 0.30×edgar + 0.25×insider + 0.20×congress + 0.15×news + 0.10×macro

Badge thresholds (Sharpe-inspired):
  HIGH BUY     ≥ 0.80
  TACTICAL BUY ≥ 0.60
  WATCHLIST    < 0.60

Cap tiers (relative within universe, sorted by market cap):
  large : top 20 by market cap
  mid   : rank 21–35
  small : rank 36+

Each section in top_lists.json is ranked by final_score descending (top 5).

Output:
  logs/top_lists.json — consumed by scripts/send_toplists_discord.py
  logs/top5.csv       — flat reference file for downstream analysis

Usage:
  python scripts/generate_top_lists.py --log-dir logs --run-id $GITHUB_RUN_ID
  python scripts/generate_top_lists.py --force --verbose
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from regime_trader.utils.io import save_json_atomic

log = logging.getLogger("generate_top_lists")

WEIGHTS: Dict[str, float] = {
    "edgar":    0.30,
    "insider":  0.25,
    "congress": 0.20,
    "news":     0.15,
    "macro":    0.10,
}

_BADGES = [
    (0.80, "HIGH BUY"),
    (0.60, "TACTICAL BUY"),
    (0.00, "WATCHLIST"),
]

# Cap-tier boundaries (relative rank in universe sorted by market cap)
_LARGE_CUTOFF = 20   # ranks 1–20
_MID_CUTOFF   = 35   # ranks 21–35; 36+ → small


def _badge(score: float) -> str:
    """Sharpe-inspired — map final_score to buy/watchlist tier."""
    for threshold, label in _BADGES:
        if score >= threshold:
            return label
    return "WATCHLIST"


def _final_score(row: Dict[str, Any]) -> float:
    """Markowitz (1990 Nobel) — weighted sum of five factor scores ∈ [0, 1]."""
    return round(
        WEIGHTS["edgar"]    * float(row.get("edgar_score",    0.30)) +
        WEIGHTS["insider"]  * float(row.get("insider_score",  0.50)) +
        WEIGHTS["congress"] * float(row.get("congress_score", 0.50)) +
        WEIGHTS["news"]     * float(row.get("news_score",     0.50)) +
        WEIGHTS["macro"]    * float(row.get("macro_score",    0.50)),
        4,
    )


def _to_entry(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a run_pipeline result row to a ranked-list entry."""
    score = _final_score(row)
    return {
        "ticker":      row.get("ticker", "?"),
        "sector":      row.get("sector", "Unknown"),
        "cap_tier":    row.get("cap_tier", "large"),
        "market_cap":  float(row.get("market_cap", 0.0)),
        "final_score": score,
        "badge":       _badge(score),
        "ceo_buy":     bool(row.get("ceo_buy", False)),
        "form4_count": int(row.get("form4_count", 0)),
        "vix":         float(row.get("vix", 20.0)),
        "factors": {
            "edgar":    float(row.get("edgar_score",    0.30)),
            "insider":  float(row.get("insider_score",  0.50)),
            "congress": float(row.get("congress_score", 0.50)),
            "news":     float(row.get("news_score",     0.50)),
            "macro":    float(row.get("macro_score",    0.50)),
        },
    }


def _assign_cap_tiers(entries: List[Dict[str, Any]]) -> None:
    """Markowitz (1990 Nobel) — assign relative cap tiers within universe.

    Sorts entries by market_cap descending and overwrites cap_tier in-place:
      ranks 1–20  → large
      ranks 21–35 → mid
      ranks 36+   → small
    """
    by_mktcap = sorted(entries, key=lambda e: e["market_cap"], reverse=True)
    for rank, entry in enumerate(by_mktcap, 1):
        if rank <= _LARGE_CUTOFF:
            entry["cap_tier"] = "large"
        elif rank <= _MID_CUTOFF:
            entry["cap_tier"] = "mid"
        else:
            entry["cap_tier"] = "small"


def generate(
    status: Dict[str, Any],
    run_id: str,
    log_dir: Path,
) -> Dict[str, Any]:
    """Markowitz (1990 Nobel) — score, rank, and tier the full ticker universe.

    Args:
        status:  Parsed intel_source_status.json from run_pipeline.py.
        run_id:  Identifier stamped into generated_at metadata (e.g. GitHub run ID).
        log_dir: Output directory for top_lists.json and top5.csv.

    Returns:
        The top_lists dict written to disk.
    """
    results = status.get("results", [])
    if not results:
        log.warning("No results found in intel_source_status.json — producing empty top_lists")

    entries = [_to_entry(row) for row in results]
    _assign_cap_tiers(entries)

    score_desc = lambda e: e["final_score"]  # noqa: E731

    top_buys = sorted(entries, key=score_desc, reverse=True)[:5]
    mid_caps = sorted(
        [e for e in entries if e["cap_tier"] == "mid"],
        key=score_desc, reverse=True,
    )[:5]
    small_caps = sorted(
        [e for e in entries if e["cap_tier"] == "small"],
        key=score_desc, reverse=True,
    )[:5]

    top_lists: Dict[str, Any] = {
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "source_run_id": run_id,
        "ticker_count":  len(entries),
        "weights":       WEIGHTS,
        "top_buys":      top_buys,
        "mid_caps":      mid_caps,
        "small_caps":    small_caps,
    }

    out_json = log_dir / "top_lists.json"
    save_json_atomic(out_json, top_lists)
    log.info(
        "Wrote %s — %d tickers, top buy: %s %.4f",
        out_json,
        len(entries),
        top_buys[0]["ticker"] if top_buys else "—",
        top_buys[0]["final_score"] if top_buys else 0.0,
    )

    out_csv = log_dir / "top5.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "rank", "ticker", "sector", "cap_tier", "market_cap",
            "final_score", "badge", "ceo_buy", "form4_count",
            "edgar", "insider", "congress", "news", "macro",
        ])
        for rank, entry in enumerate(top_buys, 1):
            f = entry["factors"]
            writer.writerow([
                rank,
                entry["ticker"],
                entry["sector"],
                entry["cap_tier"],
                entry["market_cap"],
                entry["final_score"],
                entry["badge"],
                entry["ceo_buy"],
                entry["form4_count"],
                f["edgar"], f["insider"], f["congress"], f["news"], f["macro"],
            ])
    log.info("Wrote %s", out_csv)

    return top_lists


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rank intel_source_status.json into top_lists.json"
    )
    parser.add_argument("--log-dir", type=Path, default=Path("logs"),
                        help="Directory containing intel_source_status.json (default: logs)")
    parser.add_argument("--run-id", type=str, default="local",
                        help="Identifier stamped into top_lists.json (e.g. $GITHUB_RUN_ID)")
    parser.add_argument("--force", action="store_true",
                        help="Re-generate even if top_lists.json is less than 2 hours old")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    status_path = args.log_dir / "intel_source_status.json"
    if not status_path.exists():
        log.error("intel_source_status.json not found at %s — run scripts/run_pipeline.py first", status_path)
        return 1

    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.error("Could not parse intel_source_status.json: %s", exc)
        return 1

    # Skip if fresh enough (avoids redundant re-ranks within same pipeline run)
    out = args.log_dir / "top_lists.json"
    if out.exists() and not args.force:
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
            ts = datetime.fromisoformat(
                existing.get("generated_at", "").replace("Z", "+00:00")
            )
            age_h = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
            if age_h < 2.0:
                log.info(
                    "top_lists.json is %.1fh old — skipping (use --force to override)", age_h
                )
                return 0
        except Exception:
            pass

    try:
        generate(status, args.run_id, args.log_dir)
        return 0
    except Exception as exc:
        log.exception("Failed to generate top_lists: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
