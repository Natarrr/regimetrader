# Path: src/scoring/analyst.py
"""Analyst consensus scoring from bulk NDJSON snapshot.

Reads upgrades-downgrades-consensus-bulk.ndjson pre-fetched by
fmp_bulk_prefetch.py. Never calls the per-ticker FMP endpoint.

Reference: Givoly & Lakonishok (1979) — analyst estimate revisions
precede price moves.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

_CONSENSUS_SCORE: dict[str, float] = {
    "Strong Buy":  1.00,
    "Buy":         0.75,
    "Hold":        0.50,
    "Sell":        0.25,
    "Strong Sell": 0.00,
}
_MIN_ANALYSTS = 2


def _score_record(symbol: str, record: dict) -> tuple[float, str]:
    """Score a single pre-fetched consensus record. Used by both the file
    reader (score_analyst_consensus) and direct index callers in run_pipeline.py.

    Args:
        symbol: Ticker symbol (for logging/debugging).
        record: Dict with consensus string or raw rating counts.

    Returns:
        (score [0, 1], source_tag)
    """
    consensus = (record.get("consensus") or "").strip()

    # Rating counts — two record shapes in the wild:
    #   legacy NDJSON snapshot: analystRatingsStrongBuy, analystRatingsBuy, ...
    #   live CSV bulk route:    strongBuy, buy, hold, sell, strongSell
    def _count(*keys: str) -> int:
        for key in keys:
            v = record.get(key)
            if v is None or v == "":
                continue
            try:
                return int(v)
            except (TypeError, ValueError):
                continue
        return 0

    strong_buy  = _count("analystRatingsStrongBuy",  "strongBuy")
    buy         = _count("analystRatingsBuy",        "buy")
    hold        = _count("analystRatingsHold",       "hold")
    sell        = _count("analystRatingsSell",       "sell")
    strong_sell = _count("analystRatingsStrongSell", "strongSell")
    total = strong_buy + buy + hold + sell + strong_sell

    if consensus in _CONSENSUS_SCORE:
        # Live CSV records carry no explicit count field — derive coverage
        # from the rating-count sum when analystRatingsCount/numAnalysts absent.
        analyst_count = _count("analystRatingsCount", "numAnalysts") or total
        if analyst_count < _MIN_ANALYSTS:
            return 0.0, f"insufficient_coverage:{analyst_count}"
        return _CONSENSUS_SCORE[consensus], f"consensus:{consensus}:{analyst_count}"

    if total < _MIN_ANALYSTS:
        return 0.0, f"insufficient_coverage:{total}"

    weighted = (
        strong_buy * 1.00 + buy * 0.75 + hold * 0.50 +
        sell * 0.25 + strong_sell * 0.00
    ) / total
    return round(weighted, 4), f"consensus_computed:{total}"


def score_analyst_consensus(
    symbol: str,
    bulk_cache_dir: str | Path = ".cache/bulk_snapshots",
) -> tuple[float, str]:
    """Score analyst consensus from bulk NDJSON snapshot.

    Returns (score [0, 1], source_tag).
    source_tag examples: "consensus:Strong Buy:8", "no_coverage",
                         "cache_missing", "insufficient_coverage:1",
                         "consensus_computed:10", "soft_failure"

    Dead signal convention: absent/insufficient → 0.0, not 0.5.
    0.5 means genuinely neutral (equal buy/sell analyst distribution).
    """
    try:
        cache_path = Path(bulk_cache_dir) / "upgrades-downgrades-consensus-bulk.ndjson"
        if not cache_path.exists():
            log.warning("Bulk consensus cache missing: %s", cache_path)
            return 0.0, "cache_missing"

        record = None
        with cache_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (obj.get("symbol") or "").upper() == symbol.upper():
                    record = obj
                    break

        if record is None:
            return 0.0, "no_coverage"

        return _score_record(symbol, record)

    except Exception as exc:
        log.warning("analyst_consensus soft failure for %s: %s", symbol, exc)
        return 0.0, "soft_failure"
