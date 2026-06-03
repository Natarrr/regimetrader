#!/usr/bin/env python3
# scripts/fmp_bulk_prefetch.py
"""
FMP Ultimate — Bulk endpoint pre-fetcher.

Downloads one or more bulk snapshot endpoints and writes them to a local
cache directory with a configurable TTL.  All subsequent per-ticker scoring
functions read from cache rather than making individual FMP calls.

Call reduction:  160 per-ticker calls per endpoint → 1 bulk call per endpoint.
At 7 bulk endpoints: 1,120 per-ticker calls → 7 calls total (99.4% reduction).

Supported endpoints (all Ultimate-tier, stable/ routes):
  financial-scores-bulk              → piotroskiScore, altmanZScore, etc.
  upgrades-downgrades-consensus-bulk → consensusRating, strongBuy, buy, hold, etc.
  earnings-surprises-bulk            → actualEarningResult, estimatedEarning, date
  price-target-summary-bulk          → targetHigh, targetLow, targetConsensus, count
  ratios-ttm-bulk                    → returnOnAssets, currentRatio, debtRatio, etc.
  key-metrics-ttm-bulk               → peRatioTTM, revenuePerShareTTM, etc.
  eod-bulk                           → close, volume, date for most recent EOD

Cache format: one JSON file per endpoint at {cache_dir}/{endpoint}.json
Includes a _cached_at ISO timestamp and _ttl_hours field for TTL enforcement.

Usage:
  python scripts/fmp_bulk_prefetch.py \\
      --cache-dir .cache/bulk_snapshots \\
      --ttl-hours 23 \\
      --endpoints financial-scores-bulk upgrades-downgrades-consensus-bulk \\
      --verbose
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Stable route base URL — never use /api/v3/ or /api/v4/ directly
FMP_BASE = "https://financialmodelingprep.com/stable"

# Endpoint → stable route mapping
ENDPOINT_ROUTES: dict[str, str] = {
    "financial-scores-bulk":              "financial-scores-bulk",
    "upgrades-downgrades-consensus-bulk": "upgrades-downgrades-consensus-bulk",
    "earnings-surprises-bulk":            "earnings-surprises-bulk",
    "price-target-summary-bulk":          "price-target-summary-bulk",
    "ratios-ttm-bulk":                    "ratios-ttm-bulk",
    "key-metrics-ttm-bulk":               "key-metrics-ttm-bulk",
    "eod-bulk":                           "batch-eod-prices",  # stable route name
}

# Approximate response sizes (for logging)
ENDPOINT_SIZES_MB: dict[str, float] = {
    "financial-scores-bulk":              2.0,
    "upgrades-downgrades-consensus-bulk": 4.0,
    "earnings-surprises-bulk":            5.0,
    "price-target-summary-bulk":          3.0,
    "ratios-ttm-bulk":                    15.0,
    "key-metrics-ttm-bulk":               12.0,
    "eod-bulk":                           8.0,
}


def _cache_path(cache_dir: Path, endpoint: str) -> Path:
    return cache_dir / f"{endpoint}.json"


def _is_cache_valid(cache_dir: Path, endpoint: str, ttl_hours: float) -> bool:
    """Return True if cache file exists and is within TTL."""
    if ttl_hours <= 0:
        return False
    p = _cache_path(cache_dir, endpoint)
    if not p.exists():
        return False
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        cached_at_str = data.get("_cached_at", "")
        if not cached_at_str:
            return False
        cached_at = datetime.fromisoformat(cached_at_str)
        age_h = (datetime.now(timezone.utc) - cached_at).total_seconds() / 3600
        if age_h < ttl_hours:
            logger.info(
                "Cache HIT %s (age=%.1fh < ttl=%.1fh)",
                endpoint, age_h, ttl_hours,
            )
            return True
        logger.info(
            "Cache STALE %s (age=%.1fh >= ttl=%.1fh)",
            endpoint, age_h, ttl_hours,
        )
        return False
    except Exception as exc:
        logger.warning("Cache read error for %s: %s", endpoint, exc)
        return False


def _fetch_endpoint(
    endpoint: str,
    api_key: str,
    rps: float,
    session: requests.Session,
) -> list[dict[str, Any]]:
    """Fetch a bulk endpoint from FMP stable/ route. Returns list of records."""
    route = ENDPOINT_ROUTES.get(endpoint)
    if not route:
        raise ValueError(f"Unknown endpoint: {endpoint!r}")

    url = f"{FMP_BASE}/{route}"
    params = {"apikey": api_key}

    # Respect rate limit — sleep before each call
    delay = 1.0 / rps if rps > 0 else 0
    if delay > 0:
        time.sleep(delay)

    logger.info(
        "Fetching %s (~%.0f MB expected)...",
        endpoint, ENDPOINT_SIZES_MB.get(endpoint, 0),
    )

    t0 = time.monotonic()
    resp = session.get(url, params=params, timeout=120)
    elapsed = time.monotonic() - t0

    if resp.status_code == 401:
        raise PermissionError(f"FMP 401 on {endpoint} — check FMP_API_KEY and plan tier")
    if resp.status_code == 403:
        raise PermissionError(
            f"FMP 403 on {endpoint} — endpoint may require Ultimate tier upgrade"
        )
    if resp.status_code == 404:
        raise ValueError(f"FMP 404 on {endpoint} — route {url!r} not found")
    if not resp.ok:
        raise RuntimeError(f"FMP {resp.status_code} on {endpoint}: {resp.text[:200]}")

    data = resp.json()
    if not isinstance(data, list):
        raise RuntimeError(
            f"FMP bulk {endpoint} returned {type(data).__name__}, expected list"
        )

    logger.info(
        "Fetched %s: %d records in %.1fs",
        endpoint, len(data), elapsed,
    )
    return data


def prefetch(
    endpoints: list[str],
    cache_dir: Path,
    ttl_hours: float,
    api_key: str,
    rps: float,
) -> dict[str, bool]:
    """
    Fetch all requested endpoints, writing to cache.
    Returns dict of {endpoint: success}.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, bool] = {}

    session = requests.Session()
    session.headers.update({"User-Agent": "regime-trader/1.0"})

    for endpoint in endpoints:
        if _is_cache_valid(cache_dir, endpoint, ttl_hours):
            results[endpoint] = True
            continue

        try:
            records = _fetch_endpoint(endpoint, api_key, rps, session)
            payload = {
                "_cached_at": datetime.now(timezone.utc).isoformat(),
                "_ttl_hours": ttl_hours,
                "_record_count": len(records),
                "data": records,
            }
            p = _cache_path(cache_dir, endpoint)
            p.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
            logger.info("Cached %s → %s (%d records)", endpoint, p, len(records))
            results[endpoint] = True

        except Exception as exc:
            logger.error("FAILED %s: %s", endpoint, exc)
            results[endpoint] = False

    return results


def load_bulk(cache_dir: Path, endpoint: str) -> list[dict[str, Any]]:
    """
    Load a cached bulk endpoint.  Call this from scoring functions.

    Returns list of records, or empty list if cache is absent.
    Never raises — callers must handle empty gracefully (soft failure rule).
    """
    p = _cache_path(cache_dir, endpoint)
    if not p.exists():
        logger.warning("Bulk cache absent: %s", endpoint)
        return []
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
        return payload.get("data", [])
    except Exception as exc:
        logger.warning("Bulk cache read error %s: %s", endpoint, exc)
        return []


def build_ticker_index(
    cache_dir: Path,
    endpoint: str,
    key_field: str = "symbol",
) -> dict[str, dict[str, Any]]:
    """
    Load bulk cache and return a dict keyed by ticker symbol.
    Handles both 'symbol' and 'ticker' field names across endpoints.
    """
    records = load_bulk(cache_dir, endpoint)
    index: dict[str, dict[str, Any]] = {}
    for rec in records:
        sym = rec.get(key_field) or rec.get("symbol") or rec.get("ticker") or ""
        if sym:
            index[sym.upper()] = rec
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="FMP Ultimate bulk pre-fetcher")
    parser.add_argument(
        "--cache-dir", default=".cache/bulk_snapshots",
        help="Directory to write cached bulk files",
    )
    parser.add_argument(
        "--ttl-hours", type=float, default=23.0,
        help="Cache TTL in hours (0 = always refresh)",
    )
    parser.add_argument(
        "--endpoints", nargs="+", required=True,
        choices=list(ENDPOINT_ROUTES.keys()),
        help="Bulk endpoints to fetch",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    api_key = os.environ.get("FMP_API_KEY", "")
    if not api_key:
        logger.error("FMP_API_KEY not set")
        sys.exit(1)

    rps = float(os.environ.get("FMP_MAX_RPS", "50"))

    results = prefetch(
        endpoints=args.endpoints,
        cache_dir=Path(args.cache_dir),
        ttl_hours=args.ttl_hours,
        api_key=api_key,
        rps=rps,
    )

    failures = [ep for ep, ok in results.items() if not ok]
    if failures:
        logger.error("Failed endpoints: %s", failures)
        sys.exit(1)

    logger.info(
        "Bulk pre-fetch complete: %d/%d endpoints OK",
        sum(results.values()), len(results),
    )


if __name__ == "__main__":
    main()
