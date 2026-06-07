#!/usr/bin/env python3
# scripts/fmp_bulk_prefetch.py
"""
FMP Ultimate — Bulk endpoint pre-fetcher.

Downloads bulk snapshot endpoints and writes them to a local cache directory
with a configurable TTL.  Scoring functions read from cache to avoid per-ticker
FMP round-trips.

Confirmed working stable/ routes (verified 2026-06-03):
  upgrades-downgrades-consensus-bulk → consensus field; analyst_consensus score
  ratios-ttm-bulk                    → returnOnAssets, currentRatio, debtRatio, etc.
  key-metrics-ttm-bulk               → peRatioTTM, revenuePerShareTTM, etc.

Response format: FMP returns NDJSON (one JSON object per line), not a JSON array.
The parser handles both formats transparently.

Removed endpoints (not available or no scoring consumer):
  financial-scores-bulk   — FMP stable/ 404; piotroski sourced per-ticker via FMPClient
  eod-bulk                — FMP stable/ 404; no scoring consumer
  earnings-surprises-bulk — no scoring consumer; PEAD boost uses per-ticker FMPClient
  price-target-summary-bulk — no scoring consumer; PT upside uses per-ticker FMPClient

Cache format: {cache_dir}/{endpoint}.json
Metadata: _cached_at (ISO), _ttl_hours, _record_count.

Usage:
  python scripts/fmp_bulk_prefetch.py \\
      --cache-dir .cache/bulk_snapshots \\
      --ttl-hours 23 \\
      --endpoints upgrades-downgrades-consensus-bulk ratios-ttm-bulk \\
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

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

FMP_BASE = "https://financialmodelingprep.com/stable"

# Only confirmed-working stable/ routes that have at least one scoring consumer.
ENDPOINT_ROUTES: dict[str, str] = {
    "upgrades-downgrades-consensus-bulk": "upgrades-downgrades-consensus-bulk",
    "ratios-ttm-bulk":                    "ratios-ttm-bulk",
    "key-metrics-ttm-bulk":               "key-metrics-ttm-bulk",
}

# Approximate response sizes (for logging)
ENDPOINT_SIZES_MB: dict[str, float] = {
    "upgrades-downgrades-consensus-bulk": 4.0,
    "ratios-ttm-bulk":                    15.0,
    "key-metrics-ttm-bulk":               12.0,
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
            logger.info("Cache HIT %s (age=%.1fh < ttl=%.1fh)",
                        endpoint, age_h, ttl_hours)
            return True
        logger.info("Cache STALE %s (age=%.1fh >= ttl=%.1fh)",
                    endpoint, age_h, ttl_hours)
        return False
    except Exception as exc:
        logger.warning("Cache read error for %s: %s", endpoint, exc)
        return False


def _parse_response(endpoint: str, text: str) -> list[dict[str, Any]]:
    """Parse FMP bulk response — handles both JSON array and NDJSON formats.

    FMP bulk endpoints return NDJSON (one JSON object per line).
    Attempts JSON array first for forward compatibility; falls back to NDJSON.
    """
    text = text.strip()
    if not text:
        return []

    if text.startswith("["):
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    records = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                records.append(obj)
        except json.JSONDecodeError as exc:
            logger.debug("NDJSON line skip (%s): %s", endpoint, exc)
    return records


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
    params: dict[str, Any] = {"apikey": api_key}

    delay = 1.0 / rps if rps > 0 else 0
    if delay > 0:
        time.sleep(delay)

    logger.info("Fetching %s (~%.0f MB expected)...",
                endpoint, ENDPOINT_SIZES_MB.get(endpoint, 0))

    t0 = time.monotonic()
    resp = session.get(url, params=params, timeout=120)
    elapsed = time.monotonic() - t0

    if resp.status_code == 401:
        raise PermissionError(
            f"FMP 401 on {endpoint} — check FMP_API_KEY and plan tier")
    if resp.status_code == 403:
        raise PermissionError(
            f"FMP 403 on {endpoint} — may require Ultimate tier upgrade")
    if resp.status_code == 404:
        raise ValueError(f"FMP 404 on {endpoint} — route {url!r} not found")
    if not resp.ok:
        raise RuntimeError(
            f"FMP {resp.status_code} on {endpoint}: {resp.text[:200]}")

    data = _parse_response(endpoint, resp.text)
    logger.info("Fetched %s: %d records in %.1fs",
                endpoint, len(data), elapsed)
    return data


def prefetch(
    endpoints: list[str],
    cache_dir: Path,
    ttl_hours: float,
    api_key: str,
    rps: float,
) -> dict[str, bool]:
    """Fetch all requested endpoints, writing to cache.
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
            p.write_text(json.dumps(
                payload, separators=(",", ":")), encoding="utf-8")
            logger.info("Cached %s → %s (%d records)",
                        endpoint, p, len(records))
            results[endpoint] = True

        except Exception as exc:
            logger.error("FAILED %s: %s", endpoint, exc)
            results[endpoint] = False

    return results


def load_bulk(cache_dir: Path, endpoint: str) -> list[dict[str, Any]]:
    """Load a cached bulk endpoint.

    Returns list of records, or empty list if cache is absent.
    Never raises — callers must handle empty gracefully.
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


def normalize_ticker_key(ticker: str) -> str:
    """Normalize a ticker to its base symbol for bulk lookup.

    Examples:
      ASML.AS -> ASML
      005930.KS -> 005930
    """
    if not ticker:
        return ""
    return ticker.split(".")[0].upper().strip()


def map_bulk_data_to_universe(
    universe_tickers: list[str],
    bulk_rows: list[dict[str, Any]],
    ticker_column_name: str = "symbol",
) -> dict[str, dict[str, Any]]:
    """Map bulk rows to universe tickers using exact and base-symbol matching.

    This is useful when FMP bulk snapshots index international assets using
    the base company label (ASML) instead of the exchange suffix variant
    (ASML.AS).
    """
    base_to_universe_map: dict[str, list[str]] = {}
    for ticker in universe_tickers:
        base_key = normalize_ticker_key(ticker)
        base_to_universe_map.setdefault(base_key, []).append(ticker)

    mapped_results: dict[str, dict[str, Any]] = {
        t: {} for t in universe_tickers}
    for row in bulk_rows:
        raw_symbol = (row.get(ticker_column_name) or row.get(
            "symbol") or row.get("ticker") or "").upper().strip()
        if not raw_symbol:
            continue

        # Exact match first.
        if raw_symbol in mapped_results:
            mapped_results[raw_symbol] = row
            continue

        # Fallback to stripped base symbol matching.
        base_symbol = normalize_ticker_key(raw_symbol)
        raw_suffix = raw_symbol.split(
            ".", 1)[1].upper() if "." in raw_symbol else ""
        candidates = base_to_universe_map.get(base_symbol, [])
        for target in candidates:
            if mapped_results[target]:
                continue  # already matched exactly — skip
            target_suffix = target.split(
                ".", 1)[1].upper() if "." in target else ""
            # Only accept the base-symbol match when:
            #   (a) exchange suffixes match exactly, OR
            #   (b) the bulk row carries no suffix AND this base resolves to exactly one
            #       universe ticker (unambiguous mapping).
            if raw_suffix == target_suffix or (not raw_suffix and len(candidates) == 1):
                mapped_results[target] = row

    return mapped_results


def build_ticker_index(
    cache_dir: Path,
    endpoint: str,
    key_field: str = "symbol",
) -> dict[str, dict[str, Any]]:
    """Load bulk cache and return a dict keyed by ticker symbol.
    Handles both 'symbol' and 'ticker' field names across endpoints.
    """
    records = load_bulk(cache_dir, endpoint)
    index: dict[str, dict[str, Any]] = {}
    ambiguous_bases: set[str] = set()
    for rec in records:
        sym = rec.get(key_field) or rec.get(
            "symbol") or rec.get("ticker") or ""
        if not sym:
            continue
        sym = sym.upper().strip()
        index[sym] = rec
        base_sym = normalize_ticker_key(sym)
        if base_sym and base_sym != sym:
            if base_sym in ambiguous_bases:
                pass  # already known ambiguous — never re-insert
            elif base_sym not in index:
                index[base_sym] = rec
            else:
                # Second record for this base: mark ambiguous, remove alias.
                del index[base_sym]
                ambiguous_bases.add(base_sym)
    return index


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FMP Ultimate bulk pre-fetcher")
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
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG logging")
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
