#!/usr/bin/env python3
# Path: src/ingestion/fmp_bulk_prefetch.py
"""
FMP Ultimate — Bulk endpoint pre-fetcher.

Downloads bulk snapshot endpoints and writes them to a local cache directory
with a configurable TTL.  Scoring functions read from cache to avoid per-ticker
FMP round-trips.

Confirmed working stable/ routes (verified 2026-06-03):
  upgrades-downgrades-consensus-bulk → consensus field; analyst_consensus score
  ratios-ttm-bulk                    → ratio fields; quality_piotroski score
                                       (score_quality_piotroski tolerates TTM-suffixed
                                       and unsuffixed field names)

Response format: FMP returns NDJSON (one JSON object per line), not a JSON array.
The parser handles both formats transparently.

Removed endpoints (not available or no scoring consumer):
  financial-scores-bulk   — FMP stable/ 404; piotroski sourced per-ticker via FMPClient
  eod-bulk                — FMP stable/ 404; no scoring consumer
  earnings-surprises-bulk — no scoring consumer; PEAD boost uses per-ticker
                            FMPClient.get_earnings_surprise() (stable/ "earnings")
  price-target-summary-bulk — no scoring consumer; PT upside uses per-ticker FMPClient
  key-metrics-ttm-bulk    — ~12 MB/day with zero analytical consumers; re-add to
                            ENDPOINT_ROUTES only when a factor reads its fields

Cache format: {cache_dir}/{endpoint}.json
Metadata: _cached_at (ISO), _ttl_hours, _record_count.

Usage:
  python src/ingestion/fmp_bulk_prefetch.py \\
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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
}

# Approximate response sizes (for logging)
ENDPOINT_SIZES_MB: dict[str, float] = {
    "upgrades-downgrades-consensus-bulk": 4.0,
    "ratios-ttm-bulk":                    15.0,
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


def _cached_at_str(cache_dir: Path, endpoint: str) -> str | None:
    """Return the stored ``_cached_at`` ISO string for a cached endpoint, or None."""
    try:
        data = json.loads(_cache_path(cache_dir, endpoint).read_text(encoding="utf-8"))
        return data.get("_cached_at") or None
    except Exception:
        return None


def _coerce_csv_value(raw: str | None) -> Any:
    """Coerce a CSV string cell to int/float/None so downstream scorers
    (int()/float() conversions in analyst.py, momentum_signals.py) behave
    identically to the JSON record shape."""
    if raw is None:
        return None
    s = raw.strip()
    if s == "":
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def _parse_response(endpoint: str, text: str) -> list[dict[str, Any]]:
    """Parse FMP bulk response — handles JSON array, NDJSON, and CSV.

    FMP bulk routes serve text/csv as of 2026-06-09 (confirmed live for all
    three endpoints: header row + quoted cells). Earlier snapshots were
    NDJSON; JSON array is attempted first for forward compatibility.
    Silently returning 0 records on an unrecognized format is what masked
    the bulk pipeline being dead — the CSV branch closes that gap.
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

    first_line = text.splitlines()[0].strip() if text.splitlines() else ""
    if first_line.startswith("{"):
        # NDJSON — one JSON object per line.
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

    # CSV — header row defines field names (e.g. symbol,strongBuy,...,consensus).
    import csv
    import io

    records = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        rec = {k: _coerce_csv_value(v) for k, v in row.items() if k is not None}
        if rec.get("symbol") or rec.get("ticker"):
            records.append(rec)
    if not records:
        logger.warning(
            "Bulk response for %s parsed to 0 records (unrecognized format; "
            "first line: %.80s)", endpoint, first_line,
        )
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
) -> dict[str, str]:
    """Fetch all requested endpoints, writing to cache.

    Returns ``{endpoint: status}`` where status is one of:
      "fresh"  — served from a within-TTL cache or a successful live fetch.
      "stale"  — fetch FAILED but a prior cache existed and was served. The
                 data is OLD; this is surfaced loudly (ERROR log + status
                 marker) so it is never silently treated as fresh.
      "failed" — fetch failed and no cache exists at all.

    Also writes ``{cache_dir}/bulk_prefetch_status.json`` (endpoint → status +
    cache age) so the operator/pipeline can see which feeds were served stale.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, str] = {}
    detail: dict[str, dict[str, Any]] = {}

    session = requests.Session()
    _retry = Retry(
        total=4,
        backoff_factor=2.0,
        status_forcelist={429, 500, 502, 503, 504},
        allowed_methods={"GET"},
        raise_on_status=False,
    )
    _adapter = HTTPAdapter(max_retries=_retry)
    session.mount("https://", _adapter)
    session.mount("http://", _adapter)
    session.headers.update({"User-Agent": "regime-trader/1.0"})

    for endpoint in endpoints:
        if _is_cache_valid(cache_dir, endpoint, ttl_hours):
            results[endpoint] = "fresh"
            detail[endpoint] = {
                "status": "fresh",
                "cached_at": _cached_at_str(cache_dir, endpoint),
            }
            continue

        try:
            records = _fetch_endpoint(endpoint, api_key, rps, session)
            cached_at = datetime.now(timezone.utc).isoformat()
            payload = {
                "_cached_at": cached_at,
                "_ttl_hours": ttl_hours,
                "_record_count": len(records),
                "data": records,
            }
            p = _cache_path(cache_dir, endpoint)
            p.write_text(json.dumps(
                payload, separators=(",", ":")), encoding="utf-8")
            logger.info("Cached %s → %s (%d records)",
                        endpoint, p, len(records))
            results[endpoint] = "fresh"
            detail[endpoint] = {"status": "fresh", "cached_at": cached_at}

        except Exception as exc:
            logger.error("FAILED %s: %s", endpoint, exc)
            # Fallback: serve a prior cache rather than hard-failing, but mark it
            # "stale" and log LOUDLY — downstream scores using this feed are NOT
            # fresh and must not be reported as such (see send_discord DATA age).
            stale = _cache_path(cache_dir, endpoint)
            if stale.exists():
                try:
                    payload = json.loads(stale.read_text(encoding="utf-8"))
                    cached_at = payload.get(
                        "_cached_at", "2000-01-01T00:00:00+00:00")
                    age_h = (
                        datetime.now(timezone.utc)
                        - datetime.fromisoformat(cached_at)
                    ).total_seconds() / 3600
                    logger.error(
                        "STALE CACHE served for %s (age=%.1fh); fetch failed: "
                        "%s. Scores using this feed are NOT fresh.",
                        endpoint, age_h, exc,
                    )
                    results[endpoint] = "stale"
                    detail[endpoint] = {
                        "status": "stale",
                        "cached_at": cached_at,
                        "age_hours": round(age_h, 1),
                    }
                    continue
                except Exception:
                    pass
            results[endpoint] = "failed"
            detail[endpoint] = {"status": "failed", "cached_at": None}

    try:
        (cache_dir / "bulk_prefetch_status.json").write_text(
            json.dumps(
                {"_written_at": datetime.now(timezone.utc).isoformat(),
                 "endpoints": detail},
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("could not write bulk_prefetch_status.json: %s", exc)

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

    Returns: index dict. Ambiguous base symbols are excluded (deleted from index).
    For access to the ambiguous_bases set, use build_ticker_index_with_ambiguous().
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


def build_ticker_index_with_ambiguous(
    cache_dir: Path,
    endpoint: str,
    key_field: str = "symbol",
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Load bulk cache and return (index, ambiguous_bases).

    The index maps ticker symbols (and unambiguous base symbols) to records.
    The ambiguous_bases set contains base symbols with multiple exchange variants
    that were removed from the index to prevent incorrect fallback lookups.

    Use this when you need to guard against base-symbol fallback for ambiguous bases.
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
    return index, ambiguous_bases


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

    failed = [ep for ep, st in results.items() if st == "failed"]
    stale = [ep for ep, st in results.items() if st == "stale"]
    ok_count = sum(1 for st in results.values() if st in ("fresh", "stale"))

    if failed:
        # "failed" already means fetch failed AND no cache fallback exists.
        logger.error("Failed endpoints with no cache fallback: %s", failed)
        sys.exit(1)
    if stale:
        logger.error(
            "Endpoints served from STALE cache (fetch failed, old data): %s",
            stale,
        )

    logger.info(
        "Bulk pre-fetch complete: %d/%d endpoints OK (%d stale)",
        ok_count, len(results), len(stale),
    )


if __name__ == "__main__":
    main()
