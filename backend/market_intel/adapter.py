"""backend/market_intel/adapter.py — public single-entry adapter.

Replaces the per-ticker portion of streamlit_app._run_intel_fetch.
EDGAR is primary (authoritative regulator filings); FMP is fallback.

Public functions:
    fetch_intel(ticker, ...) -> dict        — single ticker
    fetch_intel_universe(tickers, ...) -> List[dict]   — parallel batch
"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from . import config
from .edgar_ingest import fetch_edgar_for_ticker
from .edgar_parse import parse_form4_file, parse_form13f_file
from .fmp_fallback import fetch_fmp_for_ticker
from .normalizer import normalize_events
from .scorer import score_events

log = logging.getLogger("market_intel.adapter")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _try_edgar(ticker: str, limit_forms: int) -> List[Dict[str, Any]]:
    """Fetch + parse EDGAR Form-4. Returns list of raw event dicts (or [])."""
    bundle = fetch_edgar_for_ticker(ticker, limit_forms=limit_forms, include_13f=False)
    events: List[Dict[str, Any]] = []
    for filing in bundle.get("form4", []):
        path = filing.get("local_path")
        if not path:
            continue
        try:
            parsed = parse_form4_file(path)
        except Exception as exc:
            log.warning("parse_fail ticker=%s acc=%s err=%s", ticker, filing.get("accession"), exc)
            continue
        # Backfill accession if parser didn't extract it
        for ev in parsed:
            ev["filing_accession"] = ev.get("filing_accession") or filing.get("accession")
            ev["is_amendment"] = ev.get("is_amendment") or filing.get("is_amendment", False)
        events.extend(parsed)
    return events


def fetch_intel(
    ticker: str,
    *,
    limit_forms: int = config.DEFAULT_LIMIT_FORMS,
    use_fmp_fallback: bool = True,
    edgar_first: Optional[bool] = None,
) -> Dict[str, Any]:
    """Per-ticker intel: EDGAR primary, FMP fallback.

    Always returns the canonical schema:
        {
            "ticker":        str,
            "source":        "EDGAR" | "FMP" | "NONE",
            "presence":      bool,
            "is_authoritative": bool,        # True iff source == "EDGAR"
            "activity_count": int,
            "events":        [normalized InsiderEvent, ...],
            "score":         float in [0, 1],
            "score_breakdown": {...},
            "last_updated":  ISO8601,
            "errors":        [str, ...],
        }
    """
    ticker = str(ticker).upper().strip()
    errors: List[str] = []

    if edgar_first is None:
        edgar_first = config.EDGAR_FIRST

    raw_edgar: List[Dict[str, Any]] = []
    if edgar_first:
        try:
            raw_edgar = _try_edgar(ticker, limit_forms)
        except Exception as exc:
            errors.append(f"edgar_exception:{exc}")
            log.warning("edgar_exception ticker=%s err=%s", ticker, str(exc)[:200])
    else:
        errors.append("edgar_skipped:EDGAR_FIRST=false")

    if raw_edgar:
        events = normalize_events(raw_edgar, source="EDGAR")
        breakdown = score_events(events)
        return {
            "ticker":            ticker,
            "source":            "EDGAR",
            "presence":          True,
            "is_authoritative":  True,
            "activity_count":    len(events),
            "events":            events,
            "score":             breakdown["score"],
            "score_breakdown":   breakdown,
            "last_updated":      _utcnow(),
            "errors":            errors,
        }

    # ── FMP fallback ──────────────────────────────────────────────────────────
    if use_fmp_fallback and config.FMP_API_KEY:
        try:
            raw_fmp = fetch_fmp_for_ticker(ticker)
        except Exception as exc:
            errors.append(f"fmp_exception:{exc}")
            raw_fmp = []
        if raw_fmp:
            events = normalize_events(raw_fmp, source="FMP")
            breakdown = score_events(events)
            return {
                "ticker":            ticker,
                "source":            "FMP",
                "presence":          True,
                "is_authoritative":  False,
                "activity_count":    len(events),
                "events":            events,
                "score":             breakdown["score"],
                "score_breakdown":   breakdown,
                "last_updated":      _utcnow(),
                "errors":            errors,
            }

    # No data anywhere
    return {
        "ticker":            ticker,
        "source":            "NONE",
        "presence":          False,
        "is_authoritative":  False,
        "activity_count":    0,
        "events":            [],
        "score":             0.50,
        "score_breakdown":   score_events([]),
        "last_updated":      _utcnow(),
        "errors":            errors,
    }


def fetch_intel_universe(
    tickers: Iterable[str],
    *,
    max_workers: int = 4,
    limit_forms: int = config.DEFAULT_LIMIT_FORMS,
    progress_cb: Optional[Any] = None,
) -> List[Dict[str, Any]]:
    """Run fetch_intel across many tickers in parallel.

    SEC rate limit (10 req/s) is enforced by the global limiter inside
    edgar_ingest, so workers automatically queue when needed.
    """
    out: List[Dict[str, Any]] = []
    tickers = list(tickers)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(fetch_intel, t, limit_forms=limit_forms): t for t in tickers}
        for fut in as_completed(futs):
            try:
                out.append(fut.result())
            except Exception as exc:
                t = futs[fut]
                log.warning("universe_fail ticker=%s err=%s", t, str(exc)[:200])
                out.append({
                    "ticker": t, "source": "NONE", "presence": False,
                    "is_authoritative": False, "activity_count": 0,
                    "events": [], "score": 0.50, "errors": [f"exception:{exc}"],
                    "last_updated": _utcnow(), "score_breakdown": score_events([]),
                })
            if progress_cb:
                try:
                    progress_cb(len(out), len(tickers))
                except Exception:
                    pass
    out.sort(key=lambda r: r["ticker"])
    return out


# ── Persistence helpers (used by scheduler / streamlit integration) ───────────

def write_summary_files(results: List[Dict[str, Any]], log_dir: Path) -> Dict[str, Path]:
    """Write form4_summary.csv, edgar_debug_summary.json, marketintel_events.json."""
    log_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "form4_csv":   log_dir / "form4_summary.csv",
        "debug_json":  log_dir / "edgar_debug_summary.json",
        "events_json": log_dir / "marketintel_events.json",
    }

    # form4_summary.csv — one row per insider event
    import csv
    with paths["form4_csv"].open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([
            "ticker", "source", "transaction_date", "reporting_person",
            "reporting_role", "transaction_code", "shares", "price",
            "value", "acquired_disposed", "filing_accession", "is_amendment",
        ])
        for r in results:
            for ev in r.get("events", []):
                w.writerow([
                    r["ticker"], r["source"],
                    ev.get("transaction_date"), ev.get("reporting_person"),
                    ev.get("reporting_role"), ev.get("transaction_code"),
                    ev.get("shares"), ev.get("price"), ev.get("value"),
                    ev.get("acquired_disposed"), ev.get("filing_accession"),
                    ev.get("is_amendment"),
                ])

    # edgar_debug_summary.json — coverage + presence per ticker
    debug = {
        "generated_at": _utcnow(),
        "ticker_count": len(results),
        "edgar_present_count": sum(1 for r in results if r["source"] == "EDGAR"),
        "fmp_fallback_count":  sum(1 for r in results if r["source"] == "FMP"),
        "missing_count":       sum(1 for r in results if r["source"] == "NONE"),
        "per_ticker": [{
            "ticker":         r["ticker"],
            "source":         r["source"],
            "presence":       r["presence"],
            "activity_count": r["activity_count"],
            "score":          r["score"],
            "errors":         r.get("errors", []),
        } for r in results],
    }
    paths["debug_json"].write_text(json.dumps(debug, indent=2), encoding="utf-8")

    # marketintel_events.json — full normalized events for downstream consumers
    paths["events_json"].write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return paths
