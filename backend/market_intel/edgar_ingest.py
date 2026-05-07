"""backend/market_intel/edgar_ingest.py — SEC EDGAR HTTP layer.

Akerlof (2001 Nobel) "The Market for Lemons" — insider filings exist precisely
because regulators mandate disclosure to neutralize information asymmetry. This
ingest layer pulls the canonical (authoritative) source: Form-4 and 13F-HR XML
filings directly from SEC EDGAR, bypassing third-party aggregators.

Public functions:
    fetch_edgar_for_ticker(ticker, limit_forms=5) -> dict
    load_ticker_cik_map() -> dict[str, str]
    get_cik(ticker) -> str | None
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from utils.atomic_write import atomic_write_json

from . import config

# ── Logger ────────────────────────────────────────────────────────────────────

log = logging.getLogger("market_intel.edgar")
if not log.handlers:
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    _h = logging.FileHandler(config.LOG_DIR / "market_intel.log", encoding="utf-8")
    _h.setFormatter(logging.Formatter('{"ts":"%(asctime)s","lvl":"%(levelname)s","msg":%(message)s}'))
    log.addHandler(_h)
    log.setLevel(logging.INFO)


def _jlog(level: str, **fields: Any) -> None:
    """Emit a JSON-serialised structured log line."""
    try:
        body = json.dumps(fields, default=str)
    except Exception:
        body = json.dumps({"error": "log_serialize_failed"})
    getattr(log, level, log.info)(body)


# ── Circuit breaker (Engle: regime transition under repeated failures) ───────
#
# State machine: closed → open → closed (after cooldown elapses).
# On every final HTTP failure, fail_count increments. Once it reaches
# EDGAR_CB_FAIL_THRESHOLD (default 5), the breaker opens and EDGAR calls are
# short-circuited for EDGAR_CB_COOLDOWN_MIN minutes (default 15). After the
# cooldown, the next probe call resets the counter — analogous to an
# ARCH-style regime switch driven by a small recent-failure window.

_CB_PATH = Path(".monitoring/edgar_cb.json")


def _cb_default() -> Dict[str, Any]:
    return {"state": "closed", "fail_count": 0, "last_failure_ts": 0.0, "opened_at": 0.0}


def _cb_load() -> Dict[str, Any]:
    """Engle: load CB state from disk; default to closed on any read error."""
    if not _CB_PATH.exists():
        return _cb_default()
    try:
        return json.loads(_CB_PATH.read_text(encoding="utf-8"))
    except Exception:
        return _cb_default()


def _cb_save(d: Dict[str, Any]) -> None:
    """Engle: persist CB state atomically so concurrent CI jobs see a coherent file."""
    atomic_write_json(_CB_PATH, d)


def _cb_allows_calls() -> bool:
    """Engle: True if EDGAR calls are allowed; auto-resets state once cooldown elapses."""
    d = _cb_load()
    if d.get("state") != "open":
        return True
    cooldown_min = float(os.getenv("EDGAR_CB_COOLDOWN_MIN", "15"))
    if time.time() - float(d.get("opened_at", 0.0)) >= cooldown_min * 60.0:
        _cb_save(_cb_default())
        return True
    return False


def _cb_record_failure() -> None:
    """Engle: count one final HTTP failure; trip to 'open' once threshold is crossed."""
    d = _cb_load()
    fc = int(d.get("fail_count", 0)) + 1
    threshold = int(os.getenv("EDGAR_CB_FAIL_THRESHOLD", "5"))
    now = time.time()
    if fc >= threshold:
        _cb_save({"state": "open", "fail_count": fc, "last_failure_ts": now, "opened_at": now})
    else:
        _cb_save({"state": "closed", "fail_count": fc, "last_failure_ts": now, "opened_at": float(d.get("opened_at", 0.0))})


def cb_state() -> Dict[str, Any]:
    """Engle: public read of CB state for run_pipeline / metrics annotation."""
    return _cb_load()


# ── Rate limiter (simple spacing-based) ───────────────────────────────────────

_RATE_LOCK = threading.Lock()
_LAST_REQ_AT: float = 0.0


def _rate_wait() -> None:
    """Block until enough time has passed since the previous SEC request."""
    global _LAST_REQ_AT
    with _RATE_LOCK:
        now = time.monotonic()
        delta = now - _LAST_REQ_AT
        if delta < config.MIN_SPACING_S:
            time.sleep(config.MIN_SPACING_S - delta)
        _LAST_REQ_AT = time.monotonic()


# ── HTTP with retry/backoff ───────────────────────────────────────────────────

def _http_get(url: str, **kwargs: Any) -> requests.Response:
    """GET with exponential backoff. Raises on final failure."""
    headers = {"User-Agent": config.USER_AGENT, "Accept-Encoding": "gzip"}
    headers.update(kwargs.pop("headers", {}) or {})
    last_exc: Optional[Exception] = None
    for attempt in range(config.HTTP_RETRIES):
        try:
            _rate_wait()
            resp = requests.get(url, headers=headers, timeout=config.HTTP_TIMEOUT_S, **kwargs)
            if resp.status_code == 200:
                return resp
            if resp.status_code in (429, 503):
                wait_s = config.HTTP_BACKOFF_BASE_S * (2 ** attempt)
                _jlog("warning", event="rate_limit", url=url, status=resp.status_code, wait_s=wait_s)
                time.sleep(wait_s)
                continue
            resp.raise_for_status()
        except Exception as exc:
            last_exc = exc
            wait_s = config.HTTP_BACKOFF_BASE_S * (2 ** attempt)
            _jlog("warning", event="http_retry", url=url, attempt=attempt, err=str(exc)[:200])
            time.sleep(wait_s)
    _cb_record_failure()
    raise RuntimeError(f"GET failed after {config.HTTP_RETRIES} attempts: {url} ({last_exc})")


# ── Ticker → CIK mapping ──────────────────────────────────────────────────────

def load_ticker_cik_map() -> Dict[str, str]:
    """Load ticker→zero-padded-CIK map from SEC, cached on disk for TICKER_MAP_TTL_DAYS."""
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = config.CACHE_DIR / "company_tickers.json"

    fresh = (
        cache_path.exists()
        and (time.time() - cache_path.stat().st_mtime) < config.TICKER_MAP_TTL_DAYS * 86400
    )
    if fresh:
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            data = None
    else:
        data = None

    if data is None:
        url = f"{config.SEC_BASE}/files/company_tickers.json"
        resp = _http_get(url)
        data = resp.json()
        cache_path.write_text(json.dumps(data), encoding="utf-8")
        _jlog("info", event="ticker_map_refreshed", path=str(cache_path))

    out: Dict[str, str] = {}
    iter_rows = data.values() if isinstance(data, dict) else data
    for row in iter_rows:
        try:
            tkr = str(row.get("ticker", "")).upper().strip()
            cik = int(row.get("cik_str", 0))
            if tkr and cik:
                out[tkr] = f"{cik:010d}"
        except Exception:
            continue
    return out


def get_cik(ticker: str) -> Optional[str]:
    """Return zero-padded 10-digit CIK for ticker, or None if not found."""
    return load_ticker_cik_map().get(str(ticker).upper().strip())


# ── Submissions (filing index) ────────────────────────────────────────────────

def fetch_submissions(cik: str) -> Dict[str, Any]:
    """GET https://data.sec.gov/submissions/CIK{CIK}.json."""
    url = f"{config.SEC_DATA_BASE}/submissions/CIK{cik}.json"
    resp = _http_get(url)
    return resp.json()


def filter_recent_filings(
    submissions: Dict[str, Any],
    form_types: Tuple[str, ...],
    limit: int,
) -> List[Dict[str, str]]:
    """Extract recent filings matching form_types; returns up to `limit` rows.

    Each row: {form, accession, filing_date, primary_document, primary_doc_url}
    """
    recent = (submissions.get("filings") or {}).get("recent") or {}
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    dates = recent.get("filingDate", [])
    docs = recent.get("primaryDocument", [])
    cik = str(submissions.get("cik", "")).lstrip("0") or "0"

    rows: List[Dict[str, str]] = []
    for f, a, d, doc in zip(forms, accessions, dates, docs):
        if f not in form_types:
            continue
        a_clean = a.replace("-", "")
        rows.append({
            "form":             f,
            "accession":        a,
            "accession_clean":  a_clean,
            "filing_date":      d,
            "primary_document": doc,
            "primary_doc_url":  f"{config.SEC_BASE}/Archives/edgar/data/{cik}/{a_clean}/{doc}",
            "filing_index_url": f"{config.SEC_BASE}/Archives/edgar/data/{cik}/{a_clean}/",
        })
        if len(rows) >= limit:
            break
    return rows


# ── Filing download (idempotent on disk) ──────────────────────────────────────

_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_filename(name: str) -> str:
    return _SAFE_NAME_RE.sub("_", name)


def fetch_filing_doc(ticker: str, cik: str, filing: Dict[str, str]) -> Path:
    """Download the primary filing document if not already cached. Returns local path.

    Idempotent: skips download if file exists on disk under
    DATA_DIR/{ticker}/{accession_clean}/{primary_document}.
    """
    out_dir = config.DATA_DIR / ticker / filing["accession_clean"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / _safe_filename(filing["primary_document"])

    if out_path.exists() and out_path.stat().st_size > 0:
        _jlog("info", event="filing_cached", ticker=ticker, accession=filing["accession"], path=str(out_path))
        return out_path

    resp = _http_get(filing["primary_doc_url"])
    out_path.write_bytes(resp.content)
    _jlog("info", event="filing_downloaded",
          ticker=ticker, accession=filing["accession"], bytes=len(resp.content))
    return out_path


def find_form4_xml(filing_index_url: str) -> Optional[str]:
    """Locate the real ownership XML in a Form-4 filing directory.

    Important: SEC's `primaryDocument` for Form-4 returns the XSL-rendered HTML
    (named `xsl<style>_form4.xml`), NOT the raw ownership XML. The real XML is
    typically `form4.xml`, `primary_doc.xml`, or `wf-form4_<seq>.xml` and lives
    alongside the rendered file.

    Returns the absolute URL of the highest-priority candidate, or None.
    """
    idx_url = filing_index_url.rstrip("/") + "/index.json"
    try:
        resp = _http_get(idx_url)
        idx = resp.json()
    except Exception:
        return None

    candidates: List[Tuple[int, str]] = []
    for item in (idx.get("directory") or {}).get("item", []):
        name = str(item.get("name", ""))
        lower = name.lower()
        if not lower.endswith(".xml"):
            continue
        if lower.startswith("xsl"):           # XSL-rendered HTML wrapped in .xml
            continue
        if "metadata" in lower:
            continue
        # Heuristic priority — higher wins.
        if "form4" in lower or "form3" in lower or "form5" in lower:
            prio = 3
        elif "primary_doc" in lower or "ownership" in lower:
            prio = 2
        else:
            prio = 1
        candidates.append((prio, name))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return filing_index_url.rstrip("/") + "/" + candidates[0][1]


# ── Public entry point ────────────────────────────────────────────────────────

def fetch_edgar_for_ticker(
    ticker: str,
    limit_forms: int = config.DEFAULT_LIMIT_FORMS,
    include_13f: bool = True,
) -> Dict[str, Any]:
    """Fetch and persist recent Form-4 (and optionally 13F-HR) filings.

    Args:
        ticker:      Equity ticker (e.g., "AAPL").
        limit_forms: Per-form-type cap on filings fetched.
        include_13f: Also fetch institutional 13F-HR filings.

    Returns:
        {
            "ticker":   str,
            "cik":      str | None,
            "fetched_at": ISO8601,
            "form4":  [{accession, filing_date, local_path, ...}, ...],
            "form13f": [...],
            "errors": [str, ...],
        }

    The function is idempotent — already-downloaded accessions are not re-fetched.
    """
    ticker = str(ticker).upper().strip()
    out: Dict[str, Any] = {
        "ticker":     ticker,
        "cik":        None,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "form4":      [],
        "form13f":    [],
        "errors":     [],
    }

    if not _cb_allows_calls():
        out["errors"].append("edgar_cb_open")
        out["edgar_cb_open"] = True
        _jlog("warning", event="edgar_cb_open", ticker=ticker)
        return out

    cik = get_cik(ticker)
    if not cik:
        out["errors"].append(f"no_cik_for_{ticker}")
        _jlog("warning", event="no_cik", ticker=ticker)
        return out
    out["cik"] = cik

    try:
        subs = fetch_submissions(cik)
    except Exception as exc:
        out["errors"].append(f"submissions_fail:{exc}")
        _jlog("error", event="submissions_fail", ticker=ticker, err=str(exc)[:200])
        return out

    form_groups: List[Tuple[str, Tuple[str, ...]]] = [("form4", config.FORM_TYPES_INSIDER)]
    if include_13f:
        form_groups.append(("form13f", config.FORM_TYPES_INSTITUTIONAL))

    for key, form_types in form_groups:
        try:
            filings = filter_recent_filings(subs, form_types, limit_forms)
        except Exception as exc:
            out["errors"].append(f"filter_{key}_fail:{exc}")
            continue

        for filing in filings:
            try:
                primary = filing["primary_doc_url"]
                primary_lower = primary.lower()
                primary_name = primary_lower.rsplit("/", 1)[-1]
                # SEC primaryDocument for Form-4 is the XSL-rendered HTML, not raw XML.
                # The "xsl" marker can be a filename prefix (xslF345X06_form4.xml) or
                # a path segment (xslF345X06/form4.xml) — check every segment.
                segments = primary_lower.split("/")
                needs_lookup = (
                    not primary_name.endswith(".xml")
                    or any(seg.startswith("xsl") for seg in segments)
                )
                if needs_lookup:
                    xml_url = find_form4_xml(filing["filing_index_url"])
                    if xml_url:
                        filing = {**filing, "primary_doc_url": xml_url, "primary_document": xml_url.rsplit("/", 1)[-1]}
                local = fetch_filing_doc(ticker, cik, filing)
                out[key].append({
                    "accession":  filing["accession"],
                    "form":       filing["form"],
                    "filing_date": filing["filing_date"],
                    "local_path": str(local),
                    "url":        filing["primary_doc_url"],
                    "is_amendment": filing["form"].endswith("/A"),
                })
            except Exception as exc:
                out["errors"].append(f"download_{filing.get('accession')}:{exc}")
                _jlog("error", event="download_fail",
                      ticker=ticker, accession=filing.get("accession"), err=str(exc)[:200])

    _jlog("info", event="ingest_complete", ticker=ticker,
          form4=len(out["form4"]), form13f=len(out["form13f"]), errors=len(out["errors"]))
    return out
