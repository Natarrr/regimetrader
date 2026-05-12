"""regime_trader/services/edgar_index.py
SEC EDGAR daily-index bulk downloader.

Stiglitz (2001 Nobel) — asymmetric information: EDGAR daily index files expose
*every* filing submitted that day, enabling broad market surveillance that
quarterly index files cannot provide.

Design:
  - Downloads from https://www.sec.gov/Archives/edgar/daily-index/
  - Parses the fixed-width company.idx for each requested date.
  - Atomic cache under .cache/edgar/index/YYYY-MM-DD.idx  (TTL 24 h).
  - Rate-limited via regime_trader.utils.token_bucket.TokenBucket.
  - All network errors are logged; callers receive [] on failure.

Public API:
    list_filings(date)  → List[DailyFilingRef]
    list_filings_range(start, end)  → List[DailyFilingRef]  (multi-day)

Usage:
    from regime_trader.services.edgar_index import EdgarDailyIndex
    idx = EdgarDailyIndex()
    filings = idx.list_filings("2026-01-15")  # date as YYYY-MM-DD str or date obj
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import tempfile
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterator, List, Optional, TypedDict

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from regime_trader.utils.token_bucket import TokenBucket

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_BASE        = "https://www.sec.gov/Archives/edgar/daily-index"
_CACHE_ROOT  = Path(__file__).parent.parent.parent / ".cache" / "edgar" / "index"
_TTL         = 24 * 3600   # 24 h — same day's index is mutable until midnight
_MAX_RETRIES = 3
_BACKOFF     = 1.5
_DEFAULT_RATE = float(os.getenv("EDGAR_RATE_LIMIT", "0.2"))   # req/sec

_HEADERS = {
    "User-Agent": os.getenv(
        "EDGAR_USER_AGENT",
        "regime-trader-research infra-team@example.com",
    ),
    "Accept-Encoding": "gzip, deflate",
}

# company.idx column offsets (fixed-width, per SEC documentation)
# Company Name   Form Type   CIK         Date Filed  Filename
# 0              62          74          86          98
_COL_OFFSETS = (0, 62, 74, 86, 98)


# ── Typed output schema ────────────────────────────────────────────────────────

class DailyFilingRef(TypedDict):
    """A single row from the EDGAR daily company.idx."""
    date:         str    # YYYY-MM-DD
    cik:          str    # zero-padded 10-digit CIK
    company:      str
    form:         str
    filename:     str    # relative path on SEC servers
    url:          str    # absolute URL to the index page
    evidence_id:  str    # deterministic hash for caching / tracing


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _cache_path(date_str: str) -> Path:
    return _CACHE_ROOT / f"{date_str}.json"


def _cache_read(date_str: str) -> Optional[List[DailyFilingRef]]:
    """Return cached filing list or None if missing / expired."""
    p = _cache_path(date_str)
    if not p.exists():
        return None
    try:
        meta = json.loads(p.read_text(encoding="utf-8"))
        if time.time() - meta.get("_ts", 0) > _TTL:
            return None
        return meta.get("rows", [])
    except Exception:
        return None


def _cache_write(date_str: str, rows: List[DailyFilingRef]) -> None:
    """Atomically persist the filing list to the cache."""
    p = _cache_path(date_str)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {"_ts": time.time(), "rows": rows},
        indent=2,
        ensure_ascii=False,
    ).encode("utf-8")
    fd, tmp = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=str(p.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        os.replace(tmp, p)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ── Evidence ID ────────────────────────────────────────────────────────────────

def _evidence_id(date_str: str, filename: str) -> str:
    """Stable SHA-256-based ID for a single filing, used for downstream caching."""
    raw = f"{date_str}:{filename}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


# ── Parser ─────────────────────────────────────────────────────────────────────

def _parse_company_idx(text: str, date_str: str) -> List[DailyFilingRef]:
    """Parse SEC fixed-width company.idx into DailyFilingRef rows.

    Stiglitz (2001): only filings we can parse add information; silently skip
    unparseable rows rather than crashing the entire pipeline.

    Column layout:
        [0:62]   Company Name
        [62:74]  Form Type
        [74:86]  CIK
        [86:98]  Date Filed
        [98:]    Filename
    """
    results: List[DailyFilingRef] = []
    lines = text.splitlines()

    # Skip header lines (2 lines: header + separator)
    body_start = 0
    for i, line in enumerate(lines):
        if set(line.strip()) == {"-"}:
            body_start = i + 1
            break

    for line in lines[body_start:]:
        if len(line) < 100:
            continue
        try:
            company  = line[_COL_OFFSETS[0]:_COL_OFFSETS[1]].strip()
            form     = line[_COL_OFFSETS[1]:_COL_OFFSETS[2]].strip()
            cik      = line[_COL_OFFSETS[2]:_COL_OFFSETS[3]].strip().zfill(10)
            filed    = line[_COL_OFFSETS[3]:_COL_OFFSETS[4]].strip()
            filename = line[_COL_OFFSETS[4]:].strip()
        except Exception:
            continue

        if not filename:
            continue

        url = f"https://www.sec.gov/Archives/{filename}"
        eid = _evidence_id(date_str, filename)
        results.append(
            DailyFilingRef(
                date=date_str,
                cik=cik,
                company=company,
                form=form,
                filename=filename,
                url=url,
                evidence_id=eid,
            )
        )

    return results


# ── HTTP session ───────────────────────────────────────────────────────────────

def _build_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=_MAX_RETRIES,
        backoff_factor=_BACKOFF,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update(_HEADERS)
    return s


# ── Main class ─────────────────────────────────────────────────────────────────

class EdgarDailyIndex:
    """Stiglitz (2001 Nobel) — EDGAR daily bulk-index downloader.

    Downloads, parses, and caches the SEC daily-index company.idx files.
    Rate-limited to avoid SEC IP blocks; results cached 24 h.

    Args:
        rate_per_sec:  Max HTTP requests per second (default 0.2).
        cache_root:    Override cache directory (useful for tests).
        session:       Override requests.Session (for mocking in tests).
    """

    def __init__(
        self,
        rate_per_sec: float = _DEFAULT_RATE,
        cache_root:   Optional[Path] = None,
        session:      Optional[requests.Session] = None,
    ) -> None:
        self._bucket     = TokenBucket(rate_per_sec=rate_per_sec, capacity=2.0)
        self._cache_root = Path(cache_root) if cache_root else _CACHE_ROOT
        self._session    = session or _build_session()

    # ── Public API ─────────────────────────────────────────────────────────────

    def list_filings(
        self,
        target_date: "str | date",
        form_types:  Optional[List[str]] = None,
    ) -> List[DailyFilingRef]:
        """Stiglitz (2001 Nobel) — Return all filings for a single trading date.

        Downloads (or reads from cache) the SEC daily company.idx for the given
        date and returns its parsed rows. Weekends return [] immediately.

        Args:
            target_date:  Date as "YYYY-MM-DD" string or datetime.date object.
            form_types:   Optional allowlist of form types (e.g. ["4", "13F-HR"]).
                          If None, all form types are returned.

        Returns:
            List of DailyFilingRef.  Empty on weekends, SEC downtime, or error.
        """
        date_str = _to_date_str(target_date)
        d = _parse_date(date_str)
        if d.weekday() >= 5:           # Saturday = 5, Sunday = 6
            return []

        cached = self._cache_read(date_str)
        if cached is not None:
            return _filter_forms(cached, form_types)

        rows = self._fetch(date_str)
        self._cache_write(date_str, rows)
        return _filter_forms(rows, form_types)

    def list_filings_range(
        self,
        start: "str | date",
        end:   "str | date",
        form_types: Optional[List[str]] = None,
    ) -> List[DailyFilingRef]:
        """Return all filings for every trading day in [start, end].

        Args:
            start, end:  Inclusive date range.
            form_types:  Optional form-type filter.

        Returns:
            Concatenated list from all days in range.
        """
        s = _parse_date(_to_date_str(start))
        e = _parse_date(_to_date_str(end))
        result: List[DailyFilingRef] = []
        cur = s
        while cur <= e:
            result.extend(self.list_filings(cur, form_types=form_types))
            cur += timedelta(days=1)
        return result

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _cache_read(self, date_str: str) -> Optional[List[DailyFilingRef]]:
        p = self._cache_root / f"{date_str}.json"
        if not p.exists():
            return None
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
            if time.time() - meta.get("_ts", 0) > _TTL:
                return None
            return meta.get("rows", [])
        except Exception:
            return None

    def _cache_write(self, date_str: str, rows: List[DailyFilingRef]) -> None:
        p = self._cache_root / f"{date_str}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"_ts": time.time(), "rows": rows},
            indent=2,
            ensure_ascii=False,
        ).encode("utf-8")
        fd, tmp = tempfile.mkstemp(
            prefix=f".{p.name}.", suffix=".tmp", dir=str(p.parent)
        )
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(payload)
                fh.flush()
                try:
                    os.fsync(fh.fileno())
                except OSError:
                    pass
            os.replace(tmp, p)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _fetch(self, date_str: str) -> List[DailyFilingRef]:
        """Download and parse the daily company.idx for date_str."""
        d   = _parse_date(date_str)
        url = f"{_BASE}/{d.year}/QTR{_quarter(d)}/company.idx"
        self._bucket.acquire()

        try:
            resp = self._session.get(url, timeout=30)
            resp.raise_for_status()
        except Exception as exc:
            log.warning("edgar_index: failed to fetch %s — %s", url, exc)
            return []

        rows = _parse_company_idx(resp.text, date_str)
        log.info("edgar_index: %s → %d filings (URL: %s)", date_str, len(rows), url)
        return rows


# ── Utility functions ──────────────────────────────────────────────────────────

def _to_date_str(d: "str | date") -> str:
    if isinstance(d, str):
        return d
    return d.strftime("%Y-%m-%d")


def _parse_date(date_str: str) -> date:
    return datetime.strptime(date_str, "%Y-%m-%d").date()


def _quarter(d: date) -> int:
    return (d.month - 1) // 3 + 1


def _filter_forms(
    rows: List[DailyFilingRef],
    form_types: Optional[List[str]],
) -> List[DailyFilingRef]:
    if not form_types:
        return rows
    allowed = {f.upper() for f in form_types}
    return [r for r in rows if r["form"].upper() in allowed]
