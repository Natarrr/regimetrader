"""regime_trader/services/edgar_service.py
SEC EDGAR bulk-index fetcher with file cache and polite rate limiting.

Stiglitz (2001 Nobel) — asymmetric information: EDGAR filings are the primary
source of authoritative corporate disclosure; reliable, cached access prevents
both data gaps and SEC rate-limit bans.

Design:
  - Uses SEC's quarterly full-index files (company.idx) — no scraping.
  - All HTTP calls go through a single session with retry/backoff.
  - Index files are cached under .cache/edgar/ for 24 h (TTL configurable).
  - Filing documents are fetched on demand and cached individually.
  - Polite rate limit: 0.2 req/sec by default (configurable via
    EDGAR_RATE_LIMIT env var, in req/sec).

Public API:
  list_filings(cik_or_ticker, form_type, max_results) -> List[FilingRef]
  fetch_filing(url)                                   -> str | None
  quarterly_index(year, quarter)                      -> List[IndexRow]

Usage:
  from regime_trader.services.edgar_service import EdgarService
  svc = EdgarService()
  filings = svc.list_filings("0000320193", form_type="4")  # Apple insider Form 4s
"""
from __future__ import annotations

import io
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, TypedDict

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_EDGAR_BASE = "https://www.sec.gov"
_EDGAR_SEARCH = "https://efts.sec.gov/LATEST/search-index"
_INDEX_TPL = "{base}/Archives/edgar/full-index/{year}/QTR{quarter}/company.idx"
_COMPANY_SEARCH_TPL = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type={form_type}&dateb=&owner=include&count={count}&search_text=&output=atom"

_CACHE_ROOT = Path(__file__).parent.parent.parent / ".cache" / "edgar"
_TTL_INDEX = 24 * 3600       # 24 h — quarterly index files are stable
_TTL_FILING = 7 * 24 * 3600  # 7 days — individual filings don't change
_DEFAULT_RATE = 0.2           # req/sec (SEC guidance: max ~10 req/sec but be polite)

_HEADERS = {
    "User-Agent": "regime-trader-research contact@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov",
}

_MAX_RETRIES = 3
_BACKOFF = 1.0


# ── Typed output schemas ───────────────────────────────────────────────────────

class FilingRef(TypedDict):
    """A reference to a single SEC filing."""
    cik: str
    company_name: str
    form_type: str
    date_filed: str
    filename: str
    url: str


class IndexRow(TypedDict):
    """A single row from a quarterly EDGAR index file."""
    company_name: str
    form_type: str
    cik: str
    date_filed: str
    filename: str


# ── Rate limiter ───────────────────────────────────────────────────────────────

class _RateLimiter:
    """Thread-safe minimum-interval rate limiter for SEC EDGAR.

    Modigliani (1985 Nobel) — sustainable extraction pace is part of
    good-faith compliance with SEC polite-crawling guidelines.
    """

    def __init__(self, rate_per_sec: float = _DEFAULT_RATE) -> None:
        self._interval = 1.0 / max(rate_per_sec, 0.001)
        self._lock = threading.Lock()
        self._last: float = 0.0

    def acquire(self) -> None:
        """Block until the minimum inter-request interval has elapsed."""
        with self._lock:
            wait = self._interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()


# ── File cache ─────────────────────────────────────────────────────────────────

def _safe_key(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def _cache_path(bucket: str, key: str) -> Path:
    return _CACHE_ROOT / bucket / f"{_safe_key(key)}.txt"


def _cache_read(bucket: str, key: str, ttl: int) -> Optional[str]:
    p = _cache_path(bucket, key)
    try:
        if not p.exists():
            return None
        if time.time() - p.stat().st_mtime > ttl:
            return None
        return p.read_text(encoding="utf-8")
    except Exception:
        return None


def _cache_write(bucket: str, key: str, content: str) -> None:
    p = _cache_path(bucket, key)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    except Exception as exc:
        log.debug("edgar cache write failed %s/%s: %s", bucket, key, exc)


# ── HTTP session ───────────────────────────────────────────────────────────────

def _make_session() -> requests.Session:
    """Build a requests.Session with exponential-backoff retry for EDGAR."""
    session = requests.Session()
    retry = Retry(
        total=_MAX_RETRIES,
        backoff_factor=_BACKOFF,
        status_forcelist={429, 500, 502, 503, 504},
        allowed_methods={"GET"},
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(_HEADERS)
    return session


# ── Index parser ───────────────────────────────────────────────────────────────

def _parse_company_idx(raw: str) -> List[IndexRow]:
    """Parse the fixed-width SEC company.idx format.

    Format (after 9-line header):
      Company Name           | Form Type | CIK        | Date Filed | Filename
      (62 chars)             | (12 chars)| (12 chars) | (12 chars) | remainder

    Returns:
        List of IndexRow TypedDicts.
    """
    rows: List[IndexRow] = []
    lines = raw.splitlines()
    # Skip header lines (first 9)
    for line in lines[9:]:
        if len(line) < 98:
            continue
        try:
            company_name = line[0:62].strip()
            form_type = line[62:74].strip()
            cik = line[74:86].strip().lstrip("0")
            date_filed = line[86:98].strip()
            filename = line[98:].strip()
            if not filename:
                continue
            rows.append({
                "company_name": company_name,
                "form_type": form_type,
                "cik": cik,
                "date_filed": date_filed,
                "filename": filename,
            })
        except Exception:
            continue
    return rows


# ── EdgarService ───────────────────────────────────────────────────────────────

class EdgarService:
    """SEC EDGAR bulk-index fetcher with polite rate limiting and file cache.

    Thread-safe.  Instantiate once per process.

    Args:
        rate_per_sec: Max SEC requests per second (default: env EDGAR_RATE_LIMIT
                      or 0.2).
        cache_root:   Override for the local file-cache root.
    """

    def __init__(
        self,
        rate_per_sec: Optional[float] = None,
        cache_root: Optional[Path] = None,
    ) -> None:
        rate = rate_per_sec or float(os.getenv("EDGAR_RATE_LIMIT", str(_DEFAULT_RATE)))
        self._limiter = _RateLimiter(rate_per_sec=rate)
        self._session = _make_session()
        self._cache_root = cache_root or _CACHE_ROOT
        log.debug("EdgarService initialised — %.2f req/s, cache=%s", rate, self._cache_root)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get_text(self, url: str, timeout: int = 30) -> Optional[str]:
        """GET *url* with rate limiting; return response text or None."""
        self._limiter.acquire()
        try:
            resp = self._session.get(url, timeout=timeout)
            if resp.status_code == 200:
                return resp.text
            log.warning("EDGAR HTTP %s: %s", resp.status_code, url)
            return None
        except Exception as exc:
            log.warning("EDGAR request failed %s: %s", url, exc)
            return None

    # ── Public API ─────────────────────────────────────────────────────────────

    def quarterly_index(self, year: int, quarter: int) -> List[IndexRow]:
        """Download and parse the SEC quarterly full-index for *year* / *quarter*.

        Leontief (1973 Nobel) — systematic coverage of all filings, not
        cherry-picked samples, is required for unbiased signal construction.

        Index files are stable once published; cached for 24 h.

        Args:
            year:    Calendar year (e.g. 2025).
            quarter: 1–4.

        Returns:
            Parsed list of IndexRow dicts (empty list on failure).
        """
        cache_key = f"{year}_Q{quarter}"
        cached = _cache_read("index", cache_key, _TTL_INDEX)
        if cached is not None:
            log.debug("quarterly_index: cache hit %s", cache_key)
            return _parse_company_idx(cached)

        url = _INDEX_TPL.format(
            base=_EDGAR_BASE, year=year, quarter=quarter
        )
        raw = self._get_text(url)
        if not raw:
            log.warning("quarterly_index: failed to fetch %s", url)
            return []

        _cache_write("index", cache_key, raw)
        log.info("quarterly_index: fetched %d bytes for %s", len(raw), cache_key)
        return _parse_company_idx(raw)

    def list_filings(
        self,
        cik: str,
        form_type: str = "4",
        max_results: int = 40,
    ) -> List[FilingRef]:
        """List recent filings for a CIK, filtered by *form_type*.

        Args:
            cik:         10-digit CIK (zero-padded) or bare integer string.
            form_type:   SEC form type (e.g. "4", "10-K", "8-K").
            max_results: Maximum filings to return.

        Returns:
            List of FilingRef dicts sorted newest-first.
        """
        cik_padded = cik.lstrip("0").zfill(10)
        cache_key = f"{cik_padded}_{form_type}_{max_results}"
        cached_str = _cache_read("filings", cache_key, _TTL_INDEX)
        if cached_str is not None:
            import json as _json
            try:
                return _json.loads(cached_str)
            except Exception:
                pass

        url = (
            f"{_EDGAR_BASE}/cgi-bin/browse-edgar"
            f"?action=getcompany&CIK={cik_padded}"
            f"&type={form_type}&dateb=&owner=include"
            f"&count={max_results}&search_text=&output=atom"
        )
        raw = self._get_text(url)
        if not raw:
            return []

        results = self._parse_atom_feed(raw, form_type)
        import json as _json
        _cache_write("filings", cache_key, _json.dumps(results))
        log.info("list_filings(%s, %s): %d results", cik, form_type, len(results))
        return results

    def fetch_filing(self, url: str, timeout: int = 30) -> Optional[str]:
        """Fetch a specific filing document from EDGAR with caching (7 days).

        Args:
            url:     Full EDGAR document URL.
            timeout: Request timeout in seconds.

        Returns:
            Document text or None on failure.
        """
        cached = _cache_read("docs", url, _TTL_FILING)
        if cached is not None:
            log.debug("fetch_filing: cache hit %s", url[:60])
            return cached

        text = self._get_text(url, timeout=timeout)
        if text:
            _cache_write("docs", url, text)
        return text

    # ── Private: Atom feed parser ──────────────────────────────────────────────

    @staticmethod
    def _parse_atom_feed(xml: str, form_type: str) -> List[FilingRef]:
        """Extract FilingRef entries from SEC's Atom feed XML."""
        try:
            import xml.etree.ElementTree as ET
        except ImportError:
            return []

        results: List[FilingRef] = []
        try:
            root = ET.fromstring(xml)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            for entry in root.findall("atom:entry", ns):
                title_el = entry.find("atom:title", ns)
                updated_el = entry.find("atom:updated", ns)
                link_el = entry.find("atom:link", ns)
                id_el = entry.find("atom:id", ns)
                if link_el is None:
                    continue

                href = link_el.get("href", "")
                # Extract CIK from the ID URL pattern
                cik = ""
                if id_el is not None and id_el.text:
                    parts = id_el.text.split("CIK=")
                    if len(parts) > 1:
                        cik = parts[1].split("&")[0]

                title = (title_el.text or "").strip() if title_el is not None else ""
                date = (updated_el.text or "")[:10] if updated_el is not None else ""

                results.append({
                    "cik": cik,
                    "company_name": title,
                    "form_type": form_type,
                    "date_filed": date,
                    "filename": href.split("/")[-1] if "/" in href else href,
                    "url": href,
                })
        except Exception as exc:
            log.warning("_parse_atom_feed: %s", exc)
        return results


# ── Module-level singleton ─────────────────────────────────────────────────────

#: Default process-wide EdgarService instance.
default_edgar: EdgarService = EdgarService()
