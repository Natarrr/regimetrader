"""regime_trader/services/edgar_fetcher.py
XBRL-first filing fetcher with atomic cache and circuit breaker.

Stiglitz (2001 Nobel) — structured XBRL data reduces information asymmetry vs
raw HTML; the fetcher prefers machine-readable XBRL facts before falling back
to plain-text documents.

Design:
  - For each filing URL, fetches the filing index page to locate XBRL docs.
  - XBRL preferred: looks for *_htm.xml, *_10k.xml, R*.htm inline XBRL.
  - Falls back to text/HTML document if no XBRL found.
  - Atomic cache under .cache/edgar/filings/<evidence_id>.json  (TTL 7 d).
  - Circuit breaker: opens after EDGAR_CB_FAIL_THRESHOLD consecutive 5xx/timeouts.
  - Retry with exponential backoff via urllib3.

Public API:
    fetch_and_parse_filing(url, evidence_id) → ParsedFiling
    EdgarFetcher class for injection in tests.

Usage:
    from regime_trader.services.edgar_fetcher import EdgarFetcher
    fetcher = EdgarFetcher()
    result = fetcher.fetch_and_parse_filing(
        "https://www.sec.gov/Archives/edgar/data/320193/...",
        evidence_id="abc123"
    )
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from regime_trader.utils.token_bucket import TokenBucket

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

_CACHE_ROOT   = Path(__file__).parent.parent.parent / ".cache" / "edgar" / "filings"
_TTL_FILING   = 7 * 24 * 3600     # 7 days
_MAX_RETRIES  = 3
_BACKOFF      = 1.5
_DEFAULT_RATE = float(os.getenv("EDGAR_RATE_LIMIT", "0.2"))

# Circuit breaker thresholds
_CB_PATH            = Path(__file__).parent.parent.parent / ".cache" / "edgar" / "fetcher_cb.json"
_CB_FAIL_THRESHOLD  = int(os.getenv("EDGAR_CB_FAIL_THRESHOLD", "5"))
_CB_COOLDOWN_MIN    = int(os.getenv("EDGAR_CB_COOLDOWN_MIN", "15"))

_HEADERS = {
    "User-Agent": os.getenv(
        "EDGAR_USER_AGENT",
        "regime-trader-research infra-team@example.com",
    ),
    "Accept-Encoding": "gzip, deflate",
}

# XBRL namespace URIs (common ones)
_XBRL_NS = {
    "xbrli": "http://www.xbrl.org/2003/instance",
    "us-gaap": "http://fasb.org/us-gaap/2023",
    "dei": "http://xbrl.sec.gov/dei/2023",
}


# ── Typed output ───────────────────────────────────────────────────────────────

class ParsedFiling(TypedDict):
    evidence_id:  str
    url:          str
    form_type:    str          # "4", "13F-HR", "8-K", …
    source:       str          # "xbrl" | "html" | "text" | "error"
    parsed:       Dict[str, Any]
    fetched_at:   float


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _cache_path(evidence_id: str) -> Path:
    return _CACHE_ROOT / f"{evidence_id}.json"


def _cache_read(evidence_id: str) -> Optional[ParsedFiling]:
    p = _cache_path(evidence_id)
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if time.time() - data.get("fetched_at", 0) > _TTL_FILING:
            return None
        return data
    except Exception:
        return None


def _cache_write(evidence_id: str, result: ParsedFiling) -> None:
    p = _cache_path(evidence_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result, indent=2, ensure_ascii=False).encode("utf-8")
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


# ── Circuit breaker ────────────────────────────────────────────────────────────

def _cb_state() -> Dict[str, Any]:
    if not _CB_PATH.exists():
        return {"state": "closed", "fail_count": 0, "last_failure": 0.0}
    try:
        return json.loads(_CB_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"state": "closed", "fail_count": 0, "last_failure": 0.0}


def _cb_write(state: Dict[str, Any]) -> None:
    _CB_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CB_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _cb_allows_calls() -> bool:
    state = _cb_state()
    if state["state"] == "closed":
        return True
    # Check cooldown
    elapsed_min = (time.time() - state["last_failure"]) / 60.0
    threshold   = int(os.getenv("EDGAR_CB_COOLDOWN_MIN", str(_CB_COOLDOWN_MIN)))
    if elapsed_min >= threshold:
        _cb_write({"state": "closed", "fail_count": 0, "last_failure": 0.0})
        return True
    return False


def _cb_record_failure() -> None:
    state = _cb_state()
    state["fail_count"] = state.get("fail_count", 0) + 1
    state["last_failure"] = time.time()
    threshold = int(os.getenv("EDGAR_CB_FAIL_THRESHOLD", str(_CB_FAIL_THRESHOLD)))
    if state["fail_count"] >= threshold:
        state["state"] = "open"
        log.warning(
            "edgar_fetcher circuit-breaker OPEN after %d consecutive failures",
            state["fail_count"],
        )
    _cb_write(state)


def _cb_record_success() -> None:
    state = _cb_state()
    if state.get("fail_count", 0) > 0:
        _cb_write({"state": "closed", "fail_count": 0, "last_failure": 0.0})


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


# ── XBRL parser ────────────────────────────────────────────────────────────────

def _extract_xbrl_facts(text: str) -> Dict[str, Any]:
    """Parse XBRL XML and return a flat dict of tag → value for key DEI/US-GAAP facts.

    Stiglitz (2001): XBRL makes structured corporate data machine-readable,
    reducing the cost of information extraction for all market participants.

    Returns an empty dict if XML is malformed or no facts found.
    """
    facts: Dict[str, Any] = {}
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return facts

    for elem in root.iter():
        tag = elem.tag
        # Strip namespace: {http://...}TagName → TagName
        if "}" in tag:
            tag = tag.split("}", 1)[1]
        val = (elem.text or "").strip()
        if val and tag not in ("context", "unit", "xbrl"):
            facts[tag] = val

    return facts


def _extract_form4_rows(text: str) -> List[Dict[str, Any]]:
    """Parse Form-4 XML into a list of transaction rows.

    Each row contains: issuerTicker, reportingOwner, transactionDate,
    transactionCode, shares, price, acquiredDisposed.
    """
    rows: List[Dict[str, Any]] = []
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return rows

    def _text(parent: ET.Element, tag: str, default: str = "") -> str:
        el = parent.find(f".//{tag}")
        return (el.text or default).strip() if el is not None else default

    issuer  = _text(root, "issuerTradingSymbol")
    owner   = _text(root, "rptOwnerName")

    for tx in root.findall(".//nonDerivativeTransaction"):
        rows.append({
            "issuerTicker":    issuer,
            "reportingOwner":  owner,
            "transactionDate": _text(tx, "transactionDate"),
            "transactionCode": _text(tx, "transactionCode"),
            "shares":          _text(tx, "transactionShares"),
            "price":           _text(tx, "transactionPricePerShare"),
            "acquiredDisposed":_text(tx, "transactionAcquiredDisposedCode"),
        })
    return rows


def _extract_8k_items(text: str) -> List[str]:
    """Extract item numbers from an 8-K filing text (e.g. ['1.01', '8.01'])."""
    return re.findall(r"Item\s+(\d+\.\d+)", text, re.IGNORECASE)


# ── Filing index parser ────────────────────────────────────────────────────────

_XBRL_SUFFIXES  = (".xml", "_htm.xml", ".xsd")
_XBRL_KEYWORDS  = ("xbrl", "instance", "htm.xml", "R1", "R2")

def _pick_best_doc(index_text: str, base_url: str) -> tuple[str, str]:
    """From the filing index page, return (doc_url, source_type).

    Preference: XBRL > HTML > text.
    Extracts hrefs from HTML attributes OR bare filenames from plain-text index.
    """
    xbrl_url = html_url = text_url = None

    # Collect candidate URLs: HTML href attributes + plain-text filenames
    candidates: list[str] = re.findall(
        r'href=["\']([^"\'>\s]+)["\']', index_text, re.IGNORECASE
    )
    # Also pick up space-separated filenames in plain-text index listings
    for line in index_text.splitlines():
        parts = line.split()
        for p in parts:
            if "/" in p and not p.startswith("<") and len(p) > 5:
                candidates.append(p)

    for href in candidates:
        if not href.startswith("http"):
            href = base_url.rstrip("/") + "/" + href.lstrip("/")
        low = href.lower()
        if ("_htm.xml" in low or ("xbrl" in low and low.endswith(".xml"))):
            if xbrl_url is None:
                xbrl_url = href
        elif low.endswith(".htm") or low.endswith(".html"):
            if html_url is None:
                html_url = href
        elif low.endswith(".txt") and "complete" in low:
            if text_url is None:
                text_url = href

    if xbrl_url:
        return xbrl_url, "xbrl"
    if html_url:
        return html_url, "html"
    if text_url:
        return text_url, "text"
    return base_url, "text"


# ── Main class ─────────────────────────────────────────────────────────────────

class EdgarFetcher:
    """Stiglitz (2001 Nobel) — XBRL-first filing fetcher.

    Fetches filing index pages to locate XBRL documents, downloads and parses
    them, and caches results for 7 days.  Falls back to HTML/text when no XBRL
    is available.  Circuit breaker prevents cascading failures on SEC outages.

    Args:
        rate_per_sec:  HTTP request rate (default EDGAR_RATE_LIMIT env or 0.2).
        cache_root:    Override cache directory (useful in tests).
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

    def fetch_and_parse_filing(
        self,
        url:         str,
        evidence_id: str = "",
        form_type:   str = "",
    ) -> ParsedFiling:
        """Stiglitz (2001 Nobel) — Fetch and parse a single SEC filing.

        Checks cache first.  If not cached, downloads the filing index page,
        picks the best document (XBRL > HTML > text), fetches and parses it,
        then writes to cache atomically.

        Args:
            url:          Absolute URL to the filing index page on SEC servers.
            evidence_id:  Pre-computed evidence ID; derived from URL if empty.
            form_type:    Hint for the parser (e.g. "4" triggers Form-4 parser).

        Returns:
            ParsedFiling with parsed content and metadata.
        """
        if not evidence_id:
            evidence_id = hashlib.sha256(url.encode()).hexdigest()[:16]

        # Cache hit
        cached = self._cache_read(evidence_id)
        if cached is not None:
            return cached

        # Circuit breaker
        if not _cb_allows_calls():
            log.warning("edgar_fetcher CB open — skipping %s", url)
            return _error_result(evidence_id, url, form_type, "circuit_breaker_open")

        # Fetch filing index
        self._bucket.acquire()
        try:
            idx_resp = self._session.get(url, timeout=20)
            idx_resp.raise_for_status()
        except Exception as exc:
            _cb_record_failure()
            log.warning("edgar_fetcher: index fetch failed %s — %s", url, exc)
            return _error_result(evidence_id, url, form_type, str(exc))

        doc_url, source_type = _pick_best_doc(idx_resp.text, url)

        # Fetch actual document
        self._bucket.acquire()
        try:
            doc_resp = self._session.get(doc_url, timeout=30)
            doc_resp.raise_for_status()
        except Exception as exc:
            _cb_record_failure()
            log.warning("edgar_fetcher: doc fetch failed %s — %s", doc_url, exc)
            return _error_result(evidence_id, url, form_type, str(exc))

        _cb_record_success()
        parsed = self._parse(doc_resp.text, source_type, form_type)
        result: ParsedFiling = {
            "evidence_id": evidence_id,
            "url":         url,
            "form_type":   form_type or "",
            "source":      source_type,
            "parsed":      parsed,
            "fetched_at":  time.time(),
        }
        self._cache_write(evidence_id, result)
        return result

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _cache_read(self, evidence_id: str) -> Optional[ParsedFiling]:
        p = self._cache_root / f"{evidence_id}.json"
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if time.time() - data.get("fetched_at", 0) > _TTL_FILING:
                return None
            return data
        except Exception:
            return None

    def _cache_write(self, evidence_id: str, result: ParsedFiling) -> None:
        p = self._cache_root / f"{evidence_id}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(result, indent=2, ensure_ascii=False).encode("utf-8")
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

    @staticmethod
    def _parse(text: str, source_type: str, form_type: str) -> Dict[str, Any]:
        ft = form_type.upper()
        if source_type == "xbrl":
            if ft == "4":
                return {"transactions": _extract_form4_rows(text)}
            return {"facts": _extract_xbrl_facts(text)}
        if ft == "4":
            return {"transactions": _extract_form4_rows(text)}
        if ft == "8-K":
            return {"items": _extract_8k_items(text)}
        # Generic: return first 2000 chars of text
        return {"raw_preview": text[:2000]}


# ── Module-level convenience ───────────────────────────────────────────────────

def fetch_and_parse_filing(
    url:         str,
    evidence_id: str = "",
    form_type:   str = "",
) -> ParsedFiling:
    """Module-level convenience wrapper using a default EdgarFetcher instance."""
    return EdgarFetcher().fetch_and_parse_filing(url, evidence_id, form_type)


def _error_result(
    evidence_id: str,
    url:         str,
    form_type:   str,
    reason:      str,
) -> ParsedFiling:
    return ParsedFiling(
        evidence_id=evidence_id,
        url=url,
        form_type=form_type or "",
        source="error",
        parsed={"error": reason},
        fetched_at=time.time(),
    )
