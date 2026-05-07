"""backend/market_intel/edgar_parse.py — Form-4 & 13F-HR XML parsers.

Spence (2001 Nobel) — Costly signaling: insider purchases (Code P) require
the insider to spend their own money, which makes them an inherently more
credible signal than open-mouth disclosures. The parser extracts every
transaction code and lets the scorer apply the role/size weighting.

Public functions:
    parse_form4_file(path_or_url) -> List[dict]
    parse_form13f_file(path_or_url) -> List[dict]

Schema returned (matches normalizer.InsiderEvent):
    type, reporting_person, reporting_role, transaction_date,
    transaction_code, shares, price, value, filing_accession, is_amendment,
    issuer_ticker, acquired_disposed
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# ── Helpers ───────────────────────────────────────────────────────────────────

# Form-4 wraps most leaf values in <value> tags. We strip namespaces and walk.
_NS_RE = re.compile(r"^{[^}]+}")


def _local(tag: str) -> str:
    """Strip XML namespace from a tag name."""
    return _NS_RE.sub("", tag)


def _text(elem: Optional[ET.Element], default: Optional[str] = None) -> Optional[str]:
    """Return stripped text from element. Handles <value>-wrapped leaves."""
    if elem is None:
        return default
    inner = elem.find(".//{*}value")
    if inner is None:
        # try without namespace
        for child in elem:
            if _local(child.tag) == "value":
                inner = child
                break
    if inner is not None and inner.text:
        return inner.text.strip()
    return (elem.text or default or "").strip() if (elem.text or default) else default


def _find_local(parent: ET.Element, name: str) -> Optional[ET.Element]:
    """Find first descendant with matching local name (namespace-agnostic)."""
    for el in parent.iter():
        if _local(el.tag) == name:
            return el
    return None


def _findall_local(parent: ET.Element, name: str) -> List[ET.Element]:
    return [el for el in parent.iter() if _local(el.tag) == name]


def _to_float(s: Optional[str]) -> Optional[float]:
    if s is None or s == "":
        return None
    try:
        return float(str(s).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _load_xml(path_or_url: Union[str, Path]) -> ET.Element:
    """Load XML from file path, URL, or raw string."""
    s = str(path_or_url)
    if s.startswith(("http://", "https://")):
        # Lazy import requests so unit tests with file paths don't need network.
        import requests
        from . import config
        resp = requests.get(s, headers={"User-Agent": config.USER_AGENT}, timeout=config.HTTP_TIMEOUT_S)
        resp.raise_for_status()
        return ET.fromstring(resp.content)
    p = Path(s)
    if p.exists():
        return ET.parse(p).getroot()
    # Treat as raw XML string
    return ET.fromstring(s)


# ── Form-4 ────────────────────────────────────────────────────────────────────

_TRUE_VALUES = {"1", "true", "yes", "y"}


def _flag(rel: ET.Element, name: str) -> bool:
    """SEC Form-4 uses '1' or 'true' (lowercase) for boolean relationship flags."""
    val = (_text(_find_local(rel, name)) or "").strip().lower()
    return val in _TRUE_VALUES


def _derive_role(rel: Optional[ET.Element]) -> str:
    """Combine relationship flags into a single role string (CEO / CFO / Director / 10%Owner / Officer)."""
    if rel is None:
        return "Unknown"
    is_director = _flag(rel, "isDirector")
    is_officer = _flag(rel, "isOfficer")
    is_ten = _flag(rel, "isTenPercentOwner")
    title = (_text(_find_local(rel, "officerTitle")) or "").upper()

    if is_officer and ("CEO" in title or "CHIEF EXECUTIVE" in title):
        return "CEO"
    if is_officer and ("CFO" in title or "CHIEF FINANCIAL" in title):
        return "CFO"
    if is_officer and any(k in title for k in (
        "PRESIDENT", "COO", "CHAIRMAN", "CHIEF OPERATING", "CHIEF TECHNOLOGY",
        "CHIEF ACCOUNTING", "CHIEF LEGAL", "GENERAL COUNSEL",
    )):
        return "Officer-Senior"
    if is_officer:
        return "Officer"
    if is_director:
        return "Director"
    if is_ten:
        return "10%Owner"
    return "Unknown"


def parse_form4_file(path_or_url: Union[str, Path]) -> List[Dict[str, Any]]:
    """Parse a Form-4 XML and return one record per non-derivative transaction.

    Robust to namespace variants and missing fields. Empty list on parse failure.
    """
    try:
        root = _load_xml(path_or_url)
    except Exception:
        return []

    if _local(root.tag) != "ownershipDocument":
        # Some filings wrap the doc; descend until we find it.
        candidate = _find_local(root, "ownershipDocument")
        if candidate is None:
            return []
        root = candidate

    doc_type = _text(_find_local(root, "documentType")) or "4"
    is_amendment = doc_type.endswith("/A") or doc_type == "4/A"

    issuer = _find_local(root, "issuer")
    issuer_ticker = (_text(_find_local(issuer, "issuerTradingSymbol")) or "").upper() if issuer is not None else ""
    issuer_name = (_text(_find_local(issuer, "issuerName")) or "") if issuer is not None else ""

    rpt_owner = _find_local(root, "reportingOwner")
    rpt_id = _find_local(rpt_owner, "reportingOwnerId") if rpt_owner is not None else None
    rel = _find_local(rpt_owner, "reportingOwnerRelationship") if rpt_owner is not None else None
    person = _text(_find_local(rpt_id, "rptOwnerName")) if rpt_id is not None else None
    role = _derive_role(rel)

    # Accession is NOT in the ownership document — it's part of the URL path.
    # Adapter backfills filing_accession from the surrounding filing dict.
    accession: Optional[str] = None

    out: List[Dict[str, Any]] = []
    for txn in _findall_local(root, "nonDerivativeTransaction"):
        tx_date = _text(_find_local(txn, "transactionDate"))
        coding = _find_local(txn, "transactionCoding")
        code = _text(_find_local(coding, "transactionCode")) if coding is not None else None
        amounts = _find_local(txn, "transactionAmounts")
        shares = _to_float(_text(_find_local(amounts, "transactionShares"))) if amounts is not None else None
        price = _to_float(_text(_find_local(amounts, "transactionPricePerShare"))) if amounts is not None else None
        ad_code = _text(_find_local(amounts, "transactionAcquiredDisposedCode")) if amounts is not None else None

        value: Optional[float] = None
        if shares is not None and price is not None:
            value = round(shares * price, 2)

        out.append({
            "type":              "Form-4",
            "issuer_ticker":     issuer_ticker,
            "issuer_name":       issuer_name,
            "reporting_person":  person,
            "reporting_role":    role,
            "transaction_date":  tx_date,
            "transaction_code":  code,
            "shares":            shares,
            "price":             price,
            "value":             value,
            "acquired_disposed": ad_code,    # "A" = acquired, "D" = disposed
            "filing_accession":  accession,
            "is_amendment":      bool(is_amendment),
        })
    return out


# ── 13F-HR (lightweight: holdings list) ───────────────────────────────────────

def parse_form13f_file(path_or_url: Union[str, Path]) -> List[Dict[str, Any]]:
    """Parse a 13F-HR information table XML and return per-holding records.

    13F primary doc is often a cover XML; the information table is a separate
    file. This parser handles the table directly. Empty list on failure.
    """
    try:
        root = _load_xml(path_or_url)
    except Exception:
        return []

    out: List[Dict[str, Any]] = []
    for info in _findall_local(root, "infoTable"):
        name = _text(_find_local(info, "nameOfIssuer"))
        cusip = _text(_find_local(info, "cusip"))
        title = _text(_find_local(info, "titleOfClass"))
        value = _to_float(_text(_find_local(info, "value")))     # SEC reports in $thousands
        ssh = _find_local(info, "shrsOrPrnAmt")
        shares = _to_float(_text(_find_local(ssh, "sshPrnamt"))) if ssh else None
        out.append({
            "type":          "13F-HR",
            "issuer_name":   name,
            "cusip":         cusip,
            "title_class":   title,
            "value_usd":     (value * 1000.0) if value is not None else None,
            "shares":        shares,
        })
    return out
