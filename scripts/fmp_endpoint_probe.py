# Path: scripts/fmp_endpoint_probe.py
"""
FMP stable/ endpoint probe — covers every path used by fmp_client.py + bulk prefetch.

Run:
    python scripts/fmp_endpoint_probe.py

Loads FMP_API_KEY from .env automatically. Tests AAPL for US endpoints and
ASML.AS for EU endpoints. For any ❓ endpoint also tests plausible renamed
alternatives so we can find the correct stable/ path.
"""
from __future__ import annotations

import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import requests

API_KEY = os.environ.get("FMP_API_KEY", "")
if not API_KEY:
    print("ERROR: FMP_API_KEY not set (add to .env or export)")
    sys.exit(1)

BASE    = "https://financialmodelingprep.com/stable"
TICKER  = "AAPL"
EU_TICK = "ASML.AS"

_session = requests.Session()
_session.headers["User-Agent"] = "regime-trader-probe/1.0"


def probe(path: str, params: dict) -> tuple[int, str]:
    p = {**params, "apikey": API_KEY}
    try:
        r = _session.get(f"{BASE}/{path}", params=p, timeout=12)
        # Summarise: show type + count/keys for non-empty 200, else raw snippet
        if r.status_code == 200:
            try:
                body = r.json()
                if isinstance(body, list):
                    snippet = f"list({len(body)} items)"
                    if body:
                        snippet += f"  sample keys: {list(body[0].keys())[:5]}"
                elif isinstance(body, dict):
                    snippet = f"dict  keys: {list(body.keys())[:6]}"
                else:
                    snippet = repr(body)[:80]
            except Exception:
                snippet = r.text[:80]
        else:
            snippet = r.text[:80].replace("\n", " ")
        return r.status_code, snippet
    except Exception as exc:
        return -1, str(exc)


# ── All endpoints in fmp_client.py + bulk prefetch ───────────────────────────
# Format: (label, path, params)
PROBES: list[tuple[str, str, dict]] = [
    # ── Confirmed live (regression check) ──────────────────────────────────
    ("historical-price-eod/full [US]",     "historical-price-eod/full",   {"symbol": TICKER, "limit": 5}),
    ("quote [US]",                          "quote",                        {"symbol": TICKER}),
    ("batch-quote [US]",                    "batch-quote",                  {"symbols": TICKER}),
    ("grades-consensus [US]",               "grades-consensus",             {"symbol": TICKER}),
    ("ratios-ttm [US]",                     "ratios-ttm",                   {"symbol": TICKER}),
    ("enterprise-values [US]",              "enterprise-values",            {"symbol": TICKER, "limit": 1}),
    ("cash-flow-statement [US]",            "cash-flow-statement",          {"symbol": TICKER, "period": "quarter", "limit": 1}),
    ("earning-call-transcript-latest [US]", "earning-call-transcript-latest", {"symbol": TICKER, "limit": 1}),
    ("commitment-of-traders-report",        "commitment-of-traders-report", {}),
    ("news/stock [US]",                     "news/stock",                   {"symbols": TICKER, "limit": 3}),
    ("institutional-ownership/symbol-positions-summary",
                                            "institutional-ownership/symbol-positions-summary",
                                            {"symbol": TICKER, "year": "2024", "quarter": "4", "page": 0, "limit": 5}),
    # ── Insider (scoring uses /search sub-path, not bare insider-trading) ──
    ("insider-trading/search [US]",         "insider-trading/search",       {"symbol": TICKER, "page": 0, "limit": 5}),
    # ── Quarantined (should stay 404) ──────────────────────────────────────
    ("earnings-surprises [DEAD]",           "earnings-surprises",           {"symbol": TICKER, "limit": 1}),
    ("upgrades-downgrades [DEAD]",          "upgrades-downgrades",          {"symbol": TICKER, "page": 0}),
    ("senate-trading [DEAD]",               "senate-trading",               {"symbol": TICKER, "page": 0, "limit": 5}),
    # ── Unknowns — current name + alternatives ──────────────────────────────
    ("analyst-estimates [CURRENT]",         "analyst-estimates",            {"symbol": TICKER, "period": "quarter", "limit": 4}),
    ("  alt: earnings-estimates",           "earnings-estimates",           {"symbol": TICKER, "period": "quarter", "limit": 4}),
    ("  alt: analyst-forecast",             "analyst-forecast",             {"symbol": TICKER, "limit": 4}),
    ("  alt: earnings-consensus",           "earnings-consensus",           {"symbol": TICKER, "limit": 4}),
    ("  alt: earnings-analyst-estimates",   "earnings-analyst-estimates",   {"symbol": TICKER, "limit": 4}),
    ("price-target-consensus [CURRENT]",    "price-target-consensus",       {"symbol": TICKER}),
    ("  alt: price-target-summary",         "price-target-summary",         {"symbol": TICKER}),
    ("  alt: price-target",                 "price-target",                 {"symbol": TICKER, "limit": 1}),
    ("  alt: analyst-price-target",         "analyst-price-target",         {"symbol": TICKER}),
    ("  alt: price-target-latest-news",     "price-target-latest-news",     {"symbol": TICKER}),
    # ── EU/Asia spot-check ──────────────────────────────────────────────────
    ("historical-price-eod/full [EU]",      "historical-price-eod/full",    {"symbol": EU_TICK, "limit": 5}),
    ("quote [EU]",                          "quote",                        {"symbol": EU_TICK}),
    ("ratios-ttm [EU]",                     "ratios-ttm",                   {"symbol": EU_TICK}),
    ("analyst-estimates [EU]",              "analyst-estimates",            {"symbol": EU_TICK, "period": "quarter", "limit": 4}),
    ("price-target-consensus [EU]",         "price-target-consensus",       {"symbol": EU_TICK}),
    # ── Bulk endpoints (fmp_bulk_prefetch.py) ───────────────────────────────
    ("upgrades-downgrades-consensus-bulk",  "upgrades-downgrades-consensus-bulk", {}),
    ("ratios-ttm-bulk",                     "ratios-ttm-bulk",              {}),
    ("key-metrics-ttm-bulk",                "key-metrics-ttm-bulk",         {}),
]

W = 46
print(f"\nProbing {BASE}/<endpoint>\n")
print(f"{'Label':<{W}} {'HTTP':>6}  Response")
print("-" * 100)

for label, path, params in PROBES:
    status, body = probe(path, params)
    if label.startswith("  alt:"):
        flag = "  OK" if status == 200 else ("  XX" if status in (401,403,404) else "  ??")
    else:
        flag = "OK " if status == 200 else ("XX " if status in (401,403,404) else "?? ")
    print(f"{flag} {label:<{W-2}} {status:>6}  {body[:70]}")

print()
print("Legend: OK=live  XX=dead(4xx)  ??=unexpected  indent=alternative name")
