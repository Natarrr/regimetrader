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
GB_TICK = "VOD.L"     # LSE — pence (GBX) quote vs GBP target audit
JP_TICK = "7203.T"    # Toyota — tanshin quarterly cadence audit
HK_TICK = "0700.HK"   # Tencent — HKEX semi-annual interim audit

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
    # ── PEAD: get_earnings_surprise() source — expect epsActual/epsEstimated/date ──
    ("earnings [PEAD]",                     "earnings",                     {"symbol": TICKER, "limit": 4}),
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
    # ── v3.0 additions: margin_expansion / universes / intl 13F coverage ────
    ("income-statement [US qtr]",           "income-statement",             {"symbol": TICKER, "period": "quarter", "limit": 8}),
    ("income-statement [JP qtr]",           "income-statement",             {"symbol": JP_TICK, "period": "quarter", "limit": 8}),
    ("income-statement [HK qtr]",           "income-statement",             {"symbol": HK_TICK, "period": "quarter", "limit": 8}),
    ("income-statement [JP annual]",        "income-statement",             {"symbol": JP_TICK, "period": "annual", "limit": 2}),
    ("company-screener [XETRA]",            "company-screener",             {"exchange": "XETRA", "limit": 5}),
    ("  alt: stock-screener",               "stock-screener",               {"exchange": "XETRA", "limit": 5}),
    ("institutional-ownership [EU local]",  "institutional-ownership/symbol-positions-summary",
                                            {"symbol": EU_TICK, "year": "2025", "quarter": "4", "page": 0, "limit": 1}),
    ("institutional-ownership [EU base]",   "institutional-ownership/symbol-positions-summary",
                                            {"symbol": "ASML", "year": "2025", "quarter": "4", "page": 0, "limit": 1}),
    ("institutional-ownership [JP]",        "institutional-ownership/symbol-positions-summary",
                                            {"symbol": JP_TICK, "year": "2025", "quarter": "4", "page": 0, "limit": 1}),
    ("price-target-consensus [GB .L]",      "price-target-consensus",       {"symbol": GB_TICK}),
    ("quote [GB .L]",                       "quote",                        {"symbol": GB_TICK}),
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


# ── v3.0 schema audits — run after the endpoint sweep ─────────────────────────

def _fetch(path: str, params: dict):
    try:
        r = _session.get(f"{BASE}/{path}", params={**params, "apikey": API_KEY},
                         timeout=12)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def audit_income_gaps(symbol: str) -> None:
    """Discrete-vs-cumulative audit: consecutive period-end gaps should be
    ~70–110 days for discrete quarters. ~180d gaps = semi-annual rows;
    overlapping/duplicate dates = cumulative YTD reporting."""
    rows = _fetch("income-statement",
                  {"symbol": symbol, "period": "quarter", "limit": 8}) or []
    dates = [r.get("date", "?") for r in rows if isinstance(r, dict)]
    print(f"\n  {symbol} quarterly period-ends ({len(dates)} rows): {dates}")
    try:
        from datetime import date
        ds = sorted(date.fromisoformat(d) for d in dates if d and d != "?")
        gaps = [(ds[i + 1] - ds[i]).days for i in range(len(ds) - 1)]
        verdict = ("DISCRETE-OK" if gaps and all(70 <= g <= 110 for g in gaps)
                   else "REVIEW (semi-annual/cumulative suspected)")
        print(f"  {symbol} gaps (days): {gaps}  -> {verdict}")
        if rows and isinstance(rows[0], dict):
            print(f"  {symbol} filingDate present: {'filingDate' in rows[0]}  "
                  f"sample keys: {[k for k in rows[0] if 'ate' in k]}")
    except Exception as exc:
        print(f"  {symbol} gap audit failed: {exc}")


def audit_dividend_fields(symbol: str) -> None:
    """Field-name drift audit: dividend_sustain reads dividendYieldTTM /
    payoutRatioTTM — verify what the live route actually calls them."""
    rows = _fetch("ratios-ttm", {"symbol": symbol}) or []
    row = rows[0] if isinstance(rows, list) and rows else {}
    hits = sorted(k for k in row if "ividend" in k or "ayout" in k)
    print(f"\n  {symbol} ratios-ttm dividend/payout fields: {hits or 'NONE'}")


print("\n" + "=" * 100)
print("v3.0 SCHEMA AUDITS (margin_expansion discrete-quarter + dividend fields)")
for _sym in (TICKER, JP_TICK, HK_TICK, EU_TICK):
    audit_income_gaps(_sym)
for _sym in (TICKER, EU_TICK, JP_TICK):
    audit_dividend_fields(_sym)
