"""scripts/fmp_smoke_test.py
FMP Ultimate endpoint smoke-test — run BEFORE any refactor.

Pings every stable/ endpoint the pipeline depends on (plus the new premium
endpoints we want to adopt) using the real FMP_API_KEY, and prints a
PASS/FAIL/EMPTY table. This tells us, empirically and per-endpoint:

  PASS   — 200 with data       (route live, usable)
  EMPTY  — 200 with []/{}       (route live, this ticker just has no data)
  FAIL   — 401/403/404          (route dead or not in plan — DO NOT build on it)
  ERROR  — network/other

Why this matters: the whole inconsistency cascade (zeroed factors → lowered
circuit-breaker → mistrusted scores) came from NOT knowing which routes were
alive. Measure first.

Usage:
    export FMP_API_KEY=...        # your Ultimate key
    python scripts/fmp_smoke_test.py
    python scripts/fmp_smoke_test.py --ticker AAPL --eu SAP.DE --asia 7203.T
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import requests

_STABLE = "https://financialmodelingprep.com/stable"
_TIMEOUT = 15


def _probe(path: str, params: dict, key: str) -> tuple[str, str]:
    """Return (status_label, detail) for a single endpoint probe."""
    p = dict(params)
    p["apikey"] = key
    url = f"{_STABLE}/{path}"
    try:
        r = requests.get(url, params=p, timeout=_TIMEOUT)
    except Exception as exc:
        return "ERROR", str(exc)[:60]

    if r.status_code in (401, 403, 404):
        return "FAIL", f"HTTP {r.status_code}"
    if r.status_code != 200:
        return "ERROR", f"HTTP {r.status_code}"
    try:
        data = r.json()
    except Exception:
        return "ERROR", "non-JSON body"

    if isinstance(data, list):
        return ("PASS", f"{len(data)} rows") if data else ("EMPTY", "[]")
    if isinstance(data, dict):
        # FMP sometimes returns {"Error Message": "..."} with 200
        if "Error Message" in data:
            return "FAIL", str(data["Error Message"])[:60]
        return ("PASS", f"{len(data)} keys") if data else ("EMPTY", "{}")
    return "PASS", repr(data)[:40]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="AAPL", help="US test ticker")
    ap.add_argument("--eu", default="SAP.DE", help="EU test ticker (Frankfurt suffix)")
    ap.add_argument("--asia", default="7203.T", help="Asia test ticker (Tokyo suffix)")
    args = ap.parse_args()

    key = os.getenv("FMP_API_KEY", "")
    if not key:
        print("ERROR: FMP_API_KEY not set in environment.")
        return 1

    t = args.ticker

    # (label, path, params) — covers current pipeline deps + new premium endpoints
    probes: list[tuple[str, str, dict]] = [
        # ── Core 7-factor pipeline dependencies ──
        ("quote",                 "quote",                    {"symbol": t}),
        ("congress:senate",       "senate-trading",           {"symbol": t}),
        ("congress:house",        "house-trading",            {"symbol": t}),
        ("insider:search",        "insider-trading/search",   {"symbol": t, "page": 0, "limit": 100}),
        ("news:stock",            "news/stock",               {"symbols": t, "limit": 10}),
        ("profile",               "profile",                  {"symbol": t}),
        # ── New Ultimate-tier endpoints we want to adopt ──
        ("ratings:consensus",     "grades-consensus",         {"symbol": t}),
        ("price-target",          "price-target-consensus",   {"symbol": t}),
        ("key-metrics-ttm",       "key-metrics-ttm",          {"symbol": t}),
        ("ratios-ttm",            "ratios-ttm",               {"symbol": t}),
        ("13f:summary",           "institutional-ownership/symbol-positions-summary",
                                                              {"symbol": t, "page": 0, "limit": 1}),
        ("batch-quote",           "batch-quote",              {"symbols": f"{t},MSFT,NVDA"}),
        ("cash-flow",             "cash-flow-statement",      {"symbol": t, "period": "quarter", "limit": 4}),
        ("cot",                   "commitment-of-traders-report", {}),
        ("earnings-transcript",   "earning-call-transcript-latest", {"symbol": t}),
        # ── International coverage test (the market_config 403 claim) ──
        ("EU:quote",              "quote",                    {"symbol": args.eu}),
        ("EU:ratios-ttm",         "ratios-ttm",               {"symbol": args.eu}),
        ("ASIA:quote",            "quote",                    {"symbol": args.asia}),
        ("ASIA:ratios-ttm",       "ratios-ttm",               {"symbol": args.asia}),
    ]

    print(f"\nFMP stable/ smoke-test  (US={t}, EU={args.eu}, ASIA={args.asia})")
    print("=" * 70)
    print(f"{'ENDPOINT':<24} {'STATUS':<7} DETAIL")
    print("-" * 70)

    results: dict[str, str] = {}
    for label, path, params in probes:
        status, detail = _probe(path, params, key)
        results[label] = status
        marker = {"PASS": "OK", "EMPTY": "--", "FAIL": "XX", "ERROR": "!!"}.get(status, "??")
        print(f"{label:<24} {marker} {status:<5} {detail}")
        time.sleep(0.05)  # gentle on rate limit

    print("-" * 70)
    fails = [k for k, v in results.items() if v in ("FAIL", "ERROR")]
    eu_asia_live = [k for k in results if k.startswith(("EU:", "ASIA:")) and results[k] == "PASS"]

    print(f"\nSUMMARY: {sum(1 for v in results.values() if v=='PASS')} PASS, "
          f"{sum(1 for v in results.values() if v=='EMPTY')} EMPTY, "
          f"{len(fails)} FAIL/ERROR")
    if fails:
        print(f"  XX Dead/unavailable: {', '.join(fails)}")
        print("    -> Do NOT build factors on these. Investigate plan coverage.")
    if eu_asia_live:
        print(f"  OK International LIVE: {', '.join(eu_asia_live)}")
        print("    -> market_config.py '403 for non-US' claim may be WRONG. Re-test.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
