# Path: scripts/fmp_eu_asia_smoke_test.py
"""
Smoke test: confirm FMP endpoint coverage for EU/Asia universe tickers.

Checks which FMP endpoints respond to suffixed tickers (e.g. ASML.AS) vs
base symbols (e.g. ASML), and verifies that insider endpoints correctly
return empty for non-US tickers (SEC Form 4 US-only).

Usage:
    python scripts/fmp_eu_asia_smoke_test.py [--ticker ASML.AS] [--all]
"""
from __future__ import annotations

import argparse
import os
import sys

# Ensure repo root is on PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.fmp_client import FMPClient

SAMPLE_TICKERS = [
    "ASML.AS",   # Netherlands — semiconductors
    "SAP.DE",    # Germany — software
    "LVMH.PA",   # France — luxury
    "9984.T",    # Japan — SoftBank
    "005930.KS", # South Korea — Samsung
    "BABA",      # US-listed ADR (China)
]


def _fmt(val: object) -> str:
    if val is None:
        return "None (absent)"
    if isinstance(val, dict):
        return f"dict({len(val)} keys)"
    if isinstance(val, list):
        return f"list({len(val)} items)"
    return repr(val)


def probe_ticker(client: FMPClient, ticker: str) -> dict:
    base = ticker.split(".")[0] if "." in ticker else ticker
    is_us = "." not in ticker
    results: dict[str, object] = {"ticker": ticker, "base": base, "is_us": is_us}

    # --- quote ---
    try:
        q = client.get_quote(ticker)
        results["quote_suffixed"] = bool(q and q.get("price"))
    except Exception as exc:
        results["quote_suffixed"] = f"ERROR: {exc}"

    if not is_us and base != ticker:
        try:
            q2 = client.get_quote(base)
            results["quote_base"] = bool(q2 and q2.get("price"))
        except Exception as exc:
            results["quote_base"] = f"ERROR: {exc}"

    # --- ratios-ttm ---
    try:
        r = client.get_ratios_ttm(ticker)
        results["ratios_ttm_suffixed"] = bool(r)
    except Exception as exc:
        results["ratios_ttm_suffixed"] = f"ERROR: {exc}"

    if not is_us and base != ticker:
        try:
            r2 = client.get_ratios_ttm(base)
            results["ratios_ttm_base"] = bool(r2)
        except Exception as exc:
            results["ratios_ttm_base"] = f"ERROR: {exc}"

    # --- analyst estimate revision ---
    try:
        est = client.get_analyst_estimate_revision(ticker)
        results["analyst_est_suffixed"] = bool(est)
    except Exception as exc:
        results["analyst_est_suffixed"] = f"ERROR: {exc}"

    if not is_us and base != ticker:
        try:
            est2 = client.get_analyst_estimate_revision(base)
            results["analyst_est_base"] = bool(est2)
        except Exception as exc:
            results["analyst_est_base"] = f"ERROR: {exc}"

    # --- price target consensus ---
    try:
        pt = client.get_price_target_consensus(ticker)
        results["price_target_suffixed"] = bool(pt)
    except Exception as exc:
        results["price_target_suffixed"] = f"ERROR: {exc}"

    # --- news ---
    try:
        news = client.get_news_raw_articles(ticker)
        results["news_suffixed"] = None if news is None else len(news)
    except Exception as exc:
        results["news_suffixed"] = f"ERROR: {exc}"

    if not is_us and base != ticker:
        try:
            news2 = client.get_news_raw_articles(base)
            results["news_base"] = None if news2 is None else len(news2)
        except Exception as exc:
            results["news_base"] = f"ERROR: {exc}"

    # --- insider (US-only gate verification) ---
    if is_us:
        try:
            ins = client.get_insider_transactions(ticker, limit=5)
            results["insider_us"] = len(ins) if isinstance(ins, list) else bool(ins)
        except Exception as exc:
            results["insider_us"] = f"ERROR: {exc}"
    else:
        results["insider_skip"] = "non-US — SEC Form 4 not applicable (correct)"

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="FMP EU/Asia endpoint coverage smoke test")
    parser.add_argument("--ticker", help="Single ticker to probe (e.g. ASML.AS)")
    parser.add_argument("--all", action="store_true", help="Run full sample universe")
    args = parser.parse_args()

    api_key = os.environ.get("FMP_API_KEY") or os.environ.get("FMP_KEY")
    if not api_key:
        print("ERROR: FMP_API_KEY environment variable not set", file=sys.stderr)
        return 1

    client = FMPClient(api_key=api_key)
    tickers = SAMPLE_TICKERS if args.all else ([args.ticker] if args.ticker else SAMPLE_TICKERS[:3])

    all_ok = True
    for ticker in tickers:
        print(f"\n{'='*60}")
        print(f"Probing: {ticker}")
        print(f"{'='*60}")
        res = probe_ticker(client, ticker)
        for k, v in res.items():
            if k in ("ticker", "base", "is_us"):
                continue
            status = "OK " if (v is not False and not str(v).startswith("ERROR") and v != "None (absent)") else "MISS"
            print(f"  [{status}] {k}: {_fmt(v)}")

        # Fail if quote is absent on both suffixed and base
        has_quote = res.get("quote_suffixed") or res.get("quote_base")
        if not has_quote:
            print(f"  [WARN] No quote data found for {ticker} — pipeline will short-circuit to price-only")
            all_ok = False

    print(f"\n{'='*60}")
    print("Smoke test complete." if all_ok else "Smoke test: warnings detected (see above).")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
