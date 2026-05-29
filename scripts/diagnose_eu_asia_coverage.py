"""scripts/diagnose_eu_asia_coverage.py — Fix #5 diagnostic tool.

One-shot script to audit FMP Ultimate and yfinance coverage for representative
EU and Asia tickers. Run ONCE before writing Fix #5 scorers.

NOT a pipeline step. Output informs the MARKET_FACTORS configuration in
regime_trader/scoring/market_config.py.

Usage:
    python scripts/diagnose_eu_asia_coverage.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from datetime import date

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EU_TICKERS  = ["SAP.DE", "AIR.PA", "OR.PA", "ASML.AS", "MC.PA"]
ASIA_TICKERS = ["7203.T", "6758.T", "9988.HK", "2330.TW", "005930.KS"]

ALL_TICKERS = EU_TICKERS + ASIA_TICKERS


def run_diagnostic() -> None:
    api_key = os.environ.get("FMP_API_KEY", "")
    if not api_key:
        print("WARNING: FMP_API_KEY not set — FMP calls will fail.\n")

    from regime_trader.services.fmp_client import FMPClient
    import yfinance as yf

    client = FMPClient(api_key=api_key, cache_root=ROOT / ".cache" / "fmp_diag")

    print("=" * 72)
    print(f"EU/Asia Data Coverage Diagnostic — {date.today().isoformat()}")
    print("=" * 72)
    print()

    results = {}

    for ticker in ALL_TICKERS:
        market = "EU" if ticker in EU_TICKERS else "ASIA"
        print(f"[{market}] {ticker}")
        row: dict = {"ticker": ticker, "market": market}

        # ── FMP insider ───────────────────────────────────────────────────────
        try:
            usd, days = client.get_insider_purchases(ticker, lookback_days=180)
            row["fmp_insider_usd"] = usd
            row["fmp_insider_days"] = days
            print(f"  FMP insider:  ${usd:,.0f} total, {days}d since last")
        except Exception as exc:
            row["fmp_insider_usd"] = None
            row["fmp_insider_days"] = None
            print(f"  FMP insider:  ERROR — {exc}")

        time.sleep(1.1)  # rate limit

        # ── FMP news ─────────────────────────────────────────────────────────
        try:
            articles = client.get_news_raw_articles(ticker)
            row["fmp_news_count"] = len(articles)
            if articles:
                sentiments = [a.get("sentiment", "?") for a in articles[:5]]
                print(f"  FMP news:     {len(articles)} articles, sample sentiments: {sentiments}")
            else:
                print("  FMP news:     0 articles")
        except Exception as exc:
            row["fmp_news_count"] = None
            print(f"  FMP news:     ERROR — {exc}")

        time.sleep(1.1)

        # ── FMP quote ────────────────────────────────────────────────────────
        try:
            q = client.get_quote(ticker)
            row["fmp_quote_ok"] = bool(q)
            row["fmp_price"] = q.get("price")
            row["fmp_mktcap"] = q.get("marketCap")
            print(f"  FMP quote:    price={q.get('price')}, mktcap={q.get('marketCap'):,.0f}" if q.get("marketCap") else f"  FMP quote:    {q}")
        except Exception as exc:
            row["fmp_quote_ok"] = False
            print(f"  FMP quote:    ERROR — {exc}")

        time.sleep(1.1)

        # ── yfinance price history ────────────────────────────────────────────
        try:
            df = yf.download(ticker, period="13mo", progress=False, auto_adjust=True)
            n_bars = len(df) if df is not None else 0
            row["yf_bars"] = n_bars
            has_12_1m = n_bars >= 252
            row["yf_has_12_1m"] = has_12_1m
            if n_bars > 0:
                vol_cols = [c for c in df.columns if "Volume" in str(c)]
                has_vol = len(vol_cols) > 0 and df[vol_cols[0]].sum() > 0
                row["yf_has_volume"] = has_vol
                print(f"  yfinance:     {n_bars} bars, 12-1m OK={has_12_1m}, volume={has_vol}")
            else:
                row["yf_has_volume"] = False
                print("  yfinance:     0 bars (no data)")
        except Exception as exc:
            row["yf_bars"] = None
            row["yf_has_12_1m"] = False
            row["yf_has_volume"] = False
            print(f"  yfinance:     ERROR — {exc}")

        print()
        results[ticker] = row
        time.sleep(0.5)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'Ticker':<14} {'Mkt':<5} {'FMP_insider$':<15} {'FMP_news':<10} {'FMP_quote':<10} {'yf_bars':<9} {'12-1m'}")
    print("-" * 72)
    for ticker, r in results.items():
        ins_usd = f"${r.get('fmp_insider_usd') or 0:>10,.0f}" if r.get("fmp_insider_usd") is not None else "       FAIL"
        news    = str(r.get("fmp_news_count", "FAIL")).rjust(8)
        quote   = "YES" if r.get("fmp_quote_ok") else "NO"
        yf_b    = str(r.get("yf_bars", "FAIL")).rjust(7)
        m12     = "YES" if r.get("yf_has_12_1m") else "NO"
        print(f"{ticker:<14} {r['market']:<5} {ins_usd}   {news}   {quote:<10} {yf_b}   {m12}")

    print()
    print("CONCLUSIONS FOR market_config.py:")
    print("-" * 72)

    eu_insider_any    = any(results[t].get("fmp_insider_usd") for t in EU_TICKERS)
    asia_insider_any  = any(results[t].get("fmp_insider_usd") for t in ASIA_TICKERS)
    eu_insider_all    = all(results[t].get("fmp_insider_usd") is not None for t in EU_TICKERS)
    asia_insider_all  = all(results[t].get("fmp_insider_usd") is not None for t in ASIA_TICKERS)
    eu_news_any       = any((results[t].get("fmp_news_count") or 0) > 0 for t in EU_TICKERS)
    asia_news_any     = any((results[t].get("fmp_news_count") or 0) > 0 for t in ASIA_TICKERS)
    eu_yf_12_1m       = sum(1 for t in EU_TICKERS if results[t].get("yf_has_12_1m"))
    asia_yf_12_1m     = sum(1 for t in ASIA_TICKERS if results[t].get("yf_has_12_1m"))

    print(f"EU   — FMP insider coverage: {'PARTIAL' if eu_insider_any and not eu_insider_all else 'FULL' if eu_insider_all else 'NONE'} ({sum(1 for t in EU_TICKERS if results[t].get('fmp_insider_usd'))}/{len(EU_TICKERS)} tickers have data)")
    print(f"EU   — FMP news coverage:    {'YES' if eu_news_any else 'NO'} ({sum(1 for t in EU_TICKERS if (results[t].get('fmp_news_count') or 0) > 0)}/{len(EU_TICKERS)} tickers)")
    print(f"EU   — yfinance 12-1m bars:  {eu_yf_12_1m}/{len(EU_TICKERS)} tickers have ≥252 bars")
    print()
    print(f"ASIA — FMP insider coverage: {'PARTIAL' if asia_insider_any and not asia_insider_all else 'FULL' if asia_insider_all else 'NONE'} ({sum(1 for t in ASIA_TICKERS if results[t].get('fmp_insider_usd'))}/{len(ASIA_TICKERS)} tickers have data)")
    print(f"ASIA — FMP news coverage:    {'YES' if asia_news_any else 'NO'} ({sum(1 for t in ASIA_TICKERS if (results[t].get('fmp_news_count') or 0) > 0)}/{len(ASIA_TICKERS)} tickers)")
    print(f"ASIA — yfinance 12-1m bars:  {asia_yf_12_1m}/{len(ASIA_TICKERS)} tickers have ≥252 bars")
    print()
    print("Recommendation:")
    if not eu_insider_any:
        print("  EU: Remove insider_conviction_score and insider_breadth_score from MARKET_FACTORS[EUROPE].")
        print("      Only momentum_long, volume_attention, news_sentiment, news_buzz available.")
    elif not eu_insider_all:
        print("  EU: Insider available for some tickers — keep in MARKET_FACTORS but score=0.0 for missing.")
    else:
        print("  EU: Full insider coverage — keep insider in MARKET_FACTORS[EUROPE].")

    if not asia_insider_any:
        print("  ASIA: Remove insider from MARKET_FACTORS[ASIA]. Momentum + news only.")
    elif not asia_insider_all:
        print("  ASIA: Partial insider coverage — keep in MARKET_FACTORS but expect 0.0 for missing.")
    else:
        print("  ASIA: Full insider coverage — keep.")
    print()


if __name__ == "__main__":
    run_diagnostic()
