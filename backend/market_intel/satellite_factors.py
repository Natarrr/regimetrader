"""backend/market_intel/satellite_factors.py

Satellite enrichment layer for the edgar_3x pipeline.
Computes two supplementary signals:
  - Seasonal cyclicality  (win-rate / median return in the current calendar month)
  - Share cannibals       (buyback yield filtered by P/E and price proximity to 52w-low)

Called after generate_top_lists.py; writes logs/satellite_insights.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

# ── Tuning knobs ──────────────────────────────────────────────────────────────
MIN_MONTHLY_OBSERVATIONS = 8     # minimum historical month-samples for cyclicality
PE_MAX                   = 25.0  # P/E ratio ceiling for cannibal filter
PRICE_VS_52W_LOW_MAX     = 1.25  # price must be < 125% of 52-week low
TOP_N                    = 3     # tickers returned by each function

from regime_trader.services.fmp_client import FMPClient as _FMPClient  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────────
# Cyclicality
# ─────────────────────────────────────────────────────────────────────────────

def get_top_cyclical(tickers: list[str]) -> list[dict]:
    """Return up to TOP_N tickers with best win-rate in the current calendar month.

    Data source: FMP stable/historical-price-eod/full (10 years daily OHLCV).
    Monthly returns computed from first/last close of each month's trading days.
    Replaces yfinance batch monthly download.
    """
    from regime_trader.services.fmp_client import FMPClient, fmp_prices_to_arrays  # noqa: PLC0415

    client = FMPClient()
    current_month = datetime.now(timezone.utc).month
    # 10 years ≈ 2520 trading days
    results: list[dict] = []

    for ticker in tickers:
        try:
            rows = client.get_historical_prices(ticker, limit=2520)
            closes, _, dates = fmp_prices_to_arrays(rows)
            if not closes or len(closes) < 20:
                continue

            # Group by year-month → collect first and last close of each month
            monthly: dict[str, list[float]] = {}
            for i, d in enumerate(dates):
                ym = d[:7]  # "YYYY-MM"
                monthly.setdefault(ym, []).append(closes[i])

            # Filter to current calendar month only
            month_str = f"-{current_month:02d}"
            month_closes = {ym: v for ym, v in monthly.items() if ym.endswith(month_str)}
            if len(month_closes) < MIN_MONTHLY_OBSERVATIONS:
                continue

            wins = 0
            returns: list[float] = []
            for v in month_closes.values():
                if len(v) < 2:
                    continue
                first, last = v[0], v[-1]
                ret = (last - first) / first if first != 0 else 0.0
                returns.append(ret)
                if last > first:
                    wins += 1

            if not returns:
                continue

            returns.sort()
            mid = len(returns) // 2
            median_ret = returns[mid] if len(returns) % 2 else (returns[mid-1] + returns[mid]) / 2

            results.append({
                "ticker":        ticker,
                "win_rate":      round(wins / len(returns), 4),
                "median_return": round(median_ret, 4),
                "years":         len(returns),
            })
        except Exception as exc:
            log.warning("cyclical: skipping %s — %s", ticker, exc)

    results.sort(key=lambda r: (-r["win_rate"], -r["median_return"]))
    return results[:TOP_N]


# ─────────────────────────────────────────────────────────────────────────────
# Share cannibals
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_fmp_info(ticker: str, fmp_key: str) -> dict:
    """Fetch price fundamentals from FMP stable/quote + stable/ratios-ttm.

    Returns a dict matching the fields used by get_top_cannibals:
        trailingPE     ← peRatioTTM from ratios-ttm
        currentPrice   ← price from quote
        fiftyTwoWeekLow ← yearLow from quote
    """
    from regime_trader.services.fmp_client import FMPClient  # noqa: PLC0415

    client = FMPClient(api_key=fmp_key)
    quote   = client.get_quote(ticker)
    ratios  = client.get_ratios_ttm(ticker)

    pe = ratios.get("peRatioTTM") or ratios.get("priceEarningsRatioTTM")
    return {
        "trailingPE":       float(pe) if pe is not None else None,
        "currentPrice":     float(quote.get("price") or 0),
        "fiftyTwoWeekLow":  float(quote.get("yearLow") or 0),
    }


def get_top_cannibals(
    tickers: list[str],
    fmp_key: str,
    market_caps: dict[str, float],
) -> list[dict]:
    """Return up to TOP_N tickers ranked by trailing 4-quarter buyback yield.

    Filters: trailingPE < PE_MAX and currentPrice < PRICE_VS_52W_LOW_MAX * fiftyTwoWeekLow.
    Data sources: FMPClient (quote for P/E + 52w-low, cash-flow for buybacks).
    Fully on FMP Ultimate — no yfinance.
    """
    if not fmp_key:
        log.warning("FMP_API_KEY absent — skipping cannibal scan")
        return []

    try:
        fmp = _FMPClient(api_key=fmp_key)
        results: list[dict] = []

        for ticker in tickers:
            try:
                info = _fetch_fmp_info(ticker, fmp_key)
            except Exception as exc:
                log.warning(
                    "cannibal: yf.info failed for %s — %s", ticker, type(exc).__name__,
                )
                continue

            # Filter 1: P/E
            try:
                pe = info["trailingPE"]
                if pe is None or float(pe) >= PE_MAX:
                    continue
                pe = float(pe)
            except (KeyError, AttributeError, TypeError, ValueError):
                continue

            # Filter 2: price vs 52-week low
            try:
                price    = float(info["currentPrice"])
                low_52w  = float(info["fiftyTwoWeekLow"])
                if low_52w <= 0 or price >= PRICE_VS_52W_LOW_MAX * low_52w:
                    continue
                price_vs_52w = round(price / low_52w, 4)
            except (KeyError, AttributeError, TypeError, ValueError):
                continue

            # FMP cash-flow — 4 trailing quarters via FMPClient (cache + rate-limit)
            try:
                quarters = fmp.get_cash_flow_statements(ticker, limit=4)
                if not quarters:
                    continue
            except Exception as exc:
                log.warning(
                    "cannibal: FMP cash-flow failed for %s — %s",
                    ticker, type(exc).__name__,
                )
                continue

            total_repurchased = sum(
                abs(q.get("repurchasedCommonStock", 0) or 0) for q in quarters
            )

            mktcap = market_caps.get(ticker)
            if not mktcap or mktcap <= 0:
                continue

            buyback_yield = round(total_repurchased / mktcap, 6)
            results.append(
                {
                    "ticker":           ticker,
                    "buyback_yield":    buyback_yield,
                    "pe":               round(pe, 2),
                    "price_vs_52w_low": price_vs_52w,
                }
            )

        results.sort(key=lambda r: -r["buyback_yield"])
        return results[:TOP_N]

    except Exception as exc:
        log.warning("get_top_cannibals failed entirely: %s", type(exc).__name__)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate satellite insights")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    args.log_dir.mkdir(parents=True, exist_ok=True)
    top_lists_path = args.log_dir / "top_lists.json"    # merged output from cook_toplists.py
    if not top_lists_path.exists():
        log.error("top_lists.json not found at %s", top_lists_path)
        raise SystemExit(1)

    top_lists: dict = json.loads(top_lists_path.read_text(encoding="utf-8"))

    # Collect unique tickers and market-cap map from all markets + all tiers.
    # top_buys_usa/europe/asia are used instead of top_buys so that EU/Asia
    # tickers are always included regardless of cross-market score ranking.
    all_entries: list[dict] = (
        (top_lists.get("top_buys_usa") or top_lists.get("top_buys") or [])
        + (top_lists.get("top_buys_europe") or [])
        + (top_lists.get("top_buys_asia") or [])
        + (top_lists.get("mid_caps") or [])
        + (top_lists.get("small_caps") or [])
    )
    tickers: list[str] = list(dict.fromkeys(
        e["ticker"] for e in all_entries if e.get("ticker")
    ))
    market_caps: dict[str, float] = {
        e["ticker"]: float(e.get("market_cap") or 0)
        for e in all_entries
        if e.get("ticker") and e.get("market_cap")
    }

    fmp_key: str = os.getenv("FMP_API_KEY", "")

    cyclical_success = True
    cannibal_success = True

    cyclicals = get_top_cyclical(tickers)
    if not cyclicals and tickers:
        cyclical_success = False

    cannibals = get_top_cannibals(tickers, fmp_key, market_caps)
    if not cannibals and tickers and fmp_key:
        cannibal_success = False

    if cyclical_success and cannibal_success:
        status = "success"
    elif not cyclical_success and not cannibal_success:
        status = "error"
    else:
        status = "partial"

    current_month = datetime.now(timezone.utc).strftime("%B")

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "month":        current_month,
        "status":       status,
        "cyclicals":    cyclicals,
        "cannibals":    cannibals,
    }

    out_path = args.log_dir / "satellite_insights.json"
    tmp_path = out_path.with_suffix(".json.tmp")
    try:
        tmp_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(out_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    log.info(
        "satellite_insights.json written — status=%s cyclicals=%d cannibals=%d",
        status, len(cyclicals), len(cannibals),
    )


if __name__ == "__main__":
    main()
