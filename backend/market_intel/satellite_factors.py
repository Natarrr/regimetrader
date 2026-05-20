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
import time
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

# ── Tuning knobs ──────────────────────────────────────────────────────────────
MIN_MONTHLY_OBSERVATIONS = 8     # minimum historical month-samples for cyclicality
PE_MAX                   = 25.0  # P/E ratio ceiling for cannibal filter
PRICE_VS_52W_LOW_MAX     = 1.25  # price must be < 125% of 52-week low
TOP_N                    = 3     # tickers returned by each function

# ── FMP base URL ─────────────────────────────────────────────────────────────
_FMP_BASE = "https://financialmodelingprep.com"


# ─────────────────────────────────────────────────────────────────────────────
# Cyclicality
# ─────────────────────────────────────────────────────────────────────────────

def get_top_cyclical(tickers: list[str]) -> list[dict]:
    """Return up to TOP_N tickers with best win-rate in the current calendar month.

    Data source: yfinance batch download, 10 years of monthly OHLCV.
    Returns [] on any exception.
    """
    try:
        import pandas as pd
        import yfinance as yf

        current_month = datetime.now(timezone.utc).month

        raw = yf.download(
            tickers,
            period="10y",
            interval="1mo",
            auto_adjust=True,
            group_by="ticker",
            progress=False,
        )

        # Normalise index to DatetimeIndex (batch download can produce MultiIndex)
        if not isinstance(raw.index, pd.DatetimeIndex):
            raw.index = pd.to_datetime(raw.index.get_level_values(-1))

        results: list[dict] = []
        for ticker in tickers:
            try:
                # Extract per-ticker slice; column key depends on download shape
                try:
                    df = raw[[("Open", ticker), ("Close", ticker)]].copy()
                    df.columns = ["Open", "Close"]
                except KeyError:
                    if len(tickers) == 1:
                        df = raw[["Open", "Close"]].copy()
                    else:
                        log.warning("cyclical: column slice missing for %s — skipping", ticker)
                        continue

                filtered = df[df.index.month == current_month].dropna(
                    subset=["Open", "Close"]
                )
                if len(filtered) < MIN_MONTHLY_OBSERVATIONS:
                    continue

                wins = (filtered["Close"] > filtered["Open"]).sum()
                win_rate = float(wins / len(filtered))
                median_return = float(
                    ((filtered["Close"] - filtered["Open"]) / filtered["Open"]).median()
                )
                results.append(
                    {
                        "ticker":        ticker,
                        "win_rate":      round(win_rate, 4),
                        "median_return": round(median_return, 4),
                        "years":         len(filtered),
                    }
                )
            except Exception as exc:
                log.warning("cyclical: skipping %s — %s", ticker, exc)

        results.sort(key=lambda r: (-r["win_rate"], -r["median_return"]))
        return results[:TOP_N]

    except Exception as exc:
        log.warning("get_top_cyclical failed entirely: %s", exc)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Share cannibals
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_yf_info(ticker: str, max_retries: int = 3) -> dict:
    """Fetch yfinance .info with up to max_retries attempts."""
    import yfinance as yf

    last_exc: Exception = RuntimeError("no attempts")
    for attempt in range(max_retries):
        try:
            info = yf.Ticker(ticker).info
            if not isinstance(info, dict):
                raise AttributeError(f"yf.Ticker({ticker!r}).info returned non-dict")
            return info
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)   # 1s, 2s
    raise last_exc


def get_top_cannibals(
    tickers: list[str],
    fmp_key: str,
    market_caps: dict[str, float],
) -> list[dict]:
    """Return up to TOP_N tickers ranked by trailing 4-quarter buyback yield.

    Filters: trailingPE < PE_MAX and currentPrice < PRICE_VS_52W_LOW_MAX * fiftyTwoWeekLow.
    Data sources: yfinance .info (P/E, 52w-low), FMP cash-flow (repurchases), market_caps dict.
    Returns [] when fmp_key is absent or on any unrecoverable exception.
    """
    if not fmp_key:
        log.warning("FMP_API_KEY absent — skipping cannibal scan")
        return []

    try:
        import requests as req

        results: list[dict] = []

        for ticker in tickers:
            try:
                info = _fetch_yf_info(ticker)
            except Exception as exc:
                log.warning("cannibal: yf.info failed for %s — %s", ticker, exc)
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

            # FMP cash-flow — 4 trailing quarters
            try:
                url = (
                    f"{_FMP_BASE}/stable/cash-flow-statement"
                    f"?symbol={ticker}&period=quarter&limit=4&apikey={fmp_key}"
                )
                resp = req.get(url, timeout=15.0)
                resp.raise_for_status()
                quarters: list[dict] = resp.json()
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
        log.warning("get_top_cannibals failed entirely: %s", exc)
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
    top_lists_path = args.log_dir / "top_lists.json"
    if not top_lists_path.exists():
        log.error("top_lists.json not found at %s", top_lists_path)
        raise SystemExit(1)

    top_lists: dict = json.loads(top_lists_path.read_text(encoding="utf-8"))

    # Collect unique tickers and market-cap map from all three universe tiers
    all_entries: list[dict] = (
        (top_lists.get("top_buys") or [])
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
