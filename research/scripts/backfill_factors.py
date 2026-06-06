# Path: research/scripts/backfill_factors.py
"""Reconstruct 52 weeks of factor scores from FMP historical endpoints.

Run from repo root:
    FMP_API_KEY=your_key python research/scripts/backfill_factors.py

Output: research/data/backfill/factor_scores.ndjson
Each line: {"ticker": "AAPL", "snapshot_date": "2025-08-01",
            "insider_conviction": 0.72, ..., "forward_return_21d": 0.034}
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

import requests

# Import production scoring functions — ensures backfill uses identical logic.
from regime_trader.scoring.momentum_signals import (
    score_momentum_long,
    score_volume_attention,
)
from regime_trader.scoring.insider_signals import (
    score_insider_conviction,
    score_insider_breadth,
)
from regime_trader.scoring.news_signals import (
    score_news_sentiment,
    score_news_buzz,
)
from regime_trader.scoring.analyst import _score_record as score_analyst_record

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("backfill")

_FMP_BASE = "https://financialmodelingprep.com/stable"
_API_KEY = os.environ.get("FMP_API_KEY", "")
_RATE_LIMIT_DELAY = 1.0 / 30  # 30 req/s to stay safe
_OUT = Path("research/data/backfill/factor_scores.ndjson")

UNIVERSE_CSV = Path("config/universe.csv")
CONGRESS_URL = (
    "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/"
    "aggregate/all_transactions.json"
)


def _fmp_get(endpoint: str, params: dict) -> Any:
    """Single FMP stable/ GET with rate limiting."""
    params["apikey"] = _API_KEY
    url = f"{_FMP_BASE}/{endpoint}"
    time.sleep(_RATE_LIMIT_DELAY)
    r = requests.get(url, params=params, timeout=15)
    if r.status_code in (401, 403, 404):
        log.warning("FMP %s returned %d — skipping", endpoint, r.status_code)
        return None
    r.raise_for_status()
    return r.json()


def _load_universe() -> list[str]:
    import csv
    tickers = []
    with open(UNIVERSE_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            tickers.append(row["ticker"].strip())
    return tickers


def _fridays(n_weeks: int = 52) -> list[date]:
    """Last n_weeks Fridays, oldest first, excluding last 21 days (need fwd return)."""
    today = date.today()
    fridays = []
    d = today - timedelta(days=22)  # leave 21 trading days for forward return
    while len(fridays) < n_weeks:
        if d.weekday() == 4:  # Friday
            fridays.append(d)
        d -= timedelta(days=1)
    return list(reversed(fridays))


def fetch_prices(ticker: str, from_date: date, to_date: date) -> list[dict]:
    """Returns list of {date, close, volume} sorted newest-first."""
    data = _fmp_get(
        "historical-price-eod/full",
        {"symbol": ticker, "from": str(from_date), "to": str(to_date)},
    )
    if not data or not isinstance(data, list):
        return []
    return data


def compute_momentum_at(
    prices: list[dict], snapshot_date: date
) -> tuple[float, float]:
    """Return (return_12_1m, spy_relative_vol_ratio) at snapshot_date.

    prices: list of {date: str, close: float, volume: float}, newest-first.
    Returns (return_12_1m_raw, volume_5d_90d_ratio).
    """
    # Filter to prices on or before snapshot_date
    eligible = [p for p in prices if p["date"] <= str(snapshot_date)]
    if len(eligible) < 252:
        return (None, None)

    # Price at t (snapshot), t-21 (skip), t-252 (12m ago)
    p_t = float(eligible[0]["close"])
    p_skip = float(eligible[min(21, len(eligible) - 1)]["close"])
    p_12m = float(eligible[min(252, len(eligible) - 1)]["close"])

    if p_skip <= 0 or p_12m <= 0:
        return (None, None)

    return_12_1m = (p_skip - p_12m) / p_12m  # skip-month

    # Volume: 5d avg / 90d avg
    vols = [float(p.get("volume", 0)) for p in eligible[:90] if p.get("volume")]
    if len(vols) < 10:
        vol_ratio = None
    else:
        vol_5d = sum(vols[:5]) / 5
        vol_90d = sum(vols[:90]) / len(vols[:90])
        vol_ratio = vol_5d / vol_90d if vol_90d > 0 else None

    return (return_12_1m, vol_ratio)


def fetch_spy_return(from_date: date, to_date: date) -> float:
    """SPY return over [from_date, to_date] for relative momentum."""
    prices = fetch_prices("SPY", from_date - timedelta(days=400), to_date)
    eligible_from = [p for p in prices if p["date"] >= str(from_date)]
    eligible_to = [p for p in prices if p["date"] <= str(to_date)]
    if not eligible_from or not eligible_to:
        return 0.0
    p_start = float(eligible_from[-1]["close"])
    p_end = float(eligible_to[0]["close"])
    return (p_end - p_start) / p_start if p_start > 0 else 0.0


def compute_forward_return(
    prices: list[dict], snapshot_date: date, horizon: int = 21
) -> Optional[float]:
    """(price at snapshot+horizon - price at snapshot) / price at snapshot."""
    target_date = snapshot_date + timedelta(days=horizon + 5)  # buffer for weekends
    at_snapshot = [p for p in prices if p["date"] <= str(snapshot_date)]
    after_snapshot = [
        p for p in prices
        if str(snapshot_date) < p["date"] <= str(target_date)
    ]
    if not at_snapshot or not after_snapshot:
        return None
    p0 = float(at_snapshot[0]["close"])
    p1 = float(after_snapshot[-1]["close"])  # closest date after horizon
    return (p1 - p0) / p0 if p0 > 0 else None
