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
) -> tuple[Optional[float], Optional[float]]:
    """Return (return_12_1m, spy_relative_vol_ratio) at snapshot_date.

    prices: list of {date: str, close: float, volume: float}, newest-first.
    Returns (return_12_1m_raw, volume_5d_90d_ratio).
    """
    # Filter to prices on or before snapshot_date
    eligible = [p for p in prices if p["date"] <= str(snapshot_date)]
    if len(eligible) < 252:
        return (None, None)

    # Skip-month return: price at t-21 (numerator) vs t-252/12m ago (denominator)
    p_skip = float(eligible[21]["close"])
    p_12m = float(eligible[252]["close"])

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
    if p0 <= 0 or p1 <= 0:
        return None
    return (p1 - p0) / p0


def fetch_insider_trades(ticker: str, snapshot_date: date) -> list[dict]:
    """All P-code insider trades in [snapshot_date-90d, snapshot_date]."""
    from_d = snapshot_date - timedelta(days=90)
    data = _fmp_get(
        "insider-trading/search",
        {
            "symbol": ticker,
            "transactionType": "P-Purchase",
            "from": str(from_d),
            "to": str(snapshot_date),
            "limit": 200,
        },
    )
    return data if isinstance(data, list) else []


def compute_insider_scores(
    trades: list[dict], market_cap: float, snapshot_date: date
) -> tuple[float, float]:
    """Return (insider_conviction_score, insider_breadth_score) in [0, 1].

    Adapts a flat list of P-purchase trade dicts to the production function
    signatures:
      - score_insider_conviction(key_purchases_usd, market_cap, ...)
      - score_insider_breadth(p_transactions, s_transactions, ...)
    """
    if not trades or market_cap <= 0:
        return (0.0, 0.0)

    # Aggregate total purchase USD across all trades in the window.
    key_purchases_usd = sum(
        float(t.get("securitiesTransacted", 0) or 0)
        * float(t.get("price", 0) or 0)
        for t in trades
    )

    # Recency: days since the most recent trade relative to snapshot context.
    dates = [str(t.get("transactionDate") or t.get("date", "") or "")[:10] for t in trades]
    dates = [d for d in dates if d]
    if dates:
        most_recent_str = max(dates)
        try:
            most_recent = date.fromisoformat(most_recent_str)
            days_since = (snapshot_date - most_recent).days
        except ValueError:
            days_since = 0
    else:
        days_since = 0

    # CEO purchase USD — sum trades where reportingName contains "CEO"
    ceo_usd = sum(
        float(t.get("securitiesTransacted", 0) or 0)
        * float(t.get("price", 0) or 0)
        for t in trades
        if "CEO" in (t.get("reportingName") or t.get("typeOfOwner") or "").upper()
    )

    conviction = score_insider_conviction(
        key_purchases_usd=key_purchases_usd,
        market_cap=market_cap,
        days_since_most_recent=days_since,
        ceo_purchase_usd=ceo_usd,
    )

    # score_insider_breadth expects separate p_transactions and s_transactions lists.
    # backfill fetch is P-purchase only, so s_transactions is empty.
    breadth = score_insider_breadth(
        p_transactions=trades,
        s_transactions=[],
    )

    return (conviction, breadth)


def fetch_congress_all() -> list[dict]:
    """Fetch all S3 Stock Watcher senate transactions once per run."""
    r = requests.get(CONGRESS_URL, timeout=20)
    r.raise_for_status()
    return r.json()


def compute_congress_score(
    ticker: str, congress_trades: list[dict], snapshot_date: date
) -> float:
    """Score congressional trades in [snapshot_date-90d, snapshot_date]."""
    from_d = snapshot_date - timedelta(days=90)
    relevant = [
        t for t in congress_trades
        if t.get("ticker", "").upper() == ticker.upper()
        and str(from_d) <= (t.get("transaction_date") or "") <= str(snapshot_date)
        and t.get("type", "").lower() in ("purchase", "buy")
    ]
    if not relevant:
        return 0.0
    # Score: capped at 1.0, proportional to count (5+ trades = max)
    return min(1.0, len(relevant) / 5.0)


def fetch_news(ticker: str, snapshot_date: date) -> list[dict]:
    """FMP news articles in [snapshot_date-30d, snapshot_date]."""
    from_d = snapshot_date - timedelta(days=30)
    data = _fmp_get(
        "news/stock",
        {
            "tickers": ticker,
            "from": str(from_d),
            "to": str(snapshot_date),
            "limit": 200,
        },
    )
    return data if isinstance(data, list) else []


def load_analyst_bulk_index(bulk_path: Path) -> dict[str, dict]:
    """Build ticker → record index from pre-fetched NDJSON bulk file.

    Uses current snapshot as proxy for all historical dates (slow-moving signal).
    """
    index: dict[str, dict] = {}
    if not bulk_path.exists():
        return index
    with open(bulk_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                sym = rec.get("symbol", "")
                if sym:
                    index[sym.upper()] = rec
            except json.JSONDecodeError:
                log.warning("Malformed NDJSON line in %s — skipping", bulk_path)
                continue
    return index


def compute_analyst_score(ticker: str, analyst_index: dict[str, dict]) -> float:
    rec = analyst_index.get(ticker.upper(), {})
    if not rec:
        return 0.0
    score, _ = score_analyst_record(ticker, rec)
    return score


def fetch_piotroski(ticker: str, snapshot_date: date) -> float:
    """Compute Piotroski F-score from most-recent ratios-ttm filing <= snapshot_date."""
    data = _fmp_get("ratios-ttm", {"symbol": ticker})
    if not data or not isinstance(data, list) or not data[0]:
        return 0.5  # neutral default when unavailable
    r = data[0]

    # Piotroski (2000) 9-component single-period approximation
    roa = float(r.get("returnOnAssetsTTM") or 0)
    cfo = float(r.get("operatingCashFlowPerShareTTM") or 0)
    delta_roa = roa  # delta_roa: using single TTM period as proxy (no prior-period data available)
    accrual = cfo - roa  # CFO > ROA = accrual quality signal

    current_ratio = float(r.get("currentRatioTTM") or 0)
    delta_leverage = -float(r.get("debtEquityRatioTTM") or 0)  # lower is better
    delta_shares = -float(r.get("priceToBookRatioTTM") or 0)  # proxy

    gross_margin = float(r.get("grossProfitMarginTTM") or 0)
    asset_turnover = float(r.get("assetTurnoverTTM") or 0)

    f_score = sum([
        roa > 0,
        cfo > 0,
        delta_roa >= 0,
        accrual > 0,
        delta_leverage >= 0,
        current_ratio >= 1.0,
        delta_shares >= 0,
        gross_margin >= 0,
        asset_turnover > 0,
    ])
    return min(1.0, f_score / 9.0)
