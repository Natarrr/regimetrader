"""regime_trader.research.forward_returns — look-ahead-safe forward return fetcher.

Fetches N-day forward total returns for a universe of tickers on a given as-of date.
Results are cached per (date, horizon) to avoid repeated yfinance calls.

Look-ahead bias prevention: we fetch price at `as_of_date` and `as_of_date + horizon_days`
using only data that would have been observable at those points in time.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import date, timedelta
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)

_CACHE_SUBDIR = ".cache/forward_returns"
_YF_BATCH_SIZE = 40          # yfinance handles ~40 tickers per download reliably
_YF_RATE_DELAY = 1.0         # seconds between batches
_HORIZON_DAYS  = 21          # ~1 calendar month forward return (default)


class ForwardReturn(NamedTuple):
    ticker: str
    as_of_date: date
    horizon_days: int
    forward_return: float     # raw decimal, e.g. 0.043 = +4.3%


def fetch_forward_returns(
    tickers: list[str],
    as_of_date: date,
    horizon_days: int = _HORIZON_DAYS,
    cache_root: Path | None = None,
) -> dict[str, float]:
    """Return {ticker: forward_return} for the given universe as of as_of_date.

    Look-ahead safe: uses close prices on as_of_date and as_of_date + horizon_days.
    Tickers with insufficient data are silently dropped.

    Args:
        tickers: Universe of ticker symbols.
        as_of_date: Formation date (price[0]). Must be in the past.
        horizon_days: Calendar days to the target date (price[1]).
        cache_root: Project root for cache dir. Defaults to CWD.

    Returns:
        Dict of {ticker: forward_decimal_return}. Missing tickers excluded.
    """
    import yfinance as yf

    if cache_root is None:
        cache_root = Path.cwd()

    cached = _load_cache(cache_root, as_of_date, horizon_days)
    if cached is not None:
        # Filter to requested tickers only
        return {t: cached[t] for t in tickers if t in cached}

    target_date = as_of_date + timedelta(days=horizon_days)
    # Download window: 2 extra buffer days for weekends/holidays on each side
    fetch_start = as_of_date - timedelta(days=3)
    fetch_end   = target_date + timedelta(days=3)

    results: dict[str, float] = {}

    for batch_start in range(0, len(tickers), _YF_BATCH_SIZE):
        batch = tickers[batch_start : batch_start + _YF_BATCH_SIZE]
        try:
            raw = yf.download(
                batch,
                start=fetch_start.isoformat(),
                end=fetch_end.isoformat(),
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if raw.empty:
                continue

            # Multi-ticker: columns are (field, ticker); single-ticker: flat columns
            if isinstance(raw.columns, type(raw.columns)) and hasattr(raw.columns, "levels"):
                closes = raw["Close"] if "Close" in raw.columns.get_level_values(0) else raw
            else:
                closes = raw[["Close"]] if "Close" in raw.columns else raw

            for ticker in batch:
                try:
                    if hasattr(closes, "columns") and ticker in closes.columns:
                        series = closes[ticker].dropna()
                    elif len(batch) == 1:
                        series = closes.iloc[:, 0].dropna()
                    else:
                        continue

                    # Find closest available price on or just after each target date
                    p0 = _nearest_price(series, as_of_date, direction="forward")
                    p1 = _nearest_price(series, target_date, direction="forward")
                    if p0 is None or p1 is None or p0 == 0:
                        continue
                    results[ticker] = (p1 - p0) / p0
                except Exception as exc:
                    logger.debug("forward_returns: skip %s: %s", ticker, exc)

        except Exception as exc:
            logger.warning("forward_returns: batch download failed: %s", exc)

        if batch_start + _YF_BATCH_SIZE < len(tickers):
            time.sleep(_YF_RATE_DELAY)

    _save_cache(cache_root, as_of_date, horizon_days, results)
    return {t: results[t] for t in tickers if t in results}


# ── Internal helpers ─────────────────────────────────────────────────────────

def _nearest_price(series: "pd.Series", target: date, direction: str = "forward") -> float | None:
    """Return closest price to target date within a 5-day window."""
    import pandas as pd

    target_ts = pd.Timestamp(target)
    window = 5  # calendar days
    if direction == "forward":
        mask = (series.index >= target_ts) & (series.index <= target_ts + pd.Timedelta(days=window))
    else:
        mask = (series.index >= target_ts - pd.Timedelta(days=window)) & (series.index <= target_ts)

    subset = series[mask]
    if subset.empty:
        return None
    if direction == "forward":
        return float(subset.iloc[0])
    return float(subset.iloc[-1])


def _cache_path(cache_root: Path, as_of_date: date, horizon_days: int) -> Path:
    return cache_root / _CACHE_SUBDIR / f"{as_of_date.isoformat()}_{horizon_days}d.json"


def _load_cache(cache_root: Path, as_of_date: date, horizon_days: int) -> dict[str, float] | None:
    path = _cache_path(cache_root, as_of_date, horizon_days)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_cache(cache_root: Path, as_of_date: date, horizon_days: int, data: dict[str, float]) -> None:
    path = _cache_path(cache_root, as_of_date, horizon_days)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
