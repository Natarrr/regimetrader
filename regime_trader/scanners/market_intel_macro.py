"""regime_trader/market_intel_macro.py
Commodity and macro indicator data layer + conviction scoring.

Extracted from the inlined streamlit_app.py macro section.

Public API:
  fetch_commodity_prices(commodity)       -> dict | None
  fetch_macro_indicator(ticker)           -> dict | None
  calc_term_structure_score(data)         -> (score, label)
  calc_cot_proxy_score(data)              -> (score, label)
  calc_sentiment_score(etf, sent_map)     -> (score, label)
  calc_trend_score(data)                  -> (score, label)
  calc_macro_conviction(price_data, sent) -> dict
  check_macro_shocks(prices)              -> List[alert_dict]
  generate_macro_synthesis(...)           -> List[str]
  fetch_stock_pick_data(ticker, fallback) -> dict | None
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ── Static reference data ──────────────────────────────────────────────────────

COMMODITY_UNIVERSE: List[Dict[str, Any]] = [
    # cot_symbol: prefix to match FMP commitment-of-traders-report symbol field
    {"name": "Crude Oil",   "ticker": "CL=F", "etf": "USO",  "sector": "Energy",      "unit": "$/bbl",   "cot_symbol": "CL"},
    {"name": "Brent Crude", "ticker": "BZ=F", "etf": "BNO",  "sector": "Energy",      "unit": "$/bbl",   "cot_symbol": "BZ"},
    {"name": "Natural Gas", "ticker": "NG=F", "etf": "UNG",  "sector": "Energy",      "unit": "$/MMBtu", "cot_symbol": "NG"},
    {"name": "Gold",        "ticker": "GC=F", "etf": "GLD",  "sector": "Metals",      "unit": "$/oz",    "cot_symbol": "GC"},
    {"name": "Silver",      "ticker": "SI=F", "etf": "SLV",  "sector": "Metals",      "unit": "$/oz",    "cot_symbol": "SI"},
    {"name": "Copper",      "ticker": "HG=F", "etf": "CPER", "sector": "Metals",      "unit": "$/lb",    "cot_symbol": "HG"},
    {"name": "Wheat",       "ticker": "ZW=F", "etf": "WEAT", "sector": "Agriculture", "unit": "c/bu",    "cot_symbol": "ZW"},
    {"name": "Corn",        "ticker": "ZC=F", "etf": "CORN", "sector": "Agriculture", "unit": "c/bu",    "cot_symbol": "ZC"},
    {"name": "Soybeans",    "ticker": "ZS=F", "etf": "SOYB", "sector": "Agriculture", "unit": "c/bu",    "cot_symbol": "ZS"},
]

MACRO_INDICATORS: List[Dict[str, str]] = [
    {"name": "US 10Y Yield",  "ticker": "^TNX",     "unit": "%"},
    {"name": "Dollar Index",  "ticker": "DX-Y.NYB", "unit": "pts"},
    {"name": "VIX",           "ticker": "^VIX",     "unit": "pts"},
    {"name": "3M T-Bill",     "ticker": "^IRX",     "unit": "%"},
]

FUTURES_TO_ETF: Dict[str, str] = {c["ticker"]: c["etf"] for c in COMMODITY_UNIVERSE}

SECTOR_STOCKS: Dict[str, List[Dict[str, str]]] = {
    "Energy": [
        {"ticker": "XOM",  "name": "Exxon Mobil"},
        {"ticker": "CVX",  "name": "Chevron"},
        {"ticker": "PXD",  "name": "Pioneer Natural Resources"},
    ],
    "Metals": [
        {"ticker": "GOLD", "name": "Barrick Gold"},
        {"ticker": "NEM",  "name": "Newmont"},
        {"ticker": "FCX",  "name": "Freeport-McMoRan"},
    ],
    "Agriculture": [
        {"ticker": "ADM",  "name": "Archer-Daniels-Midland"},
        {"ticker": "DE",   "name": "John Deere"},
        {"ticker": "CTVA", "name": "Corteva"},
    ],
}

SECTOR_FALLBACKS: Dict[str, str] = {
    "XOM": "SHEL", "CVX": "SHEL", "PXD": "SHEL",
    "GOLD": "KL",  "NEM": "KL",   "FCX": "KL",
    "ADM": "BG",   "DE": "AGCO",  "CTVA": "MOS",
}

SECTOR_COMMODITY_MAP: Dict[str, List[str]] = {
    "Energy":      ["CL=F", "BZ=F"],
    "Metals":      ["GC=F", "HG=F"],
    "Agriculture": ["ZW=F", "ZC=F"],
}

STOCK_REGIME_MULT: Dict[str, float] = {
    "Mania":    1.00,
    "Euphoria": 1.15,
    "Bull":     1.20,
    "Neutral":  1.00,
    "Unknown":  0.90,
    "Bear":     0.65,
    "Panic":    0.40,
    "Crash":    0.00,
}

_BULL_WORDS: frozenset = frozenset([
    "beat", "beats", "exceed", "exceeds", "exceeded", "upgrade", "upgrades",
    "upgraded", "buy", "outperform", "outperforms", "strong", "record", "rally",
    "rallies", "surge", "surges", "surged", "gain", "gains", "growth", "bullish",
    "profit", "profits", "revenue", "raise", "raises", "raised", "tops",
    "topped", "jump", "jumps", "jumped", "soar", "soars", "soared", "boom",
    "positive", "breakthrough", "approval", "approved", "expands", "expansion",
])

_BEAR_WORDS: frozenset = frozenset([
    "miss", "misses", "missed", "downgrade", "downgrades", "downgraded",
    "sell", "underperform", "underperforms", "concern", "concerns", "decline",
    "declines", "declined", "weak", "weakness", "loss", "losses", "cut",
    "cuts", "cutting", "fall", "falls", "fell", "drop", "drops", "dropped",
    "recession", "layoff", "layoffs", "lawsuit", "fine", "fined", "warning",
    "warn", "warns", "risk", "risks", "volatile", "volatility", "below",
    "disappoints", "disappointing", "disappointed", "halt", "halted",
    "investigation", "probe", "fraud", "bankruptcy", "default",
])


# ── Data fetching ──────────────────────────────────────────────────────────────

def safe_download(ticker: str, period: str = "1y") -> Optional[Any]:
    """Download OHLCV data via FMP stable/historical-price-eod/full.

    Works for US, EU, Asia, indices (^VIX, ^TNX), ETFs and futures (CL=F, GC=F).
    Replaces yfinance.download — FMP Ultimate confirmed live for all these symbols.

    Args:
        ticker: FMP-compatible symbol (same as Yahoo Finance for most instruments).
        period: lookback hint ("1y", "2y", "5y", "10y", "60d", "13mo").

    Returns:
        _PriceArray with Close/Volume access, or None on failure.
    """
    from regime_trader.services.fmp_client import FMPClient, fmp_prices_to_arrays  # noqa: PLC0415
    _period_bars = {"1y": 252, "2y": 504, "5y": 1260, "10y": 2520, "60d": 60, "13mo": 280}
    limit = _period_bars.get(period, 280)
    try:
        rows = FMPClient().get_historical_prices(ticker, limit=limit)
        if not rows or len(rows) < 10:
            return None
        closes, volumes, dates = fmp_prices_to_arrays(rows)
        return _PriceArray(closes, volumes, dates)
    except Exception as exc:
        log.warning("safe_download(%s): %s", ticker, exc)
        return None


class _PriceArray:
    """Minimal DataFrame-like adapter over FMP price arrays for market_intel_macro callers."""

    class _Col:
        """List-backed column with the subset of pandas API used by this module."""

        def __init__(self, data: list) -> None:
            self._d = list(data)

        def dropna(self) -> "_PriceArray._Col":
            return _PriceArray._Col([x for x in self._d if x is not None])

        def __len__(self) -> int:
            return len(self._d)

        def __bool__(self) -> bool:
            return bool(self._d)

        @property
        def iloc(self) -> "_PriceArray._Iloc":
            return _PriceArray._Iloc(self._d)

        def rolling(self, w: int, min_periods: Optional[int] = None) -> "_PriceArray._Rolling":
            return _PriceArray._Rolling(self._d, w)

        def diff(self) -> "_PriceArray._Col":
            return _PriceArray._Col(
                [0.0] + [self._d[i] - self._d[i-1] for i in range(1, len(self._d))]
            )

        def clip(self, lower=None, upper=None) -> "_PriceArray._Col":
            d = list(self._d)
            if lower is not None:
                d = [max(lower, x) for x in d]
            if upper is not None:
                d = [min(upper, x) for x in d]
            return _PriceArray._Col(d)

        def replace(self, val, rep) -> "_PriceArray._Col":
            return _PriceArray._Col([rep if x == val else x for x in self._d])

        def mean(self) -> float:
            return sum(self._d) / max(len(self._d), 1)

        def __truediv__(self, other) -> "_PriceArray._Col":
            if isinstance(other, _PriceArray._Col):
                return _PriceArray._Col([a/b if b else 0 for a, b in zip(self._d, other._d)])
            return _PriceArray._Col([x / other for x in self._d])

        def __sub__(self, other) -> "_PriceArray._Col":
            if isinstance(other, _PriceArray._Col):
                return _PriceArray._Col([a - b for a, b in zip(self._d, other._d)])
            return _PriceArray._Col([x - other for x in self._d])

        def __neg__(self) -> "_PriceArray._Col":
            return _PriceArray._Col([-x for x in self._d])

    class _Iloc:
        def __init__(self, data: list) -> None:
            self._d = data
        def __getitem__(self, key):
            return self._d[key]
        def mean(self) -> float:
            return sum(self._d) / max(len(self._d), 1)

    class _Rolling:
        def __init__(self, data: list, w: int) -> None:
            self._d, self._w = data, w
        def mean(self) -> "_PriceArray._Col":
            result = []
            for i in range(len(self._d)):
                start = max(0, i - self._w + 1)
                window = self._d[start:i+1]
                result.append(sum(window) / len(window) if window else float("nan"))
            return _PriceArray._Col(result)
        def std(self) -> "_PriceArray._Col":
            import math as _math
            result = []
            for i in range(len(self._d)):
                start = max(0, i - self._w + 1)
                window = self._d[start:i+1]
                if len(window) < 2:
                    result.append(float("nan"))
                else:
                    mu = sum(window) / len(window)
                    result.append(_math.sqrt(sum((x-mu)**2 for x in window) / len(window)))
            return _PriceArray._Col(result)

    def __init__(self, closes: list, volumes: list, dates: list) -> None:
        self._c, self._v, self._d = closes, volumes, dates

    def __len__(self) -> int:
        return len(self._c)

    @property
    def empty(self) -> bool:
        return len(self._c) == 0

    def __getitem__(self, key: str) -> "_PriceArray._Col":
        if key in ("Close", "close"):
            return _PriceArray._Col(self._c)
        if key in ("Volume", "volume"):
            return _PriceArray._Col(self._v)
        if key in ("Open", "High", "Low"):
            return _PriceArray._Col(self._c)   # proxy: use close
        raise KeyError(key)

    @property
    def columns(self) -> list:
        return ["Close", "Volume", "Open", "High", "Low"]

    def squeeze(self) -> "_PriceArray._Col":
        return _PriceArray._Col(self._c)

    def dropna(self, subset=None) -> "_PriceArray":
        return self


def fetch_commodity_prices(commodity: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Fetch price data for a commodity, falling back from futures to ETF ticker.

    Shiller (2013 Nobel) — long price series for cyclically-adjusted valuation.

    Args:
        commodity: Entry from COMMODITY_UNIVERSE (must have 'ticker', 'etf').

    Returns:
        Dict with price, returns, moving averages, RSI, ATR; or None.
    """
    try:
        import pandas as pd
    except ImportError:
        log.warning("pandas not installed — fetch_commodity_prices unavailable")
        return None

    for ticker, source in [(commodity["ticker"], "futures"), (commodity["etf"], "etf")]:
        if not ticker:
            continue
        df = safe_download(ticker)
        if df is None:
            continue

        close = df["Close"].squeeze()
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        close = close.dropna()
        if len(close) < 20:
            continue

        price = float(close.iloc[-1])
        prev1 = float(close.iloc[-2]) if len(close) >= 2 else price
        prev5 = float(close.iloc[-6]) if len(close) >= 6 else price
        prev20 = float(close.iloc[-21]) if len(close) >= 21 else price

        window = min(252, len(close))
        high_52 = float(close.rolling(window).max().iloc[-1])
        low_52 = float(close.rolling(window).min().iloc[-1])
        pct_52 = (price - low_52) / max(high_52 - low_52, 1e-9)

        sma20 = float(close.rolling(20).mean().iloc[-1]) if len(close) >= 20 else price
        sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else price
        sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else price

        delta = close.diff().dropna()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1e-9)
        rsi14 = float(max(0.0, min(100.0, (100 - 100 / (1 + rs)).iloc[-1])))

        try:
            hi = df["High"].squeeze()
            lo = df["Low"].squeeze()
            if isinstance(hi, pd.DataFrame):
                hi = hi.iloc[:, 0]
            if isinstance(lo, pd.DataFrame):
                lo = lo.iloc[:, 0]
            tr = pd.concat(
                [(hi - lo), (hi - close.shift()).abs(), (lo - close.shift()).abs()],
                axis=1,
            ).max(axis=1)
            atr14 = float(tr.rolling(14).mean().iloc[-1])
        except Exception:
            atr14 = price * 0.015

        return {
            "name":    commodity["name"],
            "ticker":  commodity["ticker"],
            "etf":     commodity["etf"],
            "sector":  commodity["sector"],
            "unit":    commodity["unit"],
            "source":  source,
            "price":   round(price, 4),
            "ret_1d":  round((price / prev1 - 1) if prev1 > 0 else 0.0, 6),
            "ret_5d":  round((price / prev5 - 1) if prev5 > 0 else 0.0, 6),
            "ret_20d": round((price / prev20 - 1) if prev20 > 0 else 0.0, 6),
            "high_52": round(high_52, 4),
            "low_52":  round(low_52, 4),
            "pct_52":  round(pct_52, 4),
            "sma20":   round(sma20, 4),
            "sma50":   round(sma50, 4),
            "sma200":  round(sma200, 4),
            "rsi14":   round(rsi14, 2),
            "atr14":   round(atr14, 4),
            "_close":  close,
        }

    return None


def fetch_macro_indicator(ticker: str) -> Optional[Dict[str, Any]]:
    """Fetch macro time-series data for a yield / index ticker.

    Args:
        ticker: Yahoo Finance ticker (e.g. "^TNX", "^VIX").

    Returns:
        Dict with price and short-horizon returns, or None.
    """
    try:
        import pandas as pd
    except ImportError:
        return None

    df = safe_download(ticker, period="60d")
    if df is None:
        return None

    close = df["Close"].squeeze()
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = close.dropna()
    if len(close) < 2:
        return None

    price = float(close.iloc[-1])
    prev1 = float(close.iloc[-2]) if len(close) >= 2 else price
    prev5 = float(close.iloc[-6]) if len(close) >= 6 else price
    prev20 = float(close.iloc[-21]) if len(close) >= 21 else price
    high_52 = float(close.rolling(min(252, len(close))).max().iloc[-1])
    low_52 = float(close.rolling(min(252, len(close))).min().iloc[-1])

    return {
        "price":   round(price, 4),
        "ret_1d":  round((price / prev1 - 1) if prev1 > 0 else 0.0, 6),
        "ret_5d":  round((price / prev5 - 1) if prev5 > 0 else 0.0, 6),
        "ret_20d": round((price / prev20 - 1) if prev20 > 0 else 0.0, 6),
        "high_52": round(high_52, 4),
        "low_52":  round(low_52, 4),
        "_close":  close,
    }


# ── Scoring functions ──────────────────────────────────────────────────────────

def calc_term_structure_score(data: Dict[str, Any]) -> Tuple[float, str]:
    """Compute term-structure (backwardation/contango) score for a commodity.

    Keynes (1936) — normal backwardation: spot > forward implies supply tightness.

    Args:
        data: Output of fetch_commodity_prices().

    Returns:
        (score [0,1], human-readable label)
    """
    ret_5d = data.get("ret_5d", 0.0)
    ret_20d = data.get("ret_20d", 0.0)
    price = data.get("price", 1.0)
    sma50 = data.get("sma50", price)
    sma200 = data.get("sma200", price)

    daily_5d = ret_5d / 5
    daily_20d = ret_20d / 20 if ret_20d != 0 else 0.0
    acceleration = daily_5d - daily_20d
    accel_score = max(0.0, min(1.0, 0.5 + acceleration * 40))

    if price > sma200 and sma50 > sma200:
        trend_score = 0.82
    elif price > sma200:
        trend_score = 0.63
    elif price > sma50:
        trend_score = 0.42
    else:
        trend_score = 0.18

    score = round(0.50 * accel_score + 0.50 * trend_score, 4)

    if score >= 0.72:
        label = "Backwardation (up-up)"
    elif score >= 0.58:
        label = "Mild Backwardation (up)"
    elif score >= 0.42:
        label = "Flat / Neutral"
    elif score >= 0.28:
        label = "Mild Contango (down)"
    else:
        label = "Contango (down-down)"

    return score, label


def calc_cot_real_score(cot_symbol: str, fmp_key: str) -> Optional[Tuple[float, str]]:
    """Real COT commercial net-position score from FMP stable/commitment-of-traders-report.

    PASS in Phase-0 smoke-test (536 rows, weekly CFTC data).
    Uses commercial hedger net position (long - short) as the primary signal:
    commercial hedgers are informed actors who hedge real commodity exposure, so
    extreme net-short = market top, extreme net-long = market bottom.

    Score formula:
        comm_net_ratio = commLong / (commLong + commShort)  → [0, 1]
        score = 1 - comm_net_ratio  (inverted: heavy commercial shorting → bullish)

    Args:
        cot_symbol: COT contract symbol (e.g. "CL" for crude, "GC" for gold).
        fmp_key:    FMP API key.

    Returns:
        (score [0,1], label) or None if symbol not found / key absent.
    """
    try:
        from regime_trader.services.fmp_client import FMPClient as _FMPClient
        client = _FMPClient(api_key=fmp_key)
        rows = client.get_cot_report()
        if not rows:
            return None
        # Match by symbol prefix (FMP uses CLF, CL1, GCF, etc.)
        match = next(
            (r for r in rows if str(r.get("symbol", "")).startswith(cot_symbol)),
            None,
        )
        if not match:
            return None
        comm_long  = float(match.get("commPositionsLongAll",  0) or 0)
        comm_short = float(match.get("commPositionsShortAll", 0) or 0)
        total = comm_long + comm_short
        if total <= 0:
            return None
        comm_net_ratio = comm_long / total   # high = commercial long (bullish)
        score = round(1.0 - comm_net_ratio, 4)  # invert: commercial shorting = bullish for price

        if comm_net_ratio < 0.25:
            label = "STRONGLY BULLISH (heavy commercial short = classic bottom signal)"
        elif comm_net_ratio < 0.40:
            label = "Bullish — Commercial Selling Dominates"
        elif comm_net_ratio < 0.60:
            label = "Neutral"
        elif comm_net_ratio < 0.75:
            label = "Bearish — Commercial Buying Dominates"
        else:
            label = "STRONGLY BEARISH (heavy commercial long = classic top signal)"

        return score, label
    except Exception as exc:
        log.debug("calc_cot_real_score(%s) failed: %s", cot_symbol, exc)
        return None


# Fallback proxy (used when FMP COT is unavailable or fmp_key absent)
def calc_cot_proxy_score(data: Dict[str, Any]) -> Tuple[float, str]:
    """Proxy for Commitment of Traders (COT) positioning via 52-week percentile.

    Akerlof (2001 Nobel) — price position relative to range as smart-money proxy.

    Args:
        data: Output of fetch_commodity_prices() with pct_52 and rsi14.

    Returns:
        (score [0,1], human-readable label)
    """
    pct_52 = data.get("pct_52", 0.5)
    rsi14 = data.get("rsi14", 50.0)
    ret_5d = data.get("ret_5d", 0.0)

    base = 1.0 - pct_52
    if pct_52 < 0.15 and rsi14 > 32:
        score = min(0.95, base + 0.18)
        label = "STRONGLY BULLISH — Insider Accumulation"
    elif pct_52 < 0.25:
        score = min(0.85, base + 0.08)
        label = "Bullish — Commercial Buying"
    elif pct_52 < 0.45:
        score = base + 0.03
        label = "Mildly Bullish"
    elif pct_52 < 0.60:
        score = base
        label = "Neutral"
    elif pct_52 < 0.78:
        score = max(0.15, base - 0.05)
        label = "Bearish — Commercial Hedging"
    else:
        score = max(0.05, base - 0.12)
        label = "STRONGLY BEARISH — Commercial Selling"

    score += ret_5d * 0.25
    return round(max(0.0, min(1.0, score)), 4), label


def calc_sentiment_score(
    etf: str,
    sentiment_map: Dict[str, float],
) -> Tuple[float, str]:
    """Contrarian retail-sentiment score (high retail bullishness → sell signal).

    Shiller (2013 Nobel) — irrational exuberance: extreme retail bullishness
    is a contrarian sell indicator.

    Args:
        etf:           ETF ticker to look up in sentiment_map.
        sentiment_map: {etf_ticker: retail_bullish_fraction [0,1]}.

    Returns:
        (score [0,1], human-readable label)
        score=1.0 means extreme retail bearishness (strong contrarian buy).
    """
    raw = sentiment_map.get(etf, 0.5)
    score = round(1.0 - 0.80 * raw, 4)

    if raw >= 0.80:
        label = f"Extreme Retail Bullish ({raw:.0%}) — Contrarian Sell"
    elif raw >= 0.65:
        label = f"Retail Bullish ({raw:.0%}) — Caution"
    elif raw >= 0.45:
        label = f"Neutral ({raw:.0%})"
    elif raw >= 0.30:
        label = f"Retail Bearish ({raw:.0%}) — Contrarian Buy"
    else:
        label = f"Extreme Retail Bearish ({raw:.0%}) — Strong Buy Signal"

    return score, label


def calc_trend_score(data: Dict[str, Any]) -> Tuple[float, str]:
    """Technical trend score combining moving-average cross and RSI.

    Fama (2013 Nobel) — price relative to 200-day MA as regime indicator.

    Args:
        data: Output of fetch_commodity_prices() with price, sma50, sma200, rsi14.

    Returns:
        (score [0,1], human-readable label)
    """
    price = data.get("price", 1.0)
    sma50 = data.get("sma50", price)
    sma200 = data.get("sma200", price)
    rsi14 = data.get("rsi14", 50.0)

    if price > sma200 and sma50 > sma200:
        tc, tl = 0.88, "Golden Cross"
    elif price > sma200:
        tc, tl = 0.65, "Above 200MA"
    elif price > sma50:
        tc, tl = 0.40, "Between MAs"
    else:
        tc, tl = 0.15, "Death Cross"

    if rsi14 < 30:
        rc, rl = 0.90, f"Oversold ({rsi14:.0f})"
    elif rsi14 < 45:
        rc, rl = 0.72, f"Recovering ({rsi14:.0f})"
    elif rsi14 < 60:
        rc, rl = 0.55, f"Neutral ({rsi14:.0f})"
    elif rsi14 < 70:
        rc, rl = 0.33, f"Extended ({rsi14:.0f})"
    else:
        rc, rl = 0.12, f"Overbought ({rsi14:.0f})"

    score = round(0.60 * tc + 0.40 * rc, 4)
    label = f"{tl} · RSI {rl}"
    return score, label


def calc_macro_conviction(
    price_data: Dict[str, Any],
    sentiment_map: Dict[str, float],
) -> Dict[str, Any]:
    """Composite macro conviction score for a single commodity.

    Weights: term structure 30%, COT proxy 30%, sentiment 20%, trend 20%.

    Args:
        price_data:    Output of fetch_commodity_prices().
        sentiment_map: {etf_ticker: retail_bullish_fraction}.

    Returns:
        Dict with composite score, conviction label/colour, and sub-scores.
    """
    import os as _os
    ts_s, ts_l = calc_term_structure_score(price_data)
    # Use real COT data when FMP key is present; fall back to 52-week-percentile proxy.
    _cot_sym = price_data.get("cot_symbol", "")
    _fmp_key = _os.getenv("FMP_API_KEY", "")
    _cot_real = calc_cot_real_score(_cot_sym, _fmp_key) if _cot_sym and _fmp_key else None
    if _cot_real is not None:
        cot_s, cot_l = _cot_real
    else:
        cot_s, cot_l = calc_cot_proxy_score(price_data)
    etf = price_data.get("etf", "")
    sent_s, sent_l = calc_sentiment_score(etf, sentiment_map)
    tr_s, tr_l = calc_trend_score(price_data)

    composite = round(
        0.30 * ts_s + 0.30 * cot_s + 0.20 * sent_s + 0.20 * tr_s, 4
    )
    composite = max(0.0, min(1.0, composite))

    if composite >= 0.72:
        cv_lbl, cv_clr = "Strong Buy", "#00c851"
    elif composite >= 0.58:
        cv_lbl, cv_clr = "Buy", "#7cb342"
    elif composite >= 0.42:
        cv_lbl, cv_clr = "Neutral", "#9e9e9e"
    elif composite >= 0.28:
        cv_lbl, cv_clr = "Reduce", "#ff8800"
    else:
        cv_lbl, cv_clr = "Avoid", "#ff4444"

    return {
        "composite":        composite,
        "conviction_label": cv_lbl,
        "conviction_clr":   cv_clr,
        "ts_score":   ts_s,  "ts_label":   ts_l,
        "cot_score":  cot_s, "cot_label":  cot_l,
        "sent_score": sent_s, "sent_label": sent_l,
        "tr_score":   tr_s,  "tr_label":   tr_l,
    }


# ── Macro shock alerts ─────────────────────────────────────────────────────────

def check_macro_shocks(
    prices: Dict[str, Optional[Dict[str, Any]]],
) -> List[Dict[str, str]]:
    """Detect significant commodity price moves and return structured alerts.

    Leontief (1973 Nobel) — input-output linkages: commodity spikes propagate
    through the economy and affect equity sector margins.

    Args:
        prices: {ticker: fetch_commodity_prices() result | None}

    Returns:
        List of alert dicts: {level: "error"|"warning", icon, message}
    """
    alerts: List[Dict[str, str]] = []
    crude = prices.get("CL=F") or {}
    wheat = prices.get("ZW=F") or {}
    gold = prices.get("GC=F") or {}
    copper = prices.get("HG=F") or {}
    ng = prices.get("NG=F") or {}

    oil_5d = crude.get("ret_5d", 0.0)
    if oil_5d >= 0.05:
        alerts.append({
            "level": "error", "icon": "fire",
            "message": (
                f"Macro Shock — Energy Inflation: Crude Oil {oil_5d:+.1%} in 5 days. "
                "Equity margins at risk. Reduce high-beta long exposure."
            ),
        })
    elif oil_5d >= 0.03:
        alerts.append({
            "level": "warning", "icon": "warning",
            "message": (
                f"Energy Watch: Crude Oil {oil_5d:+.1%} in 5 days. "
                "Monitor margin compression in consumer/industrial sectors."
            ),
        })

    wheat_5d = wheat.get("ret_5d", 0.0)
    if wheat_5d >= 0.10:
        alerts.append({
            "level": "error", "icon": "seedling",
            "message": (
                f"Macro Shock — Food Inflation: Wheat {wheat_5d:+.1%} in 5 days. "
                "Consumer staples margins at risk. Rotate to agricultural producers."
            ),
        })
    elif wheat_5d >= 0.05:
        alerts.append({
            "level": "warning", "icon": "warning",
            "message": (
                f"Food Inflation Watch: Wheat {wheat_5d:+.1%} in 5 days. "
                "Monitor consumer staples guidance."
            ),
        })

    gold_5d = gold.get("ret_5d", 0.0)
    copper_5d = copper.get("ret_5d", 0.0)
    if gold_5d > 0.02 and copper_5d < -0.03:
        alerts.append({
            "level": "error", "icon": "chart-line-down",
            "message": (
                f"Recession Warning — Copper/Gold ratio crashing. "
                f"Gold {gold_5d:+.1%} · Copper {copper_5d:+.1%} (5d). "
                "Defensive posture recommended. Reduce cyclical exposure."
            ),
        })
    elif gold_5d > 0.01 and copper_5d < -0.01:
        alerts.append({
            "level": "warning", "icon": "warning",
            "message": (
                f"Liquidity Watch: Gold outperforming Copper — flight-to-safety pattern. "
                f"Gold {gold_5d:+.1%} · Copper {copper_5d:+.1%} (5d)."
            ),
        })

    ng_5d = ng.get("ret_5d", 0.0)
    if ng_5d >= 0.12:
        alerts.append({
            "level": "warning", "icon": "bolt",
            "message": (
                f"Natural Gas {ng_5d:+.1%} in 5 days. "
                "Utility cost pressures rising. Monitor industrial sector margins."
            ),
        })

    return alerts


# ── Macro narrative synthesis ──────────────────────────────────────────────────

def generate_macro_synthesis(
    prices: Dict[str, Optional[Dict[str, Any]]],
    convictions: Dict[str, Dict[str, Any]],
    indicators: Dict[str, Optional[Dict[str, Any]]],
) -> List[str]:
    """Build a list of natural-language macro commentary paragraphs.

    Engle (2003 Nobel) — volatility clustering informs the risk narrative.

    Args:
        prices:      {ticker: fetch_commodity_prices() result}
        convictions: {ticker: calc_macro_conviction() result}
        indicators:  {ticker: fetch_macro_indicator() result}

    Returns:
        List of paragraph strings (always at least one element).
    """
    paras: List[str] = []
    crude = prices.get("CL=F") or {}
    gold = prices.get("GC=F") or {}
    copper = prices.get("HG=F") or {}
    wheat = prices.get("ZW=F") or {}
    corn = prices.get("ZC=F") or {}
    crude_cv = convictions.get("CL=F", {})
    gold_cv = convictions.get("GC=F", {})
    tnx = indicators.get("^TNX") or {}
    dxy = indicators.get("DX-Y.NYB") or {}
    vix = indicators.get("^VIX") or {}

    if crude:
        ts_l = crude_cv.get("ts_label", "")
        cot_l = crude_cv.get("cot_label", "")
        conv_l = crude_cv.get("conviction_label", "Neutral")
        oil_5d = crude.get("ret_5d", 0.0)
        rsi = crude.get("rsi14", 50.0)
        if "Backwardation" in ts_l and "Bullish" in cot_l:
            paras.append(
                f"CRUDE OIL [{conv_l}] — Steep backwardation with commercial accumulation. "
                f"Physical supply is tight ({oil_5d:+.1%} 5-day · RSI {rsi:.0f}). "
                "Energy equities (XLE, XOP) expected to outperform."
            )
        elif "Contango" in ts_l:
            paras.append(
                f"CRUDE OIL [{conv_l}] — Market in contango — structural oversupply signal. "
                "Negative roll yield for long ETF holders (USO). "
                "Energy sector faces headwinds."
            )
        else:
            paras.append(
                f"CRUDE OIL [{conv_l}] — Mixed signals. Structure: {ts_l}. "
                f"COT proxy: {cot_l}. 5-day return {oil_5d:+.1%}."
            )

    if gold and copper:
        gold_5d = gold.get("ret_5d", 0.0)
        copper_5d = copper.get("ret_5d", 0.0)
        gold_p = gold.get("price", 0.0)
        copper_p = copper.get("price", 0.0)
        cu_au = round(copper_p / gold_p * 1000, 2) if gold_p > 0 else 0
        if gold_5d > 0.015 and copper_5d < -0.02:
            paras.append(
                f"METALS [DEFENSIVE] — Flight-to-safety divergence. "
                f"Gold {gold_5d:+.1%} vs Copper {copper_5d:+.1%} (5d). "
                f"Cu/Au ratio: {cu_au:.2f} (x1000). Dr. Copper signalling contraction. "
                "Increase defensive allocation: GLD, TLT."
            )
        elif copper_5d > 0.02:
            paras.append(
                f"METALS [RISK-ON] — Copper strength ({copper_5d:+.1%} 5d) signals "
                f"industrial demand recovery. Cu/Au ratio: {cu_au:.2f} (x1000). "
                "Bullish for materials (XLB), industrials (XLI)."
            )
        else:
            gold_conv = gold_cv.get("conviction_label", "Neutral")
            paras.append(
                f"METALS [NEUTRAL] — Gold {gold_5d:+.1%} · Copper {copper_5d:+.1%} (5d). "
                f"Cu/Au ratio {cu_au:.2f} (x1000). No strong directional signal. "
                f"Gold conviction: {gold_conv}."
            )

    if wheat or corn:
        w5 = wheat.get("ret_5d", 0.0) if wheat else 0.0
        c5 = corn.get("ret_5d", 0.0) if corn else 0.0
        if abs(w5) > 0.04 or abs(c5) > 0.04:
            direction = "surging" if (w5 + c5) > 0 else "collapsing"
            impact = "food cost pressures rising" if (w5 + c5) > 0 else "food cost pressures easing"
            paras.append(
                f"AGRICULTURE [{direction.upper()}] — Wheat {w5:+.1%} · Corn {c5:+.1%} (5d). "
                f"{impact.capitalize()}. "
                + (
                    "Monitor consumer staples margins and EM sovereign risk."
                    if (w5 + c5) > 0
                    else "Positive for restaurant chains, food manufacturers."
                )
            )

    notes: List[str] = []
    if tnx:
        tnx_val = tnx.get("price", 0.0)
        tnx_5d = tnx.get("ret_5d", 0.0)
        if tnx_val > 4.5:
            notes.append(f"10Y yield at {tnx_val:.2f}% — deeply restrictive")
        elif tnx_5d > 0.02:
            notes.append(f"10Y rising ({tnx_val:.2f}%, {tnx_5d:+.1%})")
        else:
            notes.append(f"10Y at {tnx_val:.2f}%")
    if dxy:
        dxy_val = dxy.get("price", 0.0)
        dxy_5d = dxy.get("ret_5d", 0.0)
        if dxy_5d > 0.01:
            notes.append(f"USD strengthening ({dxy_val:.1f}, {dxy_5d:+.1%}) — commodity headwind")
        elif dxy_5d < -0.01:
            notes.append(f"USD weakening ({dxy_val:.1f}, {dxy_5d:+.1%}) — commodity tailwind")
        else:
            notes.append(f"USD stable ({dxy_val:.1f})")
    if vix:
        vix_val = vix.get("price", 0.0)
        if vix_val > 30:
            notes.append(f"VIX {vix_val:.1f} — extreme fear")
        elif vix_val > 20:
            notes.append(f"VIX {vix_val:.1f} — elevated uncertainty")
        else:
            notes.append(f"VIX {vix_val:.1f} — calm conditions")
    if notes:
        paras.append("MACRO BACKDROP — " + " · ".join(notes) + ".")

    if not paras:
        paras.append(
            "Insufficient data to generate macro synthesis. "
            "Refresh macro data to fetch live prices."
        )
    return paras


# ── Equity data helper ─────────────────────────────────────────────────────────

def fetch_stock_pick_data(
    ticker: str,
    fallback: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Fetch price + fundamental quality metrics via FMP Ultimate.

    Uses stable/historical-price-eod/full for price/SMA200, stable/quote for
    current price, and stable/ratios-ttm for D/E ratio + net margin.
    Fully on FMP — no yfinance.

    Args:
        ticker:   Primary ticker.
        fallback: Secondary ticker if primary returns insufficient data.

    Returns:
        Dict with price, sma200, de_ratio, net_margin, stock_score; or None.
    """
    from regime_trader.services.fmp_client import FMPClient, fmp_prices_to_arrays  # noqa: PLC0415

    client = FMPClient()
    if not client._api_key:
        log.warning("FMP_API_KEY not set — fetch_stock_pick_data unavailable")
        return None

    for tk in ([ticker] + ([fallback] if fallback else [])):
        try:
            rows = client.get_historical_prices(tk, limit=252)
            closes, _, _ = fmp_prices_to_arrays(rows)
            if len(closes) < 2:
                continue

            price  = closes[-1]
            prev1  = closes[-2]
            ret_1d = (price / prev1 - 1.0) if prev1 > 0 else 0.0

            sma200 = None
            if len(closes) >= 200:
                sma200 = round(sum(closes[-200:]) / 200, 2)
            above_sma200 = bool(price > sma200) if sma200 is not None else False

            # Fundamentals from ratios-ttm
            ratios = client.get_ratios_ttm(tk)
            q      = client.get_quote(tk)

            company_name = q.get("name", tk)

            de_raw = ratios.get("debtEquityRatioTTM") or ratios.get("debtToEquityRatioTTM")
            de_ratio = float(de_raw) if de_raw is not None else None

            nm_raw = ratios.get("netProfitMarginTTM")
            net_margin = float(nm_raw) * 100.0 if nm_raw is not None else None

            momentum_c = 0.85 if above_sma200 else 0.25
            de_c = (
                0.25 if de_ratio is None else
                0.50 if de_ratio < 0.50 else
                0.35 if de_ratio < 1.00 else 0.10
            )
            mg_c = (
                0.20 if net_margin is None else
                0.50 if net_margin > 15 else
                0.38 if net_margin > 5 else
                0.20 if net_margin > 0 else 0.05
            )
            stock_score = round(0.50 * momentum_c + 0.50 * (de_c + mg_c), 4)

            return {
                "ticker":       tk,
                "original":     ticker,
                "name":         company_name,
                "price":        round(price, 2),
                "ret_1d":       round(ret_1d, 6),
                "sma200":       sma200,
                "above_sma200": above_sma200,
                "de_ratio":     round(de_ratio, 2) if de_ratio is not None else None,
                "net_margin":   round(net_margin, 1) if net_margin is not None else None,
                "stock_score":  stock_score,
                "close_30d":    closes[-30:],
                "is_fallback":  tk != ticker,
            }
        except Exception as exc:
            log.warning("fetch_stock_pick_data(%s): %s", tk, exc)
            continue

    return None
