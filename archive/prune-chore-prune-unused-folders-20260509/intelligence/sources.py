"""
intelligence/sources.py
───────────────────────
Individual data-source fetchers for the market-intelligence layer.

Each fetcher returns a normalised float score in [0, 1]:
  0.0 → extremely bearish / no activity
  0.5 → neutral / average activity
  1.0 → extremely bullish / peak activity

SOURCES
───────
  FlowFetcher     — Options flow (UnusualWhales API)
  SentimentFetcher— Social sentiment (ApeWisdom public API + Reddit)
  InsiderFetcher  — Insider transactions (OpenInsider free scraper)
  MacroFetcher    — Regime + VIX context (HMM label + yfinance)

API KEYS (from .env)
────────────────────
  UNUSUAL_WHALES_TOKEN — UnusualWhales bearer token
  FINTEL_API_KEY       — Fintel short-interest API key (optional)

Missing keys → that source returns 0.5 (neutral) with a warning log.
Network errors → same fallback so the whole report is never blocked.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib import request as urllib_request
import json

from log_manager.logger import get_logger

log = get_logger(__name__)

# ── Optional dependencies ─────────────────────────────────────────────────────

try:
    import requests as _req_lib   # type: ignore
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

try:
    from bs4 import BeautifulSoup  # type: ignore
    _BS4_OK = True
except ImportError:
    _BS4_OK = False

try:
    import yfinance as yf          # type: ignore
    _YF_OK = True
except ImportError:
    _YF_OK = False

# ── Shared HTTP helper ────────────────────────────────────────────────────────

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

_NEUTRAL = 0.50   # fallback score when data is unavailable


def _get(url: str, headers: Optional[Dict] = None, timeout: int = 8) -> Optional[Dict]:
    """Simple GET returning parsed JSON or None on error."""
    if not _REQUESTS_OK:
        log.warning("requests not installed — HTTP fetch skipped for {}", url)
        return None
    h = {**_DEFAULT_HEADERS, **(headers or {})}
    try:
        resp = _req_lib.get(url, headers=h, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        log.warning("HTTP GET failed ({}) : {}", url, exc)
        return None


# ── 1. Options-flow fetcher ───────────────────────────────────────────────────


class FlowFetcher:
    """
    Fetch unusual options-flow data from UnusualWhales.

    The score reflects the net bullish/bearish options premium for the symbol
    over the last 5 trading days, normalised to [0, 1].

    Free API tier: 100 req / day.  Set UNUSUAL_WHALES_TOKEN in .env.
    """

    _BASE = "https://api.unusualwhales.com/api"

    def __init__(self) -> None:
        self._token = os.getenv("UNUSUAL_WHALES_TOKEN", "")
        if not self._token:
            log.info("UNUSUAL_WHALES_TOKEN not set — flow scores will be neutral")

    def fetch(self, symbol: str) -> Tuple[float, Dict]:
        """
        Returns (score, raw_payload).
        score ∈ [0, 1]:  > 0.5 = net bullish flow, < 0.5 = net bearish flow.
        """
        if not self._token:
            return _NEUTRAL, {"source": "unusualwhales", "status": "no_token"}

        url     = f"{self._BASE}/stock/{symbol}/flow-alerts"
        headers = {"Authorization": f"Bearer {self._token}"}
        data    = _get(url, headers=headers)

        if data is None:
            return _NEUTRAL, {"source": "unusualwhales", "status": "error"}

        try:
            alerts = data.get("data", [])
            if not alerts:
                return _NEUTRAL, {"source": "unusualwhales", "count": 0}

            # Each alert has 'sentiment': 'bullish' | 'bearish'
            # and 'premium': dollar value of the flow
            total_premium = 0.0
            net_premium   = 0.0
            for alert in alerts:
                prem      = float(alert.get("premium", 0) or 0)
                sentiment = str(alert.get("sentiment", "")).lower()
                total_premium += prem
                net_premium   += prem if "bull" in sentiment else -prem

            if total_premium == 0:
                score = _NEUTRAL
            else:
                # Map [-1, 1] net ratio → [0, 1]
                ratio = net_premium / total_premium
                score = (ratio + 1.0) / 2.0

            return round(max(0.0, min(1.0, score)), 4), {
                "source":       "unusualwhales",
                "alert_count":  len(alerts),
                "total_premium": total_premium,
                "net_ratio":    ratio if total_premium > 0 else 0.0,
            }

        except Exception as exc:
            log.error("FlowFetcher parse error for {}: {}", symbol, exc)
            return _NEUTRAL, {"source": "unusualwhales", "status": "parse_error"}


# ── 2. Social sentiment fetcher ───────────────────────────────────────────────


class SentimentFetcher:
    """
    Fetch social-media sentiment from ApeWisdom (free public API).

    ApeWisdom aggregates Reddit mentions from r/wallstreetbets, r/stocks,
    r/investing, and others.  No API key required.

    Score = normalised rank × upvote-ratio  (higher rank = more bullish mention
    intensity relative to the full screened universe).
    """

    _APEWISDOM_URL = "https://apewisdom.io/api/v1.0/filter/all-stocks/page/1"
    _UNIVERSE_SIZE = 100    # ApeWisdom returns up to 100 tickers per page

    def __init__(self) -> None:
        self._cache: Optional[List[Dict]] = None
        self._cache_ts: float = 0.0
        self._cache_ttl: float = 300.0   # 5-minute cache

    def _load_cache(self) -> None:
        """Refresh the full ApeWisdom ranking list (cached 5 min)."""
        if time.time() - self._cache_ts < self._cache_ttl and self._cache:
            return
        data = _get(self._APEWISDOM_URL)
        if data and "results" in data:
            self._cache    = data["results"]
            self._cache_ts = time.time()
            log.debug("ApeWisdom cache refreshed ({} entries)", len(self._cache))
        else:
            log.warning("ApeWisdom fetch failed — using stale cache")

    def fetch(self, symbol: str) -> Tuple[float, Dict]:
        """
        Returns (score, raw).
        score ∈ [0, 1]:  based on rank position and mention volume.
        """
        self._load_cache()
        if not self._cache:
            return _NEUTRAL, {"source": "apewisdom", "status": "unavailable"}

        # Search for the symbol in the ranking list
        entry = next(
            (r for r in self._cache if r.get("ticker", "").upper() == symbol.upper()),
            None,
        )
        if entry is None:
            # Not trending at all → below neutral (low visibility)
            return 0.35, {"source": "apewisdom", "status": "not_found"}

        rank     = int(entry.get("rank", self._UNIVERSE_SIZE))
        mentions = float(entry.get("mentions", 0))
        upvotes  = float(entry.get("upvotes", 0))
        mentions_24h = float(entry.get("mentions_24h_ago", mentions) or mentions)

        # Rank score: rank 1 = 1.0, rank 100 = 0.0
        rank_score = 1.0 - (rank - 1) / self._UNIVERSE_SIZE

        # Momentum: growing vs shrinking mentions
        if mentions_24h > 0:
            momentum = min(mentions / mentions_24h, 3.0) / 3.0   # cap at 3× growth
        else:
            momentum = 0.5

        # Upvote ratio proxy (upvotes / max expected upvotes)
        upvote_score = min(upvotes / max(mentions * 10, 1), 1.0)

        score = 0.50 * rank_score + 0.30 * momentum + 0.20 * upvote_score
        return round(max(0.0, min(1.0, score)), 4), {
            "source":       "apewisdom",
            "rank":         rank,
            "mentions":     mentions,
            "upvotes":      upvotes,
            "momentum":     round(momentum, 3),
        }


# ── 3. Insider-activity fetcher ───────────────────────────────────────────────


class InsiderFetcher:
    """
    Fetch insider transaction data from OpenInsider (free public site).

    Parses the HTML table from the OpenInsider screener filtered by symbol.
    Scores based on net insider buying vs. selling in the last 90 days.
    """

    _BASE = "http://openinsider.com/screener"

    def __init__(self) -> None:
        if not _BS4_OK:
            log.info(
                "beautifulsoup4 not installed — insider scores will be neutral.  "
                "Run: pip install beautifulsoup4 lxml"
            )

    def fetch(self, symbol: str) -> Tuple[float, Dict]:
        """
        Returns (score, raw).
        score > 0.5 → net insider buying.
        score < 0.5 → net insider selling.
        score = 0.5 → no activity or parse error.
        """
        if not _BS4_OK or not _REQUESTS_OK:
            return _NEUTRAL, {"source": "openinsider", "status": "bs4_missing"}

        url = (
            f"{self._BASE}?s={symbol}&fd=90&td=0&xs=1&vl=25&"
            "is=&pl=&ph=&xp=1&xb=1&xa=1&xd=1&xs=1&xord=&sortcol=0&cnt=40&action=1"
        )

        try:
            import requests  # type: ignore
            resp = requests.get(url, headers=_DEFAULT_HEADERS, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
        except Exception as exc:
            log.warning("OpenInsider fetch failed for {}: {}", symbol, exc)
            return _NEUTRAL, {"source": "openinsider", "status": "fetch_error"}

        try:
            table = soup.find("table", {"class": "tinytable"})
            if table is None:
                return _NEUTRAL, {"source": "openinsider", "status": "no_table"}

            rows = table.find_all("tr")[1:]   # skip header
            buy_value  = 0.0
            sell_value = 0.0
            n_buys = n_sells = 0
            most_recent_date: Optional[datetime] = None

            for row in rows:
                cols = row.find_all("td")
                if len(cols) < 11:
                    continue
                tx_type = cols[3].get_text(strip=True).upper()
                val_str = cols[9].get_text(strip=True).replace(",", "").replace("$", "")
                try:
                    val = abs(float(val_str))
                except ValueError:
                    val = 0.0

                # Track most recent trade date for time-decay (col 2 = Trade Date, col 1 = Filing Date)
                for date_col in (2, 1):
                    raw_date = cols[date_col].get_text(strip=True)[:10] if len(cols) > date_col else ""
                    try:
                        dt = datetime.strptime(raw_date, "%Y-%m-%d")
                        if most_recent_date is None or dt > most_recent_date:
                            most_recent_date = dt
                        break
                    except (ValueError, IndexError):
                        pass

                if "P" in tx_type:       # Purchase
                    buy_value += val
                    n_buys    += 1
                elif "S" in tx_type:     # Sale (but not "S-" = plan sale)
                    sell_value += val
                    n_sells    += 1

            total = buy_value + sell_value
            if total == 0:
                score = _NEUTRAL
            else:
                ratio = buy_value / total          # [0, 1]
                # Apply slight penalty if mostly plan sales (less signal)
                score = 0.4 + 0.6 * ratio          # remapped to [0.4, 1.0]
                if n_buys == 0 and n_sells == 0:
                    score = _NEUTRAL

            # Data age for time-decay in engine
            ins_age_days = 0.0
            if most_recent_date:
                ins_age_days = max(
                    0.0,
                    (datetime.utcnow() - most_recent_date).total_seconds() / 86400,
                )

            return round(max(0.0, min(1.0, score)), 4), {
                "source":        "openinsider",
                "buy_value":     buy_value,
                "sell_value":    sell_value,
                "n_buys":        n_buys,
                "n_sells":       n_sells,
                "rows":          len(rows),
                "data_age_days": round(ins_age_days, 1),
            }

        except Exception as exc:
            log.error("InsiderFetcher parse error for {}: {}", symbol, exc)
            return _NEUTRAL, {"source": "openinsider", "status": "parse_error"}


# ── 4. Macro / regime fetcher ─────────────────────────────────────────────────


class MacroFetcher:
    """
    Derive a macro score from the HMM regime label and VIX level.

    This is a rule-based encoding of the current market environment:
      • High-conviction bull regime + low VIX  → score near 0.80
      • Crash/Panic regime or VIX > 30         → score near 0.10–0.20
      • Neutral regime / medium VIX            → score near 0.50

    Parameters
    ----------
    regime_label      : Current HMM regime string (e.g. "Bull").
    regime_confidence : HMM confidence score [0, 1].
    """

    _REGIME_BASE_SCORES: Dict[str, float] = {
        "Mania":    0.90,
        "Euphoria": 0.78,
        "Bull":     0.65,
        "Neutral":  0.50,
        "Bear":     0.32,
        "Panic":    0.18,
        "Crash":    0.08,
        "Unknown":  0.50,
    }

    def __init__(
        self,
        regime_label:      str   = "Unknown",
        regime_confidence: float = 0.50,
    ) -> None:
        self._regime_label      = regime_label
        self._regime_confidence = regime_confidence

    def update_regime(self, label: str, confidence: float) -> None:
        self._regime_label      = label
        self._regime_confidence = confidence

    def _fetch_vix(self) -> Optional[float]:
        """Fetch latest VIX close from yfinance."""
        if not _YF_OK:
            return None
        try:
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="2d")
            if hist.empty:
                return None
            return float(hist["Close"].iloc[-1])
        except Exception as exc:
            log.warning("VIX fetch failed: {}", exc)
            return None

    def fetch(self, symbol: str = "") -> Tuple[float, Dict]:
        """
        Returns (score, raw).
        The symbol parameter is accepted for API uniformity but not used
        (macro score is portfolio-level, not symbol-specific).
        """
        base_score = self._REGIME_BASE_SCORES.get(self._regime_label, 0.50)
        vix        = self._fetch_vix()

        # VIX adjustment: every 5 points above 15 reduces score by 0.05
        vix_adj = 0.0
        if vix is not None:
            vix_adj = -max(0.0, (vix - 15.0) / 5.0) * 0.05
            vix_adj = max(vix_adj, -0.25)   # cap downward adjustment

        # Confidence scaling: low HMM confidence pushes score toward 0.5
        confidence_pull = 1.0 - self._regime_confidence
        score = base_score + vix_adj
        score = score + confidence_pull * (0.5 - score)   # pull toward neutral

        return round(max(0.0, min(1.0, score)), 4), {
            "source":             "macro",
            "regime_label":       self._regime_label,
            "regime_confidence":  self._regime_confidence,
            "vix":                round(vix, 2) if vix else None,
            "base_score":         base_score,
            "vix_adjustment":     round(vix_adj, 4),
        }

    @property
    def vix_level(self) -> Optional[float]:
        """Quick VIX read for the report header."""
        return self._fetch_vix()


# ── 5. Congressional-trading fetcher ──────────────────────────────────────────


import re as _re


def _parse_amount_range(s: str) -> float:
    """Parse a dollar range string like '$1,001 - $15,000' → midpoint float."""
    nums = [float(n.replace(",", "")) for n in _re.findall(r"[\d,]+", str(s))]
    if not nums:
        return 0.0
    return sum(nums) / len(nums)


class CongressFetcher:
    """
    Fetch congressional trading data from QuiverQuant.

    Scores net congressional buying vs selling for a symbol, normalised to
    [0, 1].  The raw payload includes ``data_age_days`` so the engine can
    apply exponential time-decay before computing the weighted conviction.

    Set QUIVERQUANT_API_KEY in .env for live data.
    Missing key → returns neutral (0.50) with a one-time info log.
    """

    _BASE    = "https://api.quiverquant.com/beta/live/congresstrading"
    _TTL_S   = 3600.0   # refresh the full dataset once per hour

    def __init__(self) -> None:
        self._token    = os.getenv("QUIVERQUANT_API_KEY", "")
        self._cache:    Optional[List[Dict]] = None
        self._cache_ts: float = 0.0
        if not self._token:
            log.info("QUIVERQUANT_API_KEY not set — congress scores will be neutral")

    # ── Cache ─────────────────────────────────────────────────────────────────

    def _load_cache(self) -> None:
        """Download the full congressional trades list (cached 1 h)."""
        if self._cache is not None and time.time() - self._cache_ts < self._TTL_S:
            return
        if not self._token:
            return
        headers = {
            "Authorization": f"Token {self._token}",
            "Accept":        "application/json",
        }
        data = _get(self._BASE, headers=headers)
        if data and isinstance(data, list):
            self._cache    = data
            self._cache_ts = time.time()
            log.debug("QuiverQuant congress cache refreshed ({} trades)", len(self._cache))
        else:
            log.warning("QuiverQuant fetch failed — congress scores will be neutral")

    # ── Public interface ──────────────────────────────────────────────────────

    def fetch(self, symbol: str) -> Tuple[float, Dict]:
        """
        Returns (score, raw).

        score > 0.5  → net congress buying (bullish signal).
        score < 0.5  → net congress selling (bearish signal).
        score = 0.5  → no data or neutral activity.

        raw["data_age_days"] is the age of the most recent trade, used by the
        engine to apply exponential time-decay before scoring.
        """
        _no_data = {"source": "quiverquant", "status": "no_token", "data_age_days": 0.0}
        if not self._token:
            return _NEUTRAL, _no_data

        self._load_cache()
        if not self._cache:
            return _NEUTRAL, {**_no_data, "status": "unavailable"}

        sym    = symbol.upper()
        trades = [t for t in self._cache if str(t.get("Ticker", "")).upper() == sym]

        if not trades:
            return _NEUTRAL, {"source": "quiverquant", "ticker": sym,
                               "count": 0, "data_age_days": 0.0}

        buy_amt = 0.0
        sell_amt = 0.0
        most_recent: Optional[datetime] = None

        for trade in trades:
            tx  = str(trade.get("Transaction", "")).lower()
            amt = _parse_amount_range(str(trade.get("Amount", "0")))

            if "purchase" in tx or "buy" in tx:
                buy_amt += amt
            elif "sale" in tx or "sell" in tx:
                sell_amt += amt

            for date_key in ("Date", "TransactionDate", "ReportDate"):
                date_str = str(trade.get(date_key, ""))[:10]
                try:
                    dt = datetime.strptime(date_str, "%Y-%m-%d")
                    if most_recent is None or dt > most_recent:
                        most_recent = dt
                    break
                except ValueError:
                    pass

        total = buy_amt + sell_amt
        if total == 0:
            score = _NEUTRAL
        else:
            ratio = buy_amt / total
            score = 0.35 + 0.65 * ratio   # all-sell → 0.35; all-buy → 1.00

        age_days = 0.0
        if most_recent:
            age_days = max(
                0.0,
                (datetime.utcnow() - most_recent).total_seconds() / 86400,
            )

        return round(max(0.0, min(1.0, score)), 4), {
            "source":        "quiverquant",
            "ticker":        sym,
            "trade_count":   len(trades),
            "buy_amount":    buy_amt,
            "sell_amount":   sell_amt,
            "data_age_days": round(age_days, 1),
            "last_trade":    most_recent.strftime("%Y-%m-%d") if most_recent else None,
        }


# ── 6. Stocktwits social sentiment ───────────────────────────────────────────


class StockTwitsFetcher:
    """
    Fetch per-ticker sentiment from the Stocktwits public API (no key required).

    Endpoints used:
      Trending : GET https://api.stocktwits.com/api/2/trending/symbols.json
      Stream   : GET https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json

    Score = fraction of messages tagged Bullish in the most recent stream.
    Trending symbols receive a +0.05 boost (capped at 1.0).
    Rate limit: ~200 req/hour per IP (unauthenticated).

    Cache: 5 minutes per symbol, 5 minutes for the trending list.
    """

    _BASE          = "https://api.stocktwits.com/api/2"
    _TREND_CACHE_KEY = "stocktwits_trending"
    _TTL_S         = 300.0   # 5 min

    def _trending_set(self) -> set:
        """Return set of currently trending ticker symbols (upper-cased)."""
        cached = _get(f"{self._BASE}/trending/symbols.json")
        if not cached or "symbols" not in cached:
            return set()
        return {s.get("symbol", "").upper() for s in cached.get("symbols", [])}

    def fetch(self, symbol: str) -> Tuple[float, Dict]:
        """
        Returns (score, raw).
        score > 0.5  → majority of recent messages are bullish.
        score < 0.5  → majority bearish.
        score = 0.5  → neutral / no sentiment data.
        """
        data = _get(f"{self._BASE}/streams/symbol/{symbol}.json")
        if data is None or "messages" not in data:
            return _NEUTRAL, {"source": "stocktwits", "status": "error"}

        messages = data["messages"]
        bull = sum(
            1 for m in messages
            if (m.get("entities") or {}).get("sentiment", {}) and
               (m.get("entities") or {})["sentiment"].get("basic") == "Bullish"
        )
        bear = sum(
            1 for m in messages
            if (m.get("entities") or {}).get("sentiment", {}) and
               (m.get("entities") or {})["sentiment"].get("basic") == "Bearish"
        )
        total = bull + bear

        if total == 0:
            score = _NEUTRAL
        else:
            score = bull / total

        trending = symbol.upper() in self._trending_set()
        if trending:
            score = min(1.0, score + 0.05)

        return round(max(0.0, min(1.0, score)), 4), {
            "source":   "stocktwits",
            "bullish":  bull,
            "bearish":  bear,
            "total":    total,
            "trending": trending,
        }


# ── 7. CNN Fear & Greed macro indicator ──────────────────────────────────────


class FearGreedFetcher:
    """
    Fetch the CNN Fear & Greed Index (no API key required).

    Endpoint: https://production.dataviz.cnn.io/index/fearandgreed/graphdata

    Raw score 0–100:
      0–24  = Extreme Fear
      25–44 = Fear
      45–55 = Neutral
      56–74 = Greed
      75–100 = Extreme Greed

    Normalised to [0, 1] so it can feed directly into MacroFetcher or be
    used as a standalone macro signal.  A score of 0.75+ suggests crowd
    euphoria (contrarian warning); 0.25- suggests crowd panic (contrarian buy).

    Cache: 1 hour (index updates once daily).
    """

    _URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"

    def fetch(self) -> Tuple[float, Dict]:
        """Returns (score, raw) where score ∈ [0, 1]."""
        data = _get(self._URL)
        if not data:
            return _NEUTRAL, {"source": "fear_greed", "status": "error"}

        try:
            fg        = data.get("fear_and_greed") or {}
            raw_score = float(fg.get("score", 50))
            score     = round(raw_score / 100.0, 4)
            return score, {
                "source":    "fear_greed",
                "raw_score": raw_score,
                "rating":    fg.get("rating", "Unknown"),
            }
        except Exception as exc:
            log.warning("FearGreedFetcher parse error: {}", exc)
            return _NEUTRAL, {"source": "fear_greed", "status": "parse_error"}


# ── 8. Alpha Vantage news sentiment ──────────────────────────────────────────


class AlphaVantageNewsFetcher:
    """
    Fetch news sentiment from Alpha Vantage (free tier: 25 calls/day).

    Endpoint: GET https://www.alphavantage.co/query?function=NEWS_SENTIMENT
    Key: ALPHA_VANTAGE_API_KEY in .env  (free at alphavantage.co/support/#api-key)

    Score = relevance-weighted average of per-article ticker_sentiment_score,
    remapped from [-1, 1] → [0, 1].  Missing key → neutral (0.50).
    """

    _BASE = "https://www.alphavantage.co/query"

    def __init__(self) -> None:
        self._key = os.getenv("ALPHA_VANTAGE_API_KEY", "")
        if not self._key:
            log.info(
                "ALPHA_VANTAGE_API_KEY not set — Alpha Vantage news scores neutral. "
                "Free key at https://www.alphavantage.co/support/#api-key"
            )

    def fetch(self, symbol: str) -> Tuple[float, Dict]:
        """
        Returns (score, raw).
        score > 0.5 → net positive news sentiment for the symbol.
        """
        if not self._key:
            return _NEUTRAL, {"source": "alphavantage", "status": "no_key"}

        url  = (
            f"{self._BASE}?function=NEWS_SENTIMENT"
            f"&tickers={symbol}&limit=20&apikey={self._key}"
        )
        data = _get(url)
        if not data or "feed" not in data:
            return _NEUTRAL, {"source": "alphavantage", "status": "error"}

        feed = data["feed"]
        if not feed:
            return _NEUTRAL, {"source": "alphavantage", "count": 0}

        sym_upper      = symbol.upper()
        total_weight   = 0.0
        weighted_score = 0.0

        for article in feed:
            for ts in article.get("ticker_sentiment", []):
                if str(ts.get("ticker", "")).upper() != sym_upper:
                    continue
                relevance  = float(ts.get("relevance_score",        0) or 0)
                sentiment  = float(ts.get("ticker_sentiment_score", 0) or 0)
                norm_score = (sentiment + 1.0) / 2.0   # [-1,1] → [0,1]
                total_weight   += relevance
                weighted_score += relevance * norm_score

        if total_weight == 0:
            return _NEUTRAL, {"source": "alphavantage", "count": len(feed)}

        score = weighted_score / total_weight
        return round(max(0.0, min(1.0, score)), 4), {
            "source":         "alphavantage",
            "article_count":  len(feed),
            "weighted_score": round(score, 4),
        }
