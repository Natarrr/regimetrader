"""scripts/run_pipeline.py
EDGAR + FMP + yfinance daily data pipeline.

Stiglitz (2001 Nobel) — asymmetric information: insider filing activity is a
credible, costly-to-fake signal. This pipeline sources it from two layers:
  1. FMP Ultimate            — pre-parsed insider trades + congress (FMP_API_KEY)
  2. SEC EDGAR direct        — Form 4 count + CEO buy flag (always, free)

FMP budget: ≤ 80 calls per run (per-ticker profile, first 80 tickers only).
Tickers 81+ fall back to yfinance for market cap to stay within 250/day limit.

Usage:
  python scripts/run_pipeline.py --tickers-file config/universe.csv --log-dir logs
  python -m scripts.run_pipeline --tickers-file config/universe.csv --verbose
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from regime_trader.utils.io import save_json_atomic  # noqa: E402
from regime_trader.services.fmp_client import FMPClient as _FMPClient, FMPEndpointError  # noqa: E402
from backend.market_intel.validator import validate_raw  # noqa: E402

log = logging.getLogger("run_pipeline")

# ── Weights (must sum to 1.0) ──────────────────────────────────────────────────
# Fix #3/8 — Jegadeesh-Titman momentum + orthogonal news split.
# momentum 20d (anti-alpha short-term reversal) → 12-1m Jegadeesh (Tetlock 2007).
# news(0.15) split into directional sentiment(0.10) + attention buzz(0.05).
# volume extracted from momentum into its own attention tilt (0.03).
# ── Weights (must sum to 1.0) ────────────────────────────────────────────────
# Rebalanced per Grinold-Kahn Fundamental Law guidance (PATCH-08):
#
#   insider_conviction  0.18 → 0.15  (sparse ~1% density on S&P 500 large-cap;
#                                      Cohen-Malloy-Pomorski 2012: CEO purchases
#                                      carry alpha, but only ~11% of universe)
#   momentum_long       0.22 → 0.25  (strongest empirical IC; Jegadeesh-Titman
#                                      1993: 12-1m momentum IC ≈ 0.06-0.09)
#   quality_piotroski   0.06 → 0.08  (most regime-stable signal; Piotroski 2000,
#                                      Novy-Marx 2013: robust across 2008/2020/2022)
#   analyst_revision    0.06 → 0.05  (reduce sell-side triplet combined weight;
#                                      Grinold-Kahn: correlated signals overstate BR)
#   price_target_upside 0.04 → 0.03  (staleness risk on quarterly targets)
#
#   All other factors unchanged. Total remains 1.0.
WEIGHTS = {
    "insider_conviction":  0.15,   # ↓ from 0.18 — sparse on large-cap universe
    "insider_breadth":     0.12,   # unchanged
    "congress":            0.08,   # unchanged — US-only, already discounted
    "news_sentiment":      0.10,   # unchanged — Tetlock (2007) IC stable
    "news_buzz":           0.03,   # unchanged — attention tilt only
    "momentum_long":       0.25,   # ↑ from 0.22 — Jegadeesh-Titman (1993)
    "volume_attention":    0.03,   # unchanged — Barber-Odean (2008)
    "analyst_consensus":   0.04,   # unchanged
    "analyst_revision":    0.05,   # ↓ from 0.06 — reduce sell-side triplet
    "price_target_upside": 0.03,   # ↓ from 0.04 — staleness risk
    "quality_piotroski":   0.08,   # ↑ from 0.06 — Piotroski (2000) regime-robust
    "transcript_tone":     0.04,   # unchanged
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, (
    f"WEIGHTS must sum to 1.0, got {sum(WEIGHTS.values()):.8f}. "
    "Check run_pipeline.WEIGHTS after any modification."
)
# Weight sum: 0.15+0.12+0.08+0.10+0.03+0.25+0.03+0.04+0.05+0.03+0.08+0.04 = 1.00 ✓

# Human-readable migration note used in runtime logs/tests
_WEIGHTS_MIGRATION_NOTE = (
    "Weights v3 (PATCH-08): rebalanced for Grinold-Kahn IR optimization. "
    "momentum_long 0.22→0.25, quality_piotroski 0.06→0.08, "
    "insider_conviction 0.18→0.15, analyst_revision 0.06→0.05, "
    "price_target_upside 0.04→0.03.",
)

# ── Congress feed cache path (module-level so tests can monkeypatch it) ────────
CONGRESS_CACHE_PATH = ROOT / ".cache" / "congress_cache.json"

# ── SEC ticker→CIK map (fetched once, disk-cached 24 h) ───────────────────────
_CIK_CACHE_PATH  = ROOT / ".cache" / "sec_cik_map.json"
_CIK_TTL_SECONDS = 24 * 3600
_SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_cik_map: Dict[str, str] = {}   # TICKER → zero-padded 10-digit CIK
_cik_map_loaded  = False

# PATCH 08: Track structural FMP endpoint failures found inside _score_ticker().
# The existing structural_failure_seen event only covers fetch_fmp_insider_all().
# This set tracks any other endpoint (analyst, transcript, etc.) that returns 4xx.
# Written from multiple threads — using a thread-safe set via a lock.
_structural_failures_lock = threading.Lock()
_structural_failures_in_scoring: set = set()  # set of broken endpoint paths

# ── SEC rate-limited HTTP ─────────────────────────────────────────────────────
# data.sec.gov allows up to 10 req/sec; we stay at ~8 with a shared lock so
# all worker threads collectively respect the limit (not per-thread).
_SEC_RATE_LOCK: threading.Lock = threading.Lock()
_SEC_RATE_LAST: float = 0.0
_SEC_MIN_DELAY: float = 0.12   # 1/8 s ≈ 8 req/s globally

_CONGRESS_TTL_HOURS = 24
_HOUSE_URL   = "https://house-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json"
_SENATE_URL  = "https://senate-stock-watcher-data.s3-us-west-2.amazonaws.com/data/all_transactions.json"
_INVALID_TICKERS = frozenset({"N/A", "--", "", "NONE", "NO TICKER"})

# ── Bull/bear word lists for news scoring ──────────────────────────────────────
_BULL = frozenset([
    "beat", "beats", "exceed", "exceeds", "upgrade", "upgrades", "upgraded",
    "buy", "outperform", "strong", "record", "rally", "surge", "gain", "growth",
    "bullish", "profit", "revenue", "raise", "raises", "tops", "jump", "soar",
    "boom", "positive", "breakthrough", "approval", "approved", "expands",
])
_BEAR = frozenset([
    "miss", "misses", "downgrade", "downgrades", "sell", "underperform",
    "concern", "decline", "weak", "loss", "cut", "fall", "drop", "recession",
    "layoff", "lawsuit", "fine", "warning", "risk", "volatile", "below",
    "disappoints", "halt", "investigation", "fraud", "bankruptcy", "default",
])

_KEY_ROLES = frozenset([
    "CEO", "CFO", "COO", "CTO", "DIRECTOR", "PRESIDENT",
    "CHIEF EXECUTIVE", "CHIEF FINANCIAL", "CHIEF OPERATING",
    "CHAIRMAN", "FOUNDER",
])


# ── Loaders ────────────────────────────────────────────────────────────────────

def load_tickers(csv_path: Path) -> List[Dict[str, str]]:
    """Markowitz (1990 Nobel) — load stratified ticker universe from CSV."""
    rows = []
    with csv_path.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            ticker = row.get("ticker", "").strip().upper()
            if ticker:
                rows.append({
                    "ticker":   ticker,
                    "sector":   row.get("sector", "Unknown").strip(),
                    "cap_tier": row.get("cap_tier", "large").strip(),
                })
    return rows


# ── FMP fetchers ───────────────────────────────────────────────────────────────

def _fmp_get(path: str, params: Dict, timeout: int = 20) -> Any:
    """Fama (2013 Nobel) — single FMP REST call with retry."""
    try:
        import requests as _req
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        s = _req.Session()
        retry = Retry(total=2, backoff_factor=0.5, status_forcelist=[500, 502, 503])
        s.mount("https://", HTTPAdapter(max_retries=retry))
        api_key = os.getenv("FMP_API_KEY", "")
        if not api_key:
            log.warning("FMP_API_KEY not set — skipping FMP call")
            return None
        url = f"https://financialmodelingprep.com/stable/{path.lstrip('/')}"
        params["apikey"] = api_key
        r = s.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        log.warning("FMP call %s failed: %s", path, exc)
        return None


def fetch_fmp_profiles(tickers: List[str]) -> Dict[str, float]:
    """Fetch market caps via FMP stable/batch-quote (single call, all tickers).

    Uses FMPClient.get_batch_quotes() — confirmed PASS on Ultimate.
    Replaces the old N-serial-calls + yfinance-fallback approach.
    No quota tracking needed: batch-quote is one call regardless of ticker count.
    """
    if not tickers:
        return {}
    client = _FMPClient()
    if not client._api_key:
        log.warning("FMP_API_KEY not set — market caps will be 0")
        return {t: 0.0 for t in tickers}

    batch = client.get_batch_quotes(tickers)
    result: Dict[str, float] = {}
    for ticker in tickers:
        row = batch.get(ticker, {})
        cap = float(row.get("marketCap", 0) or 0)
        result[ticker] = cap

    nonzero = sum(1 for v in result.values() if v > 0)
    log.info("FMP batch market caps: %d/%d tickers", nonzero, len(tickers))
    return result




def _parse_congress_transactions(
    transactions: List[Dict],
    cutoff: str,
    by_ticker: Dict[str, Dict],
    date_key: str = "transaction_date",
    type_key: str = "type",
    ticker_key: str = "ticker",
) -> None:
    """Shared parser for Stock Watcher (S3) and Quiver Quantitative transaction lists."""
    for tx in transactions:
        date_str = str(tx.get(date_key) or tx.get("date") or "")
        if date_str[:10] < cutoff:
            continue
        ticker = str(tx.get(ticker_key) or "").upper().strip()
        if not ticker or ticker in _INVALID_TICKERS:
            continue
        tx_type = str(tx.get(type_key) or "").lower()
        is_purchase = "purchase" in tx_type
        is_sale = "sale" in tx_type
        if not (is_purchase or is_sale):
            continue
        if ticker not in by_ticker:
            by_ticker[ticker] = {"purchases": 0, "sales": 0, "total": 0}
        if is_purchase:
            by_ticker[ticker]["purchases"] += 1
        else:
            by_ticker[ticker]["sales"] += 1
        by_ticker[ticker]["total"] += 1


def _fetch_fmp_congress(ticker: str) -> Optional[Dict]:
    """Fetch congressional trades for a single ticker via FMPClient.

    Returns populated dict (with recency_days) on success, None if key absent or fails.
    """
    try:
        client = _FMPClient()
        if not client._api_key:
            return None
        return client.get_congress_trades(ticker, lookback_days=180) or None
    except Exception as exc:
        log.warning("FMP congress fetch failed for %s: %s", ticker, exc)
        return None


def fetch_congress_buys(lookback_days: int = 90) -> Dict[str, Dict]:
    """Stiglitz (2001 Nobel) — fetch congressional trading data.

    Primary:  House/Senate Stock Watcher public S3 feeds (no API key).
    Fallback: FMP Ultimate /api/v4/senate-trading + /api/v4/house-trades (FMP_API_KEY).

    Filters to the lookback window and counts purchase vs sale transactions
    per ticker.  Results are cached for 24 h.

    $\\text{net\\_score} = \\frac{(purchases - sales)}{total + 1}$

    Returns:
        Dict keyed by ticker → {"purchases": int, "sales": int, "total": int}
        Returns {} when all sources fail (score_congress will return 0.0).
    """
    import requests as _req

    # ── Check 24-hour cache ───────────────────────────────────────────────────
    # Truthy check on by_ticker: an empty {} from a failed S3 run must not be
    # served as a valid cache hit — it would silence the Quiver fallback for
    # up to 24h, leaving congress_score=0.0 for all tickers.
    if CONGRESS_CACHE_PATH.exists():
        try:
            cached = json.loads(CONGRESS_CACHE_PATH.read_text(encoding="utf-8"))
            age_h = (time.time() - float(cached.get("_ts", 0))) / 3600
            by_ticker_cached = cached.get("by_ticker", {})
            if age_h < _CONGRESS_TTL_HOURS and by_ticker_cached:
                return by_ticker_cached
        except Exception:
            pass

    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).date().isoformat()
    by_ticker: Dict[str, Dict] = {}
    s3_ok = False

    # ── Primary: Stock Watcher S3 feeds ──────────────────────────────────────
    for url, label in [(_HOUSE_URL, "house"), (_SENATE_URL, "senate")]:
        try:
            resp = _req.get(url, timeout=30)
            if resp.status_code == 403:
                log.warning(
                    "Congress feed %s returned 403 — S3 bucket restricted; "
                    "will use FMP Ultimate fallback (FMP_API_KEY).",
                    label,
                )
                continue
            resp.raise_for_status()
            _parse_congress_transactions(resp.json(), cutoff, by_ticker)
            s3_ok = True
            log.info("Congress feed %s: OK — %d tickers", label, len(by_ticker))
        except Exception as exc:
            log.warning("Congress feed %s failed: %s", label, exc)

    # ── Fallback: FMP Ultimate (when S3 yields nothing) ──────────────────────
    if not s3_ok or not by_ticker:
        log.info("S3 congress feeds unavailable — trying FMP Ultimate fallback…")
        fmp_client = _FMPClient()
        if fmp_client._api_key:
            fmp_congress: Dict[str, Dict] = {}
            for ticker_key in list(by_ticker.keys()):
                result = _fetch_fmp_congress(ticker_key)
                if result:
                    fmp_congress[ticker_key] = result
            if fmp_congress:
                by_ticker = fmp_congress
                n_tx = sum(v.get("total", 0) for v in fmp_congress.values())
                log.info("FMP congress fallback: %d transactions across %d tickers",
                         n_tx, len(fmp_congress))
        else:
            log.warning(
                "FMP_API_KEY not set and S3 feeds are down — "
                "congress factor will be 0.0 (penalised) for all tickers."
            )

    # ── Persist cache ─────────────────────────────────────────────────────────
    try:
        save_json_atomic(CONGRESS_CACHE_PATH, {"_ts": time.time(), "by_ticker": by_ticker})
    except Exception as exc:
        log.debug("Congress cache write failed: %s", exc)

    return by_ticker


def score_congress(data: Optional[Dict]) -> float:
    """Stiglitz (2001 Nobel) — congressional net buy signal $\\in [0, 1]$.

    $score = \\frac{(purchases - sales) / (total + 1) + 1}{2}$

    A ticker with only purchases scores >0.5; only sales scores <0.5.
    Equal purchases/sales (truly neutral) scores 0.5.
    Missing data (None / empty dict) scores 0.0 so the cross-sectional
    normaliser sees a dead feed and penalises rather than grants neutral credit.

    Recency weighting: trades within 30 days get full credit; older trades
    decay linearly to 0.70× signal strength at 180 days (dampens towards 0.5,
    not towards 0, so the direction is preserved but urgency is lower).
    """
    if not data:
        # Dead API or ticker not traded by congress → 0.0, not 0.5.
        # When ALL tickers return 0.0, _cross_sectional_normalize treats it as
        # a fully-failed feed (all-zero branch) rather than the "all identical
        # non-zero → neutral 0.5" branch, which would silently waste 20% weight.
        return 0.0
    purchases = int(data.get("purchases", 0))
    sales     = int(data.get("sales", 0))
    total     = purchases + sales   # compute from actual values, not stored field
    if total == 0:
        return 0.50   # data present but no net activity → genuinely neutral
    raw = (purchases - sales) / (total + 1)   # $\in (-1, 1)$
    base_score = round((raw + 1) / 2, 4)       # $\to (0, 1)$

    # Recency multiplier: full credit ≤30 days, decay to 0.70× at 180 days.
    # Dampens *towards neutral (0.5)*, not towards zero — direction is preserved.
    recency_days = data.get("recency_days")
    if recency_days is not None and recency_days > 30:
        decay = max(0.70, 1.0 - 0.30 * min(recency_days - 30, 150) / 150)
        base_score = 0.5 + (base_score - 0.5) * decay

    return round(base_score, 4)


def _fetch_spy_return() -> float:
    """SPY 12-1 month return via FMP stable/historical-price-eod/full.

    Called once per pipeline run; result shared across all worker threads.
    Returns 0.0 on failure (symmetrical with ticker fallback).
    Reference: Jegadeesh & Titman (1993), Journal of Finance 48(1).
    """
    from regime_trader.services.fmp_client import FMPClient as _FC, fmp_prices_to_arrays
    try:
        rows = _FC().get_historical_prices("SPY", limit=310)
        closes, _, _ = fmp_prices_to_arrays(rows)
        if len(closes) < 22:
            return 0.0
        idx_far  = max(0, len(closes) - 252)
        idx_near = max(1, len(closes) - 21)
        return float((closes[idx_near] - closes[idx_far]) / closes[idx_far])
    except Exception as exc:
        log.warning("FMP SPY 12-1m baseline failed: %s — using 0.0", exc)
        return 0.0


def _fetch_regional_return(ticker: str, label: str) -> float:
    """Fetch 12-1 month return for a regional benchmark ETF via FMP.

    PATCH 06: Generic helper for EU/Asia regional momentum benchmarks.
    Jegadeesh & Titman (1993): momentum should be measured relative to
    the local peer group, not a foreign index.

    Args:
        ticker: ETF symbol (e.g. "EZU" for Europe, "AAXJ" for Asia ex-Japan)
        label:  Human-readable label for log messages

    Returns:
        12-1 month return as decimal, or 0.0 on any failure.
    """
    from regime_trader.services.fmp_client import FMPClient as _FC, fmp_prices_to_arrays  # noqa: PLC0415
    try:
        rows = _FC().get_historical_prices(ticker, limit=310)
        closes, _, _ = fmp_prices_to_arrays(rows)
        if len(closes) < 252:
            log.warning(
                "Regional benchmark %s (%s): only %d bars, need 252 — falling back to 0.0",
                ticker, label, len(closes),
            )
            return 0.0
        idx_far  = max(0, len(closes) - 252)
        idx_near = max(1, len(closes) - 21)
        ret = float((closes[idx_near] - closes[idx_far]) / closes[idx_far])
        log.info("Regional benchmark %s (%s) 12-1m return: %.4f (%.1f%%)", ticker, label, ret, ret * 100)
        return ret
    except Exception as exc:
        log.warning("FMP %s (%s) 12-1m baseline failed: %s — using 0.0", ticker, label, exc)
        return 0.0


def _fetch_eu_return() -> float:
    """iShares MSCI Eurozone ETF (EZU) 12-1 month return.

    EZU tracks the MSCI EMU Index (Eurozone large/mid cap).
    Used as the benchmark for European momentum signals.
    Fallback: 0.0 (neutral — no bias to local or US market).
    """
    return _fetch_regional_return("EZU", "MSCI Eurozone")


def _fetch_asia_return() -> float:
    """iShares MSCI All Country Asia ex Japan ETF (AAXJ) 12-1 month return.

    AAXJ covers large/mid cap across China, Korea, Taiwan, India, etc.
    Used as the benchmark for Asia ex-Japan momentum signals.
    Fallback: 0.0 (neutral).
    """
    return _fetch_regional_return("AAXJ", "MSCI Asia ex-Japan")


def _compute_spy_regime(spy_return_12_1m: float, spy_return_63d: Optional[float]) -> str:
    """Classify current SPY momentum regime.

    PATCH 10: Provides early warning for bear markets that develop without
    triggering VIX >= 30 (e.g. 2022 rate shock: VIX ~37 peak, SPY -19%).
    Hull (2015): regime shifts can occur while VIX is still moderate.

    Labels:
        BEAR_CRASH    : 63d return < -20%  (rapid collapse)
        BEAR_MOMENTUM : 63d return < -10%  (persistent selling)
        BEAR_TREND    : 12-1m return < -15% (long-term downtrend)
        BULL_STRONG   : 12-1m return > +30% (strong bull — watch for reversal)
        NORMAL        : everything else

    Args:
        spy_return_12_1m: SPY 12-1 month return (skip-month adjusted)
        spy_return_63d:   SPY 63-calendar-day return (None if unavailable)

    Returns:
        Regime label string.
    """
    if spy_return_63d is not None:
        if spy_return_63d < -0.20:
            return "BEAR_CRASH"
        if spy_return_63d < -0.10:
            return "BEAR_MOMENTUM"
    if spy_return_12_1m < -0.15:
        return "BEAR_TREND"
    if spy_return_12_1m > 0.30:
        return "BULL_STRONG"
    return "NORMAL"


def _fetch_spy_full_regime() -> Tuple[float, Optional[float], str]:
    """Fetch SPY 12-1m return, 63d return, and momentum regime label.

    PATCH 10: Extends _fetch_spy_return() to also compute the 63-day return
    for momentum regime classification.

    Returns:
        (spy_return_12_1m, spy_return_63d, regime_label)
        spy_return_63d is None if < 63 bars available.
    """
    from regime_trader.services.fmp_client import FMPClient as _FC, fmp_prices_to_arrays  # noqa: PLC0415
    try:
        rows = _FC().get_historical_prices("SPY", limit=310)
        closes, _, _ = fmp_prices_to_arrays(rows)
        if len(closes) < 22:
            return 0.0, None, "NORMAL"

        # 12-1m return (Jegadeesh-Titman skip-month momentum)
        if len(closes) >= 252:
            idx_far  = max(0, len(closes) - 252)
            idx_near = max(1, len(closes) - 21)
            ret_12_1m = float((closes[idx_near] - closes[idx_far]) / closes[idx_far])
        else:
            ret_12_1m = 0.0

        # 63-day return (regime detection)
        if len(closes) >= 63:
            idx_63 = max(0, len(closes) - 63)
            ret_63d: Optional[float] = float(
                (closes[-1] - closes[idx_63]) / closes[idx_63]
            ) if closes[idx_63] != 0 else None
        else:
            ret_63d = None

        regime = _compute_spy_regime(ret_12_1m, ret_63d)
        log.info(
            "SPY regime: 12-1m=%.4f (%.1f%%) 63d=%s regime=%s",
            ret_12_1m, ret_12_1m * 100,
            f"{ret_63d:.4f}" if ret_63d is not None else "N/A",
            regime,
        )
        return ret_12_1m, ret_63d, regime

    except Exception as exc:
        log.warning("_fetch_spy_full_regime failed: %s — defaulting to NORMAL", exc)
        return 0.0, None, "NORMAL"


def fetch_price_data(ticker: str, spy_return: float = 0.0) -> Dict[str, Any]:
    """Jegadeesh-Titman (1993) — 12-1 month SPY-relative return + volume spike.

    Replaces the former 20-day return, which was short-term reversal (anti-alpha).
    period="13mo" gives ~273 trading days: 252 for signal + 21 skip-month buffer.

    Volume spike uses a 90-bar baseline excluding the 5 recent bars (no leakage).
    volume_spike hard-capped at 20.0 to prevent diagnostic outliers for thinly
    traded stocks with occasional massive spikes.

    Returns:
        return_12_1m:     None  if < 252 bars (recent IPO — dead signal, not 0.0)
        spy_return_12_1m: passed-in SPY 12-1m baseline
        volume_spike:     5d avg / 90d avg, capped at 20.0

    On any error returns the default dict with return_12_1m=None (not 0.0, so the
    caller can distinguish "no data" from "genuinely flat" in the scorer).

    Reference: Jegadeesh & Titman (1993), Journal of Finance 48(1).
    """
    from regime_trader.services.fmp_client import FMPClient as _FC, fmp_prices_to_arrays

    _default: Dict[str, Any] = {
        "return_12_1m":     None,
        "spy_return_12_1m": spy_return,
        "volume_spike":     1.0,
    }
    try:
        rows = _FC().get_historical_prices(ticker, limit=310)
        closes, volumes, _ = fmp_prices_to_arrays(rows)

        if len(closes) < 5:
            return _default
        if len(closes) < 22:
            return _default

        if len(closes) < 252:
            log.info(
                "fetch_price_data %s: %d bars < 252 — return_12_1m=None (recent IPO)",
                ticker, len(closes),
            )
            ret_12_1m = None
        else:
            idx_far  = max(0, len(closes) - 252)
            idx_near = max(1, len(closes) - 21)
            ret_12_1m = float((closes[idx_near] - closes[idx_far]) / closes[idx_far])

        volume_spike = 1.0
        if len(volumes) >= 95:
            recent_avg   = sum(volumes[-5:]) / 5.0
            baseline_avg = sum(volumes[-95:-5]) / 90.0
            if baseline_avg > 0:
                volume_spike = round(min(20.0, recent_avg / baseline_avg), 4)

        return {
            "return_12_1m":     round(ret_12_1m, 6) if ret_12_1m is not None else None,
            "spy_return_12_1m": round(spy_return, 6),
            "volume_spike":     volume_spike,
        }
    except Exception as exc:
        log.debug("fetch_price_data %s failed: %s", ticker, exc)
        return _default


def score_momentum(
    ticker_return_20d: float,
    spy_return_20d: float = 0.0,
    volume_spike: float = 1.0,
) -> float:
    """DEPRECATED (Fix #3): 20-day SPY-relative momentum was short-term reversal,
    not Jegadeesh momentum. Replaced by score_momentum_long() in
    regime_trader/scoring/momentum_signals.py. Kept for git blame and legacy
    score computation; remove in Fix #5+.
    """
    r = max(-0.30, min(0.30, ticker_return_20d - spy_return_20d))
    return_score = round((r + 0.30) / 0.60, 4)
    vol_score    = round(min(1.0, max(0.0, (volume_spike - 1.0) / 4.0)), 4)
    return round(0.65 * return_score + 0.35 * vol_score, 4)


# ── SEC helpers ────────────────────────────────────────────────────────────────

def _sec_get(url: str, timeout: int = 20):
    """Rate-limited GET to any SEC endpoint (data.sec.gov or www.sec.gov/Archives).

    Uses a module-level shared lock so all worker threads together respect the
    SEC's 10 req/sec guidance.  Lock is released BEFORE the HTTP call so
    threads can execute requests in parallel within the rate budget.

    Raises requests.HTTPError on non-200.
    """
    global _SEC_RATE_LAST
    import requests as _req
    # `or` handles both absent key AND empty-string (e.g. when secret is unset
    # and workflow sets EDGAR_USER_AGENT='').  An empty User-Agent causes 403.
    ua = os.getenv("EDGAR_USER_AGENT") or "regime-trader-research n.tardy@hotmail.fr"
    headers = {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}

    # Hold lock only for the rate-limit sleep + timestamp update, NOT for the
    # HTTP call — allows genuine parallelism across worker threads.
    with _SEC_RATE_LOCK:
        elapsed = time.time() - _SEC_RATE_LAST
        if elapsed < _SEC_MIN_DELAY:
            time.sleep(_SEC_MIN_DELAY - elapsed)
        _SEC_RATE_LAST = time.time()

    resp = _req.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp


# ── EDGAR Form 4 counter ───────────────────────────────────────────────────────

def _load_cik_map() -> Dict[str, str]:
    """Fetch SEC ticker→CIK map (disk-cached 24 h). Raises on network failure."""
    global _cik_map, _cik_map_loaded
    if _cik_map_loaded:
        return _cik_map

    if _CIK_CACHE_PATH.exists():
        try:
            cached = json.loads(_CIK_CACHE_PATH.read_text(encoding="utf-8"))
            if time.time() - float(cached.get("_ts", 0)) < _CIK_TTL_SECONDS:
                _cik_map = cached["tickers"]
                _cik_map_loaded = True
                log.info("CIK map loaded from cache: %d tickers", len(_cik_map))
                return _cik_map
        except Exception:
            pass

    resp = _sec_get(_SEC_TICKERS_URL, timeout=30)
    data = resp.json()
    ticker_map: Dict[str, str] = {
        str(e["ticker"]).upper(): str(e["cik_str"]).zfill(10)
        for e in data.values()
        if e.get("ticker") and e.get("cik_str")
    }
    _CIK_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_json_atomic(_CIK_CACHE_PATH, {"_ts": time.time(), "tickers": ticker_map})
    _cik_map = ticker_map
    _cik_map_loaded = True
    log.info("CIK map fetched from SEC: %d tickers", len(_cik_map))
    return _cik_map



# ── yfinance scorers ───────────────────────────────────────────────────────────

def _score_news_sentiment_yfinance(ticker: str) -> float:
    """Fallback directional sentiment from yfinance headlines (bull/bear word-count).

    Fix #4: dead-signal convention is 0.0, NOT 0.5. The rest of the codebase
    (score_insider_value, score_congress, score_momentum_long) returns 0.0 on
    missing/dead data so the cross-sectional normalizer triggers the all-zero
    branch (penalised) rather than the all-same-non-zero branch (silently 0.5).

    Returns 0.0 when:
      - No headlines returned by yfinance
      - All headlines contain zero bull AND zero bear words
      - yfinance raises any exception
    """
    try:
        import yfinance as yf
        news = yf.Ticker(ticker).news or []
        scores = []
        for item in news[:8]:
            content = item.get("content", {})
            title = (
                content.get("title", "") if isinstance(content, dict)
                else item.get("title", "")
            )
            if not title:
                continue
            words = set(title.lower().split())
            bull  = len(words & _BULL)
            bear  = len(words & _BEAR)
            if bull == 0 and bear == 0:
                continue  # Fix #4: skip neutral headlines, don't weight them 0.5
            scores.append(max(0.10, min(0.90, 0.50 + 0.20 * (bull - bear))))
        if not scores:
            return 0.0  # Fix #4: dead signal — no scored headlines
        return round(sum(scores) / len(scores), 4)
    except Exception:
        return 0.0


def _score_news_buzz_yfinance(ticker: str) -> float:
    """Fallback buzz signal: count of yfinance recent headlines (no sentiment).

    Returns 0.0 if no headlines returned.
    """
    try:
        import yfinance as yf
        news = yf.Ticker(ticker).news or []
        n = len(news[:50])  # cap at 50 to mirror log1p(50) saturation
        if n == 0:
            return 0.0
        return round(min(1.0, math.log1p(n) / math.log1p(50)), 4)
    except Exception:
        return 0.0


def _score_news_yfinance(ticker: str) -> float:
    """DEPRECATED (Fix #3): combined sentiment+buzz. Replaced by
    _score_news_sentiment_yfinance / _score_news_buzz_yfinance.
    Kept for git blame; remove in Fix #5+.
    """
    return _score_news_sentiment_yfinance(ticker)


def score_news_fmp(ticker: str) -> float:
    """DEPRECATED (Fix #3): combined FMP sentiment+buzz (0.60/0.40 formula).
    Replaced by score_news_sentiment_combined / score_news_buzz_combined in
    regime_trader/scoring/news_signals.py. Kept for legacy score computation;
    remove in Fix #5+.
    """
    client = _FMPClient()
    articles = client.get_news_raw_articles(ticker)
    if not articles:
        return _score_news_sentiment_yfinance(ticker)
    positive = sum(1 for a in articles if a.get("sentiment") == "Positive")
    total = len(articles)
    if total == 0:
        return _score_news_sentiment_yfinance(ticker)
    buzz_norm = min(1.0, total / 50.0)
    return round(0.60 * (positive / total) + 0.40 * buzz_norm, 4)


def score_analyst_consensus(ticker: str, client=None) -> tuple[float, str]:
    """Sell-side analyst consensus score from FMP /stable/grades-consensus.

    Womack (1996, JF): analyst upgrades/downgrades have significant post-event
    drift — the direction of the consensus is a credible, widely-used signal.
    This maps the ratings distribution to a continuous [0, 1] score weighted by
    count, treating the distribution as a probability mass over the rating scale.

    Args:
        ticker: Ticker symbol.
        client: Optional shared FMPClient instance. If None, creates a new one.
                Pass the shared pipeline client so health_report() captures
                all endpoint calls and failures in fmp_health.json.

    Rating → score mapping (linear 5-point scale):
        strongBuy  → 1.00
        buy        → 0.75
        hold       → 0.50
        sell       → 0.25
        strongSell → 0.00

    Returns (0.0, "none")   — dead signal, no coverage (total == 0).
    Returns (0.0, "sparse") — insufficient coverage (total < 3 analysts).
    Returns (score, "fmp_consensus") — valid weighted average.
    Never raises — returns (0.0, "error") on any exception.
    """
    _SCORE_MAP = {
        "strongBuy":  1.00,
        "buy":        0.75,
        "hold":       0.50,
        "sell":       0.25,
        "strongSell": 0.00,
    }
    try:
        _client = client if client is not None else _FMPClient()
        data = _client.get_analyst_ratings(ticker)
        if not data:
            return 0.0, "none"

        total = sum(int(data.get(k, 0) or 0) for k in _SCORE_MAP)
        if total == 0:
            return 0.0, "none"
        if total < 3:
            return 0.0, "sparse"

        weighted = sum(
            _SCORE_MAP[k] * int(data.get(k, 0) or 0)
            for k in _SCORE_MAP
        )
        return round(weighted / total, 4), "fmp_consensus"
    except Exception as exc:
        log.debug("score_analyst_consensus %s failed: %s", ticker, exc)
        return 0.0, "error"


def score_analyst_revision(
    revision_pct: Optional[float],
    n_analysts: int,
) -> float:
    """EPS estimate revision momentum score in [0, 1].

    Analyst estimate revision momentum is a separate alpha source from price
    momentum (Jegadeesh-Titman 1993): it captures the direction of fundamental
    re-rating by sell-side analysts, which has been shown to predict returns
    independently of past price performance.

    Reference: Chan, Jegadeesh & Lakonishok (1996, JF) "Momentum Strategies" —
    estimate revisions contribute an independent return-predictive signal,
    particularly when analyst coverage is broad (high n_analysts).

    Scoring:
      1. revision_pct is clipped to [−0.30, +0.30] (extreme revisions of
         ±30%+ are treated as ±30% to prevent outlier domination).
      2. Linear mapping to [0, 1]: (clip + 0.30) / 0.60
         −30% revision → 0.0 (maximum bearish)
          0% revision  → 0.5 (neutral)
         +30% revision → 1.0 (maximum bullish)
      3. Coverage weight: min(1.0, n_analysts / 10)
         Thin analyst coverage (< 10) reduces confidence proportionally.
         Scores from a single analyst are weighted at 0.1; 10+ analysts get
         full weight. This prevents micro-cap noise from dominating.

    Returns 0.0 (dead signal) when:
      - revision_pct is None (endpoint unavailable or < 3 estimates)
      - n_analysts < 3 (insufficient coverage — sparse, not neutral)
    """
    if revision_pct is None or n_analysts < 3:
        return 0.0
    clipped = max(-0.30, min(0.30, revision_pct))
    raw_score = (clipped + 0.30) / 0.60
    coverage_weight = min(1.0, n_analysts / 10.0)
    return round(raw_score * coverage_weight, 4)


def score_news_sentiment_combined(
    ticker: str,
) -> tuple[float, str, Optional[float], int]:
    """Recency-weighted directional sentiment boosted by EPS surprise (PEAD).

    Returns (score, source, earnings_surprise_pct, earnings_surprise_days).

    Base signal: Tetlock (2007, JF) recency-weighted bull/bear headline sentiment.
    EPS boost:   Bernard & Thomas (1989, JAE) PEAD — standardized unexpected
                 earnings (SUE) predicts returns for up to 90 days post-announcement.

    Boost formula (applied only when surprise is available AND days_since <= 90):
        boost       = clip(surprise_pct × 0.5, −0.20, +0.20)
        final_score = clip(headline_score + boost, 0.0, 1.0)

    The ×0.5 dampener keeps EPS from overwhelming the headline signal on extreme
    beats/misses. The 90-day PEAD horizon follows Bernard & Thomas (1989).

    source label: "fmp+eps" when EPS data was used, "fmp" otherwise.
    """
    from regime_trader.scoring.news_signals import score_news_sentiment  # noqa: PLC0415

    client = _FMPClient()
    articles = client.get_news_raw_articles(ticker)

    headline_score = 0.0
    base_source = "none"
    if articles:
        s = score_news_sentiment(articles)
        if s > 0.0:
            headline_score = s
            base_source = "fmp"

    # PEAD boost — FMP /stable/earnings-surprises (Bernard & Thomas 1989)
    # PATCH 04: Apply exponential decay with half-life = 20 days.
    # Bernard & Thomas (1989): drift is ~100% of peak at day 1, ~50% at day 20,
    # ~25% at day 40, and ~12% at day 60. A flat 90-day window overstates
    # the boost for old surprises. Decay formula: exp(-days * ln(2) / 20).
    surprise_pct, days_since = client.get_earnings_surprise(ticker)

    if surprise_pct is not None and days_since <= 90 and base_source != "none":
        _PEAD_HALF_LIFE_DAYS = 20.0  # Bernard & Thomas (1989), JAE
        pead_decay = math.exp(-days_since * math.log(2) / _PEAD_HALF_LIFE_DAYS)
        # Dampen to ±20pp max, then apply decay so older surprises contribute less
        boost = max(-0.20, min(0.20, surprise_pct * 0.5 * pead_decay))
        final_score = max(0.0, min(1.0, headline_score + boost))
        log.debug(
            "PEAD boost %s: surprise=%.3f days=%d decay=%.3f boost=%.4f",
            ticker, surprise_pct, days_since, pead_decay, boost,
        )
        return round(final_score, 4), "fmp+eps", round(surprise_pct, 6), days_since

    return headline_score, base_source, surprise_pct, days_since


def score_news_buzz_combined(ticker: str) -> tuple[float, str]:
    """Attention/buzz signal — volume of recent coverage (Barber-Odean 2008).

    Returns (score, source) where source ∈ {"fmp", "none"}.
    Uses FMP stable/news/stock exclusively.
    """
    from regime_trader.scoring.news_signals import score_news_buzz  # noqa: PLC0415

    client = _FMPClient()
    articles = client.get_news_raw_articles(ticker)
    if articles:
        s = score_news_buzz(articles)
        if s > 0.0:
            return s, "fmp"
    return 0.0, "none"


def score_transcript_tone(ticker: str, client=None) -> tuple[float, str]:
    """Score earnings transcript guidance tone via FMP transcript text.

    Args:
        ticker: Ticker symbol.
        client: Optional shared FMPClient instance. If None, creates a new one.
                Pass the shared pipeline client so health_report() captures
                all endpoint calls and failures in fmp_health.json.

    Returns (score, source) where source is 'fmp_transcript:<tone>' or 'none'.
    """
    try:
        _client = client if client is not None else _FMPClient()
        txt = _client.get_earnings_transcript(ticker, max_chars=3000)
        if not txt:
            return 0.0, "none"
        text = str(txt).lower()

        raise_phrases = [
            "raising guidance", "raised guidance", "increase our guidance",
            "raising our full-year", "above the high end", "raising our outlook",
            "above our guidance", "raising revenue guidance",
        ]
        lower_phrases = [
            "lowering guidance", "lowered guidance", "reduce our guidance",
            "below our guidance", "revising guidance lower", "lowering our outlook",
            "below the low end", "headwinds",
        ]
        maintain_phrases = [
            "reaffirming guidance", "reaffirm", "maintaining guidance", "on track to",
            "comfortable with our guidance", "reiterate", "confident in our",
        ]

        cnt_raise = sum(text.count(p) for p in raise_phrases)
        cnt_lower = sum(text.count(p) for p in lower_phrases)
        cnt_maint = sum(text.count(p) for p in maintain_phrases)

        log.debug("score_transcript_tone %s: raise=%d lower=%d maintain=%d",
                  ticker, cnt_raise, cnt_lower, cnt_maint)

        total = cnt_raise + cnt_lower + cnt_maint
        if total == 0:
            return 0.0, "none"

        # Majority wins; tie -> maintain
        if cnt_raise > cnt_lower and cnt_raise > cnt_maint:
            return 0.80, "fmp_transcript:raised"
        if cnt_lower > cnt_raise and cnt_lower > cnt_maint:
            return 0.20, "fmp_transcript:lowered"
        # ties and maintain majority
        return 0.55, "fmp_transcript:reaffirm"
    except Exception as exc:
        log.debug("score_transcript_tone %s failed: %s", ticker, exc)
        return 0.0, "none"


def fetch_fmp_insider_all(
    tickers: List[str],
    lookback_days: int = 180,
    max_workers: int = 10,
    client: Optional[Any] = None,
) -> Dict[str, Tuple[float, int]]:
    """Fetch insider purchase data for all tickers via FMP stable/insider-trading/search.

    Form 4 insider purchases are a credible, costly-to-fake signal (Stiglitz 2001).
    Uses FMPClient with limit=500 per ticker to cover 180-day lookback for mega-caps.
    max_workers=10: FMP Ultimate cap is 50 req/s; 10 threads at ~30 rps is safe.

    Args:
        client: Optional pre-created FMPClient instance. If None, creates one.
                Pass a shared client so health_report() captures all calls/failures.

    Returns {ticker: (total_purchases_usd, days_since_most_recent)}.
    Tickers with no qualifying purchases get (0.0, 0).
    Returns {} if FMP_API_KEY is not set.
    """
    if client is None:
        client = _FMPClient()
    if not client._api_key:
        log.info("FMP_API_KEY not set -- skipping FMP insider pre-fetch")
        return {}

    if not tickers:
        return {}

    structural_failure_seen = threading.Event()

    def _fetch_one(ticker: str) -> Tuple[str, Tuple[float, int]]:
        try:
            result = client.get_insider_purchases(ticker, lookback_days=lookback_days)
        except FMPEndpointError as exc:
            log.error(
                "FMP structural failure on insider route for %s: %s "
                "— setting structural_failure flag. Do not lower circuit-breaker.",
                ticker, exc,
            )
            structural_failure_seen.set()
            result = (0.0, 0)
        except Exception as exc:
            log.debug("FMP insider fetch failed for %s: %s", ticker, type(exc).__name__)
            result = (0.0, 0)
        log.debug("FMP insider %s: $%.0f", ticker, result[0])
        return ticker, result

    results: Dict[str, Tuple[float, int]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for ticker, value in pool.map(_fetch_one, tickers):
            results[ticker] = value

    if structural_failure_seen.is_set():
        log.error(
            "FMP insider route is structurally broken — insider factors will be zeroed. "
            "Check fmp_health.json and investigate the endpoint before the next run."
        )

    nonzero = sum(1 for v in results.values() if v[0] > 0)
    log.info(
        "FMP insider pre-fetch complete: %d/%d tickers with purchases",
        nonzero, len(tickers),
    )
    return results


# ── Per-ticker scorer ──────────────────────────────────────────────────────────

def score_edgar(form4_count: int) -> float:
    """Stiglitz (2001): normalise Form 4 filing count to [0, 0.90] using log-scale.

    0 filings -> 0.0 (penalised, not neutral).  Consistent with score_insider_value
    and score_congress: a dead/absent signal is 0.0 so the cross-sectional
    normaliser triggers the all-zero branch (treats it as a dead feed) rather than
    the all-same-non-zero branch (which silently returns 0.50 neutral for everyone).

    Log-scale prevents the old linear formula from saturating at 0.90 for any company
    with >= 5 filings (which is virtually every large-cap), making the factor useless
    cross-sectionally.  log1p(n)/log1p(200) maps [1, 200+] to roughly [0.10, 0.90]
    with good spread across the typical 5-160 filing range.
    """
    if form4_count <= 0:
        return 0.0
    return round(min(0.90, math.log1p(form4_count) / math.log1p(200)), 4)


def score_insider_value(
    key_purchases_usd: float,
    market_cap: float,
    days_since_most_recent: int = 0,
) -> float:
    """Stiglitz (2001 Nobel) — dollar conviction score for insider purchases in [0, 1].

    Maps total open-market purchase value as % of market cap to a score:
      0%      -> 0.0   (no purchases = dead signal, penalised not neutral)
      0.01%   -> ~0.30 (floor — small but credible)
      0.10%   -> ~0.65 (mid — meaningful conviction)
      1.00%+  -> ~0.90 (ceiling — exceptional conviction)

    Uses log-scale so small buys still count while large buys don't explode.
    Recency decay: purchases older than 30 days decay toward 0.50 neutral
    (direction preserved but urgency reduced — same decay as score_congress).
    """
    if key_purchases_usd <= 0 or market_cap <= 0:
        return 0.0

    pct = key_purchases_usd / market_cap
    raw = min(1.0, math.log1p(pct * 10000) / math.log1p(100))
    base_score = round(0.30 + 0.60 * raw, 4)

    if days_since_most_recent > 30:
        decay = max(0.70, 1.0 - 0.30 * min(days_since_most_recent - 30, 150) / 150)
        base_score = round(0.5 + (base_score - 0.5) * decay, 4)

    return base_score



# ── EDGAR submissions API (replaces CGI browse + yfinance insider) ─────────────

def _parse_form4_xml(cik: str, accession: str, primary_doc: str) -> List[Dict]:
    """Fetch and parse a Form 4 XML filing from EDGAR Archives.

    Returns list of dicts: {code, value, title, insider_id, date, is_ceo}.
      code:       P=open-market purchase, S=sale, A=award, F=tax withholding, etc.
      value:      shares × price (USD, approximate)
      title:      officer/director title of the reporting owner
      insider_id: rptOwnerCik — stable per-person identifier for deduplication
      date:       transactionDate/value ISO string (YYYY-MM-DD)
      is_ceo:     True when officerTitle contains CEO/CFO/COO/CTO/PRESIDENT/CHAIRMAN

    Returns [] on any failure (network, non-XML .htm filing, parse error).
    HTTP call goes through the shared _sec_get() rate limiter.

    Note: the SEC submissions API returns `primaryDocument` as the XSLT-styled
    path (e.g. "xslF345X06/form4.xml"), which serves rendered HTML, not raw XML.
    We strip any leading subdirectory prefix so we always fetch the machine-readable
    XML at the accession root (e.g. "form4.xml").
    """
    from xml.etree import ElementTree as ET
    acc_nodash = accession.replace("-", "")
    cik_int = str(int(cik))   # strip leading zeros for the Archives path segment

    # Strip XSLT subdirectory prefix — the real XML lives at the accession root.
    # "xslF345X06/form4.xml" -> "form4.xml", "form4.xml" -> "form4.xml"
    doc_filename = primary_doc.split("/")[-1] if "/" in primary_doc else primary_doc

    url = (
        f"https://www.sec.gov/Archives/edgar/data/{cik_int}"
        f"/{acc_nodash}/{doc_filename}"
    )
    try:
        resp = _sec_get(url)
    except Exception as exc:
        log.debug("Form 4 fetch failed %s: %s", url, exc)
        return []
    try:
        root = ET.fromstring(resp.text)
    except ET.ParseError:
        return []   # .htm or plain-text filing — not XML-parseable

    # Strip namespace prefixes so findall() works regardless of xmlns= declaration
    for el in root.iter():
        if "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]

    # Reporting owner — title and stable CIK identifier for breadth deduplication
    officer_title = ""
    title_el = root.find(".//officerTitle")
    if title_el is not None and title_el.text:
        officer_title = title_el.text.strip()
    if not officer_title:
        is_dir_el = root.find(".//isDirector")
        if is_dir_el is not None and is_dir_el.text == "1":
            officer_title = "Director"

    insider_id = ""
    cik_el = root.find(".//rptOwnerCik")
    if cik_el is not None and cik_el.text:
        insider_id = cik_el.text.strip()

    _CEO_TITLES = frozenset(["CEO", "CFO", "COO", "CTO", "PRESIDENT", "CHAIRMAN", "FOUNDER",
                              "CHIEF EXECUTIVE", "CHIEF FINANCIAL", "CHIEF OPERATING"])
    is_ceo = any(t in officer_title.upper() for t in _CEO_TITLES)

    transactions: List[Dict] = []
    for tx in root.findall(".//nonDerivativeTransaction"):
        code_el = tx.find(".//transactionCode")
        if code_el is None or not code_el.text:
            continue
        code = code_el.text.strip()

        shares_el = tx.find(".//transactionShares/value")
        price_el  = tx.find(".//transactionPricePerShare/value")
        date_el   = tx.find(".//transactionDate/value")
        try:
            shares = float(shares_el.text) if shares_el is not None else 0.0
        except (TypeError, ValueError):
            shares = 0.0
        try:
            price = float(price_el.text) if price_el is not None else 0.0
        except (TypeError, ValueError):
            price = 0.0
        tx_date = (date_el.text.strip()[:10] if date_el is not None and date_el.text else "")

        transactions.append({
            "code":       code,
            "value":      shares * price,
            "title":      officer_title,
            "insider_id": insider_id,
            "date":       tx_date,
            "is_ceo":     is_ceo,
        })

    return transactions


def fetch_edgar_data(
    ticker: str,
    lookback_days: int = 180,
    max_filings: int = 10,
) -> Tuple[int, float, bool, int, float, List[Dict], List[Dict]]:
    """Fetch Form 4 count and insider transactions from the SEC submissions API.

    Stiglitz (2001 Nobel) / Akerlof (2001 Nobel) — replaces both the legacy
    CGI browse endpoint (EdgarService.list_filings) and the yfinance insider
    path, both of which fail silently in GitHub Actions CI environments.

    Uses data.sec.gov/submissions/CIK{cik}.json (official programmatic API)
    for EDGAR count, then fetches up to max_filings Form 4 XML documents.
    Parses P-code (open-market purchase) and S-code (sale) transactions,
    enriched with insider_id, date, and is_ceo for orthogonal decomposition.

    Returns:
        (form4_count, total_purchases_usd, ceo_buy_flag, days_since_most_recent,
         ceo_purchase_usd, p_transactions, s_transactions)

    Raises on network failure so the caller can set _edgar_ok=False.
    """
    cik_map = _load_cik_map()
    cik = cik_map.get(ticker.upper())
    if not cik:
        log.debug("No SEC CIK for %s", ticker)
        return 0, 0.0, False, 0, 0.0, [], []

    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = _sec_get(url)   # raises on failure → caller sets _edgar_ok=False
    data = resp.json()

    recent     = data.get("filings", {}).get("recent", {})
    forms      = recent.get("form", [])
    dates      = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    prim_docs  = recent.get("primaryDocument", [])

    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).date().isoformat()

    form4_filings = [
        {
            "date":      dates[i],
            "accession": accessions[i] if i < len(accessions) else "",
            "doc":       prim_docs[i]  if i < len(prim_docs)  else "",
        }
        for i in range(len(forms))
        if forms[i] == "4" and i < len(dates) and dates[i] >= cutoff
    ]
    form4_count = len(form4_filings)
    log.debug("EDGAR %s: CIK=%s form4=%d (last %dd)", ticker, cik, form4_count, lookback_days)

    # Parse up to max_filings most-recent Form 4 XMLs — collect P and S transactions
    p_transactions: List[Dict] = []  # open-market purchases by key officers
    s_transactions: List[Dict] = []  # sales by key officers
    for filing in form4_filings[:max_filings]:
        if not filing["accession"] or not filing["doc"]:
            continue
        txs = _parse_form4_xml(cik, filing["accession"], filing["doc"])
        for tx in txs:
            if not any(role in tx["title"].upper() for role in _KEY_ROLES):
                continue
            if tx["code"] == "P":
                p_transactions.append({**tx, "date": tx.get("date") or filing["date"]})
            elif tx["code"] == "S":
                s_transactions.append({**tx, "date": tx.get("date") or filing["date"]})

    total_purchases_usd = sum(tx["value"] for tx in p_transactions)
    ceo_purchase_usd    = sum(tx["value"] for tx in p_transactions if tx.get("is_ceo"))
    # Fix #7: legacy bool preserved for 30-day backward compat; not used for scoring.
    ceo_buy             = ceo_purchase_usd > 0

    days_since_most_recent = 0
    if p_transactions:
        log.debug(
            "INSIDER %s: %d P-txs $%.0f ceo_buy=%s ceo_usd=%.0f",
            ticker, len(p_transactions), total_purchases_usd, ceo_buy, ceo_purchase_usd,
        )
        # PATCH 01: Use transaction date (not filing date) for recency.
        # SEC Form 4 filing deadline is 2 business days after the transaction,
        # so filing_date can overstate signal freshness by 0–2 days.
        # _parse_form4_xml populates tx["date"] from <transactionDate/value>.
        # Fall back to filing date only when transaction dates are unavailable.
        tx_dates = [tx.get("date", "") for tx in p_transactions if tx.get("date")]
        if tx_dates:
            most_recent_date_str = max(tx_dates)  # most recent TRANSACTION date
            log.debug("INSIDER %s: using transaction date %s for recency", ticker, most_recent_date_str)
        elif form4_filings:
            most_recent_date_str = form4_filings[0]["date"]  # fallback: filing date
            log.debug("INSIDER %s: no tx dates found, using filing date %s", ticker, most_recent_date_str)
        else:
            most_recent_date_str = ""
        if most_recent_date_str:
            try:
                from datetime import date as _date
                delta = (
                    datetime.now(timezone.utc).date()
                    - _date.fromisoformat(most_recent_date_str[:10])
                ).days
                days_since_most_recent = max(0, delta)
            except Exception:
                days_since_most_recent = 0

    return (
        form4_count, total_purchases_usd, ceo_buy, days_since_most_recent,
        ceo_purchase_usd, p_transactions, s_transactions,
    )


# ── Multi-market helpers ───────────────────────────────────────────────────────

def _load_registry_tickers() -> Dict[str, List[str]]:
    """Load EU/Asia ticker lists from config/ticker_registry.json."""
    reg_path = ROOT / "config" / "ticker_registry.json"
    try:
        data = json.loads(reg_path.read_text(encoding="utf-8"))
        return {
            "EUROPE": [e["ticker"] for e in data.get("europe", [])],
            "ASIA":   [e["ticker"] for e in data.get("asia", [])],
        }
    except Exception as exc:
        log.warning("ticker_registry load failed: %s — EU/Asia skipped", exc)
        return {"EUROPE": [], "ASIA": []}


def _registry_meta() -> Dict[str, Dict[str, Any]]:
    """Return {ticker: {sector, cap_tier, company_name}} from ticker_registry.json."""
    reg_path = ROOT / "config" / "ticker_registry.json"
    try:
        data = json.loads(reg_path.read_text(encoding="utf-8"))
        meta: Dict[str, Dict[str, Any]] = {}
        for e in data.get("europe", []) + data.get("asia", []):
            meta[e["ticker"]] = {
                "sector":       e["sector"],
                "cap_tier":     e["cap_tier"],
                "company_name": e.get("name", ""),
            }
        return meta
    except Exception:
        return {}


def _score_ticker_international(
    entry: Any,
    spy_return_baseline: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """Unified scorer for EUROPE and ASIA entries using the same factor formulas as US.

    Replaces _score_ticker_eu and _score_ticker_asia (Fix #5).

    Fix #5 diagnostic (2026-05-23): FMP Ultimate returns 403 Forbidden for all
    non-US symbols. yfinance provides price+volume universally. Therefore:

      Available factors:  momentum_long_score, volume_attention_score
      Structurally absent: insider_conviction_score, insider_breadth_score,
                           congress_score, news_sentiment_score, news_buzz_score

    Factor output semantics:
      - None  → structurally absent: factor was never computable for this market.
                 Weight excluded from renormalization denominator.
      - 0.0   → dead signal: factor is available for this market but returned no data
                 for this ticker (e.g. momentum_long=0.0 for a <252-bar ticker).
                 Weight is included but ticker is penalized in cross-sectional normalizer.

    final_score is NOT set here — it is computed downstream after cross-sectional
    neutralization with group_by=("market", "sector", "cap_tier"), then weights
    are renormalized via renormalize_weights_for_market(WEIGHTS, market).

    source_reliability is preserved as metadata; it is NOT a score multiplier (Fix #5).
    """
    from regime_trader.scoring.momentum_signals import score_momentum_long, score_volume_attention  # noqa: PLC0415

    market_str = entry.market.value  # "EUROPE" or "ASIA"
    rf = entry.raw_factors

    try:
        # ── momentum_long_score (Jegadeesh-Titman 1993) ───────────────────────
        return_12_1m = rf.get("return_12_1m")  # float | None from FMPFetcher
        if return_12_1m is not None:
            momentum_long_score = score_momentum_long(
                float(return_12_1m),
                spy_return_12_1m=spy_return_baseline,
            )
        else:
            momentum_long_score = 0.0  # dead signal — insufficient history

        # ── volume_attention_score (Barber-Odean 2008) ────────────────────────
        volume_spike = float(rf.get("volume_spike", 0.0) or 0.0)
        volume_attention_score = score_volume_attention(volume_spike)

        # ── PATCH 07: quality_piotroski via FMP ratios-ttm (PASS for EU/Asia) ────
        # Piotroski (2000): F-Score is an accounting-identity signal — universally
        # applicable regardless of exchange. FMP stable/ratios-ttm confirmed PASS
        # for SAP.DE and 7203.T in Phase-0 smoke test (2026-05-30).
        # Novy-Marx (2013): gross profitability is a cross-regime premium.
        quality_piotroski_score = 0.0
        price_target_upside_score = 0.0
        try:
            from regime_trader.services.fmp_client import FMPClient as _FC_intl  # noqa: PLC0415
            _intl_client = _FC_intl()
            if _intl_client._api_key:
                quality_piotroski_score = _intl_client.get_quality_score(entry.ticker)
                pt_upside = _intl_client.get_upside_to_target(entry.ticker)
                price_target_upside_score = pt_upside if pt_upside is not None else 0.0
                log.debug(
                    "PATCH 07 %s (%s): quality=%.4f pt_upside=%.4f",
                    entry.ticker, market_str, quality_piotroski_score, price_target_upside_score,
                )
        except Exception as exc:
            log.debug("PATCH 07 quality/pt_upside fetch %s: %s", entry.ticker, exc)
            quality_piotroski_score = 0.0
            price_target_upside_score = 0.0

        # ── structurally absent factors → None ───────────────────────────────
        # congress, insider, news: no data source for non-US markets.
        # None signals "weight excluded" in renormalize_weights_for_market.
        return {
            "ticker":                    entry.ticker,
            "company_name":              rf.get("company_name", ""),
            "sector":                    entry.sector,
            "cap_tier":                  entry.cap_tier,
            "market":                    market_str,
            "source_reliability":        entry.source_reliability,
            # Available factor scores (PATCH 07: added quality + pt_upside)
            "momentum_long_score":       momentum_long_score,
            "volume_attention_score":    volume_attention_score,
            "quality_piotroski_score":   quality_piotroski_score,   # FMP ratios-ttm ✅
            "price_target_upside_score": price_target_upside_score, # partial coverage
            # Structurally absent — None (not 0.0)
            "insider_conviction_score":  None,
            "insider_breadth_score":     None,
            "congress_score":            None,
            "news_sentiment_score":      None,
            "news_buzz_score":           None,
            "analyst_consensus_score":   None,
            "analyst_revision_score":    None,
            "transcript_tone_score":     None,
            # Raw inputs (diagnostic)
            "return_12_1m":              return_12_1m,
            "volume_spike":              volume_spike,
            "news_sentiment_source":     "none",
            "news_buzz_source":          "none",
            # Validator-required fields
            "market_cap":                1.0,
            "insider_usd":               0.0,
            # final_score computed downstream after cross-sectional neutralization
        }

    except Exception as exc:
        log.debug("_score_ticker_international skip %s: %s", entry.ticker, exc)
        return None


# Deprecated aliases — kept so any external code referencing them by name gets a
# clear log message instead of an AttributeError.
def _score_ticker_eu(entry: Any) -> Optional[Dict[str, Any]]:
    """DEPRECATED — use _score_ticker_international (Fix #5)."""
    log.warning("_score_ticker_eu is deprecated; routing to _score_ticker_international")
    return _score_ticker_international(entry)


def _score_ticker_asia(entry: Any) -> Optional[Dict[str, Any]]:
    """DEPRECATED — use _score_ticker_international (Fix #5)."""
    log.warning("_score_ticker_asia is deprecated; routing to _score_ticker_international")
    return _score_ticker_international(entry)


# ── Main ───────────────────────────────────────────────────────────────────────

def run(tickers_file: Path, log_dir: Path, max_workers: int = 8) -> Dict[str, Any]:
    """Run the full scoring pipeline; return status dict."""
    log_dir.mkdir(parents=True, exist_ok=True)
    ticker_rows = load_tickers(tickers_file)
    tickers     = [r["ticker"] for r in ticker_rows]
    cap_tier    = {r["ticker"]: r.get("cap_tier", "large") for r in ticker_rows}
    sector      = {r["ticker"]: r.get("sector", "Unknown") for r in ticker_rows}

    t0 = time.time()
    log.info("Pipeline start: %d tickers from %s", len(tickers), tickers_file)

    # ── Shared FMP client — created once so health_report() captures all calls ─
    _fmp_client = _FMPClient()

    # ── EDGAR connectivity preflight ──────────────────────────────────────────
    # Log the effective User-Agent and test data.sec.gov with AAPL (CIK 320193).
    # This diagnostic appears in verbose CI output and helps trace 403/timeout.
    _ua = os.getenv("EDGAR_USER_AGENT") or "regime-trader-research n.tardy@hotmail.fr"
    log.info("EDGAR User-Agent: %.40s%s", _ua, "…" if len(_ua) > 40 else "")
    try:
        _test = _sec_get("https://data.sec.gov/submissions/CIK0000320193.json", timeout=15)
        _d = _test.json().get("filings", {}).get("recent", {})
        _f4 = sum(1 for f in _d.get("form", []) if f == "4")
        log.info("EDGAR preflight OK — AAPL submissions: %d form4 filings", _f4)
    except Exception as _exc:
        log.warning("EDGAR preflight FAILED — data.sec.gov unreachable: %s", _exc)

    # ── FMP: per-ticker profile (up to 80/run to stay within 250/day budget) ────
    log.info("Fetching FMP profiles (per-ticker, up to 80)…")
    fmp_cap = 80
    mktcaps = fetch_fmp_profiles(tickers)
    fmp_count = sum(1 for t in tickers[:fmp_cap] if mktcaps.get(t, 0) > 0)

    # ── Congress feed ─────────────────────────────────────────────────────────
    log.info("Fetching congress trading data…")
    congress_data = fetch_congress_buys()

    # ── SPY baseline + regime — fetched once, shared across all worker threads ─
    # PATCH 10: Extended to include 63-day return for momentum regime classification.
    # Hull (2015): regime shifts can occur while VIX is moderate — the 63d return
    # provides early detection of bear markets like 2022 rate shock (VIX <30, SPY -19%).
    log.info("Fetching SPY 12-1 month return + momentum regime (Jegadeesh-Titman + PATCH 10)…")
    spy_return_baseline, _spy_return_63d, _spy_momentum_regime = _fetch_spy_full_regime()
    log.info(
        "SPY: 12-1m=%.4f (%.1f%%)  63d=%s  regime=%s",
        spy_return_baseline, spy_return_baseline * 100,
        f"{_spy_return_63d:.4f} ({_spy_return_63d*100:.1f}%)" if _spy_return_63d is not None else "N/A",
        _spy_momentum_regime,
    )
    if _spy_momentum_regime in ("BEAR_CRASH", "BEAR_MOMENTUM"):
        log.warning(
            "PATCH 10 MOMENTUM REGIME ALERT: %s — "
            "VIX may not yet be >= 30 but price action signals a bear regime. "
            "Consider manual risk reduction.",
            _spy_momentum_regime,
        )

    # ── FMP insider — primary source (Form 4, cached 12h, limit=500) ─────────────
    # get_insider_purchases() returns (total_usd, days) per ticker.
    # get_insider_transactions() returns {P: [...], S: [...]} for breadth signal.
    # Both pre-fetched here; the thread pool reads from the in-memory dicts.
    # Shared client passed in so health_report() sees all calls/failures.
    log.info("Pre-fetching FMP insider transactions for %d tickers…", len(tickers))
    fmp_insider_cache: Dict[str, Tuple[float, int]] = fetch_fmp_insider_all(
        tickers, client=_fmp_client
    )
    fmp_has_data = any(v[0] > 0 for v in fmp_insider_cache.values())
    if not fmp_has_data:
        log.info("FMP insider returned no data -- insider scoring uses EDGAR XML only")

    # Pre-fetch breadth transactions (P/S per distinct insider) for score_insider_breadth.
    # Runs only when key is set; falls back to EDGAR-derived p_transactions otherwise.
    log.info("Pre-fetching FMP insider breadth (P/S transactions) for %d tickers…", len(tickers))
    fmp_breadth_cache: Dict[str, Dict] = {}
    if _fmp_client._api_key:
        def _fetch_breadth(ticker: str) -> Tuple[str, Dict]:
            try:
                return ticker, _fmp_client.get_insider_transactions(ticker, lookback_days=90)
            except FMPEndpointError as exc:
                log.error("FMP breadth structural failure %s: %s", ticker, exc)
                return ticker, {"P": [], "S": []}
            except Exception as exc:
                log.debug("FMP breadth fetch failed %s: %s", ticker, exc)
                return ticker, {"P": [], "S": []}

        with ThreadPoolExecutor(max_workers=10) as _bp:
            for _ticker, _btx in _bp.map(_fetch_breadth, tickers):
                fmp_breadth_cache[_ticker] = _btx
        _breadth_nonzero = sum(1 for v in fmp_breadth_cache.values()
                               if v.get("P") or v.get("S"))
        log.info("FMP breadth pre-fetch: %d/%d tickers with P or S transactions",
                 _breadth_nonzero, len(tickers))

    # ── EDGAR + yfinance: parallel per-ticker ─────────────────────────────────
    results = []
    errors  = 0

    def _score_ticker(row: Dict[str, str]) -> Dict[str, Any]:
        from regime_trader.scoring.insider_signals import (  # noqa: PLC0415
            score_insider_conviction,
            score_insider_breadth,
        )
        from regime_trader.scoring.momentum_signals import (  # noqa: PLC0415
            score_momentum_long,
            score_volume_attention,
        )

        ticker = row["ticker"]
        edgar_ok    = False
        form4_count = 0
        total_purchases_usd = 0.0
        ceo_buy     = False
        days_since_most_recent = 0
        ceo_purchase_usd = 0.0
        p_transactions: List[Dict] = []
        s_transactions: List[Dict] = []
        try:
            (
                form4_count, total_purchases_usd, ceo_buy, days_since_most_recent,
                ceo_purchase_usd, p_transactions, s_transactions,
            ) = fetch_edgar_data(ticker)
            edgar_ok = True
        except Exception as exc:
            log.warning("EDGAR unreachable for %s: %s", ticker, exc)

        mktcap = mktcaps.get(ticker, 0.0)
        congress_raw = congress_data.get(ticker)

        try:
            # Insider purchases: FMP Form 4 (primary) → EDGAR XML (fallback).
            _fmp_usd, _fmp_days = fmp_insider_cache.get(ticker, (0.0, 0))
            if _fmp_usd > 0:
                total_purchases_usd    = _fmp_usd
                days_since_most_recent = _fmp_days
                # Fix #7: ceo_buy is legacy; scoring uses _ceo_purchase_significance() internally.
                ceo_buy = ceo_purchase_usd > 0
                insider_source = "fmp"
            elif total_purchases_usd > 0:
                insider_source = "edgar"
            else:
                insider_source = "none"

            # ── Fix #2: orthogonal insider signals ────────────────────────
            conviction_score = score_insider_conviction(
                key_purchases_usd=total_purchases_usd,
                market_cap=mktcap,
                days_since_most_recent=days_since_most_recent,
                ceo_purchase_usd=ceo_purchase_usd,
            )
            # Breadth: use FMP P/S transaction list (richer, deduplicated by
            # insider_id) when available; fall back to EDGAR-parsed p_transactions.
            _fmp_btx = fmp_breadth_cache.get(ticker, {})
            _p_for_breadth = _fmp_btx.get("P") or p_transactions
            _s_for_breadth = _fmp_btx.get("S") or s_transactions
            breadth_score = score_insider_breadth(_p_for_breadth, _s_for_breadth)

            # ── Fix #3: orthogonal momentum + attention signals ───────────
            price_data = fetch_price_data(ticker, spy_return=spy_return_baseline)
            mom_long_score = score_momentum_long(
                price_data["return_12_1m"],
                price_data["spy_return_12_1m"],
            )
            vol_att_score = score_volume_attention(price_data["volume_spike"])

            # ── Fix #3: orthogonal news signals ──────────────────────────
            news_sent_score, news_sent_source, _eps_pct, _eps_days = score_news_sentiment_combined(ticker)
            news_buzz_score, news_buzz_source = score_news_buzz_combined(ticker)

            # Transcript tone from FMP earnings transcript (new signal)
            transcript_tone_score, transcript_tone_source = score_transcript_tone(
                ticker, client=_fmp_client
            )

            # ── Analyst consensus (Womack 1996 JF) ───────────────────────
            analyst_consensus_score, analyst_consensus_source = score_analyst_consensus(
                ticker, client=_fmp_client
            )

            # ── Analyst revision momentum (Chan-Jegadeesh-Lakonishok 1996 JF) ─
            _rev_pct, _rev_n = _FMPClient().get_analyst_estimate_revision(ticker)
            analyst_revision_score = score_analyst_revision(_rev_pct, _rev_n)

            # Recent upgrade/downgrade (FMP) — useful catalyst signal
            recent_upg = _fmp_client.get_recent_upgrades_downgrades(ticker)

            # ── Price target upside (forward-looking analyst target signal) ────
            # None = no analyst coverage / data missing → dead signal (penalised).
            # Not the same as 0.50 (at-target with valid data).
            price_target_upside_score = _fmp_client.get_upside_to_target(ticker) or 0.0

            # Store raw PT and current price for Discord display
            # Avoids the score-reversal bug in send_toplists_discord._fmt_pt_badge
            _pt_data         = _fmp_client.get_price_target_consensus(ticker)
            _quote_data      = _fmp_client.get_quote(ticker)
            _raw_target_price  = _pt_data.get("targetConsensus") if _pt_data else None
            _raw_current_price = _quote_data.get("price") if _quote_data else None

            # quality_piotroski: Piotroski (2000) / Novy-Marx (2013) fundamental quality gate
            quality_piotroski_score = _fmp_client.get_quality_score(ticker)

            # ── Congress ─────────────────────────────────────────────────
            c_score = score_congress(congress_raw)

            # ── Legacy scalars (diagnostic, 30-day comparison window) ────
            e_score_legacy = score_edgar(form4_count)
            i_score_legacy = score_insider_value(total_purchases_usd, mktcap, days_since_most_recent)
            n_score_legacy = score_news_fmp(ticker)
            # momentum_score_legacy: need 20d return; fetch_price_data no longer returns it.
            # Approximate with return_12_1m mapped through the old 30% clip formula as a
            # directional proxy — only used for ρ diagnostics, not in WEIGHTS.
            _r12 = price_data.get("return_12_1m")
            _spy12 = float(price_data.get("spy_return_12_1m") or 0.0)
            if _r12 is not None:
                _rel = max(-0.30, min(0.30, float(_r12) - _spy12))
                m_score_legacy = round((_rel + 0.30) / 0.60, 4)
            else:
                m_score_legacy = 0.0

            quiver_evidence = {
                "congress": {
                    "purchases":       int(congress_raw.get("purchases", 0)) if congress_raw else 0,
                    "sales":           int(congress_raw.get("sales", 0)) if congress_raw else 0,
                    "net":             int(
                        congress_raw.get(
                            "net",
                            congress_raw.get("purchases", 0) - congress_raw.get("sales", 0),
                        )
                    ) if congress_raw else 0,
                    "recency_days":    congress_raw.get("recency_days") if congress_raw else None,
                    "representatives": congress_raw.get("representatives", []) if congress_raw else [],
                },
                "source": "quiver" if (congress_raw and congress_raw.get("recency_days") is not None) else "s3",
                "insider_source": insider_source,
            }

            ret_12_1m = price_data.get("return_12_1m")

            # Fix #7: relative CEO purchase significance (bps of market cap).
            _ceo_bps = (ceo_purchase_usd / mktcap * 10_000) if mktcap > 0 and ceo_purchase_usd > 0 else None
            _ceo_tier = (
                "exceptional" if _ceo_bps is not None and _ceo_bps >= 5.0 else
                "substantial" if _ceo_bps is not None and _ceo_bps >= 1.0 else
                "modest"      if _ceo_bps is not None and _ceo_bps >= 0.5 else
                "none"
            )

            return {
                "ticker":                  ticker,
                "sector":                  sector.get(ticker, "Unknown"),
                "cap_tier":                cap_tier.get(ticker, "large"),
                "market_cap":              mktcap,
                # ── Fix #2 — orthogonal insider factors (WEIGHTS keys) ────
                "insider_conviction_score": conviction_score,
                "insider_breadth_score":    breadth_score,
                # ── Fix #3 — orthogonal momentum + news factors (WEIGHTS) ─
                "momentum_long_score":      mom_long_score,
                "volume_attention_score":   vol_att_score,
                "news_sentiment_score":     news_sent_score,
                "news_buzz_score":          news_buzz_score,
                "analyst_consensus_score":  analyst_consensus_score,
                "analyst_revision_score":   analyst_revision_score,
                "analyst_revision_n":       _rev_n,
                "price_target_upside_score": price_target_upside_score,
                "quality_piotroski_score":  quality_piotroski_score,
                # ── Congress ─────────────────────────────────────────────
                "congress_score":           c_score,
                "transcript_tone_score":    transcript_tone_score,
                "transcript_tone_source":   transcript_tone_source,
                "recent_upgrade_downgrade": recent_upg,
                "target_price":             _raw_target_price,
                "current_price":            _raw_current_price,
                # ── Legacy scalars (diagnostic only — not in WEIGHTS) ─────
                "edgar_score_legacy":       e_score_legacy,
                "insider_score_legacy":     i_score_legacy,
                "news_score_legacy":        n_score_legacy,
                "momentum_score_legacy":    m_score_legacy,
                # ── Metadata ─────────────────────────────────────────────
                # Fix #7: relative CEO conviction replaces absolute $25k threshold.
                "ceo_purchase_bps":        round(_ceo_bps, 4) if _ceo_bps is not None else None,
                "ceo_conviction_tier":     _ceo_tier,
                "ceo_buy":                 ceo_buy,  # legacy bool — _deprecated: true
                "ceo_buy_deprecated":      True,
                "form4_count":             form4_count,
                "form4_purchase_count":    len(p_transactions),  # Fix #6: P-code only, for Minsky stress signal
                "quiver_evidence":         quiver_evidence,
                "news_sentiment_source":   news_sent_source,
                "news_buzz_source":        news_buzz_source,
                "analyst_consensus_source": analyst_consensus_source,
                "earnings_surprise_pct":   _eps_pct,
                "earnings_surprise_days":  _eps_days,
                "insider_usd":             float(total_purchases_usd),
                "return_12_1m":            ret_12_1m,
                "momentum_spy_relative":   float(ret_12_1m - _spy12) if ret_12_1m is not None else 0.0,
                "volume_spike":            float(price_data["volume_spike"]),
                "_edgar_ok":               edgar_ok,
                "_scoring_error":          False,
                # PATCH 09: Correlated signal flag (Grinold & Kahn 2000).
                # Both insider conviction AND congress firing on the same ticker
                # creates a double-counting risk — they share informational overlap
                # (both reflect informed buying of the same security at the same time).
                # This flag is diagnostic only — it does NOT change final_score.
                # The Discord embed and portfolio advisor use this flag to warn
                # the human reviewer about potential score inflation.
                "_correlated_signal_flag": (
                    conviction_score > 0.50 and c_score > 0.50
                ),
                "_correlated_signal_discount_advisory": (
                    # Advisory: suggested 5pp discount on final_score when both fire.
                    # Human decision — not applied automatically.
                    0.05 if (conviction_score > 0.50 and c_score > 0.50) else 0.0
                ),
            }
        except FMPEndpointError as fmp_exc:
            # PATCH 08: FMPEndpointError is a structural failure (HTTP 4xx),
            # NOT a data-absence event. Log the broken endpoint and track it
            # so fmp_health.json captures non-insider structural failures too.
            log.error(
                "FMP STRUCTURAL FAILURE in _score_ticker %s: endpoint=%s status=%d. "
                "Factor scores zeroed. This endpoint is broken — do NOT lower "
                "circuit-breaker thresholds to compensate.",
                ticker, fmp_exc.path, fmp_exc.status,
            )
            with _structural_failures_lock:
                _structural_failures_in_scoring.add(fmp_exc.path)

            return {
                "ticker":                  ticker,
                "sector":                  sector.get(ticker, "Unknown"),
                "cap_tier":                cap_tier.get(ticker, "large"),
                "market_cap":              mktcap,
                "insider_conviction_score": 0.0,
                "insider_breadth_score":    0.0,
                "momentum_long_score":      0.0,
                "volume_attention_score":   0.0,
                "news_sentiment_score":     0.0,
                "news_buzz_score":          0.0,
                "analyst_consensus_score":  0.0,
                "analyst_revision_score":   0.0,
                "analyst_revision_n":       0,
                "transcript_tone_score":    0.0,
                "transcript_tone_source":   "none",
                "recent_upgrade_downgrade": {},
                "target_price":             None,
                "current_price":            None,
                "price_target_upside_score": 0.0,
                "quality_piotroski_score":  0.0,
                "congress_score":           0.0,
                "edgar_score_legacy":       0.0,
                "insider_score_legacy":     0.0,
                "news_score_legacy":        0.0,
                "momentum_score_legacy":    0.0,
                "ceo_buy":                 ceo_buy,
                "form4_count":             form4_count,
                "form4_purchase_count":    0,
                "quiver_evidence":         {},
                "news_sentiment_source":   "none",
                "news_buzz_source":        "none",
                "analyst_consensus_source": "none",
                "earnings_surprise_pct":   None,
                "earnings_surprise_days":  0,
                "insider_usd":             float(total_purchases_usd),
                "return_12_1m":            None,
                "momentum_spy_relative":   0.0,
                "volume_spike":            1.0,
                "_edgar_ok":               edgar_ok,
                "_scoring_error":          True,
                "_fmp_structural_failure": fmp_exc.path,  # PATCH 08: track broken endpoint
            }
        except Exception as exc:
            log.warning("Scoring failed for %s: %s", ticker, exc)
            return {
                "ticker":                  ticker,
                "sector":                  sector.get(ticker, "Unknown"),
                "cap_tier":                cap_tier.get(ticker, "large"),
                "market_cap":              mktcap,
                "insider_conviction_score": 0.0,
                "insider_breadth_score":    0.0,
                "momentum_long_score":      0.0,
                "volume_attention_score":   0.0,
                "news_sentiment_score":     0.0,
                "news_buzz_score":          0.0,
                "analyst_consensus_score":  0.0,
                "analyst_revision_score":   0.0,
                "analyst_revision_n":       0,
                "transcript_tone_score":    0.0,
                "transcript_tone_source":   "none",
                "recent_upgrade_downgrade": {},
                "target_price":             None,
                "current_price":            None,
                "price_target_upside_score": 0.0,
                "quality_piotroski_score":  0.0,
                "congress_score":           0.0,
                "edgar_score_legacy":       0.0,
                "insider_score_legacy":     0.0,
                "news_score_legacy":        0.0,
                "momentum_score_legacy":    0.0,
                "ceo_buy":                 ceo_buy,
                "form4_count":             form4_count,
                "form4_purchase_count":    0,  # Fix #6: scoring failed → no parsed purchases
                "quiver_evidence":         {},
                "news_sentiment_source":   "none",
                "news_buzz_source":        "none",
                "analyst_consensus_source": "none",
                "earnings_surprise_pct":   None,
                "earnings_surprise_days":  0,
                "insider_usd":             float(total_purchases_usd),
                "return_12_1m":            None,
                "momentum_spy_relative":   0.0,
                "volume_spike":            1.0,
                "_edgar_ok":               edgar_ok,
                "_scoring_error":          True,
            }

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_score_ticker, row): row for row in ticker_rows}
        for fut in as_completed(futures):
            r = fut.result()
            if r.get("_scoring_error", False):
                errors += 1
            results.append(r)

    # ── EU / Asia scoring ─────────────────────────────────────────────────────
    registry_tickers = _load_registry_tickers()
    _meta = _registry_meta()
    log.info(
        "Multi-market registry: %d EU tickers, %d Asia tickers",
        len(registry_tickers.get("EUROPE", [])),
        len(registry_tickers.get("ASIA", [])),
    )
    if any(registry_tickers.values()):
        from regime_trader.fetchers import Orchestrator  # noqa: PLC0415
        from regime_trader.fetchers.fmp_fetcher import FMPFetcher  # noqa: PLC0415
        from regime_trader.fetchers.base import MarketEnum  # noqa: PLC0415

        fmp_key = os.environ.get("FMP_API_KEY", "")
        eu_asia_fetchers = []
        if fmp_key and registry_tickers.get("EUROPE"):
            eu_asia_fetchers.append(FMPFetcher(api_key=fmp_key, market=MarketEnum.EUROPE))
            log.info("FMPFetcher added for EUROPE (%d tickers)", len(registry_tickers["EUROPE"]))
        elif not fmp_key:
            log.warning("FMP_API_KEY absent -- EUROPE section will be empty in Discord")
        if fmp_key and registry_tickers.get("ASIA"):
            eu_asia_fetchers.append(FMPFetcher(api_key=fmp_key, market=MarketEnum.ASIA))
            log.info("FMPFetcher added for ASIA (%d tickers)", len(registry_tickers["ASIA"]))
        elif registry_tickers.get("ASIA") and not fmp_key:
            log.warning("FMP_API_KEY absent -- ASIA section will be empty in Discord")

        if eu_asia_fetchers:
            orch = Orchestrator(eu_asia_fetchers)
            raw_entries = orch.run(registry_tickers)
            eu_raw = [e for e in raw_entries if e.market.value == "EUROPE"]
            asia_raw = [e for e in raw_entries if e.market.value == "ASIA"]
            log.info("Orchestrator raw entries: %d EU, %d Asia", len(eu_raw), len(asia_raw))
            for e in raw_entries:
                m = _meta.get(e.ticker, {})
                e.sector = m.get("sector", "Unknown")
                e.cap_tier = m.get("cap_tier", "large")
                e.raw_factors["company_name"] = m.get("company_name", "")

            # PATCH 06: Fetch regional benchmarks before EU/Asia scoring.
            # Jegadeesh & Titman (1993): momentum is measured vs the local peer group.
            # EZU = iShares MSCI Eurozone, AAXJ = iShares MSCI Asia ex-Japan.
            log.info("Fetching regional momentum benchmarks for EU/Asia scoring…")
            eu_return_baseline   = _fetch_eu_return()
            asia_return_baseline = _fetch_asia_return()
            log.info(
                "Regional baselines: EU(EZU)=%.4f (%.1f%%), Asia(AAXJ)=%.4f (%.1f%%)",
                eu_return_baseline, eu_return_baseline * 100,
                asia_return_baseline, asia_return_baseline * 100,
            )
            log.info(
                "Orchestrator raw entries: %d EU, %d Asia (benchmarks: EU=EZU %.4f, Asia=AAXJ %.4f)",
                len(eu_raw), len(asia_raw),
                eu_return_baseline if eu_raw else 0.0,
                asia_return_baseline if asia_raw else 0.0,
            )

            # Fix #5 + PATCH 06: unified scorer with regional benchmark injection
            log.info(
                "Fix #5 + PATCH 06: _score_ticker_international uses regional benchmarks. "
                "EU entries benchmark vs EZU, Asia entries vs AAXJ. "
                "insider/news/congress=None (structurally absent for non-US)."
            )

            def _regional_baseline(entry) -> float:
                """Return the correct 12-1m benchmark for this market."""
                if entry.market.value == "EUROPE":
                    return eu_return_baseline
                if entry.market.value == "ASIA":
                    return asia_return_baseline
                return spy_return_baseline  # fallback — should not occur

            with ThreadPoolExecutor(max_workers=4) as eu_pool:
                eu_futures = {
                    eu_pool.submit(
                        _score_ticker_international, e, _regional_baseline(e)
                    ): e.ticker
                    for e in raw_entries
                    if e.market.value in ("EUROPE", "ASIA")
                }
                eu_scored = asia_scored = 0
                for fut in as_completed(eu_futures):
                    scored = fut.result()
                    if scored:
                        results.append(scored)
                        if scored.get("market") == "EUROPE":
                            eu_scored += 1
                        elif scored.get("market") == "ASIA":
                            asia_scored += 1
            log.info("Scored: %d EU entries, %d Asia entries added to results", eu_scored, asia_scored)
        else:
            log.warning("No EU/Asia fetchers active — both sections will be empty in Discord")

    # PATCH 08: Report any structural FMP failures found during _score_ticker().
    if _structural_failures_in_scoring:
        log.error(
            "PATCH 08: %d structural FMP endpoint failure(s) detected in scoring: %s. "
            "These endpoints returned HTTP 4xx — do NOT lower circuit-breaker thresholds.",
            len(_structural_failures_in_scoring),
            ", ".join(sorted(_structural_failures_in_scoring)),
        )
        # Inject into fmp_client health report so fmp_health.json captures it
        for path in _structural_failures_in_scoring:
            _fmp_client.endpoint_failures[path] += 1
        _structural_failures_in_scoring.clear()

    # edgar_count = tickers where EDGAR was reachable (even if 0 filings returned).
    edgar_count   = sum(1 for r in results if r.get("_edgar_ok", False))
    congress_count = len(congress_data)
    duration      = round(time.time() - t0, 2)

    # ── Fix #3 summary: missing momentum (recent IPOs / thin history) ────────
    n_missing_momentum = sum(1 for r in results if r.get("return_12_1m") is None and r.get("market", "USA") == "USA")
    if n_missing_momentum > 0:
        log.warning(
            "Momentum 12-1m missing for %d/%d US tickers (recent IPOs or insufficient history). "
            "These tickers get momentum_long_score=0.0 (dead signal, penalized in normalizer).",
            n_missing_momentum, len([r for r in results if r.get("market", "USA") == "USA"]),
        )

    # ── Orthogonality diagnostics (Fix #2 + Fix #3) ──────────────────────────
    log.warning(_WEIGHTS_MIGRATION_NOTE)
    from regime_trader.scoring.insider_signals import log_conviction_breadth_correlation  # noqa: PLC0415
    _us_results = [r for r in results if r.get("market", "USA") == "USA"]
    log_conviction_breadth_correlation(_us_results)

    # Fix #3: news_sentiment ⊥ news_buzz (expect ρ < 0.4)
    # Fix #3: momentum_long ⊥ volume_attention (expect ρ < 0.3)
    # Fix #3: momentum_long ⊥ momentum_score_legacy (expect ρ < 0.2 — temporal disjoint)
    def _pearson(xs: list[float], ys: list[float], label: str, warn_threshold: float) -> None:
        pairs = [(x, y) for x, y in zip(xs, ys) if x > 0.0 and y > 0.0]
        if len(pairs) < 5:
            log.info("ρ(%s): insufficient pairs (%d)", label, len(pairs))
            return
        n = len(pairs)
        mx = sum(p[0] for p in pairs) / n
        my = sum(p[1] for p in pairs) / n
        num   = sum((p[0] - mx) * (p[1] - my) for p in pairs)
        denom = math.sqrt(
            sum((p[0] - mx) ** 2 for p in pairs) * sum((p[1] - my) ** 2 for p in pairs)
        )
        if denom == 0:
            log.info("ρ(%s): undefined (zero variance)", label)
            return
        r = num / denom
        flag = " ⚠ EXCEEDS THRESHOLD" if abs(r) >= warn_threshold else " ✓"
        log.info("ρ(%s) = %.3f (threshold %.1f)%s", label, r, warn_threshold, flag)

    _pearson(
        [r.get("news_sentiment_score", 0.0) for r in _us_results],
        [r.get("news_buzz_score", 0.0) for r in _us_results],
        "news_sentiment,news_buzz", 0.4,
    )
    _pearson(
        [r.get("momentum_long_score", 0.0) for r in _us_results],
        [r.get("volume_attention_score", 0.0) for r in _us_results],
        "momentum_long,volume_attention", 0.3,
    )
    _pearson(
        [r.get("momentum_long_score", 0.0) for r in _us_results],
        [r.get("momentum_score_legacy", 0.0) for r in _us_results],
        "momentum_long,momentum_legacy", 0.2,
    )

    # Analyst triplet orthogonality (all three are sell-side derived — check for redundancy)
    _pearson(
        [r.get("analyst_consensus_score", 0.0) for r in _us_results],
        [r.get("analyst_revision_score", 0.0) for r in _us_results],
        "analyst_consensus,analyst_revision", 0.5,
    )
    _pearson(
        [r.get("analyst_consensus_score", 0.0) for r in _us_results],
        [r.get("price_target_upside_score", 0.0) for r in _us_results],
        "analyst_consensus,price_target_upside", 0.5,
    )
    _pearson(
        [r.get("analyst_revision_score", 0.0) for r in _us_results],
        [r.get("price_target_upside_score", 0.0) for r in _us_results],
        "analyst_revision,price_target_upside", 0.5,
    )
    # Transcript vs news sentiment (both are text-based — check overlap)
    _pearson(
        [r.get("transcript_tone_score", 0.0) for r in _us_results],
        [r.get("news_sentiment_score", 0.0) for r in _us_results],
        "transcript_tone,news_sentiment", 0.4,
    )

    # ── Fix #5: source_reliability migration notice ───────────────────────────
    log.warning(
        "Fix #5 migration: source_reliability is no longer a score multiplier. "
        "Preserved as diagnostic metadata. Re-introduce empirically via Fix #4 IC backtest "
        "if EU/Asia source quality correlates empirically with IC."
    )

    # Neutralization and final scoring moved to after validation (Stage 1 gate)

    # Diagnostic: weight_coverage distribution by market
    for mkt_str in ("USA", "EUROPE", "ASIA"):
        mkt_rows = [r for r in results if r.get("market", "USA") == mkt_str]
        if mkt_rows:
            wc_vals = [r.get("weight_coverage", 0.0) for r in mkt_rows]
            low_cov = sum(1 for r in mkt_rows if r.get("_low_coverage", False))
            log.info(
                "Fix #5 weight_coverage [%s]: n=%d, mean=%.3f, min=%.3f, max=%.3f, low_cov=%d",
                mkt_str, len(mkt_rows),
                sum(wc_vals) / len(wc_vals), min(wc_vals), max(wc_vals), low_cov,
            )

    # ── Stage 1 gate: stamp computed_at + run validate_raw ────────────────────
    # Stamp a row-level timestamp on every result so validate_dates() has a
    # per-row anchor.  Rows already carrying computed_at are left unchanged.
    pipeline_run_ts = datetime.now(timezone.utc).isoformat()
    for row in results:
        if "computed_at" not in row:
            row["computed_at"] = pipeline_run_ts

    # Build source_meta from the live run timestamps so validate_dates() can
    # check whether Quiver/Finnhub/EDGAR feeds are stale at source level.
    source_meta: Dict[str, Dict[str, Any]] = {
        "fmp":    {"last_updated": pipeline_run_ts},
        "edgar":  {"last_updated": pipeline_run_ts},
        "none":   {"last_updated": pipeline_run_ts},
    }

    # ── Stage 1 gate: BEFORE neutralization ─────────────────────────────────────
    # Quarantined tickers must not distort the peer group in cross-sectional stats.
    quarantine_count = 0
    try:
        clean_rows, quarantined_rows, val_issues = validate_raw(results, source_meta)
        quarantine_count = len(quarantined_rows)
        quarantined_tickers = {r["ticker"] for r in quarantined_rows}
        for r in results:
            r["_validation_failed"] = r["ticker"] in quarantined_tickers
        if quarantine_count:
            log.warning(
                "Stage 1 gate: %d/%d tickers quarantined pre-neutralization — %s",
                quarantine_count, len(results),
                ", ".join({i.code for i in val_issues if i.code != "STALE_DATA"}),
            )
        else:
            log.info("Stage 1 gate: all %d tickers passed", len(results))
    except Exception as exc:
        log.error("Stage 1 gate FAILED: %s", exc)
        raise

    # ── Cross-sectional neutralization (clean tickers only) ───────────────────
    from regime_trader.scoring.neutralization import neutralize_factors  # noqa: PLC0415
    from regime_trader.scoring.market_config import (  # noqa: PLC0415
        Market, PIPELINE_MARKET_MAP, renormalize_weights_for_market, LOW_COVERAGE_THRESHOLD,
    )

    _v2_factors = tuple(f"{k}_score" for k in WEIGHTS)
    clean_for_norm  = [r for r in results if not r.get("_validation_failed")]
    quarantined_out = [r for r in results if r.get("_validation_failed")]

    clean_for_norm = neutralize_factors(
        clean_for_norm,
        factors=_v2_factors,
        group_by=("market", "sector", "cap_tier"),
        min_bucket_size=5,
        fallback_group_by=("market", "cap_tier"),
    )

    # Quarantined tickers: zero out scores, mark low coverage
    for r in quarantined_out:
        for f in _v2_factors:
            r[f"{f}_neutral"] = 0.0
        r["final_score"]     = 0.0
        r["weight_coverage"] = 0.0
        r["_low_coverage"]   = True

    # Reassemble
    results = clean_for_norm + quarantined_out

    # ── Fix #5: compute final_score with market-renormalized weights ──────────
    # For each ticker, use only the factors available for its market.
    # None factor → excluded from weight denominator (structurally absent).
    # 0.0 factor → included with weight but penalized by cross-sectional normalizer.
    _renorm_cache: dict[Market, dict] = {}

    for r in results:
        market_raw = r.get("market", "USA")
        market = PIPELINE_MARKET_MAP.get(market_raw, Market.US)

        if market not in _renorm_cache:
            _renorm_cache[market] = renormalize_weights_for_market(WEIGHTS, market)
        market_weights = _renorm_cache[market]

        final_score = 0.0
        weight_sum_applied = 0.0
        for factor_short, w in market_weights.items():
            if w == 0.0:
                continue  # structurally absent for this market
            factor_neutral = f"{factor_short}_score_neutral"
            score_val = r.get(factor_neutral)
            if score_val is None:
                # Factor available for this market but missing on this specific ticker
                # (e.g. momentum_long=None for a recent IPO). Pro-rata redistribute.
                continue
            final_score += w * float(score_val)
            weight_sum_applied += w

        # Renormalize if some available factors were still missing on this ticker
        if weight_sum_applied > 0:
            final_score = final_score / weight_sum_applied
        else:
            final_score = 0.0

        r["final_score"]     = round(final_score, 4)
        r["weight_coverage"] = round(weight_sum_applied, 4)
        r["_low_coverage"]   = weight_sum_applied < LOW_COVERAGE_THRESHOLD

    # ── Fix #5: top_by_market — separate Top-20 per market ───────────────────
    # Excludes _low_coverage tickers (weight_coverage < LOW_COVERAGE_THRESHOLD).
    # Consumers (Discord report, dashboard) use these lists for per-market sections.
    def _top_n(mkt: str, n: int = 20) -> list[dict]:
        eligible = [
            r for r in results
            if r.get("market", "USA") == mkt and not r.get("_low_coverage", False)
        ]
        return sorted(eligible, key=lambda r: r.get("final_score", 0.0), reverse=True)[:n]

    top_by_market = {
        "US":     _top_n("USA"),
        "EUROPE": _top_n("EUROPE"),
        "ASIA":   _top_n("ASIA"),
    }
    log.info(
        "top_by_market: US=%d, EUROPE=%d, ASIA=%d (low_cov excluded)",
        len(top_by_market["US"]),
        len(top_by_market["EUROPE"]),
        len(top_by_market["ASIA"]),
    )

    status = {
        "run_id":              os.getenv("GITHUB_RUN_ID", "local"),
        "spy_momentum_regime": _spy_momentum_regime,  # PATCH 10: NORMAL/BEAR_MOMENTUM/etc.
        "spy_return_63d":      round(_spy_return_63d, 6) if _spy_return_63d is not None else None,
        "_edgar_meta": {
            "last_run":             pipeline_run_ts,
            "run_duration_seconds": duration,
            "ticker_count":         len(tickers),
            "edgar_count":          edgar_count,
            "fmp_count":            fmp_count,
            "congress_count":       congress_count,
            "error_count":          errors,
            "quarantine_count":     quarantine_count,
        },
        "source_meta":   source_meta,
        "weights":       WEIGHTS,
        "results":       results,   # all rows (clean + quarantined with _validation_failed flag)
        "top_by_market": top_by_market,  # Fix #5: per-market Top-20, low_coverage excluded
        "computed_at":   pipeline_run_ts,
    }

    # ── Fix #8: permanent factor orthogonality diagnostic ────────────────────
    # López de Prado (AFML ch. 8): monitor that engineered features remain
    # structurally independent on live data after every run.
    try:
        from regime_trader.monitoring.factor_orthogonality import (  # noqa: PLC0415
            compute_factor_correlation_matrix,
        )
        orthogonality_report = compute_factor_correlation_matrix(results, market_filter="US")
        status["factor_orthogonality"] = orthogonality_report

        # CEO tier distribution diagnostic (Fix #7 calibration)
        if "error" not in orthogonality_report:
            _us_rows = [r for r in results if r.get("market", "USA") in ("US", "USA")]
            if _us_rows:
                _tier_counts: dict[str, int] = {}
                for _r in _us_rows:
                    _t = _r.get("ceo_conviction_tier", "none")
                    _tier_counts[_t] = _tier_counts.get(_t, 0) + 1
                log.info(
                    "CEO conviction tier distribution (US, n=%d): %s",
                    len(_us_rows),
                    ", ".join(f"{k}={v}" for k, v in sorted(_tier_counts.items())),
                )
    except Exception as _orth_exc:
        log.warning("Factor orthogonality diagnostic failed (non-blocking): %s", _orth_exc)
        status["factor_orthogonality"] = {"error": str(_orth_exc)}

    out = log_dir / "intel_source_status.json"
    save_json_atomic(out, status)
    log.info(
        "Done in %.1fs -- tickers=%d edgar=%d fmp_calls=%d congress=%d errors=%d -> %s",
        duration, len(tickers), edgar_count, fmp_count, congress_count, errors, out,
    )

    # ── FMP health report — written every run for monitoring ──────────────────
    # A non-zero failure count means a factor is being zeroed by a dead route.
    # check_metrics reads this file and fails the canary on structural failures.
    try:
        fmp_health = _fmp_client.health_report()
        fmp_health["run_timestamp"] = pipeline_run_ts
        save_json_atomic(log_dir / "fmp_health.json", fmp_health)
        if fmp_health.get("has_structural_failure"):
            log.error(
                "FMP structural failures detected this run: %s — "
                "do NOT lower circuit-breaker thresholds; fix the endpoint.",
                fmp_health.get("failures", {}),
            )
        else:
            log.info("FMP health OK — no structural endpoint failures this run.")
    except Exception as _fmp_health_exc:
        log.warning("FMP health report write failed (non-fatal): %s", _fmp_health_exc)

    # ── Auto-archive: copy today's snapshot to logs/historical/YYYY-MM-DD/ ───
    try:
        from regime_trader.research.historical_loader import archive_current_run
        archived = archive_current_run(log_dir)
        if archived:
            log.info("Snapshot archived → %s", archived)
    except Exception as _arc_exc:
        log.warning("Auto-archive failed (non-fatal): %s", _arc_exc)

    return status


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="EDGAR+FMP+yfinance daily pipeline")
    parser.add_argument("--tickers-file", type=Path, default=Path("config/universe.csv"))
    parser.add_argument("--log-dir",      type=Path, default=Path("logs"))
    parser.add_argument("--max-workers",  type=int,  default=8)
    parser.add_argument("--verbose",      action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    try:
        run(args.tickers_file, args.log_dir, args.max_workers)
        return 0
    except Exception as exc:
        log.exception("Pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
