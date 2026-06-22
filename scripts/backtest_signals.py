"""scripts/backtest_signals.py
Weekly signal performance audit for the multi-factor scoring pipeline.

Walks logs/archive/ for all historical top_lists.json snapshots, simulates
fixed-horizon long positions at T+5/T+10/T+20, and benchmarks each trade
against SPY over the identical holding window.

Handles schema drift transparently:
  - 'macro' / 'momentum' factor rename
  - weight configurations: 30/25/20/15/10 vs 28/23/22/15/12 vs any future set
  - files missing optional keys (quiver_evidence, kill_switch, etc.)

Output:
  logs/backtest_report_latest.json   — machine-readable full results
  logs/backtest_ledger_latest.csv    — flat trade ledger (one row per signal)
  logs/backtest_summary.md           — human-readable markdown (also stdout)

Usage:
  python scripts/backtest_signals.py
  python scripts/backtest_signals.py --archive-dir logs/archive --verbose
  python scripts/backtest_signals.py --dry-run          # skip price fetch
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
    _HAS_YF = True
except ImportError:
    yf = None  # type: ignore
    _HAS_YF = False

ROOT = Path(__file__).resolve().parent.parent
log = logging.getLogger("backtest_signals")

# ── Constants ──────────────────────────────────────────────────────────────────

_HORIZONS: List[int] = [5, 10, 20]        # trading-day forward windows
_BADGE_THRESHOLDS = {
    "HIGH BUY":     0.80,
    "TACTICAL BUY": 0.60,
}
_BENCHMARK = "SPY"

# Audit P2.2 — round-trip transaction cost (entry + exit) as a fraction of
# notional, by cap tier: small/mid-caps carry wider spreads + more slippage on
# a ~20%-turnover book. Reported returns were previously GROSS (edge overstated,
# the trader's #1 objection). Net return = gross − round-trip cost, applied to
# the strategy leg only (the SPY benchmark is held passively). Env-overridable.
_ROUNDTRIP_COST: Dict[str, float] = {
    "large":   float(os.getenv("BT_COST_LARGE", "0.0020")),   # 20 bps
    "mid":     float(os.getenv("BT_COST_MID",   "0.0040")),   # 40 bps
    "small":   float(os.getenv("BT_COST_SMALL", "0.0060")),   # 60 bps
}
_DEFAULT_COST = _ROUNDTRIP_COST["large"]


def _roundtrip_cost(cap_tier: str) -> float:
    """Round-trip transaction cost for a cap tier (defaults to the large-cap
    cost for unknown/missing tiers)."""
    return _ROUNDTRIP_COST.get((cap_tier or "").lower(), _DEFAULT_COST)


_ARCHIVE_DIR  = ROOT / "logs" / "archive"
_REPORT_JSON  = ROOT / "logs" / "backtest_report_latest.json"
_LEDGER_CSV   = ROOT / "logs" / "backtest_ledger_latest.csv"
_SUMMARY_MD   = ROOT / "logs" / "backtest_summary.md"

# Canonical factors: current 9-factor schema first, then the legacy 5-factor
# names so pre-2026 archive snapshots keep parsing. 'macro' is the renamed
# 'momentum'.
_CANONICAL_FACTORS = (
    "insider_conviction", "insider_breadth", "congress", "news_sentiment",
    "news_buzz", "momentum_long", "volume_attention", "analyst_consensus",
    "quality_piotroski",
    "edgar", "insider", "news", "momentum",   # legacy eras
)
_FACTOR_ALIASES = {"macro": "momentum"}

# Entry-timing cutoff: signals generated before ~21:00 UTC (≈ NYSE close) enter
# at that day's close; signals generated at/after it enter at the next trading
# day's close (entry_next_day). Entry is therefore always at/after signal time —
# no look-ahead.
_POST_MARKET_UTC_HOUR = 21   # 21:00 UTC ≈ NYSE close


def _era_label(weights: Dict[str, float]) -> str:
    """Derive a human-readable strategy era label from the weights dict.

    Compares weights to known configurations and returns a descriptive tag.
    Unknown configurations get a fingerprint so they cluster correctly.

    Known eras:
      v1 — edgar=0.30, insider=0.25, congress=0.20, news=0.15, macro=0.10
      v2 — edgar=0.28, insider=0.23, congress=0.22, news=0.15, momentum=0.12
    """
    if not weights:
        return "unknown"
    edgar_w = round(weights.get("edgar", 0.0), 2)
    if edgar_w == 0.30:
        return "v1 (macro, 30/25/20/15/10)"
    if edgar_w == 0.28:
        return "v2 (momentum, 28/23/22/15/12)"
    # Future-proof: fingerprint from sorted rounded values
    fingerprint = "/".join(
        str(round(v * 100)) for _, v in sorted(weights.items())
    )
    return f"custom ({fingerprint})"


# Buy-list sections parsed from a snapshot: legacy US artifact (top_buys,
# mid_caps, small_caps, sector_picks) AND the combined cooked payload
# (top_buys_usa/_europe/_asia). watchlist/overflow are intentionally excluded —
# CAPITULATION survivors are force-badged WATCHLIST (not a buy signal) and would
# be dropped by the badge filter anyway.
_BUY_LIST_KEYS = (
    "top_buys", "top_buys_usa", "top_buys_europe", "top_buys_asia",
    "mid_caps", "small_caps",
)


def _iter_buy_entries(data: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Yield every buy-list entry across all snapshot schema variants."""
    for key in _BUY_LIST_KEYS:
        for entry in (data.get(key) or []):
            yield entry
    for sector_list in (data.get("sector_picks") or {}).values():
        for entry in (sector_list or []):
            yield entry


def load_archive_snapshot(path: Path) -> dict:
    """Load archive snapshot with schema version awareness.

    Applies a retroactive 0.6x score discount to pre-EU-Piotroski-gate EU/Asia
    entries (those missing quality_piotroski) so the backtest is comparable
    across the schema regime boundary (gate introduced Jun 03 2026). The
    discounted value is stored as ``final_score_adjusted`` and consumed by
    ``_parse_snapshot`` for both the badge threshold and the recorded score.
    """
    d = json.loads(path.read_text(encoding="utf-8"))
    schema = d.get("schema_version", "legacy")
    piog_eu = d.get("piotroski_eu_gate_active", False)

    if not piog_eu:
        adjusted = 0
        for entry in _iter_buy_entries(d):
            region = entry.get("region") or entry.get("market", "")
            if region in ("EU", "Asia", "EUROPE", "ASIA"):
                if entry.get("factors", {}).get("quality_piotroski") is None:
                    entry["_retroactive_piotroski_discount"] = True
                    entry["final_score_adjusted"] = round(
                        float(entry.get("final_score", 0.0)) * 0.6, 4
                    )
                    adjusted += 1
        if adjusted:
            log.info(
                "Snapshot %s (schema=%s): pre-EU-gate, retroactive 0.6x applied "
                "to %d EU/Asia score(s)", path.name, schema, adjusted,
            )
    return d


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class SignalRecord:
    """One triggered signal from a historical snapshot."""
    ticker:        str
    signal_date:   date
    badge:         str
    cap_tier:      str
    final_score:   float
    factors:       Dict[str, float]
    weights:       Dict[str, float]
    strategy_era:  str
    source_file:   str
    entry_next_day: bool   # True  → post-market, use T+1 close as entry
    market:        str = "USA"
    company_name:  str = ""

    # Filled after price download
    entry_price:   Optional[float] = None
    returns:       Dict[int, Optional[float]] = field(default_factory=dict)
    spy_returns:   Dict[int, Optional[float]] = field(default_factory=dict)
    alpha:         Dict[int, Optional[float]] = field(default_factory=dict)


@dataclass
class HorizonStats:
    horizon:       int
    count:         int
    win_rate:      float
    avg_return:    float
    median_return: float
    max_drawdown:  float
    profit_factor: float
    avg_alpha:     float


@dataclass
class SegmentStats:
    label:      str
    count:      int
    win_rate:   float
    avg_return: float


# ── Log ingestion ──────────────────────────────────────────────────────────────

def _normalize_factors(raw: Dict[str, float]) -> Dict[str, float]:
    """Rename 'macro' → 'momentum', drop unknown keys and absent (None) factors
    (INTL factor_snapshots preserve None for no-coverage factors)."""
    return {
        _FACTOR_ALIASES.get(k, k): float(v)
        for k, v in raw.items()
        if v is not None and _FACTOR_ALIASES.get(k, k) in _CANONICAL_FACTORS
    }


def _normalize_weights(raw: Dict[str, float]) -> Dict[str, float]:
    return {
        _FACTOR_ALIASES.get(k, k): v
        for k, v in raw.items()
        if _FACTOR_ALIASES.get(k, k) in _CANONICAL_FACTORS
    }


def _parse_snapshot(path: Path) -> List[SignalRecord]:
    """Parse a single top_lists.json and return qualifying SignalRecord objects."""
    records: List[SignalRecord] = []

    try:
        data: Dict[str, Any] = load_archive_snapshot(path)
    except Exception as exc:
        log.warning("Cannot parse %s: %s", path, exc)
        return records

    generated_at_raw: str = data.get("generated_at", "")
    try:
        ts = datetime.fromisoformat(generated_at_raw.replace("Z", "+00:00"))
        signal_date = ts.date()
        entry_next_day = ts.hour >= _POST_MARKET_UTC_HOUR
    except Exception:
        # Fall back to file modification time
        ts = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        signal_date = ts.date()
        entry_next_day = False
        log.debug("Bad generated_at in %s — using mtime %s", path.name, signal_date)

    weights = _normalize_weights(data.get("weights", {}))

    seen: set[str] = set()   # deduplicate ticker×date combos
    for entry in _iter_buy_entries(data):
        ticker = entry.get("ticker", "").strip().upper()
        badge  = entry.get("badge", "")
        # Consume the retroactive pre-EU-gate discount when present (set by
        # load_archive_snapshot) so threshold and recorded score share one scale.
        adj = entry.get("final_score_adjusted")
        score = float(adj if adj is not None else entry.get("final_score", 0.0))

        if not ticker or badge not in _BADGE_THRESHOLDS:
            continue
        if score < _BADGE_THRESHOLDS[badge]:
            continue

        dedup_key = f"{ticker}:{signal_date}"
        if dedup_key in seen:
            continue
        seen.add(dedup_key)

        raw_factors = entry.get("factors", {})
        records.append(SignalRecord(
            ticker        = ticker,
            signal_date   = signal_date,
            badge         = badge,
            cap_tier      = (entry.get("cap_tier") or "unknown").lower(),
            final_score   = score,
            factors       = _normalize_factors(raw_factors),
            weights       = weights,
            strategy_era  = _era_label(weights),
            source_file   = path.name,
            entry_next_day = entry_next_day,
            market        = entry.get("market", "USA"),
            company_name  = entry.get("company_name", ""),
        ))

    return records


def load_all_signals(archive_dir: Path) -> List[SignalRecord]:
    """Walk archive_dir for all top_lists*.json and parse them all."""
    if not archive_dir.exists():
        log.warning("Archive directory does not exist: %s", archive_dir)
        return []

    files = sorted(
        (p for p in archive_dir.rglob("*.json") if "top_lists" in p.name),
        key=lambda p: p.stat().st_mtime,
    )
    log.info("Found %d snapshot files in %s", len(files), archive_dir)

    all_records: List[SignalRecord] = []
    for f in files:
        batch = _parse_snapshot(f)
        log.debug("  %s -> %d qualifying signals", f.name, len(batch))
        all_records.extend(batch)

    log.info("Total qualifying signals loaded: %d", len(all_records))
    return all_records


# ── Price data ─────────────────────────────────────────────────────────────────

def _trading_day_offset(prices: pd.Series, anchor_date: date, offset: int) -> Optional[float]:
    """Return the close price at exactly `offset` trading days after anchor_date.

    Uses the DatetimeIndex of the prices Series to count calendar→trading-day
    conversion correctly. Returns None if insufficient history.
    """
    idx = prices.index
    anchor_ts = pd.Timestamp(anchor_date)

    # Find the position of the first index entry >= anchor_date
    pos_arr = idx.searchsorted(anchor_ts)
    if pos_arr >= len(idx):
        return None

    # searchsorted(side="left") already lands on the bar ON anchor_date when it
    # exists, else the first trading day after it — exactly the entry bar we want.
    target_pos = pos_arr + offset

    if target_pos >= len(idx):
        return None
    return float(prices.iloc[target_pos])


def fetch_prices(
    tickers: List[str],
    start: date,
    end: date,
    dry_run: bool = False,
) -> Dict[str, pd.Series]:
    """Download adj-close for all tickers in one batched yfinance call.

    Returns a dict of ticker → pd.Series(date-indexed, ffilled).
    Missing tickers are omitted. On dry_run returns empty dict.
    """
    if dry_run or not _HAS_YF:
        return {}

    if not tickers:
        return {}

    # Pad end by 30 calendar days to cover the T+20 window even at period edges
    end_padded = end + timedelta(days=30)
    symbols = list(set(tickers + [_BENCHMARK]))

    log.info("Fetching prices for %d symbols (%s … %s)", len(symbols), start, end_padded)
    try:
        raw = yf.download(
            tickers      = symbols,
            start        = str(start),
            end          = str(end_padded),
            auto_adjust  = True,
            progress     = False,
            threads      = True,
        )
    except Exception as exc:
        log.error("yfinance batch download failed: %s", exc)
        return {}

    # yfinance returns MultiIndex columns when multiple tickers, flat when one
    if isinstance(raw.columns, pd.MultiIndex):
        close_df: pd.DataFrame = raw["Close"]
    else:
        close_df = raw[["Close"]].rename(columns={"Close": symbols[0]})

    close_df = close_df.ffill()

    result: Dict[str, pd.Series] = {}
    for sym in symbols:
        if sym in close_df.columns:
            series = close_df[sym].dropna()
            if not series.empty:
                series.index = pd.to_datetime(series.index).normalize()
                result[sym] = series
            else:
                log.debug("Empty price series for %s", sym)
        else:
            log.debug("Ticker %s not in download result", sym)

    log.info("Prices fetched: %d / %d symbols", len(result), len(symbols))
    return result


# ── Return calculation ─────────────────────────────────────────────────────────

def compute_returns(
    records: List[SignalRecord],
    prices: Dict[str, pd.Series],
) -> None:
    """Mutate records in-place: set entry_price, returns, spy_returns, alpha."""
    spy_prices = prices.get(_BENCHMARK)

    for rec in records:
        series = prices.get(rec.ticker)
        if series is None or series.empty:
            continue

        cost = _roundtrip_cost(rec.cap_tier)   # P2.2 — net-of-cost returns

        # Entry: T+1 if post-market flag, else T+0 (same-day close)
        entry_date  = rec.signal_date
        entry_offset = 1 if rec.entry_next_day else 0

        entry_price_raw = _trading_day_offset(series, entry_date, entry_offset)
        if entry_price_raw is None or entry_price_raw == 0:
            continue
        rec.entry_price = entry_price_raw

        for h in _HORIZONS:
            exit_price = _trading_day_offset(series, entry_date, entry_offset + h)
            if exit_price is None:
                rec.returns[h] = None
                rec.spy_returns[h] = None
                rec.alpha[h] = None
                continue

            # NET of the round-trip transaction cost (P2.2) — the reported edge
            # is what the book actually keeps, not the gross paper return.
            ret = (exit_price - entry_price_raw) / entry_price_raw - cost
            rec.returns[h] = float(ret)

            # FX isolation: intl prices are listing-currency (JPY for .T,
            # GBp for .L, …). Percent returns are dimensionless and safe,
            # but alpha vs the USD-denominated SPY would embed FX drift —
            # leave alpha unset until explicit currency conversion exists.
            if rec.market != "USA":
                rec.spy_returns[h] = None
                rec.alpha[h] = None
                continue

            # SPY benchmark over identical window
            if spy_prices is not None:
                spy_entry = _trading_day_offset(spy_prices, entry_date, entry_offset)
                spy_exit  = _trading_day_offset(spy_prices, entry_date, entry_offset + h)
                if spy_entry and spy_exit and spy_entry != 0:
                    spy_ret = (spy_exit - spy_entry) / spy_entry
                    rec.spy_returns[h] = float(spy_ret)
                    rec.alpha[h]       = float(ret - spy_ret)
                else:
                    rec.spy_returns[h] = None
                    rec.alpha[h]       = None
            else:
                rec.spy_returns[h] = None
                rec.alpha[h]       = None


# ── Metrics engine ─────────────────────────────────────────────────────────────

def _horizon_stats(
    rets: List[float],
    alphas: List[Optional[float]],
    horizon: int,
) -> HorizonStats:
    if not rets:
        return HorizonStats(horizon, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    arr   = np.array(rets, dtype=float)
    wins  = arr > 0
    losses = arr[arr < 0]
    gains  = arr[arr > 0]

    profit_factor = (
        float(gains.sum() / abs(losses.sum()))
        if len(losses) > 0 and abs(losses.sum()) > 0
        else float("inf")
    )

    # Max drawdown: largest peak-to-trough during the holding period
    # (single-horizon = single return; interpret as max loss across trades)
    max_dd = float(arr.min()) if len(arr) > 0 else 0.0

    valid_alphas = [a for a in alphas if a is not None]
    avg_alpha = float(np.mean(valid_alphas)) if valid_alphas else 0.0

    return HorizonStats(
        horizon       = horizon,
        count         = len(arr),
        win_rate      = float(wins.mean()),
        avg_return    = float(arr.mean()),
        median_return = float(np.median(arr)),
        max_drawdown  = max_dd,
        profit_factor = profit_factor,
        avg_alpha     = avg_alpha,
    )


def compute_metrics(
    records: List[SignalRecord],
) -> Tuple[
    Dict[str, List[HorizonStats]],  # per badge
    Dict[str, HorizonStats],        # per cap tier at T+10
    Dict[str, HorizonStats],        # per strategy era at T+10
    List[SignalRecord],              # worst performers (false positives)
]:
    priced_t10 = [r for r in records if r.entry_price is not None and r.returns.get(10) is not None]

    # Segment by badge
    badge_records: Dict[str, List[SignalRecord]] = {}
    for rec in records:
        if rec.entry_price is None:
            continue
        badge_records.setdefault(rec.badge, []).append(rec)

    badge_stats: Dict[str, List[HorizonStats]] = {}
    for badge, recs in badge_records.items():
        badge_stats[badge] = []
        for h in _HORIZONS:
            rets   = [r.returns[h] for r in recs if r.returns.get(h) is not None]
            alphas = [r.alpha.get(h) for r in recs if r.returns.get(h) is not None]
            badge_stats[badge].append(_horizon_stats(rets, alphas, h))

    # Segment by cap tier at T+10
    cap_records: Dict[str, List[SignalRecord]] = {}
    for rec in priced_t10:
        cap_records.setdefault(rec.cap_tier, []).append(rec)

    cap_stats: Dict[str, HorizonStats] = {}
    for cap, recs in cap_records.items():
        rets   = [r.returns[10] for r in recs]   # type: ignore[misc]
        alphas = [r.alpha.get(10) for r in recs]
        cap_stats[cap] = _horizon_stats(rets, alphas, 10)

    # Segment by strategy era at T+10
    era_records: Dict[str, List[SignalRecord]] = {}
    for rec in priced_t10:
        era_records.setdefault(rec.strategy_era, []).append(rec)

    era_stats: Dict[str, HorizonStats] = {}
    for era, recs in era_records.items():
        rets   = [r.returns[10] for r in recs]   # type: ignore[misc]
        alphas = [r.alpha.get(10) for r in recs]
        era_stats[era] = _horizon_stats(rets, alphas, 10)

    # False positives: worst return at T+10 among priced signals
    worst = sorted(priced_t10, key=lambda r: r.returns[10])[:3]  # type: ignore[arg-type]

    return badge_stats, cap_stats, era_stats, worst


# ── Primary failure factor attribution ────────────────────────────────────────

def _primary_failure_factor(rec: SignalRecord) -> str:
    """Return the factor with highest weight×score contribution for a loser trade.

    The 'primary failure factor' is the factor that contributed most to the
    score but failed to predict forward returns — i.e., highest weighted score.
    """
    weights = rec.weights
    if not weights:
        # Combined cooked snapshots carry no top-level weights key. Equal-weight
        # the factors actually present so attribution reflects the highest
        # factor score instead of a fabricated legacy-weight ranking.
        present = [f for f in _CANONICAL_FACTORS if f in rec.factors]
        if present:
            weights = {f: 1.0 / len(present) for f in present}
        else:
            weights = {"edgar": 0.28, "insider": 0.23, "congress": 0.22,
                       "news": 0.15, "momentum": 0.12}
    contributions = {
        f: float(rec.factors.get(f) or 0.0) * weights.get(f, 0.0)
        for f in _CANONICAL_FACTORS
    }
    return max(contributions, key=lambda k: contributions[k])


# ── Report serialisation ───────────────────────────────────────────────────────

def _stats_to_dict(s: HorizonStats) -> Dict[str, Any]:
    return {
        "horizon":       s.horizon,
        "count":         s.count,
        "win_rate":      round(s.win_rate, 4),
        "avg_return":    round(s.avg_return, 4),
        "median_return": round(s.median_return, 4),
        "max_drawdown":  round(s.max_drawdown, 4),
        "profit_factor": round(s.profit_factor, 4) if s.profit_factor != float("inf") else None,
        "avg_alpha":     round(s.avg_alpha, 4),
    }


def build_csv_ledger(
    records:  List[SignalRecord],
    out_path: Path,
) -> None:
    """Write flat trade ledger CSV with one row per priced signal."""
    rows = []
    for r in records:
        if r.entry_price is None:
            continue
        rows.append({
            "Signal_Date":   r.signal_date.isoformat(),
            "Ticker":        r.ticker,
            "Badge":         r.badge,
            "Cap_Tier":      r.cap_tier,
            "Score":         round(r.final_score, 4),
            "Strategy_Era":  r.strategy_era,
            "Entry_Price":   round(r.entry_price, 4),
            "Return_T5":     round(r.returns[5], 4) if r.returns.get(5) is not None else None,
            "Return_T10":    round(r.returns[10], 4) if r.returns.get(10) is not None else None,
            "Return_T20":    round(r.returns[20], 4) if r.returns.get(20) is not None else None,
            "SPY_Return_T10": round(r.spy_returns[10], 4) if r.spy_returns.get(10) is not None else None,
            "Alpha_T10":     round(r.alpha[10], 4) if r.alpha.get(10) is not None else None,
        })

    if not rows:
        log.warning("No priced signals — CSV ledger will be empty")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    log.info("CSV ledger written to %s (%d rows)", out_path, len(rows))


def build_json_report(
    records:     List[SignalRecord],
    badge_stats: Dict[str, List[HorizonStats]],
    cap_stats:   Dict[str, HorizonStats],
    era_stats:   Dict[str, HorizonStats],
    worst:       List[SignalRecord],
) -> Dict[str, Any]:
    trades = []
    for r in records:
        if r.entry_price is None:
            continue
        trades.append({
            "ticker":        r.ticker,
            "signal_date":   r.signal_date.isoformat(),
            "badge":         r.badge,
            "cap_tier":      r.cap_tier,
            "strategy_era":  r.strategy_era,
            "final_score":   round(r.final_score, 4),
            "factors":       {k: round(v, 4) for k, v in r.factors.items()},
            "weights":       {k: round(v, 4) for k, v in r.weights.items()},
            "entry_price":   round(r.entry_price, 4),
            "returns":       {str(h): round(v, 4) if v is not None else None
                              for h, v in r.returns.items()},
            "spy_returns":   {str(h): round(v, 4) if v is not None else None
                              for h, v in r.spy_returns.items()},
            "alpha":         {str(h): round(v, 4) if v is not None else None
                              for h, v in r.alpha.items()},
        })

    return {
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "signal_count":  len(trades),
        "horizons":      _HORIZONS,
        # P2.2 — returns/alpha below are NET of these round-trip costs (fractions).
        "cost_model":    {"roundtrip_cost_by_tier": _ROUNDTRIP_COST},
        "badge_stats":   {
            badge: [_stats_to_dict(s) for s in stats]
            for badge, stats in badge_stats.items()
        },
        "cap_tier_stats_t10": {
            cap: _stats_to_dict(s)
            for cap, s in cap_stats.items()
        },
        "era_stats_t10": {
            era: _stats_to_dict(s)
            for era, s in era_stats.items()
        },
        "worst_trades": [
            {
                "ticker":         r.ticker,
                "signal_date":    r.signal_date.isoformat(),
                "badge":          r.badge,
                "strategy_era":   r.strategy_era,
                "return_t10":     round(r.returns.get(10, 0.0) or 0.0, 4),
                "primary_factor": _primary_failure_factor(r),
            }
            for r in worst
        ],
        "trades": trades,
    }


# ── Markdown report ────────────────────────────────────────────────────────────

def _pct(v: float) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.2f}%"


def _pf_str(v: float) -> str:
    return "∞" if v == float("inf") else f"{v:.2f}"


def build_markdown_report(
    report_date:   date,
    badge_stats:   Dict[str, List[HorizonStats]],
    cap_stats:     Dict[str, HorizonStats],
    era_stats:     Dict[str, HorizonStats],
    worst:         List[SignalRecord],
    total_signals: int,
) -> str:
    lines: List[str] = []
    lines.append("# 📊 WEEKLY PERFORMANCE AUDIT REPORT")
    lines.append(f"*Generated on: {report_date.isoformat()}*")
    lines.append("")

    # ── Global Performance ────────────────────────────────────────────────────
    lines.append("## 📈 Global Performance Metrics")
    lines.append(
        "| Badge Tier | Total Signals | Win Rate | "
        "Avg Return (T+5) | Avg Return (T+10) | "
        "Avg Alpha vs SPY (T+10) | Profit Factor |"
    )
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    badge_order = ["HIGH BUY", "TACTICAL BUY"]
    for badge in badge_order:
        stats_list = badge_stats.get(badge, [])
        stats_by_h = {s.horizon: s for s in stats_list}
        s5  = stats_by_h.get(5)
        s10 = stats_by_h.get(10)
        n   = s10.count if s10 else 0

        if not s5 or not s10 or n == 0:
            lines.append(f"| **{badge}** | 0 | — | — | — | — | — |")
            continue

        lines.append(
            f"| **{badge}** | {n} | {s10.win_rate * 100:.1f}% | "
            f"{_pct(s5.avg_return)} | {_pct(s10.avg_return)} | "
            f"{_pct(s10.avg_alpha)} | {_pf_str(s10.profit_factor)} |"
        )

    lines.append("")

    # ── Cap Tier Segmentation ─────────────────────────────────────────────────
    lines.append("## 🔍 Segmentation by Capitalization (T+10 Horizon)")
    tier_labels = [("large", "Large Caps"), ("mid", "Mid Caps"), ("small", "Small Caps")]
    for key, label in tier_labels:
        s = cap_stats.get(key)
        if s and s.count > 0:
            lines.append(
                f"- **{label}:** Win Rate: {s.win_rate * 100:.1f}% | "
                f"Avg Return: {_pct(s.avg_return)} | "
                f"Avg Alpha: {_pct(s.avg_alpha)} (Signals: {s.count})"
            )
        else:
            lines.append(f"- **{label}:** No data")

    lines.append("")

    # ── Strategy Era Breakdown ────────────────────────────────────────────────
    lines.append("## 🧬 Strategy Era Breakdown (T+10 Horizon)")
    lines.append(
        "| Era | Signals | Win Rate | Avg Return | Avg Alpha | Profit Factor |"
    )
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: |")
    for era, s in sorted(era_stats.items()):
        if s.count == 0:
            continue
        lines.append(
            f"| {era} | {s.count} | {s.win_rate * 100:.1f}% | "
            f"{_pct(s.avg_return)} | {_pct(s.avg_alpha)} | {_pf_str(s.profit_factor)} |"
        )
    if not any(s.count > 0 for s in era_stats.values()):
        lines.append("| — | No data | | | | |")

    lines.append("")

    # ── Detractors ────────────────────────────────────────────────────────────
    lines.append("## 📉 Top 3 Historical Detractors (Worst Performing Signals)")
    if not worst:
        lines.append("*No losing trades in the historical record.*")
    else:
        for i, rec in enumerate(worst, 1):
            ret_t10  = rec.returns.get(10)
            ret_str  = _pct(ret_t10) if ret_t10 is not None else "N/A"
            pf_label = _primary_failure_factor(rec)
            lines.append(
                f"{i}. Ticker: `{rec.ticker}` "
                f"(Date: {rec.signal_date.isoformat()}, Badge: {rec.badge}, "
                f"Era: {rec.strategy_era}, Return: {ret_str}). "
                f"Primary factor drag: **{pf_label}** "
                f"(score={rec.factors.get(pf_label, 0):.2f}, "
                f"weight={rec.weights.get(pf_label, 0):.0%})"
            )

    lines.append("")

    # ── Data Quality Note ─────────────────────────────────────────────────────
    lines.append("---")
    lines.append(
        f"*Total archive signals parsed: {total_signals} — "
        "priced signals reflect those with available yfinance history. "
        "Returns are **NET of estimated round-trip transaction costs** "
        f"(large {_ROUNDTRIP_COST['large']*1e4:.0f} / mid {_ROUNDTRIP_COST['mid']*1e4:.0f} / "
        f"small {_ROUNDTRIP_COST['small']*1e4:.0f} bps); the SPY benchmark is held "
        "passively. Signals with dead-feed factors are included as-is.*"
    )
    return "\n".join(lines)


# ── Discord KPI notification ───────────────────────────────────────────────────

def _score_bar(score: float, width: int = 8) -> str:
    filled = min(width, max(0, round(score * width)))
    return "▓" * filled + "░" * (width - filled)


def _pct_short(v: float) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v * 100:.1f}%"


_MARKET_FLAGS: dict[str, str] = {
    "USA": "🇺🇸",
    "EUROPE": "🇪🇺",
    "ASIA": "🇯🇵",
}


def _market_flag(market: str) -> str:
    return _MARKET_FLAGS.get(market, "🌐")


def _failure_label(ret: float) -> str:
    """Classify a negative return into a failure type."""
    if ret <= -0.05:
        return "STOPPED OUT"
    if ret <= -0.015:
        return "MEAN REVERTED"
    if ret < 0:
        return "NOISE"
    return ""


def _alpha_decay_tag(t10_avg: float, t20_avg: float) -> str:
    """Return decay warning if T+20 < T+10*0.5 or T+20 is negative."""
    if t20_avg < 0 or t20_avg < t10_avg * 0.5:
        return "⚠ ALPHA DECAY"
    return ""


def _truncate_field(text: str, limit: int = 1024) -> str:
    """Truncate a Discord embed field to stay within the character limit."""
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def build_backtest_discord_payload(
    report_date:   "date",
    badge_stats:   "Dict[str, List[HorizonStats]]",
    cap_stats:     "Dict[str, HorizonStats]",
    worst:         "List[SignalRecord]",
    total_signals: int,
    records:       "List[SignalRecord]",
) -> "Dict[str, Any]":
    """Hedge-fund-grade Discord embed: N visibility, alpha decay, capacity tiers."""

    badge_order = ["HIGH BUY", "TACTICAL BUY"]
    badge_emoji = {"HIGH BUY": "🟢", "TACTICAL BUY": "🟡"}

    # Colour: based on HIGH BUY overall win-rate at T+10
    hb_list  = badge_stats.get("HIGH BUY", [])
    hb_by_h  = {s.horizon: s for s in hb_list}
    hb_t10   = hb_by_h.get(10)
    overall_wr = hb_t10.win_rate if hb_t10 and hb_t10.count > 0 else 0.0
    if overall_wr >= 0.55:
        colour = 0x00FF00
    elif overall_wr >= 0.45:
        colour = 0xFFA500
    else:
        colour = 0xFF0000

    priced = sum(1 for r in records if r.entry_price is not None)
    description = (
        f"**Dataset: N={total_signals} signals** | "
        f"{priced} priced · Forward-Walk Validation Ledger\n"
        f"────────────────────"
    )

    fields: list[dict] = []

    # Section 1: Badge KPI table (inline, side-by-side)
    for badge in badge_order:
        stats_list = badge_stats.get(badge, [])
        stats_by_h = {s.horizon: s for s in stats_list}
        s5  = stats_by_h.get(5)
        s10 = stats_by_h.get(10)
        s20 = stats_by_h.get(20)
        n   = s10.count if s10 else 0
        emoji = badge_emoji.get(badge, "⚪")

        if n == 0:
            fields.append({"name": f"{emoji} {badge}", "value": "_No priced signals yet_", "inline": True})
            continue

        wins = round(s10.win_rate * n) if s10 else 0
        wr   = f"{s10.win_rate * 100:.1f}% ({wins}/{n})" if s10 else "—"
        pf   = f"{s10.profit_factor:.2f}" if s10 and s10.profit_factor != float("inf") else "∞"
        r5   = _pct_short(s5.avg_return)  if s5  else "—"
        r10v = _pct_short(s10.avg_return) if s10 else "—"
        r20v = s20.avg_return if s20 else 0.0
        r20s = _pct_short(r20v) if s20 else "—"

        # Alpha decay detection across T+10 → T+20
        decay_tag = ""
        if s10 and s20:
            decay_tag = _alpha_decay_tag(s10.avg_return, s20.avg_return)

        alp = _pct_short(s10.avg_alpha) if s10 else "—"

        value = _truncate_field(
            f"```\n"
            f"Metrics   | Value\n"
            f"----------|----------------\n"
            f"WR        | {wr}\n"
            f"PF        | {pf}\n"
            f"T+5 Avg   | {r5}\n"
            f"T+10 Avg  | {r10v}\n"
            f"T+20 Avg  | {r20s}{decay_tag}\n"
            f"α vs SPY  | {alp} (T+10)\n"
            f"```"
        )
        fields.append({"name": f"{emoji} {badge}  (N={n})", "value": value, "inline": True})

    # Section 2: Capacity & Liquidity Tiers
    tier_lines = ["────────────────────"]
    for key, label, flag in [
        ("large", "Large Caps", "🔵"),
        ("mid",   "Mid Caps",   "🟡"),
        ("small", "Small Caps", "🔴"),
    ]:
        s = cap_stats.get(key)
        if s and s.count > 0:
            wins_cap = round(s.win_rate * s.count)
            tier_lines.append(
                f"{flag} **{label}:** {_pct_short(s.avg_return)} avg "
                f"({wins_cap}/{s.count} WR {s.win_rate * 100:.1f}%) "
                f"| α {_pct_short(s.avg_alpha)}"
            )
        else:
            tier_lines.append(f"{flag} **{label}:** No data")
    fields.append({
        "name":   "📊 Capacity & Liquidity Tiers (T+10)",
        "value":  _truncate_field("\n".join(tier_lines)),
        "inline": False,
    })

    # Section 3: Edge Trajectory — recent BUY signals
    buy_badges = {"HIGH BUY", "TACTICAL BUY"}
    recent = sorted(
        [r for r in records if r.badge in buy_badges and r.entry_price is not None],
        key=lambda r: r.signal_date,
        reverse=True,
    )[:8]
    if recent:
        traj_lines = ["────────────────────"]
        for r in recent:
            mflag     = _market_flag(r.market)
            bar       = _score_bar(r.final_score, 10)
            t10_ret   = r.returns.get(10)
            alpha10   = r.alpha.get(10)
            ret_str   = _pct_short(t10_ret)   if t10_ret   is not None else "pending"
            alpha_str = _pct_short(alpha10)    if alpha10   is not None else "—"
            name_part = f"{r.company_name} ({r.ticker})" if r.company_name else r.ticker
            traj_lines.append(
                f"{mflag} `{name_part}` ({bar}) → T+10: **{ret_str}** | α: {alpha_str}"
            )
        fields.append({
            "name":   "⏱ Edge Trajectory — Recent BUY Signals",
            "value":  _truncate_field("\n".join(traj_lines)),
            "inline": False,
        })

    # Section 4: Risk Attribution — Top 3 Detractors
    if worst:
        risk_lines = ["────────────────────"]
        for r in worst:
            t10_ret   = r.returns.get(10)
            ret_str   = _pct_short(t10_ret) if t10_ret is not None else "—"
            fail_label = _failure_label(t10_ret) if t10_ret is not None else "—"
            pf_factor  = _primary_failure_factor(r)
            pf_score   = r.factors.get(pf_factor, 0.0)
            pf_weight  = r.weights.get(pf_factor, 0.0)
            mflag      = _market_flag(r.market)
            risk_lines.append(
                f"{mflag} `{r.ticker}` (T+10: {ret_str}) "
                f"· **{fail_label}** "
                f"· drag: {pf_factor} "
                f"(score={pf_score:.2f}, w={pf_weight:.0%})"
            )
        fields.append({
            "name":   "🚨 Risk Attribution — Top 3 Detractors",
            "value":  _truncate_field("\n".join(risk_lines)),
            "inline": False,
        })

    return {
        "embeds": [{
            "title":       "📊 STRATEGY PERFORMANCE LOG — ACTIVE EDGE TRACKING",
            "description": description,
            "color":       colour,
            "fields":      fields,
            "footer":      {
                "text": f"regime_trader · {report_date.isoformat()} · horizons: T+5 / T+10 / T+20"
            },
        }]
    }


def send_backtest_to_discord(webhook: str, payload: Dict[str, Any]) -> bool:
    """POST backtest payload to Discord webhook. Returns True on success."""
    try:
        import requests as _req
        resp = _req.post(webhook, json=payload, timeout=15)
        if resp.status_code in (200, 204):
            log.info("Backtest KPI sent to Discord")
            return True
        log.warning("Discord webhook returned %d: %s", resp.status_code, resp.text[:200])
        return False
    except Exception as exc:
        log.warning("Discord send failed: %s", exc)
        return False


# ── GitHub Actions summary helper ─────────────────────────────────────────────

def write_github_step_summary(md_path: Path) -> None:
    summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_file:
        return
    try:
        with open(summary_file, "a", encoding="utf-8") as f:
            f.write(md_path.read_text(encoding="utf-8"))
            f.write("\n")
        log.info("Appended to GITHUB_STEP_SUMMARY")
    except Exception as exc:
        log.warning("Could not write GITHUB_STEP_SUMMARY: %s", exc)


# ── CLI ────────────────────────────────────────────────────────────────────────

import os  # noqa: E402 (placed after dataclasses for readability)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Weekly signal performance backtest for regime_trader pipeline"
    )
    parser.add_argument(
        "--archive-dir", type=Path, default=_ARCHIVE_DIR,
        help=f"Directory containing historical top_lists*.json files "
             f"(default: {_ARCHIVE_DIR})",
    )
    parser.add_argument(
        "--report-json", type=Path, default=_REPORT_JSON,
        help="Output path for machine-readable JSON report",
    )
    parser.add_argument(
        "--ledger-csv", type=Path, default=_LEDGER_CSV,
        help="Output path for flat trade ledger CSV",
    )
    parser.add_argument(
        "--summary-md", type=Path, default=_SUMMARY_MD,
        help="Output path for Markdown summary",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse logs and compute metrics but skip price download (returns will be empty)",
    )
    parser.add_argument(
        "--discord-webhook", type=str,
        default=os.environ.get("DISCORD_WEBHOOK_URL", ""),
        help="Discord webhook URL for KPI notification (falls back to DISCORD_WEBHOOK_URL env var)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level   = logging.DEBUG if args.verbose else logging.INFO,
        format  = "%(asctime)s %(levelname)-8s %(message)s",
        stream  = sys.stdout,
    )

    if not _HAS_YF and not args.dry_run:
        log.error("yfinance not installed. Use --dry-run for CI or: pip install yfinance")
        return 1

    # ── 1. Load all signals ────────────────────────────────────────────────────
    records = load_all_signals(args.archive_dir)
    if not records:
        log.warning(
            "No signals found. Populate %s with historical top_lists.json "
            "snapshots and re-run.", args.archive_dir,
        )
        # Write empty reports so downstream consumers don't break
        empty_report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "signal_count": 0,
            "horizons": _HORIZONS,
            "badge_stats": {},
            "cap_tier_stats_t10": {},
            "worst_trades": [],
            "trades": [],
        }
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(empty_report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        md = (
            "# 📊 WEEKLY PERFORMANCE AUDIT REPORT\n"
            f"*Generated on: {date.today().isoformat()}*\n\n"
            "⚠️ No historical signal archive found. "
            f"Populate `{args.archive_dir}` with past `top_lists.json` snapshots."
        )
        args.summary_md.write_text(md, encoding="utf-8")
        sys.stdout.buffer.write((md + "\n").encode("utf-8", errors="replace"))
        return 0

    # ── 2. Download prices ─────────────────────────────────────────────────────
    all_tickers = list({r.ticker for r in records})
    all_dates   = [r.signal_date for r in records]
    price_start = min(all_dates)
    price_end   = max(all_dates)

    prices = fetch_prices(all_tickers, price_start, price_end, dry_run=args.dry_run)

    # ── 3. Calculate returns ───────────────────────────────────────────────────
    compute_returns(records, prices)

    priced_count = sum(1 for r in records if r.entry_price is not None)
    log.info("Priced signals: %d / %d", priced_count, len(records))

    # ── 4. Compute metrics ─────────────────────────────────────────────────────
    badge_stats, cap_stats, era_stats, worst = compute_metrics(records)

    # ── 5. Write CSV ledger ────────────────────────────────────────────────────
    build_csv_ledger(records, args.ledger_csv)

    # ── 6. Write JSON report ───────────────────────────────────────────────────
    json_report = build_json_report(records, badge_stats, cap_stats, era_stats, worst)
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(json_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log.info("JSON report written to %s", args.report_json)

    # ── 7. Write Markdown summary ──────────────────────────────────────────────
    md = build_markdown_report(
        report_date   = date.today(),
        badge_stats   = badge_stats,
        cap_stats     = cap_stats,
        era_stats     = era_stats,
        worst         = worst,
        total_signals = len(records),
    )
    args.summary_md.write_text(md, encoding="utf-8")
    log.info("Markdown summary written to %s", args.summary_md)
    sys.stdout.buffer.write((md + "\n").encode("utf-8", errors="replace"))

    write_github_step_summary(args.summary_md)

    # ── 8. Discord KPI notification ────────────────────────────────────────────
    webhook = args.discord_webhook
    if webhook and not args.dry_run:
        discord_payload = build_backtest_discord_payload(
            report_date   = date.today(),
            badge_stats   = badge_stats,
            cap_stats     = cap_stats,
            worst         = worst,
            total_signals = len(records),
            records       = records,
        )
        send_backtest_to_discord(webhook, discord_payload)
    elif args.dry_run:
        log.info("dry-run: skipping Discord notification")
    else:
        log.info("No DISCORD_WEBHOOK_URL set — skipping Discord notification")

    return 0


if __name__ == "__main__":
    sys.exit(main())
