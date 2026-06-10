"""backend/market_intel/generate_top_lists.py
Nine-factor weighted scoring → top_lists_us.json + top5.csv

Reads  logs/intel_source_status.json  (written by scripts/run_pipeline.py)
and applies Markowitz (1990 Nobel) portfolio ranking. WEIGHTS are imported
from src.weights (canonical single source of truth).

Badge thresholds (Sharpe-inspired):
  HIGH BUY     ≥ 0.80
  TACTICAL BUY ≥ 0.60
  WATCHLIST    < 0.60

Cap tiers (relative within universe, sorted by market cap):
  large : top 20 by market cap
  mid   : rank 21–35
  small : rank 36+

Each section in top_lists_us.json is ranked by final_score descending (top 5).

Output:
  logs/top_lists_us.json — consumed by scripts/send_toplists_discord.py
  logs/top5.csv          — flat reference file for downstream analysis

Usage:
  python -m backend.market_intel.generate_top_lists --log-dir logs --run-id $GITHUB_RUN_ID
  python -m backend.market_intel.generate_top_lists --force --verbose
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.scoring.normalize import normalize_score
from src.utils.io import save_json_atomic
from backend.market_intel.validator import detect_anomalies, PipelineIntegrityError  # noqa: F401 (re-exported)
# Canonical 9-factor weights — v2.1-global, single source of truth.
# Grinold & Kahn (2000): scores must be consistent across all pipeline stages.
from src.config.weights import (
    WEIGHTS, WEIGHTS_GLOBAL, WEIGHTS_VERSION,  # noqa: F401
    get_region, _piotroski_gate_multiplier,
)
from backend.market_intel.portfolio_optimizer import run_optimizer

log = logging.getLogger("generate_top_lists")

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, (
    f"WEIGHTS must sum to 1.0, got {sum(WEIGHTS.values()):.8f}. "
    "Check src/config/weights.py."
)

# Maps factor key → field name in intel_source_status.json results (12-factor schema).
FACTOR_FIELDS: Dict[str, str] = {
    "insider_conviction":  "insider_conviction_score",
    "insider_breadth":     "insider_breadth_score",
    "congress":            "congress_score",
    "news_sentiment":      "news_sentiment_score",
    "news_buzz":           "news_buzz_score",
    "momentum_long":       "momentum_long_score",
    "volume_attention":    "volume_attention_score",
    "analyst_consensus":   "analyst_consensus_score",
    "analyst_revision":    "analyst_revision_score",
    "price_target_upside": "price_target_upside_score",
    "quality_piotroski":   "quality_piotroski_score",
    "transcript_tone":     "transcript_tone_score",
    # EU/Asia fundamental value + quality signals (v2.3)
    "fcf_yield":           "fcf_yield_score",
    "amihud_shock":        "amihud_shock_score",
    "pb_value_up":         "pb_value_up_score",
    "roic_quality":        "roic_quality_score",
}

# Schema gate: a ticker is "incomplete" when more than this many factors are
# zero (i.e. missing / dead API).  Incomplete tickers are scored but flagged;
# they are NOT excluded from ranking — exclusion would distort cross-sectional
# normalization.  Instead, validation_metadata carries the signal to consumers.
_SCHEMA_MISSING_THRESHOLD = 10  # >10 zero factors → is_complete = False
# Original threshold was 6 (for 12 non-INTL factors).  When 4 INTL-only factors
# (fcf_yield, amihud_shock, pb_value_up, roic_quality) were added to FACTOR_FIELDS
# they always score 0.0 for US tickers by architectural design.  Threshold must
# rise by exactly 4 to preserve the original completeness criterion: a US ticker
# needs ≤6 of its 12 non-INTL factors missing (6 + 4 always-absent = 10 total).
# Rationale: after RT-QA-2026-REV6 (FIX 1+2), analyst_consensus and news_sentiment
# now correctly return 0.0 when absent (instead of the phantom 0.5 they returned
# before). The structurally-zero factor set for a healthy large-cap ticker is now:
#   congress (sparse trades), news_sentiment (no/neutral FMP news), news_buzz (same),
#   analyst_consensus (no bulk coverage), analyst_revision (sparse), transcript_tone
# = up to 6 legitimate zeros. Threshold raised from 4 → 6 so the gate fires only
# when the always-populated factors (momentum_long, insider_conviction, volume_attention)
# are also dead — which indicates a genuine price/EDGAR feed failure.

# Circuit breaker: if fewer than this fraction of the universe pass the schema
# gate, PipelineIntegrityError is raised and top_lists_us.json is NOT written.
_CIRCUIT_BREAKER_MIN_FRACTION = 0.40   # 40% of universe minimum
# Rationale: with live FMP stable/ routes for insider/news/quote (Phase 1 migration),
# the schema-completeness rate should be well above 40% on a healthy pipeline.
# 0.05 was a workaround for the broken /api/v4 routes that zeroed factors silently.
# Now that FMPEndpointError surfaces dead routes loudly, a low threshold is a trap
# that would allow a partially-broken pipeline to write rankings without alerting.

# Hull (2015): regime shifts precede VIX threshold breaches.
# Jegadeesh & Titman (1993): momentum factor IC turns negative in bear regimes.
# Applied when spy_momentum_regime from run_pipeline PATCH 10 indicates stress
# but VIX has not yet crossed 30 (avoids double-dampening with VIX overlay).
_MOMENTUM_REGIME_MULTIPLIERS: dict = {
    "BEAR_CRASH":    0.30,  # SPY 63d return < -20%: severe early warning
    "BEAR_MOMENTUM": 0.55,  # SPY 63d return < -10%: moderate dampening
    "BEAR_TREND":    0.70,  # SPY 12-1m return < -15%: mild dampening
    "BULL_STRONG":   0.95,  # SPY 12-1m return > +30%: reversal caution
    "NORMAL":        1.00,  # No adjustment
}

_BADGES = [
    (0.80, "HIGH BUY"),
    (0.60, "TACTICAL BUY"),
    (0.00, "WATCHLIST"),
]

# Absolute market-cap thresholds (standard institutional definitions)
_LARGE_FLOOR = 10_000_000_000   # >= $10B  → large cap
_MID_FLOOR = 2_000_000_000   # $2B–$10B → mid cap
#                                 < $2B    → small cap

_TARGET_SECTORS = [
    "Energy",
    "Materials",
    "Industrials",
    "Consumer Discretionary",
    "Consumer Staples",
    "Healthcare",
    "Financials",
    "Information Technology",
    "Communication Services",
    "Real Estate",
    "Utilities",
]

# FMP returns "Health Care" for EU/LSE tickers; universe.csv uses "Healthcare".
# Canonicalize before sector matching so both map to the same bucket.
_SECTOR_CANON: Dict[str, str] = {
    "Health Care": "Healthcare",
}


def _canon_sector(sector: str) -> str:
    return _SECTOR_CANON.get(sector, sector)


def _badge(score: float) -> str:
    for threshold, label in _BADGES:
        if score >= threshold:
            return label
    return "WATCHLIST"


def _schema_gate(
    results: List[Dict[str, Any]],
    universe_size: int,
) -> List[Dict[str, Any]]:
    """Validate each ticker's factor completeness; raise if universe collapses.

    For each ticker, counts how many of the seven raw factor scores are exactly
    0.0 (the sentinel for a missing/dead feed).  Attaches a ``_validation``
    dict to each row:

        {"is_complete": bool, "missing_sources": ["insider", "news"]}

    A ticker is incomplete when missing_sources length > _SCHEMA_MISSING_THRESHOLD.
    Incomplete tickers are NOT removed — removal would corrupt cross-sectional
    normalization by shrinking the peer group.  They are flagged so downstream
    consumers (Discord bot, audit log) can warn.

    Circuit breaker: if the complete-ticker count falls below
    _CIRCUIT_BREAKER_MIN_FRACTION of ``universe_size``, raises
    PipelineIntegrityError before any file is written.

    Returns the same list (mutated in-place with ``_validation`` added).
    """
    complete_count = 0

    for row in results:
        missing: List[str] = []
        for factor, field in FACTOR_FIELDS.items():
            val = row.get(field)
            # None or exactly 0.0 both indicate an absent/dead feed
            if val is None or float(val) == 0.0:
                missing.append(factor)

        esg_flag_raw = row.get("esg_flag")
        if isinstance(esg_flag_raw, bool):
            esg_flag = esg_flag_raw
        else:
            esg_score = esg_flag_raw
            if esg_score is None:
                esg_score = row.get("esg_score")
            if esg_score is None:
                esg_score = row.get("ESGScore")
            try:
                esg_flag = bool(
                    esg_score is not None and float(esg_score) < 30)
            except Exception:
                esg_flag = False

        is_complete = len(missing) <= _SCHEMA_MISSING_THRESHOLD
        row["_validation"] = {
            "is_complete":    is_complete,
            "missing_sources": missing,
        }
        if esg_flag:
            row["_validation"]["esg_exclusion_candidate"] = True

        if is_complete:
            complete_count += 1
        else:
            log.debug(
                "SCHEMA_GATE ticker=%s missing=%s — score may be unreliable",
                row.get("ticker", "?"),
                missing,
            )

    # Circuit breaker — abort before writing any output
    min_required = max(1, round(_CIRCUIT_BREAKER_MIN_FRACTION * universe_size))
    if complete_count < min_required:
        raise PipelineIntegrityError(
            f"Schema gate: only {complete_count}/{universe_size} tickers are complete "
            f"(threshold {_CIRCUIT_BREAKER_MIN_FRACTION:.0%} = {min_required}). "
            "Data feeds may be down. top_lists_us.json will NOT be written."
        )

    log.info(
        "Schema gate passed: %d/%d tickers complete (threshold %d)",
        complete_count, len(results), min_required,
    )
    return results


def _cross_sectional_normalize(results: List[Dict[str, Any]]) -> List[Dict[str, float]]:
    """Markowitz (1990 Nobel) — normalize each factor cross-sectionally to [0, 1].

    For each of the five factors, winsorizes at the 5th/95th percentile and
    min-max scales across the full universe.

    Edge cases:
      - Explicit null (None) in JSON → treated as 0.0, NOT as 0.5.
        A dead API feed must not get a free neutral pass.
      - All values identical AND zero (fully failed feed) → 0.0, not 0.5.
        Penalises a completely dead API rather than rewarding it with neutral credit.
      - All values identical AND non-zero → 0.5 (can't rank cross-sectionally; neutral).
      - n == 1 with a non-zero value → 0.5 (single ticker, can't rank relatively).

    $x_{norm,i} = \\frac{winsorize(x_i) - \\min}{\\max - \\min}$
    """
    n = len(results)
    if n == 0:
        return []

    normed_factors: Dict[str, np.ndarray] = {}
    for factor, field in FACTOR_FIELDS.items():
        # Fix #1 + #2: safe None handling — explicit null → 0.0 (penalise, not neutral)
        raw_values = [r.get(field) for r in results]
        raw = np.array(
            [float(v) if v is not None else 0.0 for v in raw_values])

        rmax, rmin = float(np.nanmax(raw)), float(np.nanmin(raw))

        if rmax == 0.0 and rmin == 0.0:
            # Entire factor missing / API dead → penalise with 0.0, not neutral 0.5
            normed_factors[factor] = np.full(n, 0.0)
        elif n == 1 or rmax == rmin:
            # Single ticker or all identical non-zero values → neutral 0.5
            normed_factors[factor] = np.full(n, 0.5)
        else:
            # Sparse-signal guard: insider (and similar) factors have many exact
            # zeros (dead tickers) and a small minority with real signal.  When
            # >90% of values are exactly 0.0, the 5th-percentile lower winsorize
            # clips the max to 0, collapsing everything and incorrectly triggering
            # the neutral-0.5 branch.  Disable winsorization only when the floor
            # is literally 0 (i.e. the zero tickers are "absent", not "low").
            frac_at_zero = float(np.mean(raw == 0.0))
            if frac_at_zero > 0.90:
                lo, hi = 0, 100   # no winsorization — preserve sparse signal
            else:
                lo, hi = 5, 95
            scaled = normalize_score(raw, lo_pct=lo, hi_pct=hi) / 100.0
            if float(np.nanmax(scaled)) == float(np.nanmin(scaled)):
                normed_factors[factor] = np.full(n, 0.5)
            else:
                normed_factors[factor] = scaled

    return [
        {f: round(float(normed_factors[f][i]), 4) for f in normed_factors}
        for i in range(n)
    ]


def _effective_weights(
    norm_factor_list: List[Dict[str, float]],
    weights: Dict[str, float],
) -> Dict[str, float]:
    """Redistribute weight from dead factors (all 0.0) to live ones proportionally.

    When a data feed is completely down every ticker receives the same contribution
    from that factor (0.0), wasting its allocated weight.  This function detects
    all-zero factors and rolls their weight to the live factors so the remaining
    signals retain full discriminating power.

    Example: congress dead (all 0.0), weight=0.22 redistributes pro-rata across
    the six live factors in proportion to their existing weights, so the
    surviving weights still sum to 1.0.
    """
    if not norm_factor_list:
        return dict(weights)

    dead: set = set()
    for factor in weights:
        vals = [row.get(factor, 0.0) for row in norm_factor_list]
        if max(vals) == 0.0 and min(vals) == 0.0:
            dead.add(factor)

    if not dead:
        return dict(weights)

    live = {k: v for k, v in weights.items() if k not in dead}
    dead_total = sum(weights[f] for f in dead)
    live_total = sum(live.values())

    if live_total <= 0:
        return dict(weights)   # everything dead — fall back to original

    for k in live:
        live[k] = round(live[k] + dead_total * (live[k] / live_total), 6)

    # Per-weight round(6) accumulates rounding error — correct on the largest weight
    # so the output always sums to exactly 1.0 (required by CI weight assertion).
    _live_sum = sum(live.values())
    if _live_sum != 1.0:
        _max_k = max(live, key=live.__getitem__)
        live[_max_k] = round(live[_max_k] + (1.0 - _live_sum), 6)

    return live


def _apply_vix_overlay(score: float, vix: Optional[float]) -> float:
    """Multiplicative macro-regime penalty (absolute risk layer).

    Cross-sectional ranking always produces relative winners even during a crash.
    This overlay converts relative scores to absolute risk-adjusted scores by
    dampening all signals when the VIX regime is elevated.

      VIX ≥ 40 (Crash)        : ×0.20 — almost nothing should be HIGH BUY in a crash
      VIX ≥ 30 (Panic)        : ×0.50 — significant systemic risk, dampen all buys
      VIX ≥ BEAR_THRESHOLD    : ×0.80 — elevated risk, mild penalty (bear at 20)
      VIX  < BEAR_THRESHOLD   : ×1.00 — no adjustment

    Thresholds sourced from src.risk.regime — the single source of truth.
    """
    if vix is None:
        return score
    from src.risk.regime import BEAR_THRESHOLD, CAPITULATION_THRESHOLD  # noqa: PLC0415
    if vix >= 40:
        return score * 0.20
    if vix >= CAPITULATION_THRESHOLD:
        return score * 0.50
    if vix >= BEAR_THRESHOLD:
        return score * 0.80
    return score


def _ticker_effective_weights(
    row: Dict[str, Any],
    base_weights: Dict[str, float],
) -> Dict[str, float]:
    """Compute per-ticker effective weights, redistributing weight of None factors.

    None  = API failure  → weight removed from denominator, redistributed pro-rata
    0.0   = zero signal  → kept in denominator (genuine zero, not missing data)
    """
    available = {
        f: w for f, w in base_weights.items()
        if row.get(FACTOR_FIELDS.get(f, f + "_score")) is not None
    }
    total = sum(available.values())
    if total <= 1e-9:
        n = len(base_weights)
        return {f: 1.0 / n for f in base_weights}
    return {f: w / total for f, w in available.items()}


def _to_entry(
    row: Dict[str, Any],
    norm_factors: Dict[str, float],
    vix: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
    quiver_evidence: Optional[Dict[str, Any]] = None,
    momentum_multiplier: float = 1.0,
) -> Dict[str, Any]:
    w = weights if weights is not None else WEIGHTS
    effective_w = _ticker_effective_weights(row, w)
    # Authoritative for Discord and Claude analysis outputs. See run_pipeline.py for intel_source_status.json.
    raw_score = round(
        sum(effective_w.get(f, 0.0) * norm_factors.get(f, 0.0) for f in effective_w),
        4,
    )
    # Apply VIX macro overlay first (absolute regime risk)
    score_after_vix = round(_apply_vix_overlay(raw_score, vix), 4)
    # Apply Piotroski quality gate (suppress/discount financially distressed companies)
    gate = _piotroski_gate_multiplier(row.get("quality_piotroski_raw"))
    # Apply momentum regime pre-dampening (early bear detection, avoids double-count)
    score = round(score_after_vix * momentum_multiplier * gate, 4)
    # Carry schema-gate result into the entry so it appears in top_lists_us.json
    validation = row.get(
        "_validation", {"is_complete": True, "missing_sources": []})
    esg_flag_value = row.get("esg_flag")
    esg_score_value = row.get("esg_score") if row.get(
        "esg_score") is not None else row.get("ESGScore")
    if esg_score_value is not None:
        try:
            esg_score_value = float(esg_score_value)
        except Exception:
            esg_score_value = None

    if esg_flag_value is None and esg_score_value is not None:
        esg_flag_value = bool(esg_score_value < 30)
    else:
        esg_flag_value = bool(esg_flag_value)

    esg_e_score_value = row.get("esg_e_score")
    if esg_e_score_value is None:
        esg_e_score_value = row.get("environmentalScore")
    if esg_e_score_value is not None:
        try:
            esg_e_score_value = float(esg_e_score_value)
        except Exception:
            esg_e_score_value = None

    return {
        "ticker":          row.get("ticker", "?"),
        "sector":          _canon_sector(row.get("sector", "Unknown")),
        "cap_tier":        row.get("cap_tier", "large"),
        "market_cap":      float(row.get("market_cap", 0.0)),
        "raw_score":       raw_score,
        "final_score":     score,
        "badge":           _badge(score),
        "region":          get_region(row.get("ticker", "")),
        "weights_set":     "US" if get_region(row.get("ticker", "")) == "US" else "GLOBAL",
        "ceo_buy":         bool(row.get("ceo_buy", False)),
        "form4_count":     int(row.get("form4_count", 0)),
        "factors":         norm_factors,
        "validation_metadata": validation,
        "quiver_evidence":         quiver_evidence or {},
        "news_source":             row.get("news_source", "none"),
        "insider_usd":             float(row.get("insider_usd", 0.0)),
        "momentum_spy_relative":   float(row.get("momentum_spy_relative", 0.0)),
        "volume_spike":            float(row.get("volume_spike", 1.0)),
        "market":                  row.get("market", "USA"),
        "esg_score":               esg_score_value,
        "esg_e_score":             esg_e_score_value,
        "esg_flag":                esg_flag_value,
        # PATCH-03: audit fields — momentum regime dampening
        "momentum_multiplier":     momentum_multiplier,
        "pipeline":                "US",
    }


def _assign_cap_tiers(entries: List[Dict[str, Any]]) -> None:
    """Assign cap_tier using absolute market-cap thresholds (standard definitions).

    Tickers whose fetched market_cap looks implausible (< $500M for a universe
    stock, or 0) fall back to the cap_tier already in the entry (from the CSV)
    rather than being mis-classified as small cap.
    """
    for entry in entries:
        cap = float(entry.get("market_cap") or 0)
        # Guard against obviously wrong market cap data.  yfinance sometimes returns
        # rounded/stale values (e.g. AMZN at $1B instead of $2T).  Skip any ticker
        # whose fetched cap is below $500M — implausibly low for any S&P 500 stock.
        if cap <= 500_000_000:
            continue   # keep existing cap_tier (from CSV or prior assignment)
        if cap >= _LARGE_FLOOR:
            entry["cap_tier"] = "large"
        elif cap >= _MID_FLOOR:
            entry["cap_tier"] = "mid"
        else:
            entry["cap_tier"] = "small"


def _apply_congress_boost(
    entries: List[Dict[str, Any]],
    log_dir: Path,
) -> List[Dict[str, Any]]:
    """Apply Congress conviction multiplier to final_score before ranking.

    Reads anomaly_report_latest.json (written by detect_anomalies) and looks for
    CONGRESS_CLUSTER records.  For each matching ticker:

        boosted_score = final_score × (1 + 0.10 × conviction_score)

    Rationale for 0.10 coefficient: a cluster of 5 members (conviction=1.0)
    lifts the score by exactly 10% — enough to move a 0.72 ticker ahead of a
    0.75 one when conviction is high, but not enough to manufacture a HIGH BUY
    from a weak base.  The multiplier is linear so the signal is interpretable.

    All entries receive a ``congress_boost`` field (0.0 = no boost applied) so
    the shadow score comparison is always possible even when the feed is dead.

    Never raises — a missing or unreadable anomaly report is treated as "no
    clusters found" and all boosts remain 0.0.
    """
    # Load conviction scores from the anomaly report
    conviction_map: Dict[str, float] = {}
    report_path = Path(log_dir) / "anomaly_report_latest.json"
    try:
        records = json.loads(report_path.read_text(encoding="utf-8"))
        for rec in records:
            if rec.get("flag") == "CONGRESS_CLUSTER":
                ticker = rec.get("ticker", "")
                if ticker:
                    conviction_map[ticker] = float(
                        rec.get("conviction_score", 0.0))
    except FileNotFoundError:
        pass   # no anomalies detected this run → no boost → correct
    except Exception as exc:
        log.warning("Congress boost: could not read anomaly report — %s", exc)

    if conviction_map:
        log.info(
            "Congress boost: %d ticker(s) with cluster signal: %s",
            len(conviction_map),
            ", ".join(f"{t}(conv={v:.2f})" for t, v in conviction_map.items()),
        )

    # Apply boost and attach congress_boost field to every entry
    for entry in entries:
        ticker = entry.get("ticker", "")
        conviction = conviction_map.get(ticker, 0.0)
        if conviction > 0.0:
            raw_final = entry["final_score"]
            boosted = round(raw_final * (1.0 + 0.10 * conviction), 4)
            entry["final_score"] = boosted
            entry["badge"] = _badge(boosted)
            entry["congress_boost"] = round(0.10 * conviction, 4)
            log.debug(
                "Congress boost applied: %s %.4f → %.4f (conv=%.2f)",
                ticker, raw_final, boosted, conviction,
            )
        else:
            entry["congress_boost"] = 0.0

    return entries


def _log_promoted(
    top_buys: List[Dict[str, Any]],
    shadow_top_buys: List[Dict[str, Any]],
    log_dir: Path,
    run_id: str,
) -> None:
    """Append one JSON line per promoted ticker to boost_history.log.

    A ticker is "promoted" when it appears in top_buys but not in
    shadow_top_buys — meaning the Congress boost moved it into the top-5.

    Each line contains:
      run_id, timestamp, ticker, shadow_rank (1-based, None if absent),
      boosted_rank (1-based), score_delta (boosted - shadow), congress_boost.

    Designed for append-only accumulation: after 3 months, a single
    `jq -r '...'` pass produces the full promotion history for backtesting.
    Never raises — log failures are silently swallowed.
    """
    shadow_index: Dict[str, int] = {
        e["ticker"]: i + 1 for i, e in enumerate(shadow_top_buys)}
    shadow_scores: Dict[str, float] = {
        e["ticker"]: e["final_score"] for e in shadow_top_buys}
    now_str = datetime.now(timezone.utc).isoformat()

    lines: List[str] = []
    for boosted_rank, entry in enumerate(top_buys, 1):
        ticker = entry["ticker"]
        boost = entry.get("congress_boost", 0.0)
        if boost <= 0.0:
            continue

        shadow_rank = shadow_index.get(ticker)
        # score_delta: how much the boost moved the final_score
        score_delta = round(
            boost * entry["final_score"] / (1.0 + boost), 4) if boost > 0 else 0.0
        # if ticker was already in shadow top_buys, delta is boosted - shadow score
        if ticker in shadow_scores:
            score_delta = round(
                entry["final_score"] - shadow_scores[ticker], 4)

        record = {
            "run_id":        run_id,
            "timestamp":     now_str,
            "ticker":        ticker,
            "shadow_rank":   shadow_rank,
            "boosted_rank":  boosted_rank,
            "score_delta":   score_delta,
            "congress_boost": round(boost, 4),
        }

        if shadow_rank is not None:
            msg = (
                f"Ticker {ticker} promoted from rank {shadow_rank} to rank {boosted_rank}. "
                f"Delta: +{score_delta:.4f} score (Congress Boost: +{boost:.2f})"
            )
        else:
            msg = (
                f"Ticker {ticker} entered top_buys at rank {boosted_rank} via Congress boost. "
                f"Delta: +{score_delta:.4f} score (Congress Boost: +{boost:.2f})"
            )
        log.info(msg)
        lines.append(json.dumps(record, separators=(",", ":")))

    if not lines:
        return

    history_path = Path(log_dir) / "boost_history.log"
    try:
        with history_path.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
    except Exception as exc:
        log.warning("Could not write boost_history.log: %s", exc)


def _sector_picks(entries: List[Dict[str, Any]], n: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    """Select top n tickers per target sector, ranked by final_score descending."""
    result: Dict[str, List[Dict[str, Any]]] = {}
    for sector in _TARGET_SECTORS:
        candidates = [e for e in entries if _canon_sector(e.get("sector", "")) == sector]
        result[sector] = sorted(
            candidates, key=lambda e: e["final_score"], reverse=True)[:n]
    return result


def _read_vix(log_dir: Path) -> Optional[float]:
    """Return current VIX from market_state.json, or fall back to a yfinance fetch.

    Keeping the VIX read outside the cross-sectional normaliser preserves the
    separation between relative ranking (cross-section) and absolute regime risk
    (VIX overlay).  Returns None only when both sources fail, in which case no
    overlay is applied and a warning is emitted.
    """
    # Primary: market_state.json produced by engine_worker (avoids redundant API call)
    market_state_path = log_dir / ".." / "data" / "market_state.json"
    try:
        ms = json.loads(
            market_state_path.resolve().read_text(encoding="utf-8"))
        vix = ms.get("macro_status", {}).get("vix_latest")
        if vix is not None:
            log.info("VIX overlay: %.1f (from market_state.json)", float(vix))
            return float(vix)
    except Exception:
        pass

    # Fallback: FMP stable/quote for ^VIX (real-time, no yfinance needed)
    try:
        from src.services.fmp_client import FMPClient as _FC
        q = _FC().get_quote("^VIX")
        if q:
            vix = float(q.get("price") or q.get("previousClose") or 0)
            if vix > 0:
                log.info("VIX overlay: %.1f (from FMP quote)", vix)
                return vix
    except Exception as exc:
        log.warning("FMP VIX fetch failed — no macro overlay applied: %s", exc)

    return None


def _vix_regime_label(vix: Optional[float]) -> str:
    """Canonical regime label from src.risk.regime, or 'UNKNOWN' when VIX absent."""
    if vix is None:
        return "UNKNOWN"
    try:
        from src.risk.regime import get_regime  # noqa: PLC0415
        return get_regime(float(vix)).value
    except Exception:
        return "UNKNOWN"


def _log_kill_switch_state(
    kill_switch: bool,
    vix: float,
    log_dir: Path,
    run_id: str,
) -> None:
    """Append kill-switch state to persistent NDJSON audit log.

    Written at scoring time so the audit trail covers every run, including
    those where the switch is inactive.  Consumers can grep for
    kill_switch_active=true to reconstruct suppression history.
    Regime transitions (e.g. NORMAL -> BEAR) are surfaced as warnings so they
    are visible in CI logs without parsing the NDJSON.
    """
    import os  # noqa: PLC0415 — local import avoids circular dep at module level
    regime = _vix_regime_label(vix)
    entry = {
        "event": "kill_switch_state",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kill_switch_active": kill_switch,
        "vix": round(vix, 2),
        "vix_regime": regime,
        "run_id": run_id or os.environ.get("GITHUB_RUN_ID", "local"),
    }
    audit_log = log_dir / "kill_switch_audit.ndjson"

    # Tolerant read of the previous entry — detect regime transitions.
    prev_regime: Optional[str] = None
    needs_newline = False
    try:
        if audit_log.exists():
            raw = audit_log.read_text(encoding="utf-8")
            # A truncated last line (crashed writer) must not corrupt the next
            # append — terminate it before writing the new entry.
            needs_newline = bool(raw) and not raw.endswith("\n")
            lines = raw.strip().splitlines()
            if lines:
                prev_regime = json.loads(lines[-1]).get("vix_regime")
    except Exception as exc:
        log.warning("kill_switch_audit.ndjson read failed (non-fatal): %s", exc)
    if prev_regime and prev_regime != regime:
        log.warning(
            "REGIME TRANSITION: %s -> %s (VIX %.1f)", prev_regime, regime, vix
        )

    try:
        with audit_log.open("a", encoding="utf-8") as fh:
            if needs_newline:
                fh.write("\n")
            fh.write(json.dumps(entry) + "\n")
    except Exception as exc:
        log.warning("kill_switch_audit.ndjson write failed (non-fatal): %s", exc)


def generate(
    status: Dict[str, Any],
    run_id: str,
    log_dir: Path,
) -> Dict[str, Any]:
    """Score, rank, and tier the full ticker universe."""
    results = status.get("results", [])
    if not results:
        log.warning(
            "No results found in intel_source_status.json — producing empty top_lists")

    # Fix #3: read current VIX once; passed into _to_entry for the macro overlay
    current_vix = _read_vix(log_dir)
    if current_vix is not None:
        log.info(
            "Macro overlay active -- VIX %.1f -> multiplier %.2f",
            current_vix,
            _apply_vix_overlay(1.0, current_vix),
        )

    # PATCH-03: Read SPY momentum regime from intel_source_status.json.
    # Applies a pre-VIX dampening for bear markets that develop before VIX
    # crosses 30 (e.g. 2022 rate shock: SPY -19%, VIX peak 37 but mostly < 30).
    # Hull (2015): "parameters may remain constant and then change because of a
    # regime shift" — early detection requires multi-signal confirmation.
    spy_momentum_regime = status.get("spy_momentum_regime", "NORMAL")
    momentum_multiplier = _MOMENTUM_REGIME_MULTIPLIERS.get(spy_momentum_regime, 1.0)

    # Only apply momentum dampening when VIX overlay is NOT already suppressing
    # signals (VIX >= 30 already applies 0.50x or 0.20x — avoid double-dampening)
    vix_already_active = current_vix is not None and current_vix >= 30
    if momentum_multiplier < 1.0 and not vix_already_active:
        log.warning(
            "MOMENTUM REGIME ALERT: spy_momentum_regime=%s "
            "-> applying %.2fx pre-VIX dampening (VIX=%.1f < 30). "
            "Hull (2015): regime shifts precede VIX threshold.",
            spy_momentum_regime,
            momentum_multiplier,
            current_vix if current_vix is not None else 0.0,
        )
    elif momentum_multiplier < 1.0 and vix_already_active:
        log.info(
            "Momentum regime %s detected but VIX overlay already active (VIX=%.1f). "
            "No additional dampening applied.",
            spy_momentum_regime,
            current_vix,
        )
        momentum_multiplier = 1.0  # reset — VIX overlay handles dampening

    # INTL scoring moved to StrategyEngine (run_pipeline_profile.py). This module
    # is US-only; the filter below is a safety guard against registry contamination.
    us_results = [r for r in results if r.get("market", "USA") == "USA"]

    # Schema gate: attach validation_metadata + circuit-breaker check.
    # Raises PipelineIntegrityError (before writing anything) if < 20% of the
    # universe has complete data.  Never removes rows — normalization needs the
    # full peer group.
    _schema_gate(us_results, universe_size=len(us_results))

    norm_factor_list = _cross_sectional_normalize(us_results)
    assert len(norm_factor_list) == len(us_results), (
        f"_cross_sectional_normalize returned {len(norm_factor_list)} rows for {len(us_results)} results"
    )

    # Redistribute weight from dead factors (all 0.0) to live ones
    eff_weights = _effective_weights(norm_factor_list, WEIGHTS)
    dead_factors = [f for f in WEIGHTS if f not in eff_weights]
    if dead_factors:
        log.warning(
            "Dead factor(s) detected — weight redistributed: %s",
            ", ".join(
                f"{f} {WEIGHTS[f]:.0%}→{eff_weights.get(f, 0):.0%}"
                if f not in dead_factors
                else f"{f} {WEIGHTS[f]:.0%}→dead"
                for f in WEIGHTS
            ),
        )

    entries = [
        _to_entry(row, nf, vix=current_vix, weights=eff_weights,
                  quiver_evidence=row.get("quiver_evidence"),
                  momentum_multiplier=momentum_multiplier)
        for row, nf in zip(us_results, norm_factor_list)
    ]
    # Propagate Stage 1 validation flag from raw rows into entries
    for entry, row in zip(entries, us_results):
        if row.get("_validation_failed"):
            entry["_validation_failed"] = True

    # Flat-signal detection: identical scores for all US tickers = silent fallback
    ns_scores = [
        float(r.get("news_sentiment_score", 0.0) or 0.0)
        for r in us_results
    ]
    if len(ns_scores) > 5 and len(set(round(s, 2) for s in ns_scores)) == 1:
        log.error(
            "FLAT SIGNAL DETECTED: news_sentiment identical (%.2f) for all %d US tickers. "
            "Check FMP news/stock endpoint and NLP scorer.",
            ns_scores[0], len(ns_scores),
        )

    _assign_cap_tiers(entries)

    # Exclude tickers that failed Stage 1 validation before slicing to top-N
    valid_entries = [e for e in entries if not e.get("_validation_failed")]

    # Stage 2: anomaly circuit breakers on scored universe.
    # Runs on US raw rows only — EU/Asia rows lack the fields (news_score,
    # volume_spike) that detect_anomalies expects and would cause false positives.
    try:
        detect_anomalies(us_results, run_id=run_id, log_dir=log_dir)
    except Exception as exc:
        log.warning("Stage 2 anomaly detection failed (non-fatal): %s", exc)

    # Shadow score: capture raw ranking BEFORE congress boost so callers can
    # compare which tickers the boost promoted.  Shallow-copy each entry dict
    # so the boost mutations below do not retroactively alter shadow scores.
    import copy as _copy
    shadow_top_buys = [
        _copy.copy(e)
        for e in sorted(valid_entries, key=lambda e: e["final_score"], reverse=True)[:5]
    ]

    # Congress boost: reads anomaly_report_latest.json (just written above) and
    # applies conviction multiplier to final_score in-place.  Must run AFTER
    # detect_anomalies so the report is fresh.  No-op when feed is dead.
    _apply_congress_boost(valid_entries, log_dir)

    # Portfolio construction: MVO weights for top-20 by composite_score.
    # Ties at position 20 broken alphabetically by ticker.
    _TOP_N_PORTFOLIO = 20
    _sorted_for_portfolio = sorted(
        valid_entries, key=lambda e: (-e["final_score"], e["ticker"])
    )
    _portfolio_candidates = _sorted_for_portfolio[:_TOP_N_PORTFOLIO]

    _prev_weights: dict = {}
    _prev_path = log_dir / "top_lists_us.json"
    if _prev_path.exists():
        try:
            _prev_data = json.loads(_prev_path.read_text(encoding="utf-8"))
            _LIST_KEYS = ("top_buys", "top_buys_usa", "top_buys_europe", "top_buys_asia", "mid_caps", "small_caps")
            _prev_weights = {
                e["ticker"]: e.get("portfolio_weight", 0.0)
                for key in _LIST_KEYS
                for e in _prev_data.get(key, [])
                if isinstance(e, dict) and e.get("portfolio_weight", 0.0) > 0
            }
        except Exception as _exc:
            log.debug("portfolio prev_weights load failed (non-fatal): %s", _exc)

    _portfolio_tickers, _portfolio_scores, _portfolio_sectors = zip(
        *((e["ticker"], e["final_score"], e.get("sector", "Unknown")) for e in _portfolio_candidates)
    ) if _portfolio_candidates else ([], [], [])
    _portfolio_tickers = list(_portfolio_tickers)
    _portfolio_scores = list(_portfolio_scores)
    _portfolio_sectors = list(_portfolio_sectors)
    _vix = current_vix if current_vix is not None else 20.0

    try:
        _opt_weights, _opt_method = run_optimizer(
            _portfolio_tickers, _portfolio_scores, _portfolio_sectors,
            vix=_vix, prev_weights=_prev_weights,
        )
    except Exception as _exc:
        log.warning("Portfolio optimizer raised: %s — using zeros", _exc)
        _opt_weights = {t: 0.0 for t in _portfolio_tickers}
        _opt_method = "failed"

    # Precompute sector → total weight once (O(n)) to avoid O(n²) per-entry inner sum.
    _sector_totals: dict = {}
    for _t, _s in zip(_portfolio_tickers, _portfolio_sectors):
        _sector_totals[_s] = _sector_totals.get(_s, 0.0) + _opt_weights.get(_t, 0.0)

    # Attach portfolio_weight to every entry (0.0 for non-top-20).
    _weight_set = set(_portfolio_tickers)
    for _entry in entries:
        _entry["portfolio_weight"] = round(_opt_weights.get(_entry["ticker"], 0.0), 6)
        _entry["portfolio_weight_method"] = _opt_method if _entry["ticker"] in _weight_set else "n/a"
        if _entry["ticker"] in _weight_set:
            _entry["sector_weight_contribution"] = round(
                _sector_totals.get(_entry.get("sector", "Unknown"), 0.0), 6
            )
        else:
            _entry["sector_weight_contribution"] = 0.0

    def score_desc(e): return e["final_score"]  # noqa: E731

    top_buys = sorted(valid_entries, key=score_desc, reverse=True)[:5]

    # Per-market top-5 lists — each market ranked independently so EU/Asia
    # tickers are never crowded out by the deeper US signal universe.
    top_buys_usa = sorted(
        [e for e in valid_entries if e.get("market", "USA") == "USA"],
        key=score_desc, reverse=True,
    )[:5]
    top_buys_europe = sorted(
        [e for e in valid_entries if e.get("market") == "EUROPE"],
        key=score_desc, reverse=True,
    )[:5]
    top_buys_asia = sorted(
        [e for e in valid_entries if e.get("market") == "ASIA"],
        key=score_desc, reverse=True,
    )[:5]

    mid_caps = sorted(
        [e for e in valid_entries if e["cap_tier"] == "mid"],
        key=score_desc, reverse=True,
    )[:5]
    small_caps = sorted(
        [e for e in valid_entries if e["cap_tier"] == "small"],
        key=score_desc, reverse=True,
    )[:5]

    kill_switch = current_vix is not None and current_vix >= 30
    if kill_switch:
        log.warning(
            "KILL SWITCH ACTIVE — VIX=%.1f >= 30.0. "
            "Score multiplier: %.2f. All final_scores dampened. "
            "BUY signals suppressed in Discord output. "
            "SELL signals remain active (asymmetric protection).",
            current_vix,
            _apply_vix_overlay(1.0, current_vix),
        )
    else:
        log.info(
            "Kill switch INACTIVE — VIX=%.1f < 30.0. Normal scoring active.",
            current_vix if current_vix is not None else 0.0,
        )

    _log_kill_switch_state(
        kill_switch=kill_switch,
        vix=current_vix if current_vix is not None else 0.0,
        log_dir=log_dir,
        run_id=run_id,
    )

    _log_promoted(top_buys, shadow_top_buys, log_dir, run_id)

    # Congress dead-streak tracking (FIX 6b)
    congress_dead_file = Path(".cache/congress_dead_since.txt")
    congress_scores = [
        float(r.get("congress_score") or 0.0)
        for r in us_results
    ]
    congress_is_dead = bool(congress_scores) and all(s == 0.0 for s in congress_scores)
    congress_dead_days = 0
    if congress_is_dead:
        if not congress_dead_file.exists():
            congress_dead_file.parent.mkdir(parents=True, exist_ok=True)
            congress_dead_file.write_text(datetime.now(timezone.utc).isoformat())
        try:
            dead_since = datetime.fromisoformat(congress_dead_file.read_text().strip())
            congress_dead_days = (datetime.now(timezone.utc) - dead_since).days
        except Exception:
            congress_dead_days = 0
    else:
        if congress_dead_file.exists():
            congress_dead_file.unlink()

    top_lists: Dict[str, Any] = {
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "source_run_id":   run_id,
        "ticker_count":    len(entries),
        # actual weights used (may differ from WEIGHTS if dead factors)
        "weights":         eff_weights,
        "weights_version": WEIGHTS_VERSION,
        "schema_version":           "9f-piog-eu",
        "piotroski_eu_gate_active": True,
        "vix":             current_vix,
        "vix_regime":      _vix_regime_label(current_vix),
        "kill_switch":     kill_switch,
        "vix_multiplier":  round(_apply_vix_overlay(1.0, current_vix), 2) if current_vix else 1.0,
        "spy_momentum_regime":   spy_momentum_regime,
        "momentum_multiplier":   round(momentum_multiplier, 4),
        "top_buys":        top_buys,
        "top_buys_usa":    top_buys_usa,
        "top_buys_europe": top_buys_europe,
        "top_buys_asia":   top_buys_asia,
        # Regional keys using get_region() codes (US/EU/ASIA) — audit + Discord
        "top_buys_us":     [e for e in top_buys if e.get("region") == "US"],
        "top_buys_eu":     [e for e in top_buys if e.get("region") == "EU"],
        "weights_global":  WEIGHTS_GLOBAL,
        "weights_us":      dict(WEIGHTS),
        "shadow_top_buys": shadow_top_buys,
        "mid_caps":        mid_caps,
        "small_caps":      small_caps,
        "sector_picks":    _sector_picks(entries),
        "dead_factors_detail": {
            "congress": {
                "dead":      congress_is_dead,
                "dead_days": congress_dead_days,
            }
        },
    }

    out_json = log_dir / "top_lists_us.json"
    save_json_atomic(out_json, top_lists)
    log.info(
        "Wrote %s — %d tickers, top buy: %s %.4f",
        out_json,
        len(entries),
        top_buys[0]["ticker"] if top_buys else "—",
        top_buys[0]["final_score"] if top_buys else 0.0,
    )

    out_csv = log_dir / "top5.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "rank", "ticker", "sector", "cap_tier", "market_cap",
            "final_score", "badge", "ceo_buy", "form4_count",
            "insider_conviction", "insider_breadth", "congress",
            "news_sentiment", "news_buzz", "momentum_long", "volume_attention",
            "analyst_consensus", "analyst_revision", "price_target_upside",
            "quality_piotroski", "transcript_tone",
        ])
        for rank, entry in enumerate(top_buys, 1):
            f = entry["factors"]
            writer.writerow([
                rank,
                entry["ticker"],
                entry["sector"],
                entry["cap_tier"],
                entry["market_cap"],
                entry["final_score"],
                entry["badge"],
                entry["ceo_buy"],
                entry["form4_count"],
                f.get("insider_conviction", 0.0),
                f.get("insider_breadth", 0.0),
                f.get("congress", 0.0),
                f.get("news_sentiment", 0.0),
                f.get("news_buzz", 0.0),
                f.get("momentum_long", 0.0),
                f.get("volume_attention", 0.0),
                f.get("analyst_consensus", 0.0),
                f.get("analyst_revision", 0.0),
                f.get("price_target_upside", 0.0),
                f.get("quality_piotroski", 0.0),
                f.get("transcript_tone", 0.0),
            ])
    log.info("Wrote %s", out_csv)

    return top_lists


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rank intel_source_status.json into top_lists_us.json"
    )
    parser.add_argument("--log-dir", type=Path, default=Path("logs"),
                        help="Directory containing intel_source_status.json (default: logs)")
    parser.add_argument("--run-id", type=str, default="local",
                        help="Identifier stamped into top_lists_us.json (e.g. $GITHUB_RUN_ID)")
    parser.add_argument("--bulk-cache", type=Path, default=None,
                        help="Bulk snapshot dir (informational; data already scored by run_pipeline)")
    parser.add_argument("--force", action="store_true",
                        help="Re-generate even if top_lists_us.json is less than 2 hours old")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    status_path = args.log_dir / "intel_source_status.json"
    if not status_path.exists():
        log.error(
            "intel_source_status.json not found at %s — run scripts/run_pipeline.py first",
            status_path,
        )
        return 1

    try:
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.error("Could not parse intel_source_status.json: %s", exc)
        return 1

    # Skip if fresh enough AND structurally valid.
    # A mid-write crash or empty payload must not block re-runs.
    out = args.log_dir / "top_lists_us.json"
    if out.exists() and not args.force:
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
            if "top_buys" not in existing or not existing.get("top_buys"):
                log.warning(
                    "top_lists_us.json exists but has no top_buys — treating as corrupted, regenerating"
                )
            else:
                ts = datetime.fromisoformat(
                    existing.get("generated_at", "").replace("Z", "+00:00")
                )
                age_h = (datetime.now(timezone.utc) -
                         ts).total_seconds() / 3600
                if age_h < 2.0:
                    log.info(
                        "top_lists_us.json is %.1fh old and valid — skipping (use --force to override)",
                        age_h,
                    )
                    return 0
        except Exception:
            log.warning(
                "top_lists_us.json unreadable or malformed — forcing regeneration")

    try:
        generate(status, args.run_id, args.log_dir)
        return 0
    except PipelineIntegrityError as exc:
        # Circuit breaker fired — loud, non-recoverable, must fail the CI step
        log.error("PIPELINE INTEGRITY VIOLATION: %s", exc)
        return 2   # distinct exit code so CI can distinguish from generic failure
    except Exception as exc:
        log.exception("Failed to generate top_lists: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
