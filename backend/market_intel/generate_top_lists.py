"""backend/market_intel/generate_top_lists.py
Five-factor weighted scoring → top_lists.json + top5.csv

Reads  logs/intel_source_status.json  (written by scripts/run_pipeline.py)
and applies Markowitz (1990 Nobel) portfolio ranking:

  final_score = 0.28×edgar + 0.23×insider + 0.22×congress + 0.15×news + 0.12×momentum

Badge thresholds (Sharpe-inspired):
  HIGH BUY     ≥ 0.80
  TACTICAL BUY ≥ 0.60
  WATCHLIST    < 0.60

Cap tiers (relative within universe, sorted by market cap):
  large : top 20 by market cap
  mid   : rank 21–35
  small : rank 36+

Each section in top_lists.json is ranked by final_score descending (top 5).

Output:
  logs/top_lists.json — consumed by scripts/send_toplists_discord.py
  logs/top5.csv       — flat reference file for downstream analysis

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

from regime_trader.scoring.normalize import normalize_score
from regime_trader.utils.io import save_json_atomic

log = logging.getLogger("generate_top_lists")


class PipelineIntegrityError(RuntimeError):
    """Raised when too few tickers pass the schema gate to produce a valid ranking.

    Prevents top_lists.json from being written with a near-empty or degenerate
    universe, which would cause the Discord bot to send misleading signals.
    """


WEIGHTS: Dict[str, float] = {
    "edgar":    0.28,
    "insider":  0.23,
    "congress": 0.22,
    "news":     0.15,
    "momentum": 0.12,
}

# Maps factor key → field name in intel_source_status.json results
FACTOR_FIELDS: Dict[str, str] = {
    "edgar":    "edgar_score",
    "insider":  "insider_score",
    "congress": "congress_score",
    "news":     "news_score",
    "momentum": "momentum_score",
}

# Schema gate: a ticker is "incomplete" when more than this many factors are
# zero (i.e. missing / dead API).  Incomplete tickers are scored but flagged;
# they are NOT excluded from ranking — exclusion would distort cross-sectional
# normalization.  Instead, validation_metadata carries the signal to consumers.
_SCHEMA_MISSING_THRESHOLD = 2   # >2 zero factors → is_complete = False

# Circuit breaker: if fewer than this fraction of the universe pass the schema
# gate, PipelineIntegrityError is raised and top_lists.json is NOT written.
_CIRCUIT_BREAKER_MIN_FRACTION = 0.20   # 20 % of universe minimum

_BADGES = [
    (0.80, "HIGH BUY"),
    (0.60, "TACTICAL BUY"),
    (0.00, "WATCHLIST"),
]

# Absolute market-cap thresholds (standard institutional definitions)
_LARGE_FLOOR = 10_000_000_000   # >= $10B  → large cap
_MID_FLOOR   =  2_000_000_000   # $2B–$10B → mid cap
#                                 < $2B    → small cap

_TARGET_SECTORS = [
    "Energy",
    "Materials",
    "Communication Services",
    "Healthcare",
    "Information Technology",
]


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

    For each ticker, counts how many of the five raw factor scores are exactly
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

        is_complete = len(missing) <= _SCHEMA_MISSING_THRESHOLD
        row["_validation"] = {
            "is_complete":    is_complete,
            "missing_sources": missing,
        }
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
            "Data feeds may be down. top_lists.json will NOT be written."
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
        raw = np.array([float(v) if v is not None else 0.0 for v in raw_values])

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

    Example: congress dead (all 0.0), weight=0.22 →
        edgar    0.28 → 0.28 + 0.22*(0.28/0.78) = 0.3590
        insider  0.23 → 0.23 + 0.22*(0.23/0.78) = 0.2948
        news     0.15 → 0.15 + 0.22*(0.15/0.78) = 0.1923
        momentum 0.12 → 0.12 + 0.22*(0.12/0.78) = 0.1538  (sum = 1.0)
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

    return live


def _apply_vix_overlay(score: float, vix: Optional[float]) -> float:
    """Multiplicative macro-regime penalty (absolute risk layer).

    Cross-sectional ranking always produces relative winners even during a crash.
    This overlay converts relative scores to absolute risk-adjusted scores by
    dampening all signals when the VIX regime is elevated.

      VIX ≥ 40 (Crash)  : ×0.20 — almost nothing should be HIGH BUY in a crash
      VIX ≥ 30 (Panic)  : ×0.50 — significant systemic risk, dampen all buys
      VIX ≥ 25 (Bear)   : ×0.80 — elevated risk, mild penalty
      VIX  < 25 (Normal) : ×1.00 — no adjustment
    """
    if vix is None:
        return score
    if vix >= 40:
        return score * 0.20
    if vix >= 30:
        return score * 0.50
    if vix >= 25:
        return score * 0.80
    return score


def _to_entry(
    row: Dict[str, Any],
    norm_factors: Dict[str, float],
    vix: Optional[float] = None,
    weights: Optional[Dict[str, float]] = None,
    quiver_evidence: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    w = weights if weights is not None else WEIGHTS
    raw_score = round(
        sum(w.get(f, 0.0) * norm_factors.get(f, 0.0) for f in WEIGHTS),
        4,
    )
    score = round(_apply_vix_overlay(raw_score, vix), 4)
    # Carry schema-gate result into the entry so it appears in top_lists.json
    validation = row.get("_validation", {"is_complete": True, "missing_sources": []})
    return {
        "ticker":          row.get("ticker", "?"),
        "sector":          row.get("sector", "Unknown"),
        "cap_tier":        row.get("cap_tier", "large"),
        "market_cap":      float(row.get("market_cap", 0.0)),
        "raw_score":       raw_score,
        "final_score":     score,
        "badge":           _badge(score),
        "ceo_buy":         bool(row.get("ceo_buy", False)),
        "form4_count":     int(row.get("form4_count", 0)),
        "factors":         norm_factors,
        "validation_metadata": validation,
        "quiver_evidence":         quiver_evidence or {},
        "news_source":             row.get("news_source", "none"),
        "insider_usd":             float(row.get("insider_usd", 0.0)),
        "momentum_spy_relative":   float(row.get("momentum_spy_relative", 0.0)),
        "volume_spike":            float(row.get("volume_spike", 1.0)),
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


def _sector_picks(entries: List[Dict[str, Any]], n: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    """Select top n tickers per target sector, ranked by final_score descending."""
    result: Dict[str, List[Dict[str, Any]]] = {}
    for sector in _TARGET_SECTORS:
        candidates = [e for e in entries if e.get("sector") == sector]
        result[sector] = sorted(candidates, key=lambda e: e["final_score"], reverse=True)[:n]
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
        ms = json.loads(market_state_path.resolve().read_text(encoding="utf-8"))
        vix = ms.get("macro_status", {}).get("vix_latest")
        if vix is not None:
            log.info("VIX overlay: %.1f (from market_state.json)", float(vix))
            return float(vix)
    except Exception:
        pass

    # Fallback: direct yfinance fetch (adds ~1 s to pipeline but always fresh)
    try:
        import yfinance as yf
        df = yf.download("^VIX", period="2d", interval="1d",
                         progress=False, auto_adjust=True)
        if not df.empty:
            vix = float(df["Close"].squeeze().dropna().iloc[-1])
            log.info("VIX overlay: %.1f (from yfinance fallback)", vix)
            return vix
    except Exception as exc:
        log.warning("VIX fetch failed — no macro overlay applied: %s", exc)

    return None


def generate(
    status: Dict[str, Any],
    run_id: str,
    log_dir: Path,
) -> Dict[str, Any]:
    """Score, rank, and tier the full ticker universe."""
    results = status.get("results", [])
    if not results:
        log.warning("No results found in intel_source_status.json — producing empty top_lists")

    # Fix #3: read current VIX once; passed into _to_entry for the macro overlay
    current_vix = _read_vix(log_dir)
    if current_vix is not None:
        log.info(
            "Macro overlay active -- VIX %.1f -> multiplier %.2f",
            current_vix,
            _apply_vix_overlay(1.0, current_vix),
        )

    # Schema gate: attach validation_metadata + circuit-breaker check.
    # Raises PipelineIntegrityError (before writing anything) if < 20% of the
    # universe has complete data.  Never removes rows — normalization needs the
    # full peer group.
    _schema_gate(results, universe_size=len(results))

    norm_factor_list = _cross_sectional_normalize(results)
    assert len(norm_factor_list) == len(results), (
        f"_cross_sectional_normalize returned {len(norm_factor_list)} rows for {len(results)} results"
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
                  quiver_evidence=row.get("quiver_evidence"))
        for row, nf in zip(results, norm_factor_list)
    ]
    _assign_cap_tiers(entries)

    score_desc = lambda e: e["final_score"]  # noqa: E731

    top_buys = sorted(entries, key=score_desc, reverse=True)[:5]
    mid_caps = sorted(
        [e for e in entries if e["cap_tier"] == "mid"],
        key=score_desc, reverse=True,
    )[:5]
    small_caps = sorted(
        [e for e in entries if e["cap_tier"] == "small"],
        key=score_desc, reverse=True,
    )[:5]

    # Exclude tickers that failed Stage 1 validation — prevents false-positive buy signals
    top_buys = [r for r in top_buys if not r.get("_validation_failed")]
    mid_caps = [r for r in mid_caps if not r.get("_validation_failed")]
    small_caps = [r for r in small_caps if not r.get("_validation_failed")]

    kill_switch = current_vix is not None and current_vix >= 30
    top_lists: Dict[str, Any] = {
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "source_run_id":   run_id,
        "ticker_count":    len(entries),
        "weights":         eff_weights,   # actual weights used (may differ from WEIGHTS if dead factors)
        "vix":             current_vix,
        "kill_switch":     kill_switch,
        "vix_multiplier":  round(_apply_vix_overlay(1.0, current_vix), 2) if current_vix else 1.0,
        "top_buys":        top_buys,
        "mid_caps":        mid_caps,
        "small_caps":      small_caps,
        "sector_picks":    _sector_picks(entries),
    }
    if kill_switch:
        log.warning(
            "⚠️  MACRO KILL-SWITCH ACTIVE — VIX %.1f — all scores dampened ×%.2f. "
            "Do NOT send HIGH BUY alerts to Discord.",
            current_vix,
            top_lists["vix_multiplier"],
        )

    out_json = log_dir / "top_lists.json"
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
            "edgar", "insider", "congress", "news", "momentum",
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
                f["edgar"], f["insider"], f["congress"], f["news"], f["momentum"],
            ])
    log.info("Wrote %s", out_csv)

    return top_lists


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rank intel_source_status.json into top_lists.json"
    )
    parser.add_argument("--log-dir", type=Path, default=Path("logs"),
                        help="Directory containing intel_source_status.json (default: logs)")
    parser.add_argument("--run-id", type=str, default="local",
                        help="Identifier stamped into top_lists.json (e.g. $GITHUB_RUN_ID)")
    parser.add_argument("--force", action="store_true",
                        help="Re-generate even if top_lists.json is less than 2 hours old")
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
    out = args.log_dir / "top_lists.json"
    if out.exists() and not args.force:
        try:
            existing = json.loads(out.read_text(encoding="utf-8"))
            if "top_buys" not in existing or not existing.get("top_buys"):
                log.warning(
                    "top_lists.json exists but has no top_buys — treating as corrupted, regenerating"
                )
            else:
                ts = datetime.fromisoformat(
                    existing.get("generated_at", "").replace("Z", "+00:00")
                )
                age_h = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
                if age_h < 2.0:
                    log.info(
                        "top_lists.json is %.1fh old and valid — skipping (use --force to override)",
                        age_h,
                    )
                    return 0
        except Exception:
            log.warning("top_lists.json unreadable or malformed — forcing regeneration")

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
