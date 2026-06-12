#!/usr/bin/env python3
# Path: src/delivery/cook_toplists.py
"""Merge US top_lists.json + INTL top_lists_intl.json into a unified combined payload.

Schema transformation for INTL StrategyEngine output:
  composite_score  -> final_score
  region_applied   -> market ("EUROPE" or "ASIA" via ticker_registry.json)
  factor_snapshots -> factors (congress: 0.0 injected -- audit check E)
  badge            -> computed from final_score thresholds
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.risk.regime import (
    RiskRegime,
    apply_capitulation_filter,
    get_regime,
    vix_multiplier,
)

try:
    from backend.market_intel.portfolio_optimizer import (
        run_optimizer as _run_optimizer,
        build_large_cap_anchors as _build_anchors,
    )
    from src.risk.exit_rules import enrich_with_exit_anchors as _enrich_exits
    _EXTENSIONS_AVAILABLE = True
except ImportError:
    _EXTENSIONS_AVAILABLE = False

_LARGE_CAP_THRESHOLD = 10_000_000_000
_MID_CAP_MIN         =  2_000_000_000
_MID_CAP_MAX         = 10_000_000_000
_SMALL_CAP_MIN       =    300_000_000
_SMALL_CAP_MAX       =  2_000_000_000

# ── SMID leverage sleeve (US-only) ────────────────────────────────────────────
# Composite re-rank of the already-vetted US buy list toward the small/mid-cap
# leverage profile: final_score carries the full 9-factor alpha (VIX overlay,
# Piotroski gate and momentum-regime dampening already applied upstream);
# momentum_long is re-emphasized per [Jegadeesh & Titman, 1993] (12-1m momentum
# persists strongest outside mega-caps; size premium [Banz, 1981]) and
# quality_piotroski per [Piotroski, 2000] (F-score alpha concentrates in
# small/neglected names).
_SMID_LEVERAGE_WEIGHTS = {
    "final_score":       0.50,
    "momentum_long":     0.30,
    "quality_piotroski": 0.20,
}
assert abs(sum(_SMID_LEVERAGE_WEIGHTS.values()) - 1.0) < 1e-6

_SMID_CAP_MIN = _SMALL_CAP_MIN         # $300M inclusive
_SMID_CAP_MAX = _LARGE_CAP_THRESHOLD   # $10B inclusive (large bracket starts > $10B)
_SMID_TOP_N   = 3
# Post-earnings-announcement drift: positive surprises drift upward for roughly
# 60 days post-announcement [Bernard & Thomas, 1989]. Boost is applied ONLY on
# affirmative evidence — absent earnings meta must never read as bearish.
# Mirrored in send_discord._SMID_PEAD_WINDOW_DAYS (flag must agree with boost).
_SMID_PEAD_WINDOW_DAYS = 60
_SMID_PEAD_BOOST       = 1.10

_BADGE_THRESHOLDS = [(0.80, "HIGH BUY"), (0.60, "TACTICAL BUY"), (0.00, "WATCHLIST")]


def _apply_sector_count_cap(
    entries: list, max_per_sector: int = 2
) -> tuple[list, list]:
    """Return (primary, overflow) — primary keeps ≤ max_per_sector per sector, sorted by score."""
    sorted_entries = sorted(entries, key=lambda e: e.get("final_score", 0), reverse=True)
    sector_counts: dict[str, int] = {}
    primary, overflow = [], []
    for entry in sorted_entries:
        sector = entry.get("factors", {}).get("sector", "Unknown")
        if sector_counts.get(sector, 0) < max_per_sector:
            primary.append(entry)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        else:
            overflow.append(entry)
    return primary, overflow


def _segment_by_market_cap(entries: list) -> tuple[list, list, list]:
    """Return (large_cap >$10B, mid_cap $2B-$10B, small_cap $300M-$2B). Others excluded.

    INTL entries are excluded from all brackets: their market_cap arrives in
    listing currency (FMP quote.marketCap has no FX normalization — JPY for .T,
    GBp for .L, …), so treating it as USD would misclassify brackets and skew
    MVO allocations. Lift this guard only once FX normalization exists upstream.
    """
    large, mid, small = [], [], []
    for entry in entries:
        if entry.get("pipeline") == "INTL":
            continue
        cap = entry.get("market_cap", 0) or 0
        if cap > _LARGE_CAP_THRESHOLD:
            large.append(entry)
        elif _MID_CAP_MIN <= cap <= _MID_CAP_MAX:
            mid.append(entry)
        elif _SMALL_CAP_MIN <= cap < _MID_CAP_MIN:
            small.append(entry)
    return large, mid, small


def _build_smid_leverage_pool(entries: list, top_n: int = _SMID_TOP_N) -> list:
    """Top-N US small/mid-cap entries re-ranked by the leverage composite.

    leverage_score = (0.50*final_score + 0.30*momentum_long
                      + 0.20*quality_piotroski) * pead_boost

    pead_boost = 1.10 iff earnings_surprise_pct > 0 and
    0 < earnings_surprise_days <= 60 [Bernard & Thomas, 1989]; 1.00 otherwise
    (data absence is NEVER bearish). leverage_score is a RANKING key, not a
    calibrated probability — it may exceed 1.0 (max 1.10) and is deliberately
    not gated by audit check A, which validates final_score only.

    INTL entries are excluded (same FX-normalization guard as
    _segment_by_market_cap). Returns COPIES ({**e, "leverage_score": ...}) —
    candidates are shared by reference with top_buys_usa/usa_overflow and
    must not be mutated.
    """
    w = _SMID_LEVERAGE_WEIGHTS
    scored: list = []
    for entry in entries:
        if entry.get("pipeline") == "INTL":
            continue
        cap = entry.get("market_cap", 0) or 0
        if not (_SMID_CAP_MIN <= cap <= _SMID_CAP_MAX):
            continue
        factors = entry.get("factors") or {}
        base = (
            w["final_score"]         * float(entry.get("final_score") or 0.0)
            + w["momentum_long"]     * float(factors.get("momentum_long") or 0.0)
            + w["quality_piotroski"] * float(factors.get("quality_piotroski") or 0.0)
        )
        boost = 1.0
        pct = entry.get("earnings_surprise_pct")
        days = int(entry.get("earnings_surprise_days") or 0)
        if pct is not None and float(pct) > 0 and 0 < days <= _SMID_PEAD_WINDOW_DAYS:
            boost = _SMID_PEAD_BOOST
        scored.append({**entry, "leverage_score": round(base * boost, 4)})
    scored.sort(key=lambda e: (-e["leverage_score"],
                               -float(e.get("final_score") or 0.0),
                               e.get("ticker", "")))
    return scored[:top_n]


def _badge(score: float) -> str:
    for threshold, label in _BADGE_THRESHOLDS:
        if score >= threshold:
            return label
    return "WATCHLIST"


def _build_registry_map(registry_path: Path) -> dict:
    """Return {ticker: {"market", "cap_tier", "sector"}} from ticker_registry.json."""
    if not registry_path.exists():
        return {}
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    mapping: dict = {}
    for entry in registry.get("europe", []) + registry.get("europe_mid", []):
        mapping[entry["ticker"]] = {
            "market":   "EUROPE",
            "cap_tier": entry.get("cap_tier", "large"),
            "sector":   entry.get("sector", ""),
        }
    for entry in registry.get("asia", []) + registry.get("asia_mid", []):
        mapping[entry["ticker"]] = {
            "market":   "ASIA",
            "cap_tier": entry.get("cap_tier", "large"),
            "sector":   entry.get("sector", ""),
        }
    return mapping


def _normalize_intl_entry(raw: dict, ticker_market_map: dict, vix: float) -> dict:
    """Convert StrategyEngine entry to audit_payload-compatible format.

    Applies VIX dampening to final_score. congress is forced to 0.0 (audit check E).
    """
    ticker = raw.get("ticker", "")
    # No fallback: the caller (cook) drops unregistered tickers before this
    # point — a silent EUROPE default mislabeled unknown tickers and could
    # block the whole send via audit check D (GeographicLeakageError).
    _registry_meta = ticker_market_map[ticker]
    market = _registry_meta["market"]
    cap_tier = _registry_meta.get("cap_tier", "large")
    composite_score = float(raw.get("composite_score", 0.0))
    # Single dampening point for INTL — vix_multiplier carries the Crash tier
    # (×0.20 at VIX ≥ 40) so US and INTL stay symmetric in every regime.
    # apply_capitulation_filter() no longer multiplies (filter + badge only).
    composite_score = round(composite_score * vix_multiplier(vix), 4)
    factor_snapshots = raw.get("factor_snapshots", {})
    # Strip any congress value from snapshots then pin to 0.0 (cannot be overridden)
    factors = {k: v for k, v in factor_snapshots.items() if k != "congress"}
    factors["congress"] = 0.0
    return {
        "ticker":          ticker,
        "sector":          (raw.get("sector") or _registry_meta.get("sector") or "").strip(),
        "final_score":     composite_score,
        "badge":           _badge(composite_score),
        "market":          market,
        "cap_tier":        cap_tier,
        "factors":         factors,
        "pipeline":        raw.get("pipeline", "INTL"),
        "weight_coverage": raw.get("weight_coverage", 0.0),
        # Forward raw prices for exit-anchor enrichment (src/risk/exit_rules)
        "target_price":    raw.get("target_price"),
        "current_price":   raw.get("current_price"),
        # Forward analyst meta for badge lines
        "analyst_consensus_source":    raw.get("analyst_consensus_source", "none"),
        "analyst_revision_score":      float(raw.get("analyst_revision_score") or 0.0),
        "analyst_revision_n_analysts": int(raw.get("analyst_revision_n_analysts") or 0),
        "price_target_upside_score":   raw.get("price_target_upside_score"),  # None = no analyst target
        "quality_piotroski_score":     float(raw.get("quality_piotroski_score") or 0.0),
        "earnings_surprise_pct":       raw.get("earnings_surprise_pct"),
        "earnings_surprise_days":      int(raw.get("earnings_surprise_days") or 0),
        "insider_usd":                 float(raw.get("insider_usd") or 0.0),
        "market_cap":                  float(raw.get("market_cap") or 0.0),
        # Absolute 12-1m return in the listing market — INTL has no SPY-relative
        # momentum, so momentum_spy_relative is intentionally NOT set (the old
        # mapping rendered absolute returns as "vs SPY 12m" in catalysts).
        "return_12_1m":                float(raw.get("return_12_1m") or 0.0),
        "analyst_consensus_score":     factor_snapshots.get("analyst_consensus"),
    }


def cook(
    us_input: Path,
    intl_input: Path,
    registry: Path,
    output: Path,
    mvo_enabled: bool = True,
    sector_count_cap: int = 2,
) -> None:
    # ── US payload ────────────────────────────────────────────────────────────
    us_data = json.loads(us_input.read_text(encoding="utf-8"))
    # Support both top_buys_usa (new) and top_buys (legacy) key names
    top_buys_usa = us_data.get("top_buys_usa") or us_data.get("top_buys", [])
    vix = us_data.get("vix")
    vix_regime = us_data.get("vix_regime", "Unknown")
    kill_switch = us_data.get("kill_switch", False)
    us_ticker_count = us_data.get("ticker_count", len(top_buys_usa))

    if vix is None:
        sys.exit(
            "[COOK] ERROR: US payload missing 'vix' — cannot apply macro overlay. Aborting."
        )

    # ── INTL payload ──────────────────────────────────────────────────────────
    intl_raw: list = json.loads(intl_input.read_text(encoding="utf-8"))
    ticker_market = _build_registry_map(registry)

    top_buys_europe: list = []
    top_buys_asia: list = []
    eu_mid_small: list = []
    asia_mid_small: list = []
    for raw_entry in intl_raw:
        # Registry is authoritative: unregistered tickers are dropped LOUDLY.
        # (The previous EUROPE default mislabeled them and could fail the
        # combined audit's geographic-leakage check, blocking the send.)
        if raw_entry.get("ticker") not in ticker_market:
            print(f"[COOK] WARN: {raw_entry.get('ticker')!r} not in "
                  f"ticker_registry.json — entry dropped")
            continue
        normalized = _normalize_intl_entry(raw_entry, ticker_market, vix)
        cap = normalized["cap_tier"]
        if normalized["market"] == "EUROPE":
            if cap in ("mid", "small"):
                eu_mid_small.append(normalized)
            else:
                top_buys_europe.append(normalized)
        elif normalized["market"] == "ASIA":
            if cap in ("mid", "small"):
                asia_mid_small.append(normalized)
            else:
                top_buys_asia.append(normalized)

    # Snapshot the scored intl breadth BEFORE the capitulation move and sector
    # cap mutate the lists — ticker_count reports scored coverage, and under
    # CAPITULATION the regional lists are emptied into watchlist.
    intl_scored_count = (len(top_buys_europe) + len(top_buys_asia)
                         + len(eu_mid_small) + len(asia_mid_small))

    # ── Capitulation regime gate ──────────────────────────────────────────────
    regime = get_regime(vix)
    top_buys_usa    = apply_capitulation_filter(top_buys_usa,    vix)
    top_buys_europe = apply_capitulation_filter(top_buys_europe, vix)
    top_buys_asia   = apply_capitulation_filter(top_buys_asia,   vix)
    eu_mid_small    = apply_capitulation_filter(eu_mid_small,    vix)
    asia_mid_small  = apply_capitulation_filter(asia_mid_small,  vix)
    eu_mid_small    = sorted(eu_mid_small,   key=lambda x: -x["final_score"])[:1]
    asia_mid_small  = sorted(asia_mid_small, key=lambda x: -x["final_score"])[:1]
    vix_regime      = regime.value

    # Under CAPITULATION, survivors are badged WATCHLIST by apply_capitulation_filter.
    # Move them OUT of top_buys_* (buy-signal lists) into a dedicated watchlist key
    # so the Discord embed generator does not render them as "Active Buy Signals."
    watchlist: list = []
    if regime == RiskRegime.CAPITULATION:
        for _lst in (top_buys_usa, top_buys_europe, top_buys_asia):
            watchlist.extend(_lst)
        top_buys_usa    = []
        top_buys_europe = []
        top_buys_asia   = []
        eu_mid_small    = []
        asia_mid_small  = []
        print(
            f"[COOK] CAPITULATION REGIME (VIX={vix:.1f}) — "
            f"top_buys_* emptied; {len(watchlist)} structural anchor(s) moved to watchlist."
        )

    # ── ATR / Batch Floor enrichment ──────────────────────────────────────────
    if _EXTENSIONS_AVAILABLE:
        for entry in top_buys_usa + top_buys_europe + top_buys_asia:
            _enrich_exits(entry, entry.get("atr_14"))

    # ── Sector count cap ─────────────────────────────────────────────────────
    top_buys_usa,    usa_overflow    = _apply_sector_count_cap(top_buys_usa,    sector_count_cap)
    top_buys_europe, eu_overflow     = _apply_sector_count_cap(top_buys_europe, sector_count_cap)
    top_buys_asia,   asia_overflow   = _apply_sector_count_cap(top_buys_asia,   sector_count_cap)

    # ── SMID leverage sleeve — candidates AFTER capitulation gate and sector
    # cap; overflow is included so the sleeve is NOT sector-constrained.
    # Under CAPITULATION both inputs are empty ⇒ pool is [] by construction.
    top_buys_smid = _build_smid_leverage_pool(top_buys_usa + usa_overflow)

    # ── 3-tier capital allocation ─────────────────────────────────────────────
    mvo_pools: dict = {}
    if _EXTENSIONS_AVAILABLE and mvo_enabled:
        all_candidates = top_buys_usa + top_buys_europe + top_buys_asia
        large_entries, mid_entries, small_entries = _segment_by_market_cap(all_candidates)

        if large_entries:
            mvo_pools["large_cap_anchors"] = {
                "bracket": "LARGE_CAP_ANCHOR",
                "cap_range": ">$10B",
                "positions": _build_anchors(large_entries),
            }

        if len(mid_entries) >= 2:
            mid_weights, mid_method = _run_optimizer(
                [e["ticker"] for e in mid_entries],
                [e["final_score"] for e in mid_entries],
                [e.get("factors", {}).get("sector", "Unknown") for e in mid_entries],
                vix=vix, mode="sharpe", position_ceiling=0.35,
            )
            mvo_pools["mid_cap"] = {
                "method": mid_method, "bracket": "MID_CAP", "cap_range": "$2B-$10B",
                "positions": [
                    {
                        "ticker":       e["ticker"],
                        "allocation":   round(mid_weights.get(e["ticker"], 0.0), 4),
                        "final_score":  e["final_score"],
                        "price_target": e.get("target_price") or e.get("price_target"),
                        "exit_anchors": e.get("exit_anchors", {}),
                    }
                    for e in mid_entries if mid_weights.get(e["ticker"], 0.0) > 0.001
                ],
            }

        if len(small_entries) >= 2:
            # ADV liquidity gate is currently INERT: no upstream producer
            # emits adv_20d_usd yet, and run_optimizer additionally requires
            # portfolio_value_usd (not passed here) before it caps anything.
            # Forward-compat plumbing only — do not read this as a live control.
            adv_map = {
                e["ticker"]: e.get("adv_20d_usd")
                for e in small_entries if e.get("adv_20d_usd")
            }
            small_weights, small_method = _run_optimizer(
                [e["ticker"] for e in small_entries],
                [e["final_score"] for e in small_entries],
                [e.get("factors", {}).get("sector", "Unknown") for e in small_entries],
                vix=vix, mode="min_variance",
                position_floor=0.10, position_ceiling=0.25,
                adv_20d_map=adv_map or None,
            )
            mvo_pools["small_cap"] = {
                "method": small_method, "bracket": "SMALL_CAP", "cap_range": "$300M-$2B",
                "positions": [
                    {
                        "ticker":       e["ticker"],
                        "allocation":   round(small_weights.get(e["ticker"], 0.0), 4),
                        "final_score":  e["final_score"],
                        "price_target": e.get("target_price") or e.get("price_target"),
                        "exit_anchors": e.get("exit_anchors", {}),
                    }
                    for e in small_entries if small_weights.get(e["ticker"], 0.0) > 0.001
                ],
            }

    # ── Write combined output ─────────────────────────────────────────────────
    combined = {
        "top_buys_usa":    top_buys_usa,
        "top_buys_europe": top_buys_europe,
        "top_buys_asia":   top_buys_asia,
        # US SMID leverage sleeve, ranked by leverage_score (always present —
        # [] under CAPITULATION or when no US name falls in the $300M–$10B band)
        "top_buys_smid":   top_buys_smid,
        "watchlist":       watchlist,          # populated under CAPITULATION regime
        "usa_overflow":    usa_overflow,
        "eu_overflow":     eu_overflow,
        "asia_overflow":   asia_overflow,
        "mvo_pools":       mvo_pools,
        "eu_mid_small":    eu_mid_small,
        "asia_mid_small":  asia_mid_small,
        "vix":             vix,
        "vix_regime":      vix_regime,
        "kill_switch":     kill_switch,
        # Scored coverage (universe semantics): US universe count + every
        # registered intl ticker that was scored — independent of the sector
        # cap and the CAPITULATION watchlist move.
        "ticker_count":    us_ticker_count + intl_scored_count,
        "generated_at":    datetime.now(timezone.utc).isoformat(),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    watchlist_note = f" | {len(watchlist)} watchlist" if watchlist else ""
    print(
        f"[COOK] Combined payload -> {output} "
        f"({len(top_buys_usa)} US + {len(top_buys_europe)} EU "
        f"+ {len(top_buys_asia)} Asia + {len(top_buys_smid)} smid{watchlist_note})"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge US + INTL top_lists artifacts into a combined payload"
    )
    parser.add_argument(
        "--us-input",
        default="logs/top_lists_us.json",
        help="Path to US top_lists.json (9-factor, from run_pipeline.py)",
    )
    parser.add_argument(
        "--intl-input",
        default="logs/top_lists_intl.json",
        help="Path to INTL top_lists_intl.json (6-factor StrategyEngine output)",
    )
    parser.add_argument(
        "--registry",
        default="config/ticker_registry.json",
        help="Path to ticker_registry.json for EU/Asia region classification",
    )
    parser.add_argument(
        "--output",
        default="logs/top_lists.json",
        help="Destination for combined payload (overwrites US input by default)",
    )
    args = parser.parse_args()

    us_input = Path(args.us_input)
    intl_input = Path(args.intl_input)
    registry = Path(args.registry)
    output = Path(args.output)

    if not us_input.exists():
        print(f"[COOK] ERROR: US input not found: {us_input}", file=sys.stderr)
        return 1
    if not intl_input.exists():
        print(f"[COOK] ERROR: INTL input not found: {intl_input}", file=sys.stderr)
        return 1

    cook(us_input, intl_input, registry, output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
