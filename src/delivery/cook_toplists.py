#!/usr/bin/env python3
# Path: scripts/cook_toplists.py
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
    score_multiplier,
)

try:
    from backend.market_intel.portfolio_optimizer import (
        run_optimizer as _run_optimizer,
        build_large_cap_anchors as _build_anchors,
    )
    from regime_trader.risk.exit_rules import enrich_with_exit_anchors as _enrich_exits
    _EXTENSIONS_AVAILABLE = True
except ImportError:
    _EXTENSIONS_AVAILABLE = False

_LARGE_CAP_THRESHOLD = 10_000_000_000
_MID_CAP_MIN         =  2_000_000_000
_MID_CAP_MAX         = 10_000_000_000
_SMALL_CAP_MIN       =    300_000_000
_SMALL_CAP_MAX       =  2_000_000_000

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
    """Return (large_cap >$10B, mid_cap $2B-$10B, small_cap $300M-$2B). Others excluded."""
    large, mid, small = [], [], []
    for entry in entries:
        cap = entry.get("market_cap", 0) or 0
        if cap > _LARGE_CAP_THRESHOLD:
            large.append(entry)
        elif _MID_CAP_MIN <= cap <= _MID_CAP_MAX:
            mid.append(entry)
        elif _SMALL_CAP_MIN <= cap < _MID_CAP_MIN:
            small.append(entry)
    return large, mid, small


def _badge(score: float) -> str:
    for threshold, label in _BADGE_THRESHOLDS:
        if score >= threshold:
            return label
    return "WATCHLIST"


def _build_registry_map(registry_path: Path) -> dict:
    """Return {ticker: "EUROPE"|"ASIA"} from ticker_registry.json."""
    if not registry_path.exists():
        return {}
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    mapping = {}
    for entry in registry.get("europe", []):
        mapping[entry["ticker"]] = "EUROPE"
    for entry in registry.get("asia", []):
        mapping[entry["ticker"]] = "ASIA"
    return mapping


def _normalize_intl_entry(raw: dict, ticker_market_map: dict, vix: float) -> dict:
    """Convert StrategyEngine entry to audit_payload-compatible format.

    Applies VIX dampening to final_score. congress is forced to 0.0 (audit check E).
    """
    ticker = raw.get("ticker", "")
    market = ticker_market_map.get(ticker, "EUROPE")
    composite_score = float(raw.get("composite_score", 0.0))
    regime = get_regime(vix)
    # CAPITULATION dampening is applied by apply_capitulation_filter() — skip here
    # to avoid double-dampening (BEAR and NORMAL still receive their multiplier)
    if regime != RiskRegime.CAPITULATION:
        composite_score = round(composite_score * score_multiplier(regime), 4)
    factor_snapshots = raw.get("factor_snapshots", {})
    # Strip any congress value from snapshots then pin to 0.0 (cannot be overridden)
    factors = {k: v for k, v in factor_snapshots.items() if k != "congress"}
    factors["congress"] = 0.0
    return {
        "ticker":          ticker,
        "final_score":     composite_score,
        "badge":           _badge(composite_score),
        "market":          market,
        "factors":         factors,
        "pipeline":        raw.get("pipeline", "INTL"),
        "weight_coverage": raw.get("weight_coverage", 0.0),
        # Forward raw prices for PT badge display in send_discord.py
        "target_price":    raw.get("target_price"),
        "current_price":   raw.get("current_price"),
        # Forward analyst meta for badge lines
        "analyst_consensus_source":    raw.get("analyst_consensus_source", "none"),
        "analyst_revision_score":      float(raw.get("analyst_revision_score") or 0.0),
        "analyst_revision_n_analysts": int(raw.get("analyst_revision_n_analysts") or 0),
        "price_target_upside_score":   float(raw.get("price_target_upside_score") or 0.0),
        "quality_piotroski_score":     float(raw.get("quality_piotroski_score") or 0.0),
        "earnings_surprise_pct":       raw.get("earnings_surprise_pct"),
        "earnings_surprise_days":      int(raw.get("earnings_surprise_days") or 0),
        "insider_usd":                 float(raw.get("insider_usd") or 0.0),
        "market_cap":                  float(raw.get("market_cap") or 0.0),
        "momentum_spy_relative":       float(raw.get("return_12_1m") or 0.0),
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
    for raw_entry in intl_raw:
        normalized = _normalize_intl_entry(raw_entry, ticker_market, vix)
        if normalized["market"] == "EUROPE":
            top_buys_europe.append(normalized)
        elif normalized["market"] == "ASIA":
            top_buys_asia.append(normalized)
        # Tickers absent from registry are silently dropped — registry is authoritative

    # ── Capitulation regime gate ──────────────────────────────────────────────
    regime = get_regime(vix)
    top_buys_usa    = apply_capitulation_filter(top_buys_usa,    vix)
    top_buys_europe = apply_capitulation_filter(top_buys_europe, vix)
    top_buys_asia   = apply_capitulation_filter(top_buys_asia,   vix)
    vix_regime      = regime.value

    # ── ATR / Batch Floor enrichment ──────────────────────────────────────────
    if _EXTENSIONS_AVAILABLE:
        for entry in top_buys_usa + top_buys_europe + top_buys_asia:
            _enrich_exits(entry, entry.get("atr_14"))

    # ── Sector count cap ─────────────────────────────────────────────────────
    top_buys_usa,    usa_overflow    = _apply_sector_count_cap(top_buys_usa,    sector_count_cap)
    top_buys_europe, eu_overflow     = _apply_sector_count_cap(top_buys_europe, sector_count_cap)
    top_buys_asia,   asia_overflow   = _apply_sector_count_cap(top_buys_asia,   sector_count_cap)

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
                        "price_target": e.get("price_target"),
                        "exit_anchors": e.get("exit_anchors", {}),
                    }
                    for e in mid_entries if mid_weights.get(e["ticker"], 0.0) > 0.001
                ],
            }

        if len(small_entries) >= 2:
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
                        "price_target": e.get("price_target"),
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
        "usa_overflow":    usa_overflow,
        "eu_overflow":     eu_overflow,
        "asia_overflow":   asia_overflow,
        "mvo_pools":       mvo_pools,
        "vix":             vix,
        "vix_regime":      vix_regime,
        "kill_switch":     kill_switch,
        "ticker_count":    us_ticker_count + len(top_buys_europe) + len(top_buys_asia),
        "generated_at":    datetime.now(timezone.utc).isoformat(),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(
        f"[COOK] Combined payload -> {output} "
        f"({len(top_buys_usa)} US + {len(top_buys_europe)} EU "
        f"+ {len(top_buys_asia)} Asia)"
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
