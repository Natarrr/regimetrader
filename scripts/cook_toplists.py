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

from regime_trader.risk.regime import (
    apply_capitulation_filter,
    get_regime,
    score_multiplier,
)

_BADGE_THRESHOLDS = [(0.80, "HIGH BUY"), (0.60, "TACTICAL BUY"), (0.00, "WATCHLIST")]


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
    composite_score = round(composite_score * score_multiplier(get_regime(vix)), 4)
    factor_snapshots = raw.get("factor_snapshots", {})
    # Strip any congress value from snapshots then pin to 0.0 (cannot be overridden)
    factors = {k: v for k, v in factor_snapshots.items() if k != "congress"}
    factors["congress"] = 0.0
    return {
        "ticker": ticker,
        "final_score": composite_score,
        "badge": _badge(composite_score),
        "market": market,
        "factors": factors,
        "pipeline": raw.get("pipeline", "INTL"),
    }


def cook(us_input: Path, intl_input: Path, registry: Path, output: Path) -> None:
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

    # ── Write combined output ─────────────────────────────────────────────────
    combined = {
        "top_buys_usa": top_buys_usa,
        "top_buys_europe": top_buys_europe,
        "top_buys_asia": top_buys_asia,
        "vix": vix,
        "vix_regime": vix_regime,
        "kill_switch": kill_switch,
        "ticker_count": us_ticker_count + len(top_buys_europe) + len(top_buys_asia),
        "generated_at": datetime.now(timezone.utc).isoformat(),
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
