"""backend/engine_worker.py
Autonomous market intelligence engine worker.

Fama (2013 Nobel) — efficient markets: a point-in-time state snapshot
captures all available information and enables post-mortem trade analysis.

Separation of Concerns
----------------------
This script is the *only* place where expensive API calls happen.
It writes a versioned JSON file that the Streamlit UI reads cheaply.

Schema produced — data/market_state.json:
  last_updated    : ISO-8601 UTC timestamp
  macro_status    : {regime, conviction, kill_switch_active, vix_latest}
  alpha_picks     : list of scored picks (ScanResult fields + risk_block flag)

Macro Kill Switch
-----------------
When VIX >= 30 (Panic) or >= 40 (Crash), kill_switch_active = True and
every pick in alpha_picks receives risk_block = True.

Scoring formula (already live in discovery_scanner.py):
  final_score = 0.45 × Insider + 0.35 × Institutional + 0.20 × Momentum

Usage:
    python -m backend.engine_worker
    python -m backend.engine_worker --limit 15
    python -m backend.engine_worker --force
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yfinance as yf

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from regime_trader.scanners.discovery_scanner import (
    get_top_alpha_picks_sync,
    force_refresh_sync,
)
from regime_trader.scanners.market_intel_macro import (
    COMMODITY_UNIVERSE,
    fetch_commodity_prices,
    calc_macro_conviction,
)
from regime_trader.utils.logging_cfg import configure_logging

configure_logging()
log = logging.getLogger(__name__)

_DATA_DIR  = _ROOT / "data"
_STATE_FILE = _DATA_DIR / "market_state.json"

_KILL_SWITCH_REGIMES = {"Crash", "Panic"}


# ── Macro regime detection ─────────────────────────────────────────────────────

def _vix_to_regime(vix: float) -> tuple[str, float]:
    """Map VIX level to regime label and base conviction.

    Shiller (2013 Nobel) — extreme valuations and volatility regimes predict
    long-run returns; VIX is the cleanest real-time regime proxy.

    Returns:
        (regime_label, base_conviction_float [0,1])
    """
    if vix >= 40:
        return "Crash",   0.05
    if vix >= 30:
        return "Panic",   0.15
    if vix >= 22:
        return "Bear",    0.35
    if vix >= 16:
        return "Neutral", 0.58
    return "Bull", 0.80


def _detect_macro_regime() -> Dict[str, Any]:
    """Detect current market regime from VIX + crude-oil macro conviction.

    Engle (2003 Nobel) — volatility clustering: VIX measures near-term
    fear; blending it with commodity conviction gives a multi-dimensional
    regime signal.

    Returns:
        macro_status dict for market_state.json
    """
    # ── VIX ──────────────────────────────────────────────────────────────────
    try:
        vix_df = yf.download("^VIX", period="5d", interval="1d",
                             progress=False, auto_adjust=True)
        if isinstance(vix_df.columns, pd.MultiIndex):
            vix_df.columns = vix_df.columns.get_level_values(0)
        vix_latest = float(vix_df["Close"].squeeze().dropna().iloc[-1])
    except Exception as exc:
        log.warning("[ENGINE] VIX fetch failed — defaulting to 20.0: %s", exc)
        vix_latest = 20.0

    regime, conviction = _vix_to_regime(vix_latest)
    log.info("[ENGINE] VIX=%.1f → regime=%s base_conviction=%.2f",
             vix_latest, regime, conviction)

    # ── Crude-oil macro conviction (optional blend) ───────────────────────
    try:
        crude = next(c for c in COMMODITY_UNIVERSE if c["name"] == "Crude Oil")
        crude_data = fetch_commodity_prices(crude)
        if crude_data:
            mac = calc_macro_conviction(crude_data, {})
            oil_conv = mac["composite"]
            # 60% VIX-derived, 40% commodity macro signal
            conviction = round(0.60 * conviction + 0.40 * oil_conv, 4)
            log.info("[ENGINE] Oil conviction=%.2f → blended=%.2f", oil_conv, conviction)
    except Exception as exc:
        log.debug("[ENGINE] Oil conviction blend skipped: %s", exc)

    kill_switch = regime in _KILL_SWITCH_REGIMES

    return {
        "regime":             regime,
        "conviction":         round(conviction, 4),
        "kill_switch_active": kill_switch,
        "vix_latest":         round(vix_latest, 2),
    }


# ── Alpha pick builder ─────────────────────────────────────────────────────────

def _build_alpha_picks(results: List[Dict], kill_switch: bool) -> List[Dict]:
    """Serialize ScanResults into the market_state schema.

    Adds risk_block flag — True when macro kill switch is active (Crash/Panic).
    Each alpha pick retains the full 5-field score breakdown for the UI.

    Tirole (2014 Nobel) — institutional accumulation signals must survive
    a macro risk filter before being actionable.
    """
    picks = []
    for r in results:
        picks.append({
            "symbol":                   r.get("symbol", ""),
            "smart_money_score":        round(float(r.get("smart_money_score", 0.0)), 4),
            "insider_score":            round(float(r.get("insider_score", 0.0)), 4),
            "institutional_score":      round(float(r.get("institutional_score", 0.0)), 4),
            "momentum_score":           round(float(r.get("momentum_score", 0.0)), 4),
            "insider_value_usd":        float(r.get("insider_value_usd", 0.0)),
            "insider_value_pct_mcap":   float(r.get("insider_value_pct_mcap", 0.0)),
            "key_insider_roles":        list(r.get("key_insider_roles", [])),
            "institutional_net_shares": float(r.get("institutional_net_shares", 0.0)),
            "institutional_pct_change": float(r.get("institutional_pct_change", 0.0)),
            "volume_spike":             float(r.get("volume_spike", 0.0)),
            "price_change_pct":         float(r.get("price_change_pct", 0.0)),
            "market_cap":               float(r.get("market_cap", 0.0)),
            "source_flags":             list(r.get("source_flags", [])),
            "risk_block":               kill_switch,
        })
    return picks


# ── Main engine entry ──────────────────────────────────────────────────────────

def run_engine(limit: int = 20, force: bool = False) -> Path:
    """Run a full market state computation and write data/market_state.json.

    Lucas (1995 Nobel) — rational expectations: agents act on all available
    information; this worker aggregates it into a single coherent state file.

    Args:
        limit: Maximum number of alpha picks to include.
        force: If True, bypass the discovery cache and run a fresh scan.

    Returns:
        Path to the written market_state.json file.
    """
    log.info("[ENGINE] ═══════════════════════════════════════════════")
    log.info("[ENGINE] Market intelligence engine starting (limit=%d force=%s)",
             limit, force)

    # ── 1. Discovery scan ────────────────────────────────────────────────────
    try:
        if force:
            log.info("[ENGINE] Force-refresh: bypassing discovery cache")
            payload = force_refresh_sync(limit=limit)
        else:
            payload = get_top_alpha_picks_sync(limit=limit)
        results = payload.get("results", [])
        log.info("[ENGINE] Scan complete — %d picks (cached=%s computed_at=%s)",
                 len(results), payload.get("cached"), payload.get("computed_at"))
    except Exception as exc:
        log.error("[ENGINE] Discovery scan failed: %s", exc)
        results = []

    # ── 2. Macro regime + kill switch ────────────────────────────────────────
    log.info("[ENGINE] Detecting macro regime…")
    macro_status = _detect_macro_regime()
    log.info("[ENGINE] Kill switch %s (regime=%s VIX=%.1f)",
             "ACTIVE ⛔" if macro_status["kill_switch_active"] else "clear ✅",
             macro_status["regime"], macro_status["vix_latest"])

    # ── 3. Build state ───────────────────────────────────────────────────────
    alpha_picks = _build_alpha_picks(results, macro_status["kill_switch_active"])

    state: Dict[str, Any] = {
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "macro_status": macro_status,
        "alpha_picks":  alpha_picks,
    }

    # ── 4. Atomic write ──────────────────────────────────────────────────────
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
    tmp.replace(_STATE_FILE)

    log.info("[ENGINE] Wrote %s (%d picks | %d bytes)",
             _STATE_FILE, len(alpha_picks), _STATE_FILE.stat().st_size)
    log.info("[ENGINE] ═══════════════════════════════════════════════")

    return _STATE_FILE


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regime Trader — autonomous market intelligence engine worker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--limit", type=int, default=20,
                        help="Number of alpha picks to produce")
    parser.add_argument("--force", action="store_true",
                        help="Bypass discovery cache and run a fresh scan")
    args = parser.parse_args()
    out = run_engine(limit=args.limit, force=args.force)
    print(f"State written → {out}")


if __name__ == "__main__":
    main()
