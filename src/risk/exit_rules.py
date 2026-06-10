# Path: src/risk/exit_rules.py
"""Batch Rebalancing Stop Floors, PT profit-taking, and breakout extension logic.

OPERATIONAL CONSTRAINT:
These are BATCH REBALANCING STOP FLOORS, not real-time intraday stops.
The pipeline runs at discrete cron intervals (3× daily via GitHub Actions).
Gap-down events between runs cannot be caught programmatically. The computed
floor prices must be placed as GTC (Good 'Til Cancelled) stop-market or
stop-limit orders directly on the brokerage desk between execution runs.

ATR source: FMP /stable/technical-indicator/daily/{symbol}?type=atr&period=14
RSI source: FMP /stable/technical-indicator/daily/{symbol}?type=rsi&period=14
VWAP source: FMP /stable/technical-indicator/daily/{symbol}?type=vwap
"""
from __future__ import annotations
from typing import Optional

_ATR_MULTIPLIER:         float = 2.5
_PT_PROFIT_THRESHOLD:    float = 0.05    # 5% within PT → partial exit signal
_BREAKOUT_ATR_SHIFT:     float = 1.5     # breakout extension: PT + 1.5 × ATR
_ACCUMULATION_RSI_FLOOR: float = 50.0   # RSI > 50 = institutional accumulation signal
_ACCUMULATION_VWAP_FLOOR: float = 1.0   # price/VWAP > 1.0 = trading above VWAP


def compute_batch_floor(
    current_price: float,
    atr_14: float,
    multiplier: float = _ATR_MULTIPLIER,
) -> float:
    """Return 2.5×ATR Batch Rebalancing Stop Floor price."""
    return round(current_price - multiplier * atr_14, 4)


def compute_pt_signal(
    current_price: float,
    price_target: Optional[float],
    threshold_pct: float = _PT_PROFIT_THRESHOLD,
) -> dict:
    if price_target is None or price_target <= 0 or current_price <= 0:
        return {"upside_pct": None, "take_profit_alert": False}
    upside = (price_target - current_price) / current_price
    return {
        "upside_pct":        round(upside * 100, 2),
        "take_profit_alert": bool(0.0 <= upside <= threshold_pct),
    }


def compute_breakout_extension(
    current_price: float,
    price_target: Optional[float],
    atr_14: float,
    rsi_14: Optional[float],
    vwap_ratio: Optional[float],   # current_price / VWAP
) -> dict:
    """Detect PT breakout with institutional accumulation; shift target by 1.5×ATR.

    Conditions for BREAKOUT EXTENSION flag:
      1. current_price > price_target   (broken above consensus PT)
      2. rsi_14 > 50                    (momentum confirmation)
      3. vwap_ratio > 1.0               (trading above VWAP — institutional flow)

    Extended target = price_target + 1.5 × ATR_14
    """
    if not price_target or current_price <= price_target:
        return {"breakout_extension": False, "extended_target": None}

    accumulation = (
        (rsi_14 is not None and rsi_14 > _ACCUMULATION_RSI_FLOOR) and
        (vwap_ratio is not None and vwap_ratio > _ACCUMULATION_VWAP_FLOOR)
    )
    if not accumulation:
        return {"breakout_extension": False, "extended_target": None}

    extended = round(float(price_target) + _BREAKOUT_ATR_SHIFT * atr_14, 4)
    return {"breakout_extension": True, "extended_target": extended}


def enrich_with_exit_anchors(entry: dict, atr_14: Optional[float]) -> dict:
    """Attach exit_anchors to entry. Reads current_price, price_target, rsi_14, vwap_ratio."""
    price  = float(entry.get("current_price") or 0)
    pt_raw = entry.get("price_target") or entry.get("factors", {}).get("price_target_consensus")
    pt     = float(pt_raw) if pt_raw else None
    rsi    = entry.get("rsi_14") or entry.get("factors", {}).get("rsi_14")
    vwap_r = entry.get("vwap_ratio") or entry.get("factors", {}).get("vwap_ratio")

    batch_floor = compute_batch_floor(price, atr_14) if price and atr_14 else None
    pt_signal   = compute_pt_signal(price, pt)
    breakout    = compute_breakout_extension(price, pt, atr_14 or 0.0, rsi, vwap_r)

    entry["exit_anchors"] = {
        "atr_14":             round(float(atr_14), 4) if atr_14 is not None else None,
        "batch_floor":        batch_floor,
        "upside_pct":         pt_signal["upside_pct"],
        "take_profit_alert":  pt_signal["take_profit_alert"],
        "breakout_extension": breakout["breakout_extension"],
        "extended_target":    breakout.get("extended_target"),
    }
    return entry


def format_card_line(entry: dict) -> str:
    """Format one ticker card for Discord institutional block.

    Format: [TICKER] | Spot: XX.XX | Target: XX.XX (+X.X% to PT) | Batch Floor: XX.XX (2.5 ATR)
    Appends [BREAKOUT EXTENSION → $XX.XX] when triggered.
    """
    ticker  = entry.get("ticker", "???")
    price   = entry.get("current_price")
    pt      = entry.get("price_target")
    anchors = entry.get("exit_anchors", {})
    upside  = anchors.get("upside_pct")
    floor   = anchors.get("batch_floor")
    breakout    = anchors.get("breakout_extension", False)
    ext_tgt = anchors.get("extended_target")

    spot_str   = f"{float(price):.2f}" if price else "N/A"
    pt_str     = f"{float(pt):.2f}" if pt else "N/A"
    if upside is not None:
        upside_str = f"+{upside:.1f}% to PT" if upside >= 0 else f"{upside:.1f}% to PT"
    else:
        upside_str = "N/A"
    floor_str = f"{float(floor):.2f} (2.5 ATR)" if floor is not None else "N/A"

    line = f"[{ticker}] | Spot: {spot_str} | Target: {pt_str} ({upside_str}) | Batch Floor: {floor_str}"
    if breakout and ext_tgt:
        line += f" [BREAKOUT EXTENSION → ${ext_tgt:.2f}]"
    return line
