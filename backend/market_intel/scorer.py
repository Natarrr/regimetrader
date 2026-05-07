"""backend/market_intel/scorer.py — insider event aggregation → 0–1 score.

Spence (2001 Nobel) — Costly Signaling: open-market purchases (Code P) are
the cleanest insider signal because the insider commits personal capital.
Sales (Code S) are noisier (diversification, taxes, planned 10b5-1 plans).
Awards (A), gifts (G), exercises (M) are excluded from the directional score.

Akerlof (2001 Nobel) — Asymmetric Information: weighting role × value lets
the score capture *who* holds the private information, not just notional.

Score formula:
    net = sum_buy(role_weight × value)  −  sum_sell(role_weight × value)
    score = clip(0.50 + tanh(net / SCALE) × 0.50,  0, 1)

A pure neutral filing yields 0.50; aggressive CEO buying near-1; CFO selling
near-0. Robust to a single outsized transaction via tanh saturation.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from . import config


# ── Role weights ──────────────────────────────────────────────────────────────

_ROLE_WEIGHTS: Dict[str, float] = {
    "CEO":              2.00,
    "CFO":              1.50,
    "Officer-Senior":   1.20,   # President, COO, Chairman
    "Officer":          1.00,
    "Director":         1.00,
    "10%Owner":         1.20,
    "Unknown":          0.50,
}

# tanh saturation scale — net values above this approach ±1.
# $50M net buying / selling is "very strong cluster" for a single ticker.
_SATURATION_USD: float = 50_000_000.0

_NEUTRAL: float = 0.50


def _role_weight(role: Optional[str]) -> float:
    return _ROLE_WEIGHTS.get(str(role or "Unknown"), 0.5)


def _is_buy(ev: Dict[str, Any]) -> bool:
    """Open-market purchase: code P AND acquired."""
    code = str(ev.get("transaction_code") or "").upper()
    ad = str(ev.get("acquired_disposed") or "").upper()
    return code == "P" and ad in ("A", "")


def _is_sell(ev: Dict[str, Any]) -> bool:
    """Open-market sale: code S AND disposed."""
    code = str(ev.get("transaction_code") or "").upper()
    ad = str(ev.get("acquired_disposed") or "").upper()
    return code == "S" and ad in ("D", "")


def _within_window(ev: Dict[str, Any], days: int) -> bool:
    """Keep events within `days` of today (UTC)."""
    d = ev.get("transaction_date")
    if not d:
        return True   # if no date, don't drop
    try:
        dt = datetime.strptime(str(d)[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        return True
    return (datetime.now(timezone.utc) - dt) <= timedelta(days=days)


# ── Public API ────────────────────────────────────────────────────────────────

def score_events(
    events: List[Dict[str, Any]],
    *,
    window_days: int = config.INSIDER_WINDOW_DAYS,
    saturation_usd: float = _SATURATION_USD,
) -> Dict[str, Any]:
    """Aggregate insider events into a 0–1 directional score with breakdown.

    Returns:
        {
            "score":          float in [0, 1],
            "buy_value":      float,
            "sell_value":     float,
            "net_value":      float,
            "buy_count":      int,
            "sell_count":     int,
            "events_in_window": int,
            "ceo_buy":        bool,    # CEO open-market buy ≥ $50k present
            "amendment_count": int,
        }
    """
    if not events:
        return _empty_breakdown()

    in_window = [e for e in events if _within_window(e, window_days)]
    if not in_window:
        return _empty_breakdown()

    buy_value = 0.0
    sell_value = 0.0
    buy_count = 0
    sell_count = 0
    ceo_buy = False
    amend = 0

    for ev in in_window:
        if ev.get("is_amendment"):
            amend += 1
        val = float(ev.get("value") or 0.0)
        if val <= 0:
            continue
        w = _role_weight(ev.get("reporting_role"))
        if _is_buy(ev):
            buy_value += w * val
            buy_count += 1
            if str(ev.get("reporting_role")) == "CEO" and val >= 50_000:
                ceo_buy = True
        elif _is_sell(ev):
            sell_value += w * val
            sell_count += 1

    net = buy_value - sell_value
    # tanh in [-1, 1] → score in [0, 1] centred on 0.50
    score = _NEUTRAL + 0.5 * math.tanh(net / max(saturation_usd, 1.0))

    # CEO-buy floor: a real CEO open-market buy lifts score off neutral.
    if ceo_buy:
        score = max(score, 0.62)

    return {
        "score":            round(max(0.0, min(1.0, score)), 4),
        "buy_value":        round(buy_value, 2),
        "sell_value":       round(sell_value, 2),
        "net_value":        round(net, 2),
        "buy_count":        buy_count,
        "sell_count":       sell_count,
        "events_in_window": len(in_window),
        "ceo_buy":          ceo_buy,
        "amendment_count":  amend,
    }


def _empty_breakdown() -> Dict[str, Any]:
    return {
        "score":            _NEUTRAL,
        "buy_value":        0.0,
        "sell_value":       0.0,
        "net_value":        0.0,
        "buy_count":        0,
        "sell_count":       0,
        "events_in_window": 0,
        "ceo_buy":          False,
        "amendment_count":  0,
    }
