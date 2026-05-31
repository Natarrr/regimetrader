"""regime_trader/scoring/momentum_signals.py
Orthogonal momentum and attention signals.

Theory:
    Jegadeesh & Titman (1993), "Returns to Buying Winners and Selling Losers",
    Journal of Finance 48(1) pp. 65–91:
        The 12-1 month formation period (skip-month) produces a robust positive
        cross-sectional premium. The skip-month (t-21 to t) is excluded because
        it exhibits short-term reversal (De Bondt & Thaler 1985, Jegadeesh 1990)
        — using it with a positive weight is anti-alpha.

    Barber & Odean (2008), "All That Glitters: The Effect of Attention and News
    on the Buying Behavior of Individual and Institutional Investors",
    Review of Financial Studies 21(2) pp. 785–818:
        Volume spikes are an attention signal, not a directional signal. They
        predict short-term buying pressure from retail investors, not sustained
        outperformance. Belongs in a separate low-weight attention bucket.
"""
from __future__ import annotations

import logging
import math

log = logging.getLogger(__name__)


def score_momentum_long(
    return_12_1m: float | None,
    spy_return_12_1m: float = 0.0,
) -> float:
    """Jegadeesh-Titman (1993) 12-1 month momentum, SPY-relative, in [0, 1].

    Formula:
        excess  = return_12_1m - spy_return_12_1m
        clipped = max(-0.60, min(+0.60, excess))   # ±60% practical bound
        score   = (clipped + 0.60) / 1.20           # linear map to [0, 1]

    Returns 0.0 if return_12_1m is None or NaN (dead signal — recent IPO or
    insufficient history). Consistent with dead-signal treatment for insider
    and congress: 0.0 is penalised in the cross-sectional normalizer, not
    given a free neutral pass.

    Reference: Jegadeesh & Titman (1993), Journal of Finance 48(1).
    """
    if return_12_1m is None:
        return 0.0
    try:
        r = float(return_12_1m)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(r):
        return 0.0

    excess  = r - float(spy_return_12_1m)
    clipped = max(-0.60, min(0.60, excess))
    return round((clipped + 0.60) / 1.20, 4)


def score_volume_attention(volume_spike: float) -> float:
    """Pure attention signal based on 5d/90d volume ratio, in [0, 1].

    Formula:
        score = min(1.0, max(0.0, (volume_spike - 1.0) / 4.0))

    Mapping:
        1.0× (flat)  → 0.0
        2.0×         → 0.25
        3.0×         → 0.50
        5.0× spike   → 1.0

    Returns 0.0 if volume_spike <= 1.0 (no attention spike at all — dead signal,
    not neutral). Used as secondary tilt only (weight 0.03 in WEIGHTS).

    Reference: Barber & Odean (2008), Review of Financial Studies 21(2).
    """
    try:
        v = float(volume_spike)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(v) or v <= 1.0:
        return 0.0
    return round(min(1.0, (v - 1.0) / 4.0), 4)


def score_price_target_upside(
    target_price: float | None,
    current_price: float | None,
) -> float:
    """Analyst consensus price target upside, in [0, 1].

    Captures the forward-looking dimension that backward-looking price momentum
    (Jegadeesh-Titman 1993, 12-1m returns) cannot: where sell-side analysts
    collectively expect the price to go. These two signals are orthogonal —
    a stock can have strong past momentum and low analyst upside (priced in)
    or weak momentum and high analyst upside (re-rating candidate).

    Formula:
        upside  = (target_price - current_price) / current_price
        clipped = max(-0.50, min(+0.50, upside))   # ±50% practical bounds
        score   = round((clipped + 0.50) / 1.00, 4) # linear map → [0, 1]

    Score semantics:
        1.00 = 50%+ upside to target
        0.75 = 25% upside
        0.50 = at target (no upside/downside)
        0.25 = 25% downside
        0.00 = 50%+ downside OR dead signal

    Returns 0.0 (dead signal) when either argument is None, zero, or
    non-numeric. Consistent with score_momentum_long and score_volume_attention:
    a missing/zero input is penalised rather than granted a neutral pass.

    Source: FMPClient.get_price_target_consensus() → stable/price-target-consensus.
    """
    try:
        t = float(target_price)
        c = float(current_price)
    except (TypeError, ValueError):
        return 0.0
    if not t or not c:
        return 0.0
    upside  = (t - c) / c
    clipped = max(-0.50, min(0.50, upside))
    return round((clipped + 0.50) / 1.00, 4)
