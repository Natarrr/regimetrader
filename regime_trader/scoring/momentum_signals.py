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
    if math.isnan(t) or math.isnan(c):
        return 0.0
    upside  = (t - c) / c
    clipped = max(-0.50, min(0.50, upside))
    return round((clipped + 0.50) / 1.00, 4)


def score_quality_piotroski(ratios: dict) -> float:
    """Simplified 9-point Piotroski F-score, in [0, 1].

    Captures fundamental quality as a value-trap gate: high-conviction insider
    buying in a deteriorating business is a false signal. Piotroski (2000)
    showed that a simple binary F-score on financial statement data separates
    winners from losers among high book-to-market stocks. Novy-Marx (2013)
    extended this: gross profitability is the strongest single quality predictor.
    Ilmanen (2011) documents quality as a cross-regime premium independent of
    momentum — which makes it a natural complement to score_momentum_long.

    9 binary points (each worth 1/9 of the final score):
        Profitability (4):
          1. returnOnAssetsTTM > 0         — profitable at all
          2. returnOnAssetsTTM > 0.05      — strong ROA (> 5%)
          3. operatingCashFlowPerShareTTM > 0  — positive operating cash flow
             (or operatingProfitMarginTTM > 0 as proxy when CFO/share unavailable)
          4. CFO-to-assets proxy > ROA     — accruals quality signal
             (grossProfitMarginTTM > returnOnAssetsTTM * 1.5 as proxy)
        Leverage/Liquidity (3):
          5. debtEquityRatioTTM < 1.0      — manageable leverage
          6. debtEquityRatioTTM < 0.5      — low leverage (bonus)
          7. currentRatioTTM > 1.5         — liquid balance sheet
        Efficiency (2):
          8. grossProfitMarginTTM > 0.30   — 30%+ gross margin = pricing power
          9. netProfitMarginTTM > 0.05     — profitable after all costs

    score = round(points_earned / 9.0, 4)

    Partial-data handling: a missing or None field contributes 0 for its
    point(s) but does not collapse the entire score. A company with 7 of 9
    fields and 6 passing scores 6/9 = 0.667.

    Negative D/E (negative book equity) fails both leverage points — it
    signals structural distress, not low debt.

    Returns 0.0 (dead signal) when ratios is None, not a dict, or every
    relevant field is None/missing. Consistent with score_momentum_long:
    missing input is penalised, not granted a neutral pass.

    References:
        Piotroski (2000), "Value Investing: The Use of Historical Financial
        Statement Information to Separate Winners from Losers", JAR 38(1).
        Novy-Marx (2013), "The Other Side of Value", JFE 108(1).
        Ilmanen (2011), "Expected Returns", Wiley.

    Source: FMPClient.get_ratios_ttm() → stable/ratios-ttm (24h cache).
    """
    if not isinstance(ratios, dict) or not ratios:
        return 0.0

    def _get(field: str) -> float | None:
        v = ratios.get(field)
        if v is None:
            return None
        try:
            f = float(v)
            return None if math.isnan(f) else f
        except (TypeError, ValueError):
            return None

    roa           = _get("returnOnAssetsTTM")
    cfo_per_share = _get("operatingCashFlowPerShareTTM")
    opm           = _get("operatingProfitMarginTTM")
    de            = _get("debtEquityRatioTTM")
    cr            = _get("currentRatioTTM")
    gpm           = _get("grossProfitMarginTTM")
    npm           = _get("netProfitMarginTTM")

    # Guard: all fields missing → dead signal
    if all(v is None for v in (roa, cfo_per_share, opm, de, cr, gpm, npm)):
        return 0.0

    points = 0

    # Profitability (4 points)
    if roa is not None and roa > 0:
        points += 1
    if roa is not None and roa > 0.05:
        points += 1

    # Point 3: positive operating cash flow
    # Use CFO/share if available; fall back to operating margin proxy
    if cfo_per_share is not None:
        if cfo_per_share > 0:
            points += 1
    elif opm is not None and opm > 0:
        points += 1

    # Point 4: accruals quality proxy (Piotroski 2000: CFO > ROA)
    # grossProfitMarginTTM > returnOnAssetsTTM * 1.5 approximates cash earnings
    # exceeding accrual earnings when CFO/assets is unavailable from ratios-ttm.
    if gpm is not None and roa is not None and roa > 0:
        if gpm > roa * 1.5:
            points += 1

    # Leverage/Liquidity (3 points)
    if de is not None and 0 <= de < 1.0:  # negative D/E fails both leverage points
        points += 1
    if de is not None and 0 <= de < 0.5:  # negative D/E fails: 0 <= de is False
        points += 1
    if cr is not None and cr > 1.5:
        points += 1

    # Efficiency (2 points)
    if gpm is not None and gpm > 0.30:
        points += 1
    if npm is not None and npm > 0.05:
        points += 1

    return round(points / 9.0, 4)
