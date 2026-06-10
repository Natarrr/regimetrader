# Path: src/scoring/fundamental_signals.py
"""src/scoring/fundamental_signals.py
Fundamental value and quality signals for EU/Asia INTL pipeline.

Theory:
    Damodaran (2006), "Damodaran on Valuation":
        FCF Yield = TTM Free Cash Flow / Enterprise Value. Higher yield indicates
        cheap asset relative to its cash-generating ability. Distinct from earnings
        yield because FCF excludes non-cash accruals (Sloan 1996 — accruals predictor).

    Amihud (2002), "Illiquidity and stock returns: cross-section and time-series effects",
    Journal of Financial Markets 5 pp. 31-56:
        Illiquidity ratio = |R_t| / (V_t × P_t). Daily illiquidity shock ratio vs
        rolling median captures latent sell-side pressure and forced liquidation signals.

    Fama & French (1992), "The Cross-Section of Expected Stock Returns",
    Journal of Finance 47(2) pp. 427-465:
        Book-to-market is the strongest cross-sectional predictor after size.
        P/B inversion maps to B/M in [0,1] — lower P/B = higher score.

    Greenblatt (2005), "The Little Book That Beats the Market":
        Magic formula quality = ROIC. Blend with ROE when ROCE is available.
        Normalized 0–50% operating ROE range maps cleanly to [0,1].
"""
from __future__ import annotations

import logging
import math
from typing import Optional

log = logging.getLogger(__name__)


# ── FCF Yield ─────────────────────────────────────────────────────────────────

def score_fcf_yield(ttm_fcf_usd: float, enterprise_value_usd: float) -> float:
    """Damodaran FCF Yield = TTM_FCF / EV, normalized to [0, 1].

    Formula:
        raw_yield  = ttm_fcf_usd / enterprise_value_usd
        clipped    = max(0.0, min(0.20, raw_yield))   # practical [0%, 20%] bound
        score      = clipped / 0.20                    # linear map to [0, 1]

    Returns 0.0 (dead signal) when either input is ≤ 0 (loss-making or no EV data).
    Consistent with dead-signal treatment: 0.0 is penalised, not neutral-passed.

    Reference: Damodaran (2006), "Damodaran on Valuation".
    """
    try:
        fcf = float(ttm_fcf_usd)
        ev  = float(enterprise_value_usd)
    except (TypeError, ValueError):
        return 0.0
    if ev <= 0 or fcf <= 0:
        return 0.0
    if math.isnan(fcf) or math.isnan(ev):
        return 0.0
    raw_yield = fcf / ev
    clipped   = max(0.0, min(0.20, raw_yield))
    return round(clipped / 0.20, 4)


# ── Amihud Illiquidity Shock ───────────────────────────────────────────────────

def score_amihud_shock(
    price_history: list[float],
    volume_history: list[float],
    return_history: Optional[list[float]] = None,
    lookback_baseline: int = 20,
) -> float:
    """Amihud (2002) illiquidity shock ratio, mapped to [0, 1].

    Illiquidity ratio for each day t:
        amihud_t = |R_t| / (V_t × P_t)

    Shock ratio:
        shock = amihud_today / median(amihud[-lookback_baseline:])

    Score mapping (piece-wise linear):
        shock < 1.0×  →  0.0   (liquidity normal or improving)
        shock = 1.5×  →  0.5
        shock ≥ 3.0×  →  1.0   (significant liquidity disruption)

    A high shock alongside a large absolute return = genuine institutional
    activity or forced deleveraging — elevated score is intentional.

    Returns 0.0 if price_history, volume_history have < (lookback_baseline + 1)
    observations or volumes are all zero.

    Reference: Amihud (2002), Journal of Financial Markets 5 pp. 31-56.
    """
    n = lookback_baseline + 1
    if len(price_history) < n or len(volume_history) < n:
        return 0.0

    prices  = price_history[-n:]
    volumes = volume_history[-n:]

    if return_history and len(return_history) >= n:
        rets = return_history[-n:]
    else:
        rets = []
        for i in range(1, n):
            p_prev = prices[i - 1]
            p_cur  = prices[i]
            if p_prev and p_prev > 0:
                rets.append(abs((p_cur - p_prev) / p_prev))
            else:
                rets.append(0.0)
        rets = [0.0] + rets

    amihud_series: list[float] = []
    for ret, vol, price in zip(rets, volumes, prices):
        denom = vol * price
        if denom and denom > 0:
            amihud_series.append(abs(ret) / denom)
        else:
            amihud_series.append(0.0)

    baseline = amihud_series[:-1]
    today    = amihud_series[-1]

    non_zero = [x for x in baseline if x > 0]
    if not non_zero:
        return 0.0

    sorted_baseline = sorted(non_zero)
    mid             = len(sorted_baseline) // 2
    if len(sorted_baseline) % 2 == 0:
        median_amihud = (sorted_baseline[mid - 1] + sorted_baseline[mid]) / 2
    else:
        median_amihud = sorted_baseline[mid]

    if median_amihud <= 0:
        return 0.0

    shock = today / median_amihud

    # Piece-wise linear: [0, 1.0) → 0.0; [1.0, 3.0] → [0.0, 1.0]; >3.0 → 1.0
    if shock < 1.0:
        return 0.0
    score = min(1.0, (shock - 1.0) / 2.0)
    return round(score, 4)


# ── Dynamic P/B "Value-Up" ────────────────────────────────────────────────────

def score_pb_value_up(
    book_value_per_share: float,
    current_price: float,
) -> float:
    """Fama & French (1992) P/B value signal, mapped to [0, 1].

    Formula:
        pb_ratio  = current_price / book_value_per_share
        clipped   = max(0.5, min(3.0, pb_ratio))      # practical [0.5×, 3.0×] bound
        base      = 1 - (clipped - 0.5) / 2.5         # invert: low P/B → high score
        bonus     = +0.10 when pb_ratio < 1.0 (below book floor — Fama & French value trigger)
        score     = min(1.0, base + bonus)

    Returns 0.0 (dead signal) when book_value_per_share ≤ 0 (asset impairment,
    negative equity, or missing data).

    Reference: Fama & French (1992), Journal of Finance 47(2).
    """
    try:
        bvps  = float(book_value_per_share)
        price = float(current_price)
    except (TypeError, ValueError):
        return 0.0
    if bvps <= 0 or price <= 0:
        return 0.0
    if math.isnan(bvps) or math.isnan(price):
        return 0.0

    pb_ratio = price / bvps
    clipped  = max(0.5, min(3.0, pb_ratio))
    base     = 1.0 - (clipped - 0.5) / 2.5
    bonus    = 0.10 if pb_ratio < 1.0 else 0.0
    return round(min(1.0, base + bonus), 4)


# ── ROIC / ROE Quality ────────────────────────────────────────────────────────

def score_roic_quality(
    return_on_equity: float,
    return_on_capital_employed: Optional[float] = None,
) -> float:
    """Greenblatt (2005) quality component: ROE/ROIC blend, mapped to [0, 1].

    Formula:
        If ROCE is available:
            quality_ratio = (return_on_equity + return_on_capital_employed) / 2
        Else:
            quality_ratio = return_on_equity

        clipped = max(0.0, min(0.50, quality_ratio))   # practical [0%, 50%] bound
        score   = clipped / 0.50                        # linear map to [0, 1]

    Returns 0.0 (dead signal) when ROE ≤ 0 (loss-making quarter).
    ROCE being None or ≤ 0 falls back gracefully to ROE-only scoring.

    Reference: Greenblatt (2005), "The Little Book That Beats the Market".
    """
    try:
        roe = float(return_on_equity)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(roe) or roe <= 0:
        return 0.0

    quality_ratio = roe
    if return_on_capital_employed is not None:
        try:
            roce = float(return_on_capital_employed)
            if roce > 0 and not math.isnan(roce):
                quality_ratio = (roe + roce) / 2.0
        except (TypeError, ValueError):
            pass

    clipped = max(0.0, min(0.50, quality_ratio))
    return round(clipped / 0.50, 4)
