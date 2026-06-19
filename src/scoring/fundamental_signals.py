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


# ── DuPont quality composite (v3.0, US P1 anchor) ─────────────────────────────

def _f(value) -> Optional[float]:
    """float() that maps None/garbage/NaN to None."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def score_quality_dupont(
    roa: Optional[float],
    npm: Optional[float],
    asset_turnover: Optional[float],
    debt_to_equity: Optional[float],
) -> float:
    """DuPont-derived quality composite, negative-range preserving.

    roa_eff = returnOnAssetsTTM, else DuPont npm × asset_turnover.
    roa_c   = (clip(roa_eff, −0.10, +0.20) + 0.10) / 0.30
    npm_c   = (clip(npm,     −0.10, +0.25) + 0.10) / 0.35
    lev     = 1.0 if 0 ≤ D/E < 0.5; 0.6 if < 1.0; 0.2 if < 2.0; else 0.0
    score   = 0.5·roa_c + 0.3·npm_c + 0.2·lev, renormalized over the
              components that are present.

    The negative clip floor (−10%) preserves downside cross-sectional
    variance: ROA −0.5% must outrank ROA −45% instead of both flattening
    to the same 0 (component mass-point). All components missing → 0.0
    (quality data is universal; total absence = broken feed; UNSIGNED dead).
    """
    roa_v = _f(roa)
    npm_v = _f(npm)
    at_v = _f(asset_turnover)
    de_v = _f(debt_to_equity)

    roa_eff = roa_v
    if roa_eff is None and npm_v is not None and at_v is not None:
        roa_eff = npm_v * at_v  # DuPont identity fallback

    parts: list[tuple[float, float]] = []  # (weight, component)
    if roa_eff is not None:
        roa_c = (max(-0.10, min(0.20, roa_eff)) + 0.10) / 0.30
        parts.append((0.5, roa_c))
    if npm_v is not None:
        npm_c = (max(-0.10, min(0.25, npm_v)) + 0.10) / 0.35
        parts.append((0.3, npm_c))
    if de_v is not None:
        if 0.0 <= de_v < 0.5:
            lev = 1.0
        elif de_v < 1.0:
            lev = 0.6
        elif de_v < 2.0:
            lev = 0.2
        else:
            lev = 0.0  # includes negative equity (D/E < 0) — distressed
        parts.append((0.2, lev))

    if not parts:
        return 0.0
    total_w = sum(w for w, _ in parts)
    score = sum(w * c for w, c in parts) / total_w
    return max(0.0, min(1.0, score))


# ── Operating margin expansion (v3.0, ASIA P1 anchor) ─────────────────────────

_QTR_GAP_MIN_DAYS = 70
_QTR_GAP_MAX_DAYS = 110


def _valid_statement_rows(rows: list[dict]) -> list[dict]:
    """Rows with the fields margin math needs, newest-first by period end."""
    out = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        if not row.get("date") or not row.get("filingDate"):
            continue  # filingDate required — look-ahead anchoring (CLAUDE.md)
        if _f(row.get("revenue")) is None or _f(row.get("operatingIncome")) is None:
            continue
        out.append(row)
    return sorted(out, key=lambda r: str(r["date"]), reverse=True)


def _ttm_opm(rows: list[dict]) -> Optional[float]:
    rev = sum(_f(r["revenue"]) for r in rows)
    op = sum(_f(r["operatingIncome"]) for r in rows)
    return op / rev if rev and rev > 0 else None


def score_margin_expansion(
    quarterly_rows: list[dict],
    annual_rows: list[dict],
) -> Optional[float]:
    """TTM operating-margin trajectory: Δ = OPM(q0–3) − OPM(q4–7).

    score = 0.5 + clip(Δ, −0.10, +0.10) / 0.20

    Discrete-quarter validation BEFORE any summation: international rows can
    be cumulative YTD (HKEX semi-annual mandates, JP tanshin) — consecutive
    period-ends must be 70–110 days apart and strictly decreasing, else the
    quarterly track is rejected (no heuristic differencing; mis-detection
    risk exceeds the benefit at 0.13 weight) and the ticker drops to the
    annual track: Δ = OPM(FY0) − OPM(FY−1), filingDate-anchored.

    SIGNED factor: None only when BOTH tracks are unavailable.
    """
    from datetime import date as _date

    quarters = _valid_statement_rows(quarterly_rows)
    if len(quarters) >= 8:
        window = quarters[:8]
        try:
            ends = [_date.fromisoformat(str(r["date"])[:10]) for r in window]
            gaps = [(ends[i] - ends[i + 1]).days for i in range(len(ends) - 1)]
            discrete = all(_QTR_GAP_MIN_DAYS <= g <= _QTR_GAP_MAX_DAYS
                           for g in gaps)
        except ValueError:
            discrete = False
        if discrete:
            opm_now = _ttm_opm(window[:4])
            opm_prior = _ttm_opm(window[4:8])
            if opm_now is not None and opm_prior is not None:
                delta = max(-0.10, min(0.10, opm_now - opm_prior))
                return 0.5 + delta / 0.20
        else:
            log.debug(
                "margin_expansion: quarterly rows fail discrete-window "
                "validation (cumulative/semi-annual suspected) — annual track"
            )

    annuals = _valid_statement_rows(annual_rows)
    if len(annuals) >= 2 and str(annuals[0]["date"]) != str(annuals[1]["date"]):
        opm_now = _ttm_opm(annuals[:1])
        opm_prior = _ttm_opm(annuals[1:2])
        if opm_now is not None and opm_prior is not None:
            delta = max(-0.10, min(0.10, opm_now - opm_prior))
            return 0.5 + delta / 0.20

    return None


# ══════════════════════════════════════════════════════════════════════════════
# CANDIDATE FACTORS — shadow-first (computed, NOT yet weighted). Added to widen
# valuation breadth (A1) and add growth/earnings-quality (A2) per the model
# audit. Each must pass a de-overlapped IC gate (src/research/ic_metrics) before
# any weight is allocated in WEIGHTS_* / FACTOR_MATRIX_V3.
# ══════════════════════════════════════════════════════════════════════════════

# ── A1 · Earnings Yield (E/P) ─────────────────────────────────────────────────

def score_earnings_yield(net_income_ttm: float, market_cap: float) -> float:
    """Basu (1977) earnings yield E/P = TTM net income / market cap → [0, 1].

    Formula:
        ey      = net_income_ttm / market_cap
        clipped = max(0.0, min(0.15, ey))    # practical [0%, 15%] band
        score   = clipped / 0.15             # linear map to [0, 1]

    UNSIGNED dead-signal: 0.0 when net income ≤ 0 (loss-making) or market cap ≤ 0.
    Distinct from fcf_yield (cash/EV-based): E/P captures accrual earnings and
    diversifies the value pillar beyond FCF yield + P/B.

    Reference: Basu (1977), Journal of Finance 32(3).
    """
    ni = _f(net_income_ttm)
    mc = _f(market_cap)
    if ni is None or mc is None or ni <= 0 or mc <= 0:
        return 0.0
    clipped = max(0.0, min(0.15, ni / mc))
    return round(clipped / 0.15, 4)


# ── A1 · Enterprise multiple (EV/EBITDA) ──────────────────────────────────────

def score_ev_ebitda(enterprise_value: float, ebitda_ttm: float) -> float:
    """Loughran & Wellman (2011) enterprise multiple EV/EBITDA, inverted → [0, 1].

    Formula:
        ratio   = enterprise_value / ebitda_ttm
        clipped = max(5.0, min(35.0, ratio))      # ≈ 10th–90th pct of large/mid-caps
        score   = 1 - (clipped - 5.0) / 30.0      # low multiple → high score

    The [5×, 35×] band spans the realistic cross-section: a tighter [4, 20]
    floors every premium mega-cap (AAPL ≈28×, ASML ≈32×) at 0.0, collapsing the
    top half of the universe to a single mass-point that kills cross-sectional
    discrimination (and IC).

    UNSIGNED dead-signal: 0.0 when EBITDA ≤ 0 (no positive operating earnings)
    or EV ≤ 0. Capital-structure-neutral complement to E/P and FCF yield.

    Reference: Loughran & Wellman (2011), JFQA 46(6).
    """
    ev = _f(enterprise_value)
    eb = _f(ebitda_ttm)
    if ev is None or eb is None or ev <= 0 or eb <= 0:
        return 0.0
    clipped = max(5.0, min(35.0, ev / eb))
    return round(1.0 - (clipped - 5.0) / 30.0, 4)


# ── A2 · TTM YoY growth (revenue / earnings) ──────────────────────────────────

def _valid_field_rows(rows: list[dict], field: str) -> list[dict]:
    """Rows carrying date + filingDate + `field`, newest-first by period end.
    filingDate is required — look-ahead anchoring (CLAUDE.md §3)."""
    out = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        if not row.get("date") or not row.get("filingDate"):
            continue
        if _f(row.get(field)) is None:
            continue
        out.append(row)
    return sorted(out, key=lambda r: str(r["date"]), reverse=True)


def _ttm_field_sum(rows: list[dict], field: str) -> Optional[float]:
    vals = [_f(r.get(field)) for r in rows]
    if any(v is None for v in vals):
        return None
    return sum(vals)


def _ttm_growth_score(
    quarterly_rows: list[dict],
    annual_rows: list[dict],
    field: str,
    band: float,
) -> Optional[float]:
    """SIGNED TTM YoY growth of `field`: g = TTM(q0–3) / TTM(q4–7) − 1.

        score = 0.5 + clip(g, −band, +band) / (2·band)

    Same discrete-quarter validation as score_margin_expansion (70–110d gaps;
    rejects cumulative/semi-annual rows), with an annual fallback FY0 vs FY−1.
    SIGNED: None when both tracks unavailable OR prior TTM ≤ 0 (growth off a
    non-positive base is undefined — must not read as a level).
    """
    from datetime import date as _date

    quarters = _valid_field_rows(quarterly_rows, field)
    if len(quarters) >= 8:
        window = quarters[:8]
        try:
            ends = [_date.fromisoformat(str(r["date"])[:10]) for r in window]
            gaps = [(ends[i] - ends[i + 1]).days for i in range(len(ends) - 1)]
            discrete = all(_QTR_GAP_MIN_DAYS <= g <= _QTR_GAP_MAX_DAYS
                           for g in gaps)
        except ValueError:
            discrete = False
        if discrete:
            now = _ttm_field_sum(window[:4], field)
            prior = _ttm_field_sum(window[4:8], field)
            if now is not None and prior is not None and prior > 0:
                g = max(-band, min(band, now / prior - 1.0))
                return 0.5 + g / (2.0 * band)

    annuals = _valid_field_rows(annual_rows, field)
    if len(annuals) >= 2 and str(annuals[0]["date"]) != str(annuals[1]["date"]):
        now = _f(annuals[0].get(field))
        prior = _f(annuals[1].get(field))
        if now is not None and prior is not None and prior > 0:
            g = max(-band, min(band, now / prior - 1.0))
            return 0.5 + g / (2.0 * band)

    return None


def score_revenue_growth(
    quarterly_rows: list[dict], annual_rows: list[dict]
) -> Optional[float]:
    """TTM revenue YoY growth, SIGNED → [0, 1] (band ±30%). filingDate-anchored.

    Reference: Lakonishok, Shleifer & Vishny (1994) — fundamental growth axis.
    """
    return _ttm_growth_score(quarterly_rows, annual_rows, "revenue", band=0.30)


def score_eps_growth(
    quarterly_rows: list[dict], annual_rows: list[dict]
) -> Optional[float]:
    """TTM earnings (net income) YoY growth, SIGNED → [0, 1] (band ±50% — EPS is
    more volatile than revenue). Prior TTM ≤ 0 → None (growth off a loss base
    is undefined). filingDate-anchored.
    """
    return _ttm_growth_score(quarterly_rows, annual_rows, "netIncome", band=0.50)


# ── A2 · Accruals anomaly (Sloan 1996) ────────────────────────────────────────

def score_accruals(
    net_income_ttm: float, cfo_ttm: float, total_assets: float
) -> float:
    """Sloan (1996) accruals anomaly: accruals = (NI − CFO) / total assets.

    HIGH accruals predict LOWER future returns (earnings less cash-backed), so
    the score INVERTS the ratio:
        ratio   = (net_income_ttm − cfo_ttm) / total_assets
        clipped = max(−0.20, min(0.20, ratio))
        score   = 1 − (clipped + 0.20) / 0.40    # low accruals → high score

    UNSIGNED dead-signal: 0.0 when total assets ≤ 0 or NI/CFO missing (broken
    feed). Earnings-quality complement to the Piotroski gate.

    Reference: Sloan (1996), The Accounting Review 71(3).
    """
    ni = _f(net_income_ttm)
    cfo = _f(cfo_ttm)
    ta = _f(total_assets)
    if ni is None or cfo is None or ta is None or ta <= 0:
        return 0.0
    clipped = max(-0.20, min(0.20, (ni - cfo) / ta))
    return round(1.0 - (clipped + 0.20) / 0.40, 4)
