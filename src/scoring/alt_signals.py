# Path: src/scoring/alt_signals.py
"""src/scoring/alt_signals.py
v3.0 Alternative / Micro-Structural Flow scorers (pillar P3).

Theory:
    Cohen, Malloy & Pomorski (2012), "Decoding Inside Information", JF 67(3):
        Opportunistic insider PURCHASES predict returns; routine sales carry
        no positive information (compensation vesting, diversification).
        All volume terms here are therefore P-code (purchase) only.

    Lakonishok & Lee (2001), "Are Insider Trades Informative?", RFS 14(1):
        Insider purchase breadth predicts best in small/value names — the
        breadth residual (Gram-Schmidt vs conviction) enters the composite.

    Boudoukh, Michaely, Richardson & Roberts (2007), JF 62(2) — pre-registered
        C2 successor (shareholder_yield) if intl 13F coverage < 30%.

Missing-data semantics (factor_matrix.FactorSpec):
    insider_alpha / congress / dividend_sustain are UNSIGNED: 0.0 = dead
    signal, excluded from bucket stats (mass-point defense).
    inst_flow_13f / inst_concentration are SIGNED: unavailability is None,
    a computed low value is a real observation.
"""
from __future__ import annotations

import logging
import math
from typing import Optional

log = logging.getLogger(__name__)


# ── Insider alpha composite (US P3 anchor) ────────────────────────────────────

def score_insider_alpha(
    conviction: float,
    breadth_residual: float,
    p_count_30d: int,
    p_count_31_180d: int,
    usd_buy_csuite: float,
    usd_buy_total: float,
) -> float:
    """Cohen/Lakonishok composite + velocity / C-suite micro-structure term.

    Dead short-circuit FIRST: zero P-code purchases across the 180-day
    lookback returns exactly 0.0 (the zero_is_dead exclusion tier). Without
    it, the neutral csuite fallback would leak 0.15·0.4·0.5 = 0.03 into
    every inactive name — a mass-point that bypasses the zero-exclusion and
    collapses bucket σ.

    Active names:
        velocity    = min(1, log1p(P_30d / max(1, P_31_180d·30/150)) / log1p(5))
        csuite_tilt = 0.5 + 0.5·(USD_buy_csuite / USD_buy_total)
                      (0.5 only when purchases exist but USD unparseable)
        micro       = 0.6·velocity + 0.4·csuite_tilt
        score       = clip(0.55·conviction + 0.30·breadth_residual
                           + 0.15·micro, 0, 1)

    All inputs are purchase-side (P-code) only — S-code sales never enter
    [Cohen, Malloy & Pomorski 2012].
    """
    p_total = int(p_count_30d or 0) + int(p_count_31_180d or 0)
    if p_total <= 0:
        return 0.0

    baseline = max(1.0, float(p_count_31_180d or 0) * 30.0 / 150.0)
    velocity = min(
        1.0, math.log1p(float(p_count_30d or 0) / baseline) / math.log1p(5.0)
    )

    if usd_buy_total and usd_buy_total > 0:
        csuite_tilt = 0.5 + 0.5 * (
            max(0.0, float(usd_buy_csuite or 0.0)) / float(usd_buy_total)
        )
        csuite_tilt = min(1.0, csuite_tilt)
    else:
        # Purchases exist (p_total > 0) but USD values were unparseable.
        csuite_tilt = 0.5

    micro = 0.6 * velocity + 0.4 * csuite_tilt
    raw = (
        0.55 * float(conviction or 0.0)
        + 0.30 * float(breadth_residual or 0.0)
        + 0.15 * micro
    )
    return max(0.0, min(1.0, raw))


# ── Congressional flow surge multiplier (US P3) ───────────────────────────────

def congress_surge_multiplier(net_buys_30d: int, net_buys_180d: int) -> float:
    """Boost factor for an acceleration in congressional net buying.

    surge = net_buys_30d / max(1, net_buys_180d·30/180)
    mult  = 1 + 0.25·min(1, (surge − 1)/2)   when surge > 1, else 1.0

    Bounded [1.0, 1.25]; applied multiplicatively to the existing congress
    score (result clipped to [0, 1] by the caller).
    """
    baseline = max(1.0, float(net_buys_180d or 0) * 30.0 / 180.0)
    surge = float(net_buys_30d or 0) / baseline
    if surge <= 1.0:
        return 1.0
    return 1.0 + 0.25 * min(1.0, (surge - 1.0) / 2.0)


# ── 13F institutional flow delta (US P3) ──────────────────────────────────────

def score_inst_flow_13f(summary: Optional[dict]) -> Optional[float]:
    """13F position-delta composite from symbol-positions-summary fields.

    score = clip(0.5 + 0.2·clip(10·ΔinvestorsHolding/investorsHolding, −1, 1)
                     + 0.2·(inc − red)/max(1, inc + red)
                     + 0.1·clip(ΔownershipPercent, −5, 5)/5, 0, 1)

    SIGNED factor: empty payload → None (data unavailable, reweighted);
    a computed 0.0 is an extreme-outflow observation, not a dead signal.
    """
    if not summary:
        return None
    ih = float(summary.get("investorsHolding") or 0.0)
    dih = float(summary.get("investorsHoldingChange") or 0.0)
    inc = float(summary.get("increasedPositions") or 0.0)
    red = float(summary.get("reducedPositions") or 0.0)
    dop = float(summary.get("ownershipPercentChange") or 0.0)

    t1 = 0.2 * max(-1.0, min(1.0, 10.0 * dih / ih)) if ih > 0 else 0.0
    t2 = 0.2 * (inc - red) / max(1.0, inc + red)
    t3 = 0.1 * max(-5.0, min(5.0, dop)) / 5.0
    return max(0.0, min(1.0, 0.5 + t1 + t2 + t3))


# ── Insider acquired-vs-disposed spike (US overlay, not weighted) ─────────────

def score_insider_npr_spike(
    stats: Optional[list], baseline_quarters: int = 4,
) -> Optional[dict]:
    """Acquired-vs-disposed Net Purchase Ratio spike from insider statistics.

    NPR = acquiredTransactions / (acquiredTransactions + disposedTransactions)
    per quarter [Lakonishok & Lee 2001 — buys are far more informative than
    sells]. ``spike`` compares the latest quarter's NPR to the trailing
    ``baseline_quarters`` mean; a large positive spike flags unusual insider
    cluster buying — the 🐋 whale-accumulation trigger.

    This is a DISPLAY/BADGE overlay, not a weighted scoring factor (it overlaps
    the Form-4 insider_conviction/breadth factors; adding it to WEIGHTS would
    breach the orthogonality monitor, CLAUDE.md §3). Returns
    {npr, spike, acquired, disposed} for the latest quarter, or None when no
    statistics exist (absence is silent, never bearish).
    """
    if not stats:
        return None
    latest = stats[0]
    acq = int(latest.get("acquiredTransactions") or 0)
    dis = int(latest.get("disposedTransactions") or 0)
    if acq + dis <= 0:
        return None
    npr = acq / (acq + dis)

    base_nprs = []
    for q in stats[1:1 + baseline_quarters]:
        a = int(q.get("acquiredTransactions") or 0)
        d = int(q.get("disposedTransactions") or 0)
        if a + d > 0:
            base_nprs.append(a / (a + d))
    baseline = sum(base_nprs) / len(base_nprs) if base_nprs else npr

    return {
        "npr":      round(npr, 4),
        "spike":    round(npr - baseline, 4),
        "acquired": acq,
        "disposed": dis,
    }


# ── 13F ownership concentration (EU/ASIA synthetic alternative, P3) ───────────

def score_inst_concentration(summary: Optional[dict]) -> Optional[float]:
    """Institutional ownership concentration level for intl synthetic alt.

    score = clip(0.4·clip(ownershipPercent, 0, 80)/80
                 + 0.3·min(1, log1p(investorsHolding)/log1p(500))
                 + 0.3·(0.5 + 0.5·(inc − red)/max(1, inc + red)), 0, 1)

    The min(1, ·) caps the breadth term for rosters above 500 institutions
    (a 2,500-holder mega-cap would otherwise push the log ratio to ≈1.26
    and the composite past 1.0, breaking the bounded-[0,1] contract).
    SIGNED factor: empty payload → None (13F coverage of intl local lines
    is partial — None-reweighting absorbs the sparsity).
    """
    if not summary:
        return None
    own = float(summary.get("ownershipPercent") or 0.0)
    ih = max(0.0, float(summary.get("investorsHolding") or 0.0))
    inc = float(summary.get("increasedPositions") or 0.0)
    red = float(summary.get("reducedPositions") or 0.0)

    t1 = 0.4 * max(0.0, min(80.0, own)) / 80.0
    t2 = 0.3 * min(1.0, math.log1p(ih) / math.log1p(500.0))
    t3 = 0.3 * (0.5 + 0.5 * (inc - red) / max(1.0, inc + red))
    return max(0.0, min(1.0, t1 + t2 + t3))


# ── Dividend sustainability (EU/ASIA P3, income-quality value tilt) ───────────

def score_dividend_sustain(
    dividend_yield: Optional[float],
    payout_ratio: Optional[float],
    fcf_ttm: Optional[float],
    dividends_paid_ttm: Optional[float],
) -> Optional[float]:
    """Payout sustainability composite.

    Payers:
        score = 0.45·(1 − clip(|payout − 0.45| / 0.55, 0, 1))
              + 0.35·clip(FCF/|dividendsPaid| − 1, 0, 2)/2
              + 0.20·clip(yield, 0, 0.08)/0.08

    Non-payers → 0.0 (deliberate value-vector tilt; UNSIGNED dead signal,
    excluded from bucket μ/σ so clustered non-payers cannot collapse σ).
    Payers with missing inputs → None (data unavailable, never penalized).
    Field names verified against live ratios-ttm (2026-06-10 probe):
    dividendYieldTTM / dividendPayoutRatioTTM.
    """
    if (dividend_yield is None and payout_ratio is None
            and fcf_ttm is None and dividends_paid_ttm is None):
        return None

    dy = float(dividend_yield or 0.0)
    paid = abs(float(dividends_paid_ttm or 0.0))
    if dy <= 0.0 and paid <= 0.0:
        return 0.0  # non-payer — dead by design

    if (dividend_yield is None or payout_ratio is None
            or fcf_ttm is None or dividends_paid_ttm is None or paid <= 0.0):
        return None  # payer with incomplete data — unavailable, not bearish

    payout = float(payout_ratio)
    coverage = float(fcf_ttm) / paid

    t1 = 0.45 * (1.0 - max(0.0, min(1.0, abs(payout - 0.45) / 0.55)))
    t2 = 0.35 * max(0.0, min(2.0, coverage - 1.0)) / 2.0
    t3 = 0.20 * max(0.0, min(0.08, dy)) / 0.08
    return max(0.0, min(1.0, t1 + t2 + t3))
