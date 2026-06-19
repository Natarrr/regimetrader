"""src/scoring/momentum_signals.py
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


def compute_beta(
    asset_closes: list[float] | None,
    benchmark_closes: list[float] | None,
    window: int = 30,
) -> float | None:
    """Trailing rolling beta of an asset vs a benchmark (CAPM slope).

        beta = Cov(r_asset, r_bench) / Var(r_bench)

    over the most recent `window` daily simple returns. Both series must be
    oldest-first and aligned to the SAME trading calendar (e.g. a US ticker vs
    SPY) — for cross-calendar pairs (INTL vs SPY) the caller must align by date
    first. Returns None when fewer than `window` return pairs exist or the
    benchmark variance is ~0 (beta undefined).

    Producer for the CAPITULATION low-beta gate
    (src/risk/regime._is_capitulation_survivor), which is inert without a beta
    factor: a name with beta > 1.2 is dropped from the crash-regime shortlist.
    """
    if not asset_closes or not benchmark_closes:
        return None
    a = [float(x) for x in asset_closes[-(window + 1):]]
    b = [float(x) for x in benchmark_closes[-(window + 1):]]
    ra = [a[i] / a[i - 1] - 1.0 for i in range(1, len(a)) if a[i - 1]]
    rb = [b[i] / b[i - 1] - 1.0 for i in range(1, len(b)) if b[i - 1]]
    m = min(len(ra), len(rb))
    if m < window:
        return None
    ra, rb = ra[-window:], rb[-window:]
    mean_a = sum(ra) / window
    mean_b = sum(rb) / window
    cov = sum((ra[i] - mean_a) * (rb[i] - mean_b) for i in range(window)) / window
    var_b = sum((rb[i] - mean_b) ** 2 for i in range(window)) / window
    if var_b < 1e-12:
        return None
    return round(cov / var_b, 4)


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
) -> float | None:
    """Analyst consensus price target upside, in [0, 1], or None when absent.

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
        0.00 = 50%+ downside to target

    SIGNED factor (`price_target_upside` ∈ SIGNED_FACTORS, src/config/factor_matrix.py).
    Per CLAUDE.md §2, data absence must never read as bearish: a missing/zero/
    non-numeric input therefore returns None ("unavailable"), NOT 0.0 — a 0.0
    here is a real observation of 50%+ downside, and coercing absence to 0.0
    would silently mark every uncovered ticker maximally bearish. The pipeline
    (run_pipeline._ticker_effective_weights) redistributes the weight pro-rata
    when this factor is None; the v3 neutralizer treats None as unavailable.

    Source: FMPClient.get_price_target_consensus() → stable/price-target-consensus.
    """
    try:
        t = float(target_price)
        c = float(current_price)
    except (TypeError, ValueError):
        return None
    if not t or not c:
        return None
    if math.isnan(t) or math.isnan(c):
        return None
    upside  = (t - c) / c
    clipped = max(-0.50, min(0.50, upside))
    return round((clipped + 0.50) / 1.00, 4)


def score_quality_piotroski(ratios: dict) -> tuple[float, int]:
    """Simplified 8-point Piotroski F-score, in [0, 1], with raw count.

    Captures fundamental quality as a value-trap gate: high-conviction insider
    buying in a deteriorating business is a false signal. Piotroski (2000)
    showed that a simple binary F-score on financial statement data separates
    winners from losers among high book-to-market stocks. Novy-Marx (2013)
    extended this: gross profitability is the strongest single quality predictor.
    Ilmanen (2011) documents quality as a cross-regime premium independent of
    momentum — which makes it a natural complement to score_momentum_long.

    8 binary points (each worth 1/8 of the final score):
        1. ROA > 0                        — profitable at all
        2. ROA > 0.05                     — strong ROA (>5%)
        3. operatingProfitMarginTTM > 0   — positive operating income (OCF proxy)
        4. debtToEquityRatioTTM < 1.0     — manageable leverage
        5. debtToEquityRatioTTM < 0.5     — low leverage (bonus)
        6. currentRatioTTM > 1.5          — liquid balance sheet
        7. grossProfitMarginTTM > 0.30    — 30%+ gross margin = pricing power
        8. netProfitMarginTTM > 0.05      — profitable after all costs

    Field names (verified live 2026-06-09, identical for per-ticker
    stable/ratios-ttm and ratios-ttm-bulk):
        - The leverage field is debtToEquityRatioTTM. The legacy name
          debtEquityRatioTTM is kept as a fallback candidate for old fixtures.
        - There is NO returnOnAssets* field in the live payload. ROA is
          derived via DuPont: ROA = netProfitMargin × assetTurnover
          (netProfitMarginTTM × assetTurnoverTTM). Without this derivation
          points 1–2 could never be awarded — the bug that pinned the whole
          universe at raw=4.
        - Unsuffixed variants (e.g. grossProfitMargin) are also accepted in
          case a future bulk snapshot drops the TTM suffix.

    score = round(points_earned / 8.0, 4)

    Returns:
        (score, raw_count): score in [0, 1] and the raw integer point count
        (0–8). The raw count is used by _piotroski_gate_multiplier to apply
        the suppress/discount gate independently of the normalised score.

    Partial-data handling: a missing or None field contributes 0 for its
    point(s) but does not collapse the entire score. A company with 6 of 8
    fields and 5 passing scores 5/8 = 0.625.

    Negative D/E (negative book equity) fails both leverage points — it
    signals structural distress, not low debt.

    Returns (0.0, 0) (dead signal) when ratios is None, not a dict, or every
    relevant field is None/missing. Consistent with score_momentum_long:
    missing input is penalised, not granted a neutral pass.

    References:
        Piotroski (2000), "Value Investing: The Use of Historical Financial
        Statement Information to Separate Winners from Losers", JAR 38(1).
        Novy-Marx (2013), "The Other Side of Value", JFE 108(1).
        Ilmanen (2011), "Expected Returns", Wiley.

    Source: FMPClient.get_ratios_ttm() → stable/ratios-ttm (24h cache), or a
    ratios-ttm-bulk snapshot record (run_pipeline bulk index — same shape).
    """
    if not isinstance(ratios, dict) or not ratios:
        return 0.0, 0

    def _get(*fields: str) -> float | None:
        """First parseable value among candidate names, each tried with and
        without the TTM suffix (per-ticker and bulk shapes)."""
        for field in fields:
            for key in (field, field.removesuffix("TTM")):
                v = ratios.get(key)
                if v is None:
                    continue
                try:
                    f = float(v)
                    if not math.isnan(f):
                        return f
                except (TypeError, ValueError):
                    continue
        return None

    roa  = _get("returnOnAssetsTTM")
    opm  = _get("operatingProfitMarginTTM")
    de   = _get("debtToEquityRatioTTM", "debtEquityRatioTTM")  # live name first; legacy for old fixtures
    cr   = _get("currentRatioTTM")
    gpm  = _get("grossProfitMarginTTM")
    npm  = _get("netProfitMarginTTM")

    if roa is None:
        # Live stable/ratios-ttm has no ROA field — derive via DuPont:
        # ROA = net profit margin × asset turnover.
        at = _get("assetTurnoverTTM")
        if npm is not None and at is not None:
            roa = npm * at

    # Guard: all fields missing → dead signal
    if all(v is None for v in (roa, opm, de, cr, gpm, npm)):
        return 0.0, 0

    points = 0
    if roa is not None and roa > 0:
        points += 1
    if roa is not None and roa > 0.05:
        points += 1
    if opm is not None and opm > 0:
        points += 1
    if de is not None and 0 <= de < 1.0:  # negative D/E fails both leverage points
        points += 1
    if de is not None and 0 <= de < 0.5:  # negative D/E fails: 0 <= de is False
        points += 1
    if cr is not None and cr > 1.5:
        points += 1
    if gpm is not None and gpm > 0.30:
        points += 1
    if npm is not None and npm > 0.05:
        points += 1

    return round(points / 8.0, 4), points
