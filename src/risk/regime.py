# Path: src/risk/regime.py
"""Shared RiskRegime — single source of truth for VIX thresholds.

NORMAL      : VIX < 20   — full positioning
BEAR        : 20 ≤ VIX < 30 — graduated caution
CAPITULATION: VIX ≥ 30   — distressed alpha regime (NOT a blind kill-switch)

At CAPITULATION, the pipeline does NOT go silent. It surfaces the highest-quality,
lowest-beta structural anchors — assets historically offering asymmetric risk-reward
during market capitulation events (Greenblatt 1980; Druckenmiller drawdown theory).

Criteria for CAPITULATION survivors:
  - quality_piotroski (normalized) ≥ 0.70  OR  debt_to_equity (normalized) ≤ 0.30
  - Beta ≤ 1.2 (trailing 30-day rolling vs SPY/benchmark) — enforced only when
    the producer supplies a beta factor; no current producer does, so this
    gate is inert until beta is added to the factor payloads
"""
from __future__ import annotations

import math
from enum import Enum
from typing import List

_BEAR_THRESHOLD = 20.0
_CAPITULATION_THRESHOLD = 30.0

# Public aliases — consumers must import these instead of hardcoding VIX levels.
BEAR_THRESHOLD = _BEAR_THRESHOLD
CAPITULATION_THRESHOLD = _CAPITULATION_THRESHOLD
_BETA_MAX = 1.2       # assets above this beta are filtered in CAPITULATION
_PIOTROSKI_FLOOR = 0.70   # normalized score threshold (≈ F-Score ≥ 7)
_DE_CEILING = 0.30    # normalized D/E — bottom quintile qualifier


class RiskRegime(str, Enum):
    NORMAL = "NORMAL"
    BEAR = "BEAR"
    CAPITULATION = "CAPITULATION"


def get_regime(vix: float) -> RiskRegime:
    if not isinstance(vix, (int, float)) or math.isnan(vix) or vix < 0:
        raise ValueError(f"Invalid VIX: {vix!r}")
    if vix >= _CAPITULATION_THRESHOLD:
        return RiskRegime.CAPITULATION
    if vix >= _BEAR_THRESHOLD:
        return RiskRegime.BEAR
    return RiskRegime.NORMAL


def is_panic(vix: float) -> bool:
    """True when VIX triggers CAPITULATION regime (≥ 30). Alias for legacy compatibility."""
    return get_regime(vix) == RiskRegime.CAPITULATION


def score_multiplier(regime: RiskRegime) -> float:
    return {
        RiskRegime.NORMAL: 1.00,
        RiskRegime.BEAR: 0.80,
        RiskRegime.CAPITULATION: 0.50,
    }[regime]


_CRASH_THRESHOLD = 40.0
CRASH_THRESHOLD = _CRASH_THRESHOLD


def vix_multiplier(vix: float) -> float:
    """Single source of truth for the VIX macro-overlay multiplier.

    Unlike score_multiplier (regime-level, 3 tiers), this carries the
    additional Crash tier so the US pipeline (generate_top_lists) and the
    INTL cook (cook_toplists) apply byte-identical dampening:

      VIX ≥ 40 (Crash) : ×0.20
      VIX ≥ 30 (Panic) : ×0.50
      VIX ≥ 20 (Bear)  : ×0.80
      VIX <  20        : ×1.00

    Raises ValueError on NaN/negative VIX (NaN comparisons are all False in
    Python and would silently bypass dampening).
    """
    get_regime(vix)  # validates NaN / negative / non-numeric
    if vix >= _CRASH_THRESHOLD:
        return 0.20
    if vix >= _CAPITULATION_THRESHOLD:
        return 0.50
    if vix >= _BEAR_THRESHOLD:
        return 0.80
    return 1.00


def strategy_label(regime: RiskRegime) -> str:
    return {
        RiskRegime.NORMAL: "NORMAL / FULL POSITIONING",
        RiskRegime.BEAR: "DEFENSIVE / GRADUATED POSITIONING ACTIVATED",
        RiskRegime.CAPITULATION: "CAPITULATION DISTRESSED REGIME / HIGH-QUALITY ANCHORS ONLY",
    }[regime]


def _is_capitulation_survivor(entry: dict) -> bool:
    """True if entry qualifies for CAPITULATION distressed alpha shortlist.

    Survivor criteria:
      beta ≤ 1.2            — applied only when the producer supplies beta;
                              neither the US pipeline (FACTOR_FIELDS) nor the
                              INTL profile emits it today, so the gate is
                              inert until beta lands in factors
      debt_to_equity ≤ 0.30 — same producer caveat; qualifies when present
      piotroski ≥ 0.70      — the only always-produced gate (US: normalized
                              cross-sectional; INTL: raw profile score)
    """
    factors = entry.get("factors", {})
    beta_raw = factors.get("beta") or factors.get("beta_30d")
    if beta_raw is not None and float(beta_raw) > _BETA_MAX:
        return False
    de_raw = factors.get("debt_to_equity")
    if de_raw is not None and float(de_raw) <= _DE_CEILING:
        return True
    return float(factors.get("quality_piotroski") or 0.0) >= _PIOTROSKI_FLOOR


def apply_capitulation_filter(entries: List[dict], vix: float) -> List[dict]:
    """Apply CAPITULATION DISTRESSED REGIME filter when VIX ≥ 30.

    Pure filter + badge pass — it does NOT multiply final_score. The regime
    multiplier is applied exactly once upstream: US entries by
    generate_top_lists._apply_vix_overlay, INTL entries by
    cook_toplists._normalize_intl_entry (both via vix_multiplier). Multiplying
    here again double-dampened US scores 0.25×/0.10× vs INTL 0.50×.

    All survivors are force-badged WATCHLIST — no new BUY signals during
    Panic/Crash. (Upstream dampening keeps survivor scores ≤ 0.50, below the
    0.60 TACTICAL BUY threshold, so the forced badge matches audit check B.)
    Callers (cook_toplists.py) must move survivors from top_buys_* into a
    watchlist key so the Discord embed does not render them as buy signals.
    Non-capitulation regime: returns entries unchanged.
    """
    if not is_panic(vix):
        return entries

    result = []
    for entry in entries:
        if not _is_capitulation_survivor(entry):
            continue
        e = dict(entry)
        e["badge"] = "WATCHLIST"
        e["_capitulation_survivor"] = True
        result.append(e)
    return result


# ── Market-regime nowcast (DISPLAY-ONLY) ──────────────────────────────────────
# A directional Bull/Euphoria/Bear read from VIX + trailing index momentum, used
# only to give the daily brief a market-state header. It is deliberately NOT
# wired into score_multiplier / vix_multiplier: the VIX RiskRegime remains the
# sole sizing lever, so a frothy or defensive nowcast never silently re-scales
# alpha. Thresholds are module constants (tunable; no hardcoded levels at call
# sites). Standard risk-on/off nowcast: complacent vol + strong trend = froth;
# elevated vol or negative trend = risk-off.

_EUPHORIA_VIX_MAX = 14.0    # complacent volatility ceiling for the froth signature
_EUPHORIA_MOM_MIN = 0.08    # +8% trailing-63d required on BOTH SPY and QQQ
_BULL_MOM_MIN = 0.02        # +2% trailing-63d SPY → risk-on trend intact
_BEAR_MOM_MAX = -0.05       # -5% trailing-63d SPY → risk-off


class MarketRegime(str, Enum):
    EUPHORIA = "EUPHORIA"
    BULL = "BULL"
    NEUTRAL = "NEUTRAL"
    BEAR = "BEAR"
    CAPITULATION = "CAPITULATION"


def _is_missing(x: object) -> bool:
    """True for None / NaN / non-numeric — a regime is never inferred from these."""
    if x is None or isinstance(x, bool):
        return True
    if not isinstance(x, (int, float)):
        return True
    return math.isnan(float(x))


def classify_market_regime(
    vix: float,
    spy_ret_63d: float | None,
    qqq_ret_63d: float | None = None,
) -> MarketRegime:
    """Directional market-state nowcast from VIX + 63-day index momentum.

    DISPLAY-ONLY — never feeds a score multiplier (see section note above).

    VIX is mandatory (it is always present in the brief); a regime is never
    guessed from momentum alone. The stressed regimes are a volatility fact and
    hold without momentum, but EUPHORIA/BULL require the SPY/QQQ trend.

    Ladder (first match wins):
      NEUTRAL      : VIX missing/invalid (insufficient evidence)
      CAPITULATION : VIX ≥ CAPITULATION_THRESHOLD   (parity with RiskRegime)
      (momentum missing) BEAR if VIX ≥ BEAR_THRESHOLD else NEUTRAL
      EUPHORIA     : VIX < _EUPHORIA_VIX_MAX and SPY & QQQ 63d ≥ _EUPHORIA_MOM_MIN
      BEAR         : VIX ≥ BEAR_THRESHOLD or SPY 63d ≤ _BEAR_MOM_MAX
      BULL         : SPY 63d ≥ _BULL_MOM_MIN
      NEUTRAL      : otherwise
    """
    if _is_missing(vix) or float(vix) < 0:
        return MarketRegime.NEUTRAL
    v = float(vix)
    if v >= _CAPITULATION_THRESHOLD:
        return MarketRegime.CAPITULATION
    if _is_missing(spy_ret_63d):
        # No trend data: stressed regime still holds on vol; never fabricate BULL.
        return MarketRegime.BEAR if v >= _BEAR_THRESHOLD else MarketRegime.NEUTRAL
    spy = float(spy_ret_63d)
    # QQQ confirms the froth signature; fall back to SPY when QQQ is unavailable.
    qqq = spy if _is_missing(qqq_ret_63d) else float(qqq_ret_63d)
    if v < _EUPHORIA_VIX_MAX and spy >= _EUPHORIA_MOM_MIN and qqq >= _EUPHORIA_MOM_MIN:
        return MarketRegime.EUPHORIA
    if v >= _BEAR_THRESHOLD or spy <= _BEAR_MOM_MAX:
        return MarketRegime.BEAR
    if spy >= _BULL_MOM_MIN:
        return MarketRegime.BULL
    return MarketRegime.NEUTRAL


_MARKET_REGIME_LABEL: dict = {
    MarketRegime.EUPHORIA:     ("🟣", "risk-on extreme — froth / mean-reversion caution"),
    MarketRegime.BULL:         ("🟢", "risk-on — trend intact"),
    MarketRegime.NEUTRAL:      ("⚪", "mixed — range-bound"),
    MarketRegime.BEAR:         ("🟠", "risk-off — defensive"),
    MarketRegime.CAPITULATION: ("🔴", "distressed — high-quality anchors only"),
}


def market_regime_label(regime: MarketRegime) -> tuple[str, str]:
    """(emoji, one-line blurb) for the daily-brief market-regime banner."""
    return _MARKET_REGIME_LABEL[regime]
