"""regime/credit_regime_detector.py
Credit-market stress regime detection — standalone signal for the ensemble.

Signal sources (free / yfinance, no paid API):
  HY proxy  : JNK  (iShares High Yield Corp Bond ETF)
  IG proxy  : LQD  (iShares Investment Grade Corp Bond ETF)
  MOVE proxy: ^MOVE (ICE BofA MOVE Index) — optional, skipped on failure

Features per series:
  z_60  — rolling 60-day z-score of log-price (negative price trend = stress)
  chg5  — 5-day log-return negated (falling price = positive stress)
  chg20 — 20-day log-return negated
  slope_20 — OLS slope of last 20 log-prices, negated & normalised

Credit Stress Score:
  credit_raw = 0.35*z_hy + 0.25*z_ig + 0.20*slope_hy_norm
             + 0.10*hy_ig_ratio_norm + 0.10*z_move
  credit_score = sigmoid-clamp of weighted mean ∈ [0, 1]

Regime thresholds:
  NORMAL   score < 0.40
  CAUTION  0.40 <= score < 0.60
  STRESS   0.60 <= score < 0.75
  CRISIS   score >= 0.75

Asymmetric persistence filter:
  Enter STRESS  : 2 consecutive STRESS signals
  Enter CRISIS  : 3 consecutive CRISIS signals
  Exit to lower : 1 signal sufficient  (fast de-escalation)

Engle (2003 Nobel) — volatility and credit spreads are co-integrated
during systemic stress; combining both signals reduces false-positive
regime switches present in VIX-only detection.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Tickers (env-overridable for testing) ─────────────────────────────────────
_HY_TICKER   = os.getenv("CREDIT_HY_TICKER",   "JNK")
_IG_TICKER   = os.getenv("CREDIT_IG_TICKER",   "LQD")
_MOVE_TICKER = os.getenv("CREDIT_MOVE_TICKER",  "^MOVE")

# ── Regime enum ───────────────────────────────────────────────────────────────

class CreditRegime(str, Enum):
    """Credit-market stress regimes, ordered by severity."""
    NORMAL  = "normal"
    CAUTION = "caution"
    STRESS  = "stress"
    CRISIS  = "crisis"


# Severity ordering (0 = calm, 3 = crisis) — used by persistence filter
_SEVERITY: Dict[CreditRegime, int] = {
    CreditRegime.NORMAL:  0,
    CreditRegime.CAUTION: 1,
    CreditRegime.STRESS:  2,
    CreditRegime.CRISIS:  3,
}

# Minimum consecutive signals required to commit to each regime level
# (asymmetric: upgrades are slow, downgrades are fast)
_PERSIST_REQUIRED: Dict[CreditRegime, int] = {
    CreditRegime.NORMAL:  1,   # immediate de-escalation
    CreditRegime.CAUTION: 1,   # immediate de-escalation
    CreditRegime.STRESS:  2,   # 2 consecutive STRESS signals
    CreditRegime.CRISIS:  3,   # 3 consecutive CRISIS signals
}

# ── Feature container ─────────────────────────────────────────────────────────

@dataclass
class CreditFeatures:
    """Compressed credit-market features for a single day.

    All signed so that POSITIVE values indicate MORE stress.

    Black-Scholes (1997 Nobel) — credit spreads embed forward-looking
    default probabilities; these features extract that information
    in a model-free way.
    """
    z_hy: Optional[float] = None              # HY log-price z-score (negated)
    z_ig: Optional[float] = None              # IG log-price z-score (negated)
    slope_hy_norm: Optional[float] = None     # HY OLS slope, negated & normalised
    hy_ig_ratio_norm: Optional[float] = None  # HY/IG ratio z-score (negated)
    z_move: Optional[float] = None            # MOVE z-score (positive = stress)
    n_sources: int = 0
    source_flags: Dict[str, bool] = field(default_factory=dict)


# ── Pure math helpers ─────────────────────────────────────────────────────────

def _zscore_series(series: pd.Series, window: int = 60) -> pd.Series:
    """Rolling z-score.

    Markowitz (1952 Nobel) — standardisation makes heterogeneous
    time-series comparable for portfolio (here: ensemble) combination.

    z_t = (x_t - mu_{t-window:t}) / sigma_{t-window:t}
    """
    mean = series.rolling(window, min_periods=window // 2).mean()
    std  = series.rolling(window, min_periods=window // 2).std()
    return (series - mean) / (std + 1e-8)


def _ols_slope(series: pd.Series) -> float:
    """OLS slope of the series vs a linear trend (per-period).

    Granger (2003 Nobel) — slope of log-price encodes directional momentum
    in a causal-regression framework.

    Returns NaN if series is too short.
    """
    n = len(series)
    if n < 2:
        return float("nan")
    x = np.arange(n, dtype=float)
    y = series.values.astype(float)
    mask = np.isfinite(y)
    if mask.sum() < 2:
        return float("nan")
    x, y = x[mask], y[mask]
    coeffs = np.polyfit(x, y, 1)
    return float(coeffs[0])  # slope in log-price-per-day units


def _slope_norm(series: pd.Series, window: int = 20) -> float:
    """Compute OLS slope of last `window` obs, normalised by rolling std.

    Returns value ~ in [-3, 3] units; negative = price falling = stress.
    Negated so that POSITIVE value = stress direction.
    """
    if len(series) < window:
        return float("nan")
    sub = series.iloc[-window:]
    slope = _ols_slope(sub)
    if not np.isfinite(slope):
        return float("nan")
    # Normalise by typical magnitude (std of 20-day log-returns)
    log_ret_std = float(sub.diff().std()) + 1e-8
    # slope is in log-price/day; log_ret_std is in log-price/day
    normalised = slope / (3.0 * log_ret_std)  # ÷3σ so ±1 = ±3σ slope
    return float(-np.clip(normalised, -3.0, 3.0))   # negate: falling = positive


def _component_to_01(x: float) -> float:
    """Soft-map component value ∈ (-∞,+∞) to probability ∈ [0, 1].

    Uses a rescaled sigmoid centred at 0, with scale=2 so that:
      x = -2  → ~0.12   (low stress)
      x =  0  → 0.50    (neutral)
      x = +2  → ~0.88   (high stress)

    Clipped strictly to [0, 1].
    """
    return float(np.clip(1.0 / (1.0 + np.exp(-x / 2.0 * 3.0)), 0.0, 1.0))


# ── Score & classification ────────────────────────────────────────────────────

def compute_credit_stress_score(features: CreditFeatures) -> float:
    """Return credit stress score in [0, 1].

    Weights (total 1.0 when all sources present):
      HY z-score         0.35
      IG z-score         0.25
      HY OLS slope       0.20
      HY/IG ratio        0.10
      MOVE z-score       0.10

    When a component is unavailable (None or NaN), its weight is
    redistributed among the available components so the score is
    always comparable regardless of data availability.

    Returns 0.5 (neutral/caution boundary) when NO data is available.

    Engle (2003 Nobel) — credit volatility is jointly determined with
    equity volatility; a scalar composite score captures this joint signal.
    """
    _WEIGHTS = [
        ("z_hy",            0.35),
        ("z_ig",            0.25),
        ("slope_hy_norm",   0.20),
        ("hy_ig_ratio_norm",0.10),
        ("z_move",          0.10),
    ]

    total_weight = 0.0
    weighted_sum = 0.0

    for attr, w in _WEIGHTS:
        val = getattr(features, attr)
        if val is not None and np.isfinite(val):
            weighted_sum += w * _component_to_01(val)
            total_weight += w

    if total_weight == 0.0:
        log.warning("[CREDIT] No features available — returning neutral score 0.5")
        return 0.5

    score = weighted_sum / total_weight
    return float(np.clip(score, 0.0, 1.0))


def classify_credit_regime(score: float) -> CreditRegime:
    """Map credit stress score to CreditRegime.

    Thresholds:
      CRISIS   score >= 0.75
      STRESS   0.60 <= score < 0.75
      CAUTION  0.40 <= score < 0.60
      NORMAL   score < 0.40

    Lucas (1995 Nobel) — regime thresholds should be set at economically
    meaningful discontinuities, not arbitrary quantiles.
    """
    if score >= 0.75:
        return CreditRegime.CRISIS
    if score >= 0.60:
        return CreditRegime.STRESS
    if score >= 0.40:
        return CreditRegime.CAUTION
    return CreditRegime.NORMAL


# ── Asymmetric persistence filter ─────────────────────────────────────────────

def apply_credit_persistence_filter(
    regimes: List[CreditRegime],
) -> List[CreditRegime]:
    """Asymmetric persistence filter for credit regimes.

    Escalation requires sustained signals; de-escalation is immediate.

    Rules (consecutive signals required to COMMIT):
      NORMAL  : 1   (fast de-escalation from any level)
      CAUTION : 1
      STRESS  : 2   (require 2 consecutive STRESS signals)
      CRISIS  : 3   (require 3 consecutive CRISIS signals)

    Lucas (1995 Nobel) — asymmetric adjustment costs justify asymmetric
    persistence: entering a stress positioning is costly, exiting is free.

    Args:
        regimes: Raw daily signal list (oldest first).

    Returns:
        List of committed CreditRegime labels, same length as input.
    """
    if not regimes:
        return []

    committed: CreditRegime = regimes[0]
    candidate: CreditRegime = regimes[0]
    count: int = 1
    result: List[CreditRegime] = [committed]

    for regime in regimes[1:]:
        if regime == candidate:
            count += 1
        else:
            candidate = regime
            count = 1

        n_required = _PERSIST_REQUIRED[candidate]
        going_up = _SEVERITY[candidate] > _SEVERITY[committed]

        if going_up:
            if count >= n_required:
                committed = candidate
        else:
            # De-escalation: 1 signal always sufficient
            committed = candidate

        result.append(committed)

    return result


# ── Ensemble bridge: credit score → VIX probability vector ───────────────────

def credit_score_to_vix_proba(score: float) -> np.ndarray:
    """Map credit stress score to a soft probability vector over VIX regimes.

    Uses the effective-VIX mapping:
      effective_vix = 8 + score * (65 - 8)

    Then delegates to vix_proba() for the soft probability distribution.
    This bridges the credit signal into the existing VIX ensemble machinery.

    Args:
        score: Credit stress score ∈ [0, 1].

    Returns:
        Probability vector (length 6) over ["Crash","Panic","Bear",
        "Neutral","Bull","Euphoria"], sums to 1.0.
    """
    # Import here to avoid circular dependency at module load
    from regime.regime_detector import vix_proba as _vix_proba
    effective_vix = 8.0 + float(np.clip(score, 0.0, 1.0)) * (65.0 - 8.0)
    return _vix_proba(effective_vix)


# ── Override rules ────────────────────────────────────────────────────────────

def apply_credit_overrides(
    ensemble_label: str,
    credit_regime: CreditRegime,
    latest_vix: Optional[float] = None,
) -> str:
    """Apply credit-based override rules to the ensemble regime label.

    Rules:
      1. CRISIS  → output must be at least "Bear" (cannot output Neutral/Bull/Euphoria)
      2. STRESS + VIX < 20 → force at least "Bear" (early-warning: credit stress
         while equity vol is still low signals early systemic deterioration)

    Merton (1997 Nobel) — credit and equity markets are structurally linked;
    credit-market stress is a leading indicator of equity-volatility regimes.

    Args:
        ensemble_label: Raw ensemble output label.
        credit_regime:  Committed credit regime.
        latest_vix:     Most recent VIX level (for rule 2).

    Returns:
        Possibly overridden regime label.
    """
    _SEVERITY_VIX = {
        "Euphoria": 0, "Bull": 1, "Neutral": 2,
        "Bear": 3,    "Panic": 4, "Crash": 5,
    }
    _MINIMUM_LABEL = "Bear"  # floor for crisis / early-warning

    current_sev = _SEVERITY_VIX.get(ensemble_label, 2)
    floor_sev   = _SEVERITY_VIX[_MINIMUM_LABEL]

    if credit_regime == CreditRegime.CRISIS:
        if current_sev < floor_sev:
            log.info(
                "[CREDIT OVERRIDE] CRISIS → forcing %s → %s",
                ensemble_label, _MINIMUM_LABEL,
            )
            return _MINIMUM_LABEL

    if credit_regime == CreditRegime.STRESS and latest_vix is not None:
        if latest_vix < 20.0 and current_sev < floor_sev:
            log.info(
                "[CREDIT OVERRIDE] STRESS + VIX=%.1f < 20 → early warning → %s",
                latest_vix, _MINIMUM_LABEL,
            )
            return _MINIMUM_LABEL

    return ensemble_label


# ── Data fetching helpers ─────────────────────────────────────────────────────

def _safe_yf_download(ticker: str, period: str = "2y") -> Optional[pd.Series]:
    """Download a single ticker's adjusted close from yfinance.

    Returns None (with warning) on any error.
    """
    try:
        import yfinance as yf
        df = yf.download(ticker, period=period, interval="1d",
                         progress=False, auto_adjust=True)
        if df is None or df.empty:
            log.warning("[CREDIT] %s: empty download", ticker)
            return None
        col = "Close"
        if col not in df.columns:
            log.warning("[CREDIT] %s: 'Close' column missing", ticker)
            return None
        series = df[col].squeeze().dropna()
        if len(series) < 30:
            log.warning("[CREDIT] %s: only %d rows — insufficient", ticker, len(series))
            return None
        return series
    except Exception as exc:
        log.warning("[CREDIT] %s download failed: %s", ticker, exc)
        return None


# ── Main detector class ───────────────────────────────────────────────────────

class CreditRegimeDetector:
    """Credit-market stress regime detector.

    Uses JNK (HY), LQD (IG), and optionally ^MOVE as free market proxies
    to compute a daily Credit Stress Score and map it to a CreditRegime label.

    Designed for use as an optional fourth signal in RegimeDetector ensemble.

    Merton (1997 Nobel) — structural credit models link default probability
    to equity volatility; this detector operationalises that link with
    observable ETF prices as spread proxies.
    """

    def __init__(
        self,
        hy_ticker:   str = _HY_TICKER,
        ig_ticker:   str = _IG_TICKER,
        move_ticker: str = _MOVE_TICKER,
        zscore_window: int = 60,
        slope_window:  int = 20,
    ) -> None:
        self.hy_ticker    = hy_ticker
        self.ig_ticker    = ig_ticker
        self.move_ticker  = move_ticker
        self.zscore_window = zscore_window
        self.slope_window  = slope_window

    # ── Price → log-price helpers ─────────────────────────────────────────────

    @staticmethod
    def _log_price(series: pd.Series) -> pd.Series:
        return np.log(series.clip(lower=1e-8))

    # ── Feature computation from price series ─────────────────────────────────

    def compute_features_from_prices(
        self,
        hy_prices:   Optional[pd.Series] = None,
        ig_prices:   Optional[pd.Series] = None,
        move_prices: Optional[pd.Series] = None,
    ) -> CreditFeatures:
        """Compute the latest CreditFeatures snapshot from raw price series.

        All inputs are optional; missing series produce None features.
        Score is always computable as long as at least one series is available.

        Args:
            hy_prices:   Daily close prices for HY proxy (e.g. JNK).
            ig_prices:   Daily close prices for IG proxy (e.g. LQD).
            move_prices: Daily close for MOVE proxy.

        Returns:
            CreditFeatures with signed components (positive = more stress).
        """
        feats = CreditFeatures()
        sources = 0

        # ── HY features ───────────────────────────────────────────────────────
        if hy_prices is not None and len(hy_prices) >= self.zscore_window // 2:
            lp = self._log_price(hy_prices)
            z  = _zscore_series(lp, self.zscore_window)
            feats.z_hy = float(-z.iloc[-1]) if np.isfinite(z.iloc[-1]) else None
            feats.slope_hy_norm = _slope_norm(lp, self.slope_window)
            feats.source_flags["hy"] = True
            sources += 1
        else:
            feats.source_flags["hy"] = False

        # ── IG features ───────────────────────────────────────────────────────
        if ig_prices is not None and len(ig_prices) >= self.zscore_window // 2:
            lp = self._log_price(ig_prices)
            z  = _zscore_series(lp, self.zscore_window)
            feats.z_ig = float(-z.iloc[-1]) if np.isfinite(z.iloc[-1]) else None
            feats.source_flags["ig"] = True
            sources += 1
        else:
            feats.source_flags["ig"] = False

        # ── HY/IG ratio feature ───────────────────────────────────────────────
        if feats.z_hy is not None and feats.z_ig is not None:
            # Ratio stress: HY spreads widening faster than IG
            # z_hy and z_ig are both positively signed for stress
            # Ratio stress = z_hy - z_ig (excess HY stress vs IG)
            feats.hy_ig_ratio_norm = float(feats.z_hy - feats.z_ig)
            feats.source_flags["ratio"] = True

        # ── MOVE features ─────────────────────────────────────────────────────
        if move_prices is not None and len(move_prices) >= self.zscore_window // 2:
            z = _zscore_series(move_prices, self.zscore_window)
            # MOVE rising = stress (positive z → high stress)
            feats.z_move = float(z.iloc[-1]) if np.isfinite(z.iloc[-1]) else None
            feats.source_flags["move"] = True
            sources += 1
        else:
            feats.source_flags["move"] = False

        feats.n_sources = sources
        return feats

    def compute_features_series(
        self,
        hy_prices:   Optional[pd.Series] = None,
        ig_prices:   Optional[pd.Series] = None,
        move_prices: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Compute daily credit stress scores for a full historical price series.

        Returns a pd.Series of floats ∈ [0, 1] aligned to the hy_prices index
        (or ig_prices if hy is unavailable).

        Args:
            hy_prices:   Daily close prices for HY proxy.
            ig_prices:   Daily close prices for IG proxy.
            move_prices: Daily close for MOVE proxy.

        Returns:
            pd.Series of credit stress scores, DatetimeIndex.
        """
        # Determine index from available series
        ref = hy_prices if hy_prices is not None else ig_prices
        if ref is None:
            raise ValueError("At least one price series (hy or ig) is required")

        idx = ref.index
        scores = []

        # Pre-compute z-score series for efficiency
        lp_hy   = self._log_price(hy_prices)   if hy_prices   is not None else None
        lp_ig   = self._log_price(ig_prices)   if ig_prices   is not None else None

        z_hy_full   = _zscore_series(lp_hy,   self.zscore_window) if lp_hy   is not None else None
        z_ig_full   = _zscore_series(lp_ig,   self.zscore_window) if lp_ig   is not None else None
        z_move_full = _zscore_series(move_prices, self.zscore_window) if move_prices is not None else None

        for i in range(len(idx)):
            feats = CreditFeatures()

            if z_hy_full is not None and i < len(z_hy_full):
                v = float(z_hy_full.iloc[i])
                feats.z_hy = -v if np.isfinite(v) else None
                # slope: use window ending at i
                start = max(0, i - self.slope_window + 1)
                if lp_hy is not None and (i - start + 1) >= 3:
                    feats.slope_hy_norm = _slope_norm(lp_hy.iloc[start:i+1], self.slope_window)

            if z_ig_full is not None and i < len(z_ig_full):
                v = float(z_ig_full.iloc[i])
                feats.z_ig = -v if np.isfinite(v) else None

            if feats.z_hy is not None and feats.z_ig is not None:
                feats.hy_ig_ratio_norm = feats.z_hy - feats.z_ig

            if z_move_full is not None and i < len(z_move_full):
                v = float(z_move_full.iloc[i])
                feats.z_move = v if np.isfinite(v) else None

            scores.append(compute_credit_stress_score(feats))

        return pd.Series(scores, index=idx, name="credit_score")

    def regime_series(
        self,
        hy_prices:    Optional[pd.Series] = None,
        ig_prices:    Optional[pd.Series] = None,
        move_prices:  Optional[pd.Series] = None,
        apply_filter: bool = True,
    ) -> pd.Series:
        """Compute daily CreditRegime labels over a historical price series.

        Args:
            hy_prices:    HY proxy daily prices.
            ig_prices:    IG proxy daily prices.
            move_prices:  MOVE proxy daily prices.
            apply_filter: Apply asymmetric persistence filter (default True).

        Returns:
            pd.Series of CreditRegime values.
        """
        scores = self.compute_features_series(hy_prices, ig_prices, move_prices)
        raw_regimes = [classify_credit_regime(s) for s in scores]
        if apply_filter:
            raw_regimes = apply_credit_persistence_filter(raw_regimes)
        return pd.Series(raw_regimes, index=scores.index, name="credit_regime")

    def predict_latest(
        self,
        window: int = 300,
    ) -> Tuple[CreditRegime, float, CreditFeatures]:
        """Fetch latest market data and return current credit regime.

        Falls back gracefully if any ticker is unavailable.

        Args:
            window: Lookback period in trading days (approximate).

        Returns:
            Tuple of (CreditRegime, stress_score, CreditFeatures).
        """
        period = f"{max(window // 252 + 1, 2)}y"

        hy_prices   = _safe_yf_download(self.hy_ticker,   period)
        ig_prices   = _safe_yf_download(self.ig_ticker,   period)
        move_prices = _safe_yf_download(self.move_ticker, period)

        if hy_prices is None and ig_prices is None:
            log.error(
                "[CREDIT] Both HY (%s) and IG (%s) unavailable — returning NORMAL",
                self.hy_ticker, self.ig_ticker,
            )
            return CreditRegime.NORMAL, 0.5, CreditFeatures()

        feats = self.compute_features_from_prices(hy_prices, ig_prices, move_prices)
        score = compute_credit_stress_score(feats)
        regime = classify_credit_regime(score)

        log.info(
            "[CREDIT] score=%.3f  regime=%s  sources=%d  flags=%s",
            score, regime.value, feats.n_sources, feats.source_flags,
        )
        return regime, score, feats
