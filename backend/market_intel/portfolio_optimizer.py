# Path: backend/market_intel/portfolio_optimizer.py
"""MVO portfolio optimizer — no qlib dependency.

Imported by generate_top_lists.py to add portfolio_weight to top_lists.json.

Fallback chain (always produces a valid weight vector):
    MVO (SLSQP) → risk parity → score-proportional

VIX vol-targeting applied post-optimization (scales weights down, never up).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.optimize import minimize

log = logging.getLogger(__name__)

_COVARIANCE_PATH = Path("logs/covariance_matrix.npz")
_IC_REPORT_PATH = Path("research/ic_report.json")

_MAX_POSITION = 0.10        # max 10% per position
_MAX_SECTOR_WEIGHT = 0.30   # max 30% per sector
_MAX_TURNOVER = 0.20        # max 20% portfolio turnover

TARGET_VOL: dict[str, float] = {
    "normal": 0.15,   # VIX < 25
    "bear":   0.10,   # VIX 25–30
    "panic":  0.05,   # VIX >= 30
}


def _vix_regime(vix: float) -> str:
    if vix >= 30:
        return "panic"
    if vix >= 25:
        return "bear"
    return "normal"


def _score_proportional(scores: np.ndarray) -> np.ndarray:
    w = np.clip(scores, 0.0, None)
    total = w.sum()
    if total <= 0:
        n = len(scores)
        return np.ones(n) / n
    w = w / total
    w = np.clip(w, 0.0, _MAX_POSITION)
    s = w.sum()
    return w / s if s > 0 else np.ones(len(scores)) / len(scores)


def _risk_parity(cov: np.ndarray) -> np.ndarray:
    vols = np.sqrt(np.diag(cov))
    vols = np.where(vols <= 1e-8, 1e-8, vols)
    w = 1.0 / vols
    return w / w.sum()


def _mvo(
    expected_returns: np.ndarray,
    cov: np.ndarray,
    sector_ids: list[int],
    n_sectors: int,
    prev_weights: Optional[np.ndarray],
) -> np.ndarray:
    n = len(expected_returns)

    def neg_sharpe(w: np.ndarray) -> float:
        ret = float(w @ expected_returns)
        vol = float(np.sqrt(w @ cov @ w + 1e-10))
        return -ret / vol

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]

    for s in range(n_sectors):
        mask = np.array([1.0 if sector_ids[i] == s else 0.0 for i in range(n)])
        if mask.sum() > 0:
            constraints.append(
                {"type": "ineq", "fun": lambda w, m=mask: _MAX_SECTOR_WEIGHT - float((w * m).sum())}
            )

    if prev_weights is not None and len(prev_weights) == n:
        constraints.append(
            {"type": "ineq", "fun": lambda w: _MAX_TURNOVER - float(np.abs(w - prev_weights).sum())}
        )

    bounds = [(0.0, _MAX_POSITION)] * n
    w0 = np.ones(n) / n

    result = minimize(
        neg_sharpe, w0, method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )
    if not result.success:
        raise RuntimeError(f"MVO did not converge: {result.message}")
    return np.array(result.x)


def _calibrate_scores_to_returns(
    scores: np.ndarray,
    lower_return: float = 0.04,
    upper_return: float = 0.18,
) -> np.ndarray:
    """Convert ordinal factor scores [0, 1] to cardinal expected annual return estimates.

    Winsorize → Z-score → linear map to [+4%, +18%] annualized return bounds.
    Direct MVO input of raw scores implies linear return proportionality which
    distorts the efficient frontier. [Grinold & Kahn, IR = IC × √BR]
    """
    lo = float(np.percentile(scores, 5))
    hi = float(np.percentile(scores, 95))
    clipped = np.clip(scores, lo, hi)
    mu = clipped.mean()
    sigma = clipped.std() + 1e-8
    z = (clipped - mu) / sigma
    mid   = (upper_return + lower_return) / 2.0
    scale = (upper_return - lower_return) / 6.0   # ±3σ spans full range
    return np.clip(mid + z * scale, lower_return, upper_return)


def build_async_covariance(price_series: dict[str, list[float]]) -> Optional[np.ndarray]:
    """Build covariance matrix using 2-day rolling log-return window for global assets.

    Smooths non-overlapping timezone trading deltas (Tokyo/London/New York).
    log_ret_2d[t] = log(price[t] / price[t-2] + ε).
    Returns None if fewer than 5 observations.
    """
    if not price_series:
        return None
    tickers = list(price_series.keys())
    min_len = min(len(v) for v in price_series.values())
    if min_len < 5:
        return None

    returns_matrix = []
    for t in tickers:
        prices = np.array(price_series[t][-min_len:], dtype=float)
        log_ret_2d = np.log(prices[2:] / prices[:-2] + 1e-10)
        returns_matrix.append(log_ret_2d)

    R = np.array(returns_matrix)   # (n_tickers, n_periods − 2)
    return np.cov(R)


_LARGE_CAP_THRESHOLD = 10_000_000_000   # $10B


def build_large_cap_anchors(entries: list[dict]) -> list[dict]:
    """Equal-weight Structural Core Anchor allocation for large-cap entries (>$10B).

    These bypass MVO to preserve mid/small-cap efficient frontier integrity.
    """
    large = [e for e in entries if (e.get("market_cap") or 0) > _LARGE_CAP_THRESHOLD]
    if not large:
        return []
    alloc = round(1.0 / len(large), 4)
    return [
        {
            "ticker":       e["ticker"],
            "allocation":   alloc,
            "final_score":  e.get("final_score", 0.0),
            "price_target": e.get("price_target"),
            "exit_anchors": e.get("exit_anchors", {}),
            "tier":         "LARGE_CAP_ANCHOR",
        }
        for e in large
    ]


def adv_capacity_ceiling(
    current_ceiling: float,
    adv_20d_usd: float,
    portfolio_value_usd: float,
    max_adv_pct: float = 0.03,
) -> float:
    """Return min(current_ceiling, max_adv_pct × ADV_20d / portfolio_value).

    Prevents illiquid small-cap allocations exceeding 3% of trailing 20-day ADV
    (standard hedge-fund liquidity capacity constraint).
    """
    if portfolio_value_usd <= 0 or adv_20d_usd <= 0:
        return current_ceiling
    adv_ceiling = (max_adv_pct * adv_20d_usd) / portfolio_value_usd
    return min(current_ceiling, adv_ceiling)


def _min_variance(
    cov: np.ndarray,
    sector_ids: list[int],
    n_sectors: int,
    prev_weights: Optional[np.ndarray],
    position_floor: float,
    position_ceiling: float,
) -> np.ndarray:
    """Min-variance optimizer (SLSQP). Suitable for small-cap illiquid pools."""
    n = cov.shape[0]

    def portfolio_variance(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    constraints: list[dict] = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    for s in range(n_sectors):
        mask = np.array([1.0 if sector_ids[i] == s else 0.0 for i in range(n)])
        if mask.sum() > 0:
            constraints.append(
                {"type": "ineq", "fun": lambda w, m=mask: _MAX_SECTOR_WEIGHT - float((w * m).sum())}
            )
    if prev_weights is not None and len(prev_weights) == n:
        constraints.append(
            {"type": "ineq", "fun": lambda w: _MAX_TURNOVER - float(np.abs(w - prev_weights).sum())}
        )

    bounds = [(position_floor, position_ceiling)] * n
    w0 = np.ones(n) / n

    result = minimize(
        portfolio_variance, w0, method="SLSQP",
        bounds=bounds, constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-9},
    )
    if not result.success:
        raise RuntimeError(f"Min variance did not converge: {result.message}")
    return np.array(result.x)


def _load_covariance(tickers: list[str]) -> Optional[np.ndarray]:
    if not _COVARIANCE_PATH.exists():
        log.debug("No covariance matrix at %s — will use identity fallback", _COVARIANCE_PATH)
        return None
    try:
        data = np.load(_COVARIANCE_PATH, allow_pickle=True)
        stored_tickers: list[str] = list(data["tickers"])
        full_cov: np.ndarray = data["covariance"]
        idx = [stored_tickers.index(t) for t in tickers if t in stored_tickers]
        if len(idx) != len(tickers):
            log.warning("Covariance ticker mismatch: expected %d, found %d", len(tickers), len(idx))
            return None
        return full_cov[np.ix_(idx, idx)]
    except Exception as exc:
        log.warning("Failed to load covariance matrix: %s", exc)
        return None


def _ic_estimate() -> float:
    try:
        data = json.loads(_IC_REPORT_PATH.read_text())
        ics = [v["mean_ic"] for v in data.values() if isinstance(v, dict) and "mean_ic" in v]
        return max(0.01, float(np.mean(ics))) if ics else 0.03
    except Exception:
        return 0.03  # conservative default


def _clip_sector_weights(
    weights: np.ndarray,
    sector_ids: list[int],
    n_sectors: int,
) -> np.ndarray:
    """Iteratively scale down over-weight sectors until all sectors ≤ _MAX_SECTOR_WEIGHT.

    Excess weight within a sector is removed proportionally from that sector's
    positions. The total portfolio weight may decrease below 1.0 (cash buffer).
    """
    w = weights.copy()
    for s in range(n_sectors):
        mask = np.array([i for i, sid in enumerate(sector_ids) if sid == s])
        if mask.size == 0:
            continue
        sector_total = w[mask].sum()
        if sector_total > _MAX_SECTOR_WEIGHT + 1e-8:
            scale = _MAX_SECTOR_WEIGHT / sector_total
            w[mask] *= scale
    return w


def run_optimizer(
    tickers: list[str],
    scores: list[float],
    sectors: list[str],
    vix: float,
    prev_weights: Optional[dict[str, float]] = None,
    mode: str = "sharpe",
    position_floor: float = 0.0,
    position_ceiling: float = _MAX_POSITION,
    price_series: Optional[dict[str, list[float]]] = None,
    adv_20d_map: Optional[dict[str, float]] = None,
    portfolio_value_usd: Optional[float] = None,
) -> tuple[dict[str, float], str]:
    """Compute portfolio weights for the given tickers.

    Args:
        tickers:            Ticker list (top-20 by composite score).
        scores:             Composite scores, same order as tickers.
        sectors:            Sector strings, same order as tickers.
        vix:                Current VIX level for vol-targeting.
        prev_weights:       Previous weight dict (ticker → weight) for turnover control.
        mode:               "sharpe" (Sharpe-Max MVO) or "min_variance" (Min-Var MVO).
        position_floor:     Minimum weight per position (default 0.0).
        position_ceiling:   Maximum weight per position (default _MAX_POSITION).
        price_series:       {ticker: [price, ...]} for async 2-day covariance estimation.
        adv_20d_map:        {ticker: adv_usd} for ADV liquidity gate (small-cap only).
        portfolio_value_usd: Portfolio value for ADV gate computation.

    Returns:
        (weights_dict, method_used)
        method_used in {"sharpe", "min_variance", "risk_parity", "score_proportional"}
    """
    n = len(tickers)
    scores_arr = np.array(scores, dtype=float)

    unique_sectors = sorted(set(sectors))
    sector_map = {s: i for i, s in enumerate(unique_sectors)}
    sector_ids = [sector_map[s] for s in sectors]

    # Use async 2-day cov if price_series provided, else load from file
    cov: Optional[np.ndarray] = None
    if price_series:
        cov = build_async_covariance({t: price_series[t] for t in tickers if t in price_series})
        if cov is not None and cov.shape != (n, n):
            cov = None
    if cov is None:
        cov = _load_covariance(tickers)
    if cov is None:
        cov = np.eye(n) * 0.04  # 20% annual vol fallback

    # Calibrate ordinal factor scores to cardinal expected return estimates
    expected_returns = _calibrate_scores_to_returns(scores_arr)

    prev_arr = (
        np.array([prev_weights.get(t, 0.0) for t in tickers])
        if prev_weights else None
    )

    method = mode if mode in {"sharpe", "min_variance"} else "sharpe"
    try:
        if method == "min_variance":
            weights = _min_variance(
                cov, sector_ids, len(unique_sectors), prev_arr,
                position_floor=position_floor, position_ceiling=position_ceiling,
            )
        else:
            weights = _mvo(expected_returns, cov, sector_ids, len(unique_sectors), prev_arr)
    except Exception as exc:
        log.warning("%s failed (%s), trying risk parity", method, exc)
        method = "risk_parity"
        try:
            weights = _risk_parity(cov)
        except Exception as exc2:
            log.warning("Risk parity failed (%s), using score-proportional", exc2)
            method = "score_proportional"
            weights = _score_proportional(scores_arr)

    # Sector cap enforcement — applied to all methods
    weights = _clip_sector_weights(weights, sector_ids, len(unique_sectors))

    # ADV liquidity gate — applied after sector cap for small-cap pools
    if adv_20d_map and portfolio_value_usd:
        for i, t in enumerate(tickers):
            adv = adv_20d_map.get(t)
            if adv:
                cap = adv_capacity_ceiling(position_ceiling, adv, portfolio_value_usd)
                weights[i] = min(weights[i], cap)
        total = weights.sum()
        if total > 1e-8:
            weights = weights / total

    # VIX vol-targeting — scale down only, never up
    regime_key = _vix_regime(vix)
    target_vol = TARGET_VOL[regime_key]
    port_vol = float(np.sqrt(weights @ cov @ weights + 1e-10))
    if port_vol > 1e-8:
        scale = min(1.0, target_vol / port_vol)
        weights = weights * scale

    return {t: float(w) for t, w in zip(tickers, weights)}, method
