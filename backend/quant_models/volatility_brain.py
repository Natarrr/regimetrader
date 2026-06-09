# Path: backend/quant_models/volatility_brain.py
"""GJR-GARCH volatility regime model.

Theory:
    Glosten, Jagannathan & Runkle (1993), "On the Relation Between the Expected
    Value and the Volatility of the Nominal Excess Return on Stocks",
    Journal of Finance 48(5) pp. 1779-1801.

    Persistence = α + β + γ/2 where γ captures the asymmetric leverage effect.
    Persistence ≥ 0.98 ≈ integrated GARCH (IGARCH) — volatility shocks are permanent.
    This is the Minsky condition #1 trigger.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_MIN_OBS = 252   # 1 year minimum for meaningful GARCH fit


def fit_gjr_garch(log_returns: pd.Series) -> dict:
    """Fit a GJR-GARCH(1,1) model on *log_returns*.

    Returns a dict with:
        persistence   — α + β + γ/2 (Engle & Ng 1993)
        omega, alpha, beta, gamma — estimated parameters
        status        — "ok" | "fallback" | "failed"
        message       — diagnostic string

    On arch/fit failure, falls back to realised-variance persistence estimate.
    """
    returns = log_returns.dropna()
    if len(returns) < _MIN_OBS:
        return {"persistence": 0.85, "status": "fallback",
                "message": f"insufficient obs ({len(returns)} < {_MIN_OBS})"}
    try:
        from arch import arch_model  # noqa: PLC0415
        scaled = returns * 100     # arch works on percentage returns
        model  = arch_model(scaled, vol="GARCH", p=1, o=1, q=1, dist="normal")
        result = model.fit(disp="off", show_warning=False)
        params = result.params
        alpha = float(params.get("alpha[1]", 0.0) or 0.0)
        beta  = float(params.get("beta[1]",  0.0) or 0.0)
        gamma = float(params.get("gamma[1]", 0.0) or 0.0)
        omega = float(params.get("omega",    0.0) or 0.0)
        persistence = alpha + beta + gamma / 2.0
        return {
            "omega": omega, "alpha": alpha, "beta": beta, "gamma": gamma,
            "persistence": round(min(1.0, persistence), 6),
            "status": "ok", "message": "GJR-GARCH(1,1) fitted",
        }
    except Exception as exc:
        log.warning("GJR-GARCH fit failed: %s — using realised-variance fallback", exc)
        return _realised_variance_fallback(returns)


def _realised_variance_fallback(returns: pd.Series) -> dict:
    """Estimate GARCH-like persistence from realised variance autocorrelation."""
    try:
        sq = returns ** 2
        corr = float(np.corrcoef(sq[:-1].values, sq[1:].values)[0, 1])
        persistence = max(0.0, min(0.999, 0.9 + corr * 0.09))
        return {
            "persistence": round(persistence, 6),
            "status": "fallback",
            "message": "realised-variance AR(1) persistence estimate",
        }
    except Exception:
        return {"persistence": 0.85, "status": "failed", "message": "all estimates failed"}


def volatility_regime(persistence: float) -> str:
    """Classify GARCH persistence into a volatility state.

    Thresholds:
        persistence ≥ 0.98 → CLUSTERING (IGARCH — permanent shocks, Minsky trigger)
        persistence ≥ 0.90 → EXPANDING  (elevated and rising)
        persistence <  0.90 → STABLE     (mean-reverting, normal)
    """
    if persistence >= 0.98:
        return "CLUSTERING"
    if persistence >= 0.90:
        return "EXPANDING"
    return "STABLE"
