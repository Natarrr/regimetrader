"""backend/quant_models/volatility_brain.py
Module B — The Volatility Brain (Engle / Merton).

GJR-GARCH(1,1) for volatility clustering persistence, and Merton's
Distance-to-Default for systemic bank solvency risk.

All functions are pure: accept array/scalar inputs, return plain dicts.
No global state; safe for multi-threaded FastAPI handlers.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
from arch import arch_model
from scipy import optimize, stats

from regime_trader.utils.volatility import annualise_vol_from_condvar


# ── GJR-GARCH ─────────────────────────────────────────────────────────────────

def fit_gjr_garch(returns: np.ndarray) -> Dict[str, float | np.ndarray]:
    """Engle (2003 Nobel) — ARCH/GARCH family captures volatility clustering.

    GJR-GARCH(1,1) extends standard GARCH with a leverage term γ that gives
    extra weight to negative shocks (the asymmetric leverage effect).

    # $\\sigma^2_t = \\omega + (\\alpha + \\gamma \\mathbf{1}[\\varepsilon_{t-1}<0])
    #               \\varepsilon^2_{t-1} + \\beta \\sigma^2_{t-1}$

    # Persistence $= \\alpha + \\beta + \\gamma/2$
    # Stationarity requires persistence $< 1$.

    Uses arch.arch_model with vol='GARCH', p=1, o=1, q=1 (GJR variant).

    Args:
        returns: 1-D array of log-returns (e.g. np.log(close/close.shift(1))).
                 Should be scaled to percentage points (multiply by 100) for
                 numerical stability in the arch library.

    Returns:
        {omega, alpha, gamma, beta, persistence,
         conditional_vol_series (annualised √252),
         latest_conditional_vol_ann}
    """
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)] * 100  # arch lib expects %-point returns

    am = arch_model(r, vol="Garch", p=1, o=1, q=1, dist="Normal", rescale=False)
    res = am.fit(disp="off", show_warning=False)

    params = res.params
    omega = float(params["omega"])
    alpha = float(params["alpha[1]"])
    gamma = float(params.get("gamma[1]", 0.0))
    beta = float(params["beta[1]"])
    persistence = alpha + beta + gamma / 2.0

    # arch returns conditional_volatility as daily std in %-pt.
    # Square → daily variance in %-pt², then convert via canonical util.
    # annualise_vol_from_condvar returns annualised vol in plain %-pt (e.g. 10.25).
    h_t_pct2 = np.asarray(res.conditional_volatility) ** 2
    cond_vol_ann_pct = annualise_vol_from_condvar(h_t_pct2, units="percent")
    # Divide by 100 → annualised decimal (e.g. 0.1025) to keep existing interface.
    cond_vol_ann = cond_vol_ann_pct.to_numpy() / 100.0
    latest_ann = float(cond_vol_ann[-1])

    return {
        "omega": omega,
        "alpha": alpha,
        "gamma": gamma,
        "beta": beta,
        "persistence": persistence,
        # h_t_pct2: raw daily conditional variance in %-pt² (input to annualise_vol_from_condvar)
        "h_t_pct2": h_t_pct2,
        # conditional_vol_series: annualised decimal vol (e.g. 0.1025 = 10.25 %/year)
        "conditional_vol_series": cond_vol_ann,
        "latest_conditional_vol_ann": latest_ann,
    }


def volatility_regime(persistence: float) -> str:
    """Engle (2003 Nobel) — GARCH persistence above 0.98 signals regime break.

    # $\\text{regime} = \\begin{cases}
    #   \\text{CLUSTERING} & \\text{if persistence} > 0.98 \\\\
    #   \\text{STABLE}     & \\text{otherwise}
    # \\end{cases}$
    """
    return "CLUSTERING" if persistence > 0.98 else "STABLE"


# ── Merton Distance-to-Default ─────────────────────────────────────────────────

def merton_distance_to_default(
    equity_value: float,
    face_value_debt: float,
    risk_free_rate: float,
    equity_vol: float,
    T: float = 1.0,
) -> Dict[str, float]:
    """Merton (1997 Nobel) — Structural credit model treats equity as a call option.

    The firm's equity E is modelled as a European call on total asset value V:
    # $E = V \\cdot N(d_1) - F e^{-rT} \\cdot N(d_2)$

    where
    # $d_1 = \\frac{\\ln(V/F) + (r + \\sigma_V^2/2)T}{\\sigma_V \\sqrt{T}}$,
    # $d_2 = d_1 - \\sigma_V \\sqrt{T}$

    The system (E, σ_E) → (V, σ_V) is solved iteratively via:
    # $\\sigma_E = \\frac{V}{E} N(d_1) \\sigma_V$

    Distance-to-Default:
    # $D2D = \\frac{\\ln(V/F) + (\\mu - \\sigma_V^2/2)T}{\\sigma_V \\sqrt{T}}$

    D2D < 1.5 is conventionally the distress zone for large financial institutions.

    Args:
        equity_value:    Market cap in consistent currency units.
        face_value_debt: Total face value of liabilities (short + long term debt).
        risk_free_rate:  Annual risk-free rate as decimal (e.g. 0.045).
        equity_vol:      Annualised equity volatility as decimal (e.g. 0.28).
        T:               Time horizon in years (default 1.0).

    Returns:
        {asset_value, asset_vol, d1, d2, d2d, prob_default}
    """
    r = risk_free_rate
    F = face_value_debt
    E = equity_value
    sigma_e = equity_vol

    def _bs_call(V: float, sigma_v: float) -> float:
        if V <= 0 or sigma_v <= 0:
            return 0.0
        d1 = (np.log(V / F) + (r + 0.5 * sigma_v ** 2) * T) / (sigma_v * np.sqrt(T))
        d2 = d1 - sigma_v * np.sqrt(T)
        return V * stats.norm.cdf(d1) - F * np.exp(-r * T) * stats.norm.cdf(d2)

    def _system(x: np.ndarray) -> np.ndarray:
        V, sigma_v = x[0], x[1]
        if V <= 0 or sigma_v <= 0:
            return [1e10, 1e10]
        d1 = (np.log(V / F) + (r + 0.5 * sigma_v ** 2) * T) / (sigma_v * np.sqrt(T))
        eq1 = _bs_call(V, sigma_v) - E
        eq2 = (V / E) * stats.norm.cdf(d1) * sigma_v - sigma_e
        return [eq1, eq2]

    V0 = E + F
    sigma_v0 = sigma_e * E / V0
    try:
        sol = optimize.fsolve(_system, [V0, sigma_v0], full_output=True)
        V_sol, sigma_v_sol = sol[0]
    except Exception:
        V_sol, sigma_v_sol = V0, sigma_v0

    V_sol = abs(V_sol)
    sigma_v_sol = abs(sigma_v_sol)

    mu = r  # use risk-free as drift under risk-neutral measure
    d2d = (np.log(V_sol / F) + (mu - 0.5 * sigma_v_sol ** 2) * T) / (
        sigma_v_sol * np.sqrt(T)
    )
    prob_default = float(stats.norm.cdf(-d2d))

    d1 = (np.log(V_sol / F) + (r + 0.5 * sigma_v_sol ** 2) * T) / (
        sigma_v_sol * np.sqrt(T)
    )
    d2 = d1 - sigma_v_sol * np.sqrt(T)

    return {
        "asset_value": float(V_sol),
        "asset_vol": float(sigma_v_sol),
        "d1": float(d1),
        "d2": float(d2),
        "d2d": float(d2d),
        "prob_default": prob_default,
    }
