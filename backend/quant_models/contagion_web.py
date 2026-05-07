"""backend/quant_models/contagion_web.py
Module D — The Contagion Web (Leontief / Tirole).

Input-Output matrix model of inter-sector shock propagation across the S&P 500.
Uses a calibrated approximation of BEA I-O tables for 11 GICS sectors.

All functions are pure: accept numpy inputs, return numpy/dict outputs.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np


# ── Sector definitions ─────────────────────────────────────────────────────────

SECTORS: List[str] = [
    "Energy",
    "Materials",
    "Industrials",
    "Consumer_Disc",
    "Consumer_Stap",
    "Healthcare",
    "Financials",
    "IT",
    "Comm_Services",
    "Utilities",
    "Real_Estate",
]

# Calibrated BEA I-O table approximation (11×11).
# a_ij = fraction of sector j's inputs sourced from sector i.
# Row i = "who supplies sector j"; column j = "sector j's input mix".
# Values sourced from BEA 2022 Use Tables (simplified).
_BEA_A_MATRIX: np.ndarray = np.array([
    # ENE   MAT   IND   CDis  CSta  HLT   FIN   IT    COM   UTL   RE
    [0.05, 0.04, 0.06, 0.01, 0.02, 0.01, 0.01, 0.01, 0.02, 0.15, 0.03],  # Energy
    [0.01, 0.08, 0.10, 0.04, 0.05, 0.02, 0.00, 0.02, 0.01, 0.02, 0.05],  # Materials
    [0.03, 0.05, 0.12, 0.06, 0.04, 0.03, 0.02, 0.04, 0.03, 0.04, 0.06],  # Industrials
    [0.01, 0.02, 0.03, 0.08, 0.06, 0.02, 0.01, 0.03, 0.04, 0.01, 0.02],  # Consumer Disc
    [0.01, 0.02, 0.02, 0.03, 0.10, 0.04, 0.01, 0.01, 0.02, 0.01, 0.01],  # Consumer Stap
    [0.00, 0.01, 0.01, 0.01, 0.02, 0.15, 0.01, 0.02, 0.01, 0.00, 0.01],  # Healthcare
    [0.01, 0.02, 0.03, 0.03, 0.03, 0.03, 0.08, 0.03, 0.04, 0.02, 0.06],  # Financials
    [0.01, 0.01, 0.04, 0.05, 0.02, 0.04, 0.03, 0.12, 0.08, 0.02, 0.02],  # IT
    [0.01, 0.01, 0.02, 0.03, 0.02, 0.02, 0.02, 0.05, 0.10, 0.02, 0.02],  # Comm Services
    [0.03, 0.02, 0.03, 0.01, 0.01, 0.01, 0.01, 0.02, 0.01, 0.06, 0.03],  # Utilities
    [0.01, 0.01, 0.02, 0.02, 0.01, 0.01, 0.04, 0.01, 0.02, 0.02, 0.08],  # Real Estate
], dtype=float)


# ── I-O Model ──────────────────────────────────────────────────────────────────

def build_io_matrix(sector_weights: Dict[str, float] | None = None) -> np.ndarray:
    """Leontief (1973 Nobel) — Technical coefficient matrix for inter-sector flows.

    Returns the calibrated BEA approximation A matrix, optionally scaled by
    user-supplied sector output weights.

    # $a_{ij} = \\frac{z_{ij}}{x_j}$
    # where $z_{ij}$ = flow from sector $i$ to sector $j$, $x_j$ = total output of $j$.

    Args:
        sector_weights: Optional {sector_name: weight} to re-scale columns.
                        If None, uses equal weights (the raw BEA approximation).

    Returns:
        (11, 11) technical coefficient matrix A.
    """
    A = _BEA_A_MATRIX.copy()
    if sector_weights:
        for j, name in enumerate(SECTORS):
            w = sector_weights.get(name, 1.0)
            A[:, j] *= w
    return A


def leontief_inverse(A: np.ndarray) -> np.ndarray:
    """Leontief (1973 Nobel) — Compute the Leontief multiplier matrix.

    The inverse $(I - A)^{-1}$ captures both direct and all indirect inter-sector
    dependencies. Element $(i,j)$ is the total output required from sector $i$
    per unit of final demand for sector $j$.

    # $L = (I - A)^{-1}$, where $x = L \\cdot d$

    Args:
        A: Technical coefficient matrix (n×n). Must satisfy spectral radius < 1
           for economic viability.

    Returns:
        Leontief inverse matrix L (n×n).

    Raises:
        ValueError: If (I-A) is singular or spectral radius ≥ 1.
    """
    n = A.shape[0]
    I_minus_A = np.eye(n) - A
    spectral_radius = np.max(np.abs(np.linalg.eigvals(A)))
    if spectral_radius >= 1.0:
        raise ValueError(
            f"A matrix spectral radius {spectral_radius:.3f} ≥ 1: "
            "system is not productive (Hawkins-Simon condition violated)."
        )
    return np.linalg.inv(I_minus_A)


def shock_propagation(
    A: np.ndarray,
    demand_shock: Dict[str, float],
    baseline_demand: np.ndarray | None = None,
) -> Dict[str, float]:
    """Tirole (2014 Nobel) — Systemic risk via interconnected balance sheet contagion.

    Applies a demand shock to one or more sectors and propagates it through the
    Leontief inverse to compute total output impact per sector.

    # $\\Delta x = (I - A)^{-1} \\cdot \\Delta d$
    # where $\\Delta d_i = -0.20 \\cdot d_i$ for shocked sectors.

    Args:
        A:              Technical coefficient matrix (from build_io_matrix()).
        demand_shock:   {sector_name: shock_fraction} e.g. {"Energy": -0.20}.
        baseline_demand: Sector baseline demand vector. Defaults to equal weights.

    Returns:
        {sector_name: pct_output_impact} — negative = output loss.
    """
    n = len(SECTORS)
    if baseline_demand is None:
        baseline_demand = np.ones(n)

    L = leontief_inverse(A)

    delta_d = np.zeros(n)
    for sector, shock in demand_shock.items():
        if sector in SECTORS:
            idx = SECTORS.index(sector)
            delta_d[idx] = shock * baseline_demand[idx]

    delta_x = L @ delta_d
    pct_impact = {}
    for i, name in enumerate(SECTORS):
        base = baseline_demand[i]
        pct_impact[name] = round(float(delta_x[i] / base) * 100, 3) if base != 0 else 0.0
    return pct_impact


def critical_nodes(L: np.ndarray, top_n: int = 3) -> List[str]:
    """Leontief (1973 Nobel) — Identify sectors with the highest multiplier effect.

    The row-sum of the Leontief inverse gives the total output generated across
    the economy per unit of final demand for each sector (output multiplier).

    # $m_i = \\sum_j L_{ij}$

    Args:
        L:     Leontief inverse matrix (from leontief_inverse()).
        top_n: Number of critical nodes to return.

    Returns:
        List of sector names ranked by output multiplier (descending).
    """
    multipliers = L.sum(axis=1)
    ranked_indices = np.argsort(multipliers)[::-1][:top_n]
    return [SECTORS[i] for i in ranked_indices]


def total_gdp_impact(shock_impacts: Dict[str, float], gdp_weights: Dict[str, float] | None = None) -> float:
    """Leontief (1973 Nobel) — Aggregate GDP impact from sector-level shocks.

    # $\\Delta GDP = \\sum_i w_i \\cdot \\Delta x_i$

    Args:
        shock_impacts: {sector: pct_impact} from shock_propagation().
        gdp_weights:   {sector: gdp_share} — defaults to equal weights.

    Returns:
        Weighted aggregate GDP impact as percent.
    """
    n = len(SECTORS)
    weights = gdp_weights or {s: 1.0 / n for s in SECTORS}
    total = sum(shock_impacts.get(s, 0.0) * weights.get(s, 1.0 / n) for s in SECTORS)
    return round(total, 3)
