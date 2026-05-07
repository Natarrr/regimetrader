"""backend/routers/contagion.py
Module D router — The Contagion Web (Leontief / Tirole).

GET /api/contagion/shock — Propagate a sector demand shock through the I-O matrix.
GET /api/contagion/nodes — Return critical node rankings.
"""
from __future__ import annotations

import backend.dependencies  # noqa: F401

from typing import Optional
from fastapi import APIRouter, HTTPException, Query

from backend.data.schemas import ContagionOut
from backend.quant_models.contagion_web import (
    SECTORS,
    build_io_matrix,
    leontief_inverse,
    shock_propagation,
    critical_nodes,
    total_gdp_impact,
)

router = APIRouter(tags=["Contagion Web"])


@router.get("/shock", response_model=ContagionOut)
def get_shock_propagation(
    sector: str = Query(description=f"Sector to shock. Options: {', '.join(SECTORS)}"),
    shock_pct: float = Query(default=-20.0, le=0, ge=-100,
                             description="Demand shock in percent (negative = contraction)"),
) -> ContagionOut:
    """Leontief (1973 Nobel) + Tirole (2014 Nobel) — Supply chain shock propagation.

    Applies a demand shock to the specified GICS sector and propagates it through
    the Leontief inverse to compute total output impact across all 11 sectors.
    """
    if sector not in SECTORS:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown sector '{sector}'. Valid options: {SECTORS}",
        )

    A = build_io_matrix()
    L = leontief_inverse(A)
    impacts = shock_propagation(A, {sector: shock_pct / 100.0})
    nodes = critical_nodes(L, top_n=3)
    gdp_impact = total_gdp_impact(impacts)

    return ContagionOut(
        shock_sector=sector,
        sector_impacts=impacts,
        critical_nodes=nodes,
        total_gdp_impact_pct=gdp_impact,
    )


@router.get("/nodes")
def get_critical_nodes(top_n: int = Query(default=3, ge=1, le=11)) -> dict:
    """Leontief (1973 Nobel) — Return sectors with the highest multiplier effect."""
    A = build_io_matrix()
    L = leontief_inverse(A)
    nodes = critical_nodes(L, top_n=top_n)
    multipliers = L.sum(axis=1)
    return {
        "critical_nodes": [
            {
                "sector": s,
                "output_multiplier": round(float(multipliers[SECTORS.index(s)]), 4),
            }
            for s in nodes
        ]
    }
