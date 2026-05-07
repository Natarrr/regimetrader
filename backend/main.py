"""backend/main.py
The Laureate Engine — FastAPI application entry point.

Five Nobel-grounded modules expose a composable REST API:
  /api/monetary  — Friedman / Kuznets / Prescott (yield curve, M2, HP filter)
  /api/volatility — Engle / Merton (GJR-GARCH, Distance-to-Default)
  /api/valuation  — Shiller / Thaler (CAPE, Excess CAPE Yield)
  /api/contagion  — Leontief / Tirole (I-O shock propagation)
  /api/regime     — Lucas / Sargent + Minsky Moment alert

Run from repo root:      uvicorn backend.main:app --reload --port 8000
Run from backend/ dir:   uvicorn main:app --reload --port 8000
"""
from __future__ import annotations

import os
import sys

# Wire project root into sys.path so backend.* and utils.* resolve regardless
# of whether uvicorn is invoked from repo root or from inside backend/.
_here = os.path.dirname(os.path.abspath(__file__))   # .../backend
_root = os.path.dirname(_here)                        # .../regime_trader
if _root not in sys.path:
    sys.path.insert(0, _root)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import monetary, volatility, valuation, contagion, regime

app = FastAPI(
    title="The Laureate Engine",
    description=(
        "Full-stack predictive dashboard for market regime detection and "
        "crisis anticipation. Powered by Nobel Prize-winning quantitative models."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(monetary.router, prefix="/api/monetary")
app.include_router(volatility.router, prefix="/api/volatility")
app.include_router(valuation.router, prefix="/api/valuation")
app.include_router(contagion.router, prefix="/api/contagion")
app.include_router(regime.router, prefix="/api/regime")


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {
        "service": "The Laureate Engine",
        "status": "running",
        "docs": "/docs",
        "modules": ["monetary", "volatility", "valuation", "contagion", "regime"],
    }
