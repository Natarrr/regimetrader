# Path: src/mcp/schemas.py
"""Pydantic input schemas for the MCP tools (Zod → pydantic).

These validate untrusted LLM-supplied arguments at the tool boundary — the one
place external input enters the system (internal callers stay type-hinted and
unvalidated, per the comparison plan). Each model maps 1:1 to an ArtifactStore
query's parameters; `model_json_schema()` feeds the MCP tool registration.
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class TickerScoreInput(BaseModel):
    ticker: str = Field(..., min_length=1,
                        description="Ticker symbol, e.g. AAPL or ASML.AS")


class ToplistsInput(BaseModel):
    market: Optional[str] = Field(
        None, description="Filter to one market: usa | europe | asia")
    badge: Optional[str] = Field(
        None, description="Filter to one badge, e.g. buy | watch")


class RegimeInput(BaseModel):
    """No arguments — returns the current VIX safety-gate state."""


class SourceHealthInput(BaseModel):
    """No arguments — returns data-source freshness + EDGAR run metadata."""


class SearchUniverseInput(BaseModel):
    min_score: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="Only return names with final_score >= this floor (0–1)")
    market: Optional[str] = Field(
        None, description="Filter to one market: usa | europe | asia")
