# Path: src/mcp/tools.py
"""MCP tool registry + uniform handler factory.

Adopts the external fmp-mcp `createToolHandler` pattern: every tool is wrapped so
that (1) inputs are validated against a pydantic schema, (2) the underlying
ArtifactStore query runs, (3) the result is a structured envelope
{"ok": bool, "data" | "error"}. The server layer (server.py) serialises the
envelope to JSON text content; these handlers are pure and SDK-free so they can
be unit-tested without the MCP runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Type

from pydantic import BaseModel, ValidationError

from src.mcp.artifacts import ArtifactStore
from src.mcp.schemas import (
    RegimeInput,
    SearchUniverseInput,
    SourceHealthInput,
    TickerScoreInput,
    ToplistsInput,
)

Handler = Callable[[Dict[str, Any]], Dict[str, Any]]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_model: Type[BaseModel]
    handler: Handler

    @property
    def input_schema(self) -> Dict[str, Any]:
        """JSON schema for MCP tool registration."""
        return self.input_model.model_json_schema()


def _format_validation(exc: ValidationError) -> str:
    return "invalid input: " + "; ".join(
        f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()
    )


def make_handler(method: Callable[..., Any], model: Type[BaseModel]) -> Handler:
    """Wrap an ArtifactStore method as a validated, error-isolated handler."""
    def handler(raw: Dict[str, Any] | None) -> Dict[str, Any]:
        try:
            parsed = model.model_validate(raw or {})
        except ValidationError as exc:
            return {"ok": False, "error": _format_validation(exc)}
        try:
            return {"ok": True, "data": method(**parsed.model_dump())}
        except Exception as exc:  # never leak a traceback to the LLM
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return handler


def build_tools(store: ArtifactStore) -> List[Tool]:
    """Construct the tool set bound to a given artifact store."""
    return [
        Tool("get_ticker_score",
             "Composite score, badge and per-factor breakdown for one ticker, "
             "from the latest pipeline run. Null when the ticker is not in the "
             "scored universe.",
             TickerScoreInput, make_handler(store.ticker_score, TickerScoreInput)),
        Tool("get_toplists",
             "Ranked buy/watch names across markets plus the current regime; "
             "optionally filter by market or badge.",
             ToplistsInput, make_handler(store.toplists, ToplistsInput)),
        Tool("get_regime",
             "Current VIX safety-gate state (vix, vix_regime, kill_switch).",
             RegimeInput, make_handler(store.regime, RegimeInput)),
        Tool("get_source_health",
             "Data-source freshness and EDGAR run metadata for the latest run.",
             SourceHealthInput, make_handler(store.source_health, SourceHealthInput)),
        Tool("search_universe",
             "Ranked names filtered by score floor and/or market, "
             "score-descending.",
             SearchUniverseInput, make_handler(store.search_universe, SearchUniverseInput)),
    ]
