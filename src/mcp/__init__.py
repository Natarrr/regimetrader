# Path: src/mcp/__init__.py
"""Read-only Model Context Protocol (MCP) surface over regime_trader artifacts.

Exposes the pipeline's COMPUTED OUTPUTS (scores, top-lists, regime, source
health) to an LLM as MCP tools. It is strictly read-only over the committed
artifacts under logs/ — it NEVER calls FMP live and never re-runs scoring, so it
honours CLAUDE.md §1 (status from artifact state, not live scraping) and cannot
bypass the safety_gate. A generic FMP MCP (e.g. simonpierreboucher02/fmp-mcp)
exposes raw FMP; this exposes OUR signals, which that cannot.
"""
from src.mcp.artifacts import ArtifactStore

__all__ = ["ArtifactStore"]
