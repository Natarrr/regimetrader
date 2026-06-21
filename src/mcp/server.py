# Path: src/mcp/server.py
"""stdio MCP server exposing regime_trader's signals to an LLM (read-only).

Run:
    python -m src.mcp.server            # serves over stdio

Reads the committed artifacts under logs/ (override with REGIME_TRADER_LOGS_DIR).
It performs NO FMP calls and never re-scores — it only surfaces what the pipeline
already produced and passed through the safety_gate (CLAUDE.md §1). The tool set
and validation live in tools.py / schemas.py and are fully unit-tested without
the MCP SDK; this module is the thin transport binding.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from src.mcp.artifacts import ArtifactStore
from src.mcp.tools import build_tools


def _logs_dir() -> Path:
    override = os.getenv("REGIME_TRADER_LOGS_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "logs"


async def _amain() -> None:
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import TextContent
        from mcp.types import Tool as MCPTool
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise SystemExit(
            "The 'mcp' package is required to run the server: pip install mcp"
        ) from exc

    store = ArtifactStore(_logs_dir())
    tools = build_tools(store)
    by_name = {t.name: t for t in tools}
    server = Server("regime-trader")

    @server.list_tools()
    async def list_tools() -> list:  # noqa: D401 - SDK callback
        return [MCPTool(name=t.name, description=t.description,
                        inputSchema=t.input_schema) for t in tools]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict | None) -> list:
        tool = by_name.get(name)
        if tool is None:
            result = {"ok": False, "error": f"unknown tool: {name}"}
        else:
            result = tool.handler(arguments or {})
        return [TextContent(type="text", text=json.dumps(result, default=str))]

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream,
                         server.create_initialization_options())


def main() -> None:
    import asyncio
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
