# regime_trader MCP server (read-only signals)

A [Model Context Protocol](https://modelcontextprotocol.io) server that exposes
**our** computed signals — composite scores, factor breakdowns, ranked top-lists,
regime/VIX gate, source health — to an LLM (e.g. Claude Desktop).

It is deliberately the inverse of a generic FMP MCP (e.g.
`simonpierreboucher02/fmp-mcp`, which exposes raw FMP): this server exposes the
*outputs of our pipeline*, which a generic FMP wrapper cannot.

## Safety model

- **Read-only over artifacts.** It reads `logs/intel_source_status.json` and
  `logs/top_lists.json` only. It makes **no FMP calls** and never re-runs
  scoring, so it honours CLAUDE.md §1 (status from artifact state, not live
  scraping) and cannot bypass the `safety_gate`. No tool can spend FMP quota.
- **Validated boundary.** All tool inputs are validated with pydantic
  (`schemas.py`); errors return a structured `{"ok": false, "error": ...}`
  envelope rather than a traceback.

> **CLAUDE.md note:** current project rules keep FMP out of the LLM path. This
> server does not contradict that — it serves *post-gate artifacts*, not the FMP
> client. Recommend adding a short MCP clause to CLAUDE.md documenting that the
> read-only artifact server is permitted while live FMP calls remain barred.

## Tools

| Tool | Args | Returns |
|---|---|---|
| `get_ticker_score` | `ticker` | composite `final_score`, `badge`, per-factor breakdown |
| `get_toplists` | `market?`, `badge?` | ranked buy/watch names + regime |
| `get_regime` | — | `vix`, `vix_regime`, `kill_switch` |
| `get_source_health` | — | source freshness + EDGAR run metadata |
| `search_universe` | `min_score?`, `market?` | ranked names filtered, score-descending |

## Run

```bash
pip install mcp                       # SDK (also in requirements.txt)
python -m src.mcp.server             # serves over stdio
```

Override the artifacts directory with `REGIME_TRADER_LOGS_DIR`.

## Claude Desktop registration

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "regime-trader": {
      "command": "python",
      "args": ["-m", "src.mcp.server"],
      "cwd": "c:/Users/ntard/Projects/Trading dashboard/regime_trader",
      "env": { "REGIME_TRADER_LOGS_DIR": "c:/Users/ntard/Projects/Trading dashboard/regime_trader/logs" }
    }
  }
}
```

## Architecture

```
src/mcp/
  artifacts.py   ArtifactStore — pure, SDK-free reads over logs/*.json
  schemas.py     pydantic input models (Zod → pydantic)
  tools.py       Tool registry + createToolHandler-style handler factory
  server.py      thin stdio transport binding (imports the mcp SDK)
```

`artifacts.py`, `schemas.py`, `tools.py` are unit-tested (`tests/mcp/`) without
the MCP SDK installed; `server.py` is the only module that imports `mcp`.
