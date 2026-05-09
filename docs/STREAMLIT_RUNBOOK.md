# Streamlit Dashboard — Runbook

## Overview

The Streamlit dashboard is the human-facing surface of the Regime Trader system.
It surfaces:

- **Live Monitor** — current regime, portfolio, risk
- **Market Intel** — Smart Money discovery shortlist (FMP-powered)
- **Macro Intel** — commodity conviction + macro shock detection
- **Trade Log** — historical executions
- **Regime History** — regime transition timeline
- **Portfolio Sync** — broker CSV reconciliation

Heavy I/O is delegated to:

- [regime_trader.discovery_scanner](../regime_trader/discovery_scanner.py)
- [regime_trader.market_intel_macro](../regime_trader/market_intel_macro.py)

The Streamlit module itself orchestrates rendering and caches results via
`st.cache_data`.

---

## Run Locally

### Prerequisites

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
```

Python 3.11+. The `streamlit` and `plotly` packages must come from
`requirements.txt`, not `requirements-ci.txt`.

### Environment

Create `.env` at the repo root (NEVER commit it):

```dotenv
FMP_API_KEY=...               # Financial Modeling Prep — discovery scanner
ANTHROPIC_API_KEY=...         # Optional — if using LLM tabs locally
SEC_USER_AGENT=Your Name your@email.com
```

The app loads `.env` automatically via `python-dotenv`. Missing keys
degrade gracefully — the app still starts and shows fallback data.

### Start

Either entry point works (the root file is a shim into the package):

```bash
streamlit run streamlit_app.py
# equivalent to:
streamlit run regime_trader/ui/streamlit_app.py
```

Default port: `http://localhost:8501`.

For a headless run on a custom port:

```bash
streamlit run streamlit_app.py --server.port 8502 --server.headless true
```

---

## Caching

| Layer | Mechanism | TTL | Bypass |
| ----- | --------- | --- | ------ |
| Discovery picks | `@st.cache_data(ttl=6h)` (UI) + file-backed cache (data) | 6 h | UI: "🔄 Refresh" button → `force_refresh_sync()` |
| Commodity prices | `@st.cache_data(ttl=1h)` | 1 h | Restart Streamlit process |
| Regime detector | computed on-demand | n/a | n/a |

The data-layer cache lives at `logs/discovery_cache.json` (atomic write via
`save_json_atomic`). The UI cache is in-process only.

---

## Environment Variables Reference

| Variable | Purpose | Used by |
| -------- | ------- | ------- |
| `FMP_API_KEY` | Financial Modeling Prep API | discovery_scanner, market_intel_macro |
| `ANTHROPIC_API_KEY` | Claude SDK | analysis.claude_client (LLM tabs) |
| `SEC_USER_AGENT` | EDGAR rate-limit compliance | discovery_scanner (EDGAR proxy) |
| `ALPACA_KEY_ID` / `ALPACA_SECRET` | Live broker positions | live_monitor tab |
| `LOG_DIR` | NDJSON log directory | live_monitor tab |

**Never** log these values. The test
[test_streamlit_app_smoke.py::test_configure_logging_does_not_emit_environment](../tests/test_streamlit_app_smoke.py)
guards against env-var leaks in the default formatter.

---

## DST / Scheduler Note

The hybrid pipeline runs at **08:30 ET**, scheduled in
[hybrid_pipeline.yml](../.github/workflows/hybrid_pipeline.yml) as cron
`30 12 * * 1-5` (12:30 UTC). This corresponds to:

- 08:30 EDT (UTC-4) — March → November
- 07:30 EST (UTC-5) — November → March

The 1-hour drift during EST is intentional: the workflow runs **before**
US market open in both regimes, but in EST it fires 1 h earlier vs ET-clock-time.
If pre-open timing matters (e.g., for earnings releases), adjust the cron
twice yearly or migrate to a timezone-aware scheduler.

The Streamlit dashboard itself is **timezone-naïve** in its UI — timestamps are
shown in UTC unless the user's browser locale overrides display.

---

## Debugging

### App fails to start

```bash
# 1. Verify imports
python -c "import regime_trader.ui.streamlit_app"

# 2. Check Streamlit version
streamlit version   # must be >= 1.35

# 3. Run with verbose logging
streamlit run streamlit_app.py --logger.level=debug
```

### Market Intel tab shows "API key missing"

`FMP_API_KEY` not set or `.env` not loaded:

```bash
# Verify
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('FMP key:', bool(os.getenv('FMP_API_KEY')))"
```

### Stale data after FMP outage

Force-refresh:

```python
from regime_trader.discovery_scanner import force_refresh_sync
force_refresh_sync(limit=5)
```

Or delete the cache file: `logs/discovery_cache.json`.

### "ScriptRunContext! This warning can be ignored when running in bare mode."

Harmless. Emitted when the module is imported in tests/REPL outside Streamlit's
runner. Streamlit attaches the context at runtime.

---

## Deploying

The dashboard is meant for local / single-operator use. For team deployment:

1. Run inside a container (`uvicorn`-style — Streamlit has its own server)
2. Reverse-proxy through nginx or Traefik with auth
3. Mount `.env` as a Docker secret, **never** bake it into the image
4. Read-only volume for `data/` if you don't need write-back caches
