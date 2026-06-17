# regime_trader

[![ci](https://github.com/Natarrr/regimetrader/actions/workflows/ci.yml/badge.svg)](https://github.com/Natarrr/regimetrader/actions/workflows/ci.yml)
[![canary](https://github.com/Natarrr/regimetrader/actions/workflows/canary.yml/badge.svg)](https://github.com/Natarrr/regimetrader/actions/workflows/canary.yml)
[![nightly_edgar](https://github.com/Natarrr/regimetrader/actions/workflows/nightly_edgar.yml/badge.svg)](https://github.com/Natarrr/regimetrader/actions/workflows/nightly_edgar.yml)
[![edgar_3x](https://github.com/Natarrr/regimetrader/actions/workflows/edgar_3x.yml/badge.svg)](https://github.com/Natarrr/regimetrader/actions/workflows/edgar_3x.yml)
[![daily_trading_pipeline](https://github.com/Natarrr/regimetrader/actions/workflows/daily_trading_pipeline.yml/badge.svg)](https://github.com/Natarrr/regimetrader/actions/workflows/daily_trading_pipeline.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)

EDGAR-first market intelligence and regime detection pipeline.

## Workflows

| Workflow | Trigger | Purpose |
| --- | --- | --- |
| `ci` | push to main, PR | Lint (ruff) + import sanity + full test suite gate |
| `canary` | schedule (06:00, 12:00, 18:00 UTC) | End-to-end pipeline health check (10 tickers) |
| `daily_trading_pipeline` | schedule (00:30, 08:30, 16:30 UTC weekdays) | Consolidated US + INTL scoring → Discord delivery |
| `edgar_3x` | schedule (00:00, 08:00, 16:00 UTC) | 3× daily data fetch + top_lists artifact |
| `hybrid_pipeline` | schedule (12:30 UTC weekdays) | Claude AI analysis on top of edgar_3x artifact |
| `nightly_edgar` | manual dispatch only | Full universe EDGAR backfill with retry |
| `test_daily_toplists_absence` | push (send_discord paths), manual | Live integration test of the DATA UNAVAILABLE alert path |
| `weekly_backtest` | schedule (Friday 21:00 UTC) | Weekly signal backtest validation |

## Quick start

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt

# Run the canary locally
powershell -ExecutionPolicy Bypass -File run_canary.ps1  # loads .env values like FMP_API_KEY if present

# Generate top_lists locally (after an EDGAR run)
python -m backend.market_intel.generate_top_lists --log-dir logs --force

# Preview Discord message (no webhook needed)
python -m src.delivery.send_discord --dry-run
```

## Test suite

```bash
# CI-equivalent (lightweight deps)
pip install -r requirements-ci.txt
pytest tests/ -v
```

## Project layout

```text
src/                    Core package (single namespace)
  config/               Factor weights (weights.py — canonical SSOT)
  core/                 Fetcher base classes (fetchers_base.py)
  delivery/             audit_payload, cook_toplists, send_discord
  engine/               StrategyEngine + profile_runner (INTL scoring)
  fetchers/             Market fetcher orchestrator
  ingestion/            run_pipeline, fmp_fetcher, fmp_bulk_prefetch
  monitoring/           In-pipeline QA (factor_orthogonality)
  research/             historical_loader (run archiving)
  risk/                 regime.py (VIX thresholds SSOT), exit_rules.py
  scoring/              Signal modules (insider, momentum, news, analyst, …)
  services/             FMP API client
  utils/                io, formatting helpers
backend/market_intel/   generate_top_lists.py scoring orchestrator + validator
backend/data/           market_service (SPY/QQQ snapshot for Discord embeds)
monitoring/             Ops monitoring: metrics export, threshold checks, alerts
scripts/                CLI utilities (check_imports, backtest_signals, …)
tests/                  Test suite (pytest)
config/                 Ticker lists (universe.csv, canary_top10.csv)
.github/workflows/      CI/CD (ci, canary, daily_trading_pipeline, edgar_3x, …)
```

## Operation — Daily Checkup

### What it does

On weekdays at **00:30, 08:30 and 16:30 UTC** (30 min after each `edgar_3x` cache write), the `daily_trading_pipeline` workflow:

1. Reuses the bulk cache produced by `edgar_3x` (3× daily)
2. Runs `src/delivery/send_discord.py` to format and POST a Discord embed
3. The embed contains three ranked lists — Top 5 Buys, Top 5 Mid Caps, Top 5 Small Caps — each with 5-factor breakdown

### Five-factor scoring formula

```text
final_score = 0.30 × edgar  +  0.25 × insider  +  0.20 × congress
            + 0.15 × news   +  0.10 × momentum
```

| Factor | Source | Signal |
| --- | --- | --- |
| **EDGAR** | Form-4 filings via SEC (existing pipeline) | Open-market buy/sell dollar amounts, role-weighted |
| **Insider** | Derived from EDGAR score_breakdown | CEO buy presence (+0.30), buy/sell ratio, amendment penalty |
| **Congress/Inst** | FMP `/v4/senate-disclosure` + `/v3/institutional-holder` | Senate trade recency + institutional net-change direction |
| **News** | yfinance headline sentiment | Bull/bear word-list score over last 10 articles |
| **Momentum** | Price & volume trend (cross-sectional normalization) | Relative strength vs universe over rolling window |

Weights can be overridden: `--weights '{"edgar":0.40,"insider":0.20,...}'`

### Market cap tiers

The top-50 universe is all large/mega caps. Tiers are **relative within the universe**:

- **Large**: top 40% by market cap
- **Mid**: middle 35%
- **Small**: bottom 25% (still multi-billion — labeled as "smallest in this universe")

### Secrets to configure

In GitHub → Settings → Secrets → Actions:

| Secret | Required | Description |
| --- | --- | --- |
| `DISCORD_WEBHOOK_URL` | **Yes** | Discord channel webhook URL |
| `EDGAR_USER_AGENT` | **Yes** | SEC user-agent string (e.g. `YourName Pipeline email@example.com`) |
| `FMP_API_KEY` | Optional | Financial Modeling Prep API key (enables congress + institutional factors) |
| `SLACK_WEBHOOK_URL` | Optional | Slack webhook for Minsky CRITICAL alerts |

### Local test

```bash
# 1. Generate top_lists from existing EDGAR output
python -m backend.market_intel.generate_top_lists --log-dir logs --force --verbose

# 2. Preview the Discord embed (no webhook, no network)
python -m src.delivery.send_discord \
  --input logs/top_lists.json \
  --dry-run

# 3. Send for real (requires DISCORD_WEBHOOK_URL in environment)
DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..." \
  python -m src.delivery.send_discord --input logs/top_lists.json

# 4. Run unit tests
pytest tests/test_send_toplists_discord.py -v
```

### Artifact retention policy

| Artifact | Workflow | Retention |
| --- | --- | --- |
| `top-lists` | `edgar_3x` | 7 days |
| `edgar-3x-debug-*` | `edgar_3x` | 14 days |
| `discord-send-log-*` | `daily_trading_pipeline` | 7 days |
| `nightly-edgar-*` | `nightly_edgar` | 90 days |
| `canary-*` | `canary` | 14 days |
