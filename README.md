# regime_trader

[![market_intel](https://github.com/Natarrr/regimetrader/actions/workflows/market_intel.yml/badge.svg)](https://github.com/Natarrr/regimetrader/actions/workflows/market_intel.yml)
[![canary](https://github.com/Natarrr/regimetrader/actions/workflows/canary.yml/badge.svg)](https://github.com/Natarrr/regimetrader/actions/workflows/canary.yml)
[![nightly_edgar](https://github.com/Natarrr/regimetrader/actions/workflows/nightly_edgar.yml/badge.svg)](https://github.com/Natarrr/regimetrader/actions/workflows/nightly_edgar.yml)
[![edgar_3x](https://github.com/Natarrr/regimetrader/actions/workflows/edgar_3x.yml/badge.svg)](https://github.com/Natarrr/regimetrader/actions/workflows/edgar_3x.yml)
[![daily_discord](https://github.com/Natarrr/regimetrader/actions/workflows/daily_toplists_discord.yml/badge.svg)](https://github.com/Natarrr/regimetrader/actions/workflows/daily_toplists_discord.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)

EDGAR-first market intelligence and regime detection pipeline.

## Workflows

| Workflow | Trigger | Purpose |
| --- | --- | --- |
| `market_intel` | push, PR, schedule (3×/day weekdays) | Unit tests + EDGAR data fetch |
| `canary` | schedule (06:00, 12:00, 18:00 UTC) | End-to-end pipeline health check (10 tickers) |
| `nightly_edgar` | schedule (00:00 UTC daily) | Full 50-ticker EDGAR backfill with retry |
| `edgar_3x` | schedule (00:00, 08:00, 16:00 UTC) | 3× daily data fetch + top_lists artifact |
| `daily_toplists_discord` | schedule (13:00 UTC daily) | Daily 14:00 London market checkup → Discord |

## Quick start

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt

# Run the canary locally
powershell -ExecutionPolicy Bypass -File run_canary.ps1

# Generate top_lists locally (after an EDGAR run)
python -m backend.market_intel.generate_top_lists --log-dir logs --force

# Preview Discord message (no webhook needed)
python scripts/send_toplists_discord.py --dry-run
```

## Test suite

```bash
# CI-equivalent (lightweight deps)
pip install -r requirements-ci.txt
pytest backend/tests/market_intel/ -v --noconftest
pytest tests/ -v
```

## Project layout

```text
backend/market_intel/   EDGAR ingestion pipeline + generate_top_lists.py
monitoring/             Metrics export, threshold checks, alert state, minsky_alert
scripts/                send_toplists_discord.py — daily Discord checkup
tests/                  Canary hardening suite + top_lists tests
tests/data/             Sample JSON fixtures for unit tests
config/                 Ticker lists (canary_top10.csv)
.github/workflows/      CI/CD (market_intel, canary, nightly_edgar, edgar_3x, discord)
```

## Operation — Daily Checkup

### What it does

Every day at **14:00 London time (BST)**, the `daily_toplists_discord` workflow:

1. Downloads the latest `top-lists` artifact from `edgar_3x` (produced 3× daily)
2. Runs `scripts/send_toplists_discord.py` to format and POST a Discord embed
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

### DST note

The schedule `cron: "0 13 * * *"` targets **14:00 BST** (UTC+1, last Sunday March – last Sunday October).

During **winter (GMT, UTC+0)**, 13:00 UTC = 13:00 London. To maintain 14:00 London year-round,
either:

- Switch to `cron: "0 14 * * *"` in winter and revert in spring (manual update)
- Use an external scheduler (Inngest, Zapier, AWS EventBridge) that resolves `Europe/London` timezone before calling `workflow_dispatch`

### Local test

```bash
# 1. Generate top_lists from existing EDGAR output
python -m backend.market_intel.generate_top_lists --log-dir logs --force --verbose

# 2. Preview the Discord embed (no webhook, no network)
python scripts/send_toplists_discord.py \
  --input logs/top_lists.json \
  --dry-run

# 3. Send for real (requires DISCORD_WEBHOOK_URL in environment)
DISCORD_WEBHOOK_URL="https://discord.com/api/webhooks/..." \
  python scripts/send_toplists_discord.py --input logs/top_lists.json

# 4. Run unit tests
pytest tests/test_top_lists.py -v
```

### Artifact retention policy

| Artifact | Workflow | Retention |
| --- | --- | --- |
| `top-lists` | `edgar_3x` | 7 days |
| `edgar-3x-debug-*` | `edgar_3x` | 14 days |
| `discord-send-log-*` | `daily_toplists_discord` | 7 days |
| `nightly-edgar-*` | `nightly_edgar` | 90 days |
| `canary-*` | `canary` | 14 days |
