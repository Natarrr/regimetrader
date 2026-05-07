[![market_intel](https://github.com/Natarrr/regimetrader/actions/workflows/market_intel.yml/badge.svg)](https://github.com/Natarrr/regimetrader/actions/workflows/market_intel.yml)
[![canary](https://github.com/Natarrr/regimetrader/actions/workflows/canary.yml/badge.svg)](https://github.com/Natarrr/regimetrader/actions/workflows/canary.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)

# regime_trader

EDGAR-first market intelligence and regime detection pipeline.

## Workflows

| Workflow | Trigger | Purpose |
|---|---|---|
| `market_intel` | push, PR, schedule (3×/day weekdays) | Unit tests + EDGAR data fetch |
| `canary` | schedule (06:00, 12:00, 18:00 UTC) | End-to-end pipeline health check |

## Quick start

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1        # Windows
pip install -r requirements.txt

# Run the canary locally
powershell -ExecutionPolicy Bypass -File run_canary.ps1
```

## Test suite

```bash
# CI-equivalent (lightweight deps)
pip install -r requirements-ci.txt
pytest backend/tests/market_intel/ -v --noconftest
pytest tests/ -v
```

## Project layout

```
backend/market_intel/   EDGAR ingestion pipeline
monitoring/             Metrics export, threshold checks, alert state
tests/                  Canary hardening suite
config/                 Ticker lists (canary_top10.csv)
.github/workflows/      CI/CD (market_intel, canary)
```
