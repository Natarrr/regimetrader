# FMP stable/ Endpoint Status

**Date:** 2026-05-30  
**Test tickers:** US=AAPL, EU=SAP.DE, ASIA=7203.T  
**Plan:** FMP Ultimate ($139/mo)

## Results

| Endpoint | Status | Detail | Pipeline role |
|---|---|---|---|
| `quote` | **PASS** | 1 row | Market cap, price for conviction scoring |
| `congress:senate` | **FAIL** | HTTP 404 | Congress factor — route dead |
| `congress:house` | **FAIL** | HTTP 404 | Congress factor — route dead |
| `insider:search` | **PASS** | 100 rows | Insider conviction + breadth signals |
| `news:stock` | **PASS** | 10 rows | News sentiment + buzz signals |
| `profile` | **PASS** | 1 row | Market cap fallback |
| `ratings:consensus` | **PASS** | 1 row | New: analyst consensus signal |
| `price-target` | **PASS** | 1 row | New: price target consensus |
| `key-metrics-ttm` | **PASS** | 1 row | New: quality metrics (P/E, etc.) |
| `ratios-ttm` | **PASS** | 1 row | New: financial ratios (D/E, margins) |
| `13f:summary` | **PASS** (year+quarter required) | Requires `?year=YYYY&quarter=Q` — HTTP 400 without them | Institutional ownership — fixed |
| `batch-quote` | **PASS** | 3 rows | Bulk quote (throughput optimization) |
| `cash-flow` | **PASS** | 4 rows | New: cash flow for cannibal filter |
| `cot` | **PASS** | 536 rows | COT data — real feed available |
| `earnings-transcript` | **PASS** | 100 rows | New: earnings transcript for Claude NLP |
| `EU:quote` (SAP.DE) | **PASS** | 1 row | EU coverage confirmed live |
| `EU:ratios-ttm` (SAP.DE) | **PASS** | 1 row | EU fundamentals confirmed live |
| `ASIA:quote` (7203.T) | **PASS** | 1 row | Asia coverage confirmed live |
| `ASIA:ratios-ttm` (7203.T) | **PASS** | 1 row | Asia fundamentals confirmed live |

## Summary

- **16 PASS, 0 EMPTY, 3 FAIL/ERROR**

## Decision gate (per Phase 0 rules)

### FAIL / ERROR — Do NOT build factors on these

| Endpoint | Verdict | Action |
|---|---|---|
| `congress:senate` (HTTP 404) | FMP has not migrated senate-trading to stable/ yet | `get_congress_trades()` now fetches directly from public S3 Stock Watcher feeds (no API key needed). File FMP support ticket to request stable/ migration. |
| `congress:house` (HTTP 404) | Same — house-trading not on stable/ | Same S3 fallback. |
| `13f:summary` — **FIXED** | Required `year` + `quarter` params | Now uses `get_institutional_ownership()` in FMPClient with auto-computed quarter. |

### International — market_config.py claim is WRONG

`EU:quote`, `EU:ratios-ttm`, `ASIA:quote`, `ASIA:ratios-ttm` all **PASS**.  
The comment in `market_config.py` ("FMP 403 for all non-US") was based on old
`/api/v3`/`/api/v4` behavior. Under `stable/`, FMP Ultimate covers EU and Asia
with quote + ratios. **Phase 4 applies** — EUROPE/ASIA market config can be
extended to include FMP-backed factors (momentum + volume_attention +
quality ratios).

### New endpoints available for Phase 4

| Endpoint | Use case |
|---|---|
| `cot` | Replace the 52-week-percentile COT proxy in `market_intel_macro` |
| `key-metrics-ttm`, `ratios-ttm` | Replace yfinance `.info` scraping in `satellite_factors.py` |
| `batch-quote` | Replace serial per-ticker quote loops (throughput) |
| `earnings-transcript` | Claude NLP feed for earnings analysis |
| `ratings:consensus` | Analyst consensus as optional signal |
| `cash-flow` | Cannibal filter (buyback yield) |
