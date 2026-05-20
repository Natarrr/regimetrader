# Satellite Factors — Design Spec

**Date:** 2026-05-20
**Status:** Approved

---

## Goal

Add a "Satellite" enrichment layer to the edgar_3x pipeline that computes two
supplementary signals — **seasonal cyclicality** and **share buyback yield** —
for the tickers already ranked by the 5-factor scoring engine. The signals are
appended to the daily Discord embed without touching `generate_top_lists.py` or
the core scoring logic.

---

## Architecture Overview

```text
fetch-and-rank job (edgar_3x.yml)
  Step 6: generate_top_lists.py  → logs/top_lists.json          (unchanged)
  Step 7: satellite_factors.py   → logs/satellite_insights.json  (NEW, continue-on-error: true)
  Step 8: minsky_alert                                           (unchanged)
  Step 9: artifact upload — extended to include satellite_insights.json

daily_toplists_discord.yml
  send_toplists_discord.py  reads logs/top_lists.json            (unchanged)
                            + conditionally reads logs/satellite_insights.json (NEW)
```

**Isolation guarantee:** `continue-on-error: true` on the satellite step ensures
that any network failure, rate-limit, or FMP outage never blocks the main
Top Buys Discord message from being sent.

---

## Output Schema — `logs/satellite_insights.json`

```json
{
  "generated_at": "2026-05-20T08:41:00+00:00",
  "month":        "May",
  "status":       "success",
  "cyclicals": [
    {"ticker": "PLTR", "win_rate": 0.75, "median_return": 0.031, "years": 9}
  ],
  "cannibals": [
    {"ticker": "SQ", "buyback_yield": 0.048, "pe": 18.2, "price_vs_52w_low": 1.18}
  ]
}
```

`status` is one of:

- `"success"` — both sub-tasks completed without error
- `"partial"` — one sub-task failed (e.g. FMP key absent), other succeeded
- `"error"` — both sub-tasks failed; lists will be empty

---

## Module: `backend/market_intel/satellite_factors.py`

### Constants (top of file — tuning knobs)

```python
MIN_MONTHLY_OBSERVATIONS = 8    # minimum historical month-samples for cyclicality
PE_MAX                   = 25.0  # P/E ratio ceiling for cannibal filter
PRICE_VS_52W_LOW_MAX     = 1.25  # price must be < 125% of 52-week low
TOP_N                    = 3     # tickers returned by each function
```

### `get_top_cyclical(tickers: list[str]) -> list[dict]`

**Data source:** yfinance — one batch `yf.download(tickers, period="10y", interval="1mo", auto_adjust=True, group_by="ticker")`.

**Algorithm:**

1. Normalise the DataFrame index to `pd.DatetimeIndex` (guard against multi-level index from batch download).
2. For each ticker extract the column slice `[("Open", ticker), ("Close", ticker)]` → per-ticker DataFrame.
3. Filter rows where `index.month == current_calendar_month`.
4. Drop rows with NaN in Open or Close.
5. If `len(filtered) < MIN_MONTHLY_OBSERVATIONS` → skip ticker (handles IPOs, delistings, missing data).
6. Compute:
   - `win_rate = (filtered["Close"] > filtered["Open"]).sum() / len(filtered)`
   - `median_return = ((filtered["Close"] - filtered["Open"]) / filtered["Open"]).median()`
7. Sort descending by `win_rate`, then `median_return`. Return top `TOP_N` as list of dicts:
   `{"ticker", "win_rate", "median_return", "years"}` where `years = len(filtered)`.

**Error handling:** entire function wrapped in `try/except`; any exception returns `[]` and logs a warning.

### `get_top_cannibals(tickers: list[str], fmp_key: str, market_caps: dict[str, float]) -> list[dict]`

**Data sources:**

- **yfinance** `yf.Ticker(t).info` → `trailingPE`, `fiftyTwoWeekLow`, `currentPrice`
- **FMP** `/stable/cash-flow-statement?symbol={t}&period=quarter&limit=4` → `repurchasedCommonStock`
- **market_cap** taken from `top_lists.json` (already loaded by caller — zero extra API calls)

**FMP budget:** 1 call per ticker × ≤15 tickers × 3 runs/day = 45 calls/day maximum. Well within 200/day limit.

**Algorithm (per ticker):**

1. If `fmp_key` is absent or empty → skip entire function, return `[]`, log warning.
2. Fetch yfinance `info` in a `try/except`; if absent/raises → skip ticker.
3. Apply filter 1: `trailingPE` is not None and `trailingPE < PE_MAX` — skip if fails.
4. Apply filter 2: `currentPrice` and `fiftyTwoWeekLow` both present, and `currentPrice < PRICE_VS_52W_LOW_MAX * fiftyTwoWeekLow` — skip if fails.
5. Fetch FMP cash-flow (4 quarters) in a `try/except`; if HTTP error or empty → skip ticker.
6. `total_repurchased = sum(abs(q.get("repurchasedCommonStock", 0) or 0) for q in quarters)`
7. `market_cap` from `market_caps` dict; if value is None or ≤ 0 → skip ticker (no ZeroDivisionError).
8. `buyback_yield = total_repurchased / market_cap`
9. Collect passing tickers, sort by `buyback_yield DESC`, return top `TOP_N` as list of dicts:
   `{"ticker", "buyback_yield", "pe", "price_vs_52w_low"}`.

**Error handling:** entire function wrapped in `try/except`; any exception returns `[]` and logs a warning.

### `main()`

```python
def main() -> None:
    # 1. Load logs/top_lists.json → extract unique tickers + market_cap map
    # 2. Call get_top_cyclical(tickers) → cyclicals ([] on failure)
    # 3. Call get_top_cannibals(tickers, fmp_key, market_caps) → cannibals ([] on failure)
    # 4. Determine status: "success" / "partial" / "error"
    # 5. Write logs/satellite_insights.json atomically
```

Accepts CLI args: `--log-dir` (default `logs`), `--verbose`.

No global state. No imports from `generate_top_lists.py`.

---

## File: `scripts/send_toplists_discord.py` — changes

### New helper: `_load_satellite(log_dir: Path) -> dict | None`

```python
def _load_satellite(log_dir: Path) -> dict | None:
    """Load satellite_insights.json if present. Returns None on any failure."""
    path = log_dir / "satellite_insights.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        return data
    except Exception as exc:
        log.warning("satellite_insights.json unreadable: %s", exc)
        return None
```

### Change to `build_payload(top_lists, satellite=None)`

Add optional `satellite: dict | None = None` parameter.

If `satellite` is not None and contains non-empty `cyclicals` or `cannibals`, append up to **2 new embed fields** before the Factor Legend field:

```text
🌀 Seasonal Cyclicals — May
1. PLTR  Win-rate: 75%  Median: +3.1%  (9 yr)
2. AMD   Win-rate: 67%  Median: +2.4%  (10 yr)
3. SQ    Win-rate: 63%  Median: +1.8%  (8 yr)

🐷 Share Cannibals — Buyback Yield
1. PLTR  Yield: 4.8%  P/E: 18.2  Price/52wLow: 1.18×
2. AMD   Yield: 3.1%  P/E: 22.4  Price/52wLow: 1.09×
```

Fields only added if the list is non-empty. If `satellite` is `None` or both lists are empty, `build_payload` behaves exactly as before.

The entire satellite block in `build_payload` is wrapped in `try/except` — any formatting error falls through silently and the embed is sent without satellite fields.

### Change to `main()`

```python
satellite = _load_satellite(args.log_dir)
payload   = build_payload(top_lists, satellite=satellite)
```

`--log-dir` already exists on the CLI, so no new argument needed.

---

## `edgar_3x.yml` changes

### New step (after step 6, before step 7)

```yaml
- name: Generate satellite insights
  continue-on-error: true
  env:
    FMP_API_KEY: ${{ secrets.FMP_API_KEY || '' }}
  run: |
    python -m backend.market_intel.satellite_factors \
      --log-dir logs \
      --verbose
```

### Extended artifact upload (step 9)

```yaml
path: |
  logs/top_lists.json
  logs/top5.csv
  logs/satellite_insights.json
```

The `if-no-files-found: warn` on the artifact step already handles the case where `satellite_insights.json` was not generated (step failed or was skipped).

---

## Dependencies

All already in `requirements.txt`:

- `yfinance` — cyclicality batch download + P/E and 52w-low via `.info`
- `requests` — FMP cash-flow API calls
- `pandas` — DataFrame operations
- `numpy` — median computation (already in env)

No new dependencies.

---

## Tests

`tests/test_satellite_factors.py` — unit tests using monkeypatched yfinance and mocked FMP HTTP:

- `test_cyclical_filters_insufficient_history` — ticker with < 8 observations excluded
- `test_cyclical_win_rate_calculation` — known OHLC data → expected win_rate
- `test_cannibal_filters_high_pe` — ticker with PE ≥ 25 excluded
- `test_cannibal_filters_price_above_52w_band` — ticker above 125% excluded
- `test_cannibal_zero_market_cap_skipped` — no ZeroDivisionError
- `test_cannibal_missing_fmp_key_returns_empty` — graceful degradation
- `test_main_writes_satellite_json` — integration: reads fixture top_lists.json, writes satellite_insights.json
- `test_satellite_status_partial_when_one_fails` — status="partial" when one sub-task returns []

`tests/test_send_toplists_discord.py` — extend existing tests:

- `test_build_payload_with_satellite` — satellite fields appear in embed
- `test_build_payload_without_satellite` — embed unchanged when satellite=None
- `test_load_satellite_returns_none_on_missing_file`
- `test_load_satellite_returns_none_on_corrupt_json`

---

## Error Handling Summary

| Scenario | Behaviour |
| --- | --- |
| yfinance timeout/network error | `cyclicals: []`, `status: "partial"` or `"error"` |
| FMP_API_KEY absent | `cannibals: []`, log warning, `status: "partial"` |
| FMP HTTP 4xx/5xx | ticker skipped, others continue |
| `market_cap` is 0 or None | ticker skipped (no ZeroDivisionError) |
| `satellite_insights.json` absent when Discord sends | `_load_satellite` returns None, embed sent without satellite fields |
| Corrupt `satellite_insights.json` | `_load_satellite` returns None, same as above |
| Exception inside `build_payload` satellite block | caught, embed sent without satellite fields |
| satellite step fails in CI (exit 1) | `continue-on-error: true` — pipeline continues, artifact upload skipped for that file |
