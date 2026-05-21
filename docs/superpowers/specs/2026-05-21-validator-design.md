# Pipeline Validator & Anomaly Detection — Design Spec

**Date:** 2026-05-21
**Status:** Approved

---

## Goal

Add a two-stage data quality gate to the regime_trader pipeline that catches bad data before it contaminates scoring, produces a machine-actionable anomaly audit trail, and never silently produces false-positive buy signals from dirty inputs.

## Architecture

```
run_pipeline.py
  │
  ├─ [Stage 1] validate_raw(rows, source_meta, failure_threshold=0.20)
  │     ├─ validate_dates(rows, source_meta, max_age_days=5, source_stale_hours=48)
  │     ├─ validate_amounts(rows)
  │     └─ validate_tickers(rows)
  │     → patches rows in-place (_validation_failed, _stale_source flags)
  │     → updates intel_source_status.json with per-source health flags
  │     → raises PipelineIntegrityError with structured summary if >threshold fail
  │
  ├─ [Normalize] Normalizer.log_scale_insider(amount, market_cap, tier)
  │              Normalizer.winsorize(series, limits)
  │              Normalizer.cross_sectional_norm(series)
  │
  ├─ [Stage 2] detect_anomalies(rows, run_id, log_dir)
  │     → writes anomaly_report_{run_id}.json  (permanent, never overwritten)
  │     → writes anomaly_report_latest.json    (always current run, for Discord)
  │     → returns list[AnomalyRecord]
  │
  └─ generate_top_lists.py
        → excludes _validation_failed tickers from top_buys ranking
        → Discord formatter reads anomaly_report_latest.json, appends ⚠️ to flagged tickers
        → writes top_lists.json
```

---

## Files

| Path | Role |
|------|------|
| `backend/market_intel/validator.py` | New — all validation, normalization, anomaly detection |
| `tests/test_validator.py` | New — full test suite |
| `backend/market_intel/generate_top_lists.py` | Modified — exclude `_validation_failed` tickers from `top_buys` |
| `scripts/send_toplists_discord.py` | Modified — read `anomaly_report_latest.json`, append ⚠️ to flagged tickers |
| `scripts/run_pipeline.py` | Modified — call `validate_raw()` after writing `intel_source_status.json` |

---

## Section 1 — Validation Functions

### `validate_dates(rows, source_meta, max_age_days=5, source_stale_hours=48)`

Checks three things:

1. **Parse validity**: each row's timestamp is a valid ISO-8601 string. Failure → `INVALID_DATE`, ticker quarantined.
2. **Future-dating**: timestamp is not in the future (clock skew tolerance: 60s). Failure → `FUTURE_DATE`, ticker quarantined.
3. **Row-level staleness**: age > `max_age_days` → `STALE_DATA` flag on the row. Action: `flag_only` (stays in pipeline, logged).
4. **Source-level staleness**: `source_meta[source_name]["last_updated"]` age > `source_stale_hours` → `STALE_SOURCE` flag. Action: **quarantine all rows from that source** (`_stale_source=True`, `_validation_failed=True`). This is a source-level decision, not ticker-level.

`source_meta` shape:
```python
{
  "quiver":  {"last_updated": "2026-05-19T10:00:00Z"},
  "fmp":     {"last_updated": "2026-05-21T08:00:00Z"},
  "edgar":   {"last_updated": "2026-05-21T09:00:00Z"},
}
```

Returns `(ok: bool, issues: list[ValidationIssue])`. Never raises.

### `validate_amounts(rows)`

For each row, checks `insider_usd` and `market_cap`:
- `None`, `NaN`, negative, or zero → set field to `float("nan")`, tag row `_validation_failed=True`
- Logs each incident as a `ValidationIssue` with ticker, field name, original value

Returns `(ok: bool, issues: list[ValidationIssue])`. Never raises.

### `validate_tickers(rows)`

- Empty string, non-string, or not matching `^[A-Z]{1,5}$` → `_validation_failed=True`
- Returns `(ok: bool, issues: list[ValidationIssue])`

### `validate_raw(rows, source_meta, failure_threshold=0.20)`

Orchestrates all three validators. After all checks:

1. Counts tickers with `_validation_failed=True`
2. If `failed / total > failure_threshold` → raises `PipelineIntegrityError` with structured summary:

```
PipelineIntegrityError: 25/83 tickers failed validation (30.1% > 20.0% threshold)
  STALE_SOURCE:    15 tickers  [source: quiver — last_updated 61h ago]
  MISSING_AMOUNT:   8 tickers  [field: insider_usd]
  INVALID_TICKER:   2 tickers
Aborting pipeline to prevent degenerate top_lists.json.
```

3. Below threshold: returns `(clean_rows, quarantined_rows, issues)`. Pipeline continues on `clean_rows` only.

---

## Section 2 — `Normalizer` class

Thin wrapper. Delegates to existing code where possible. Only `log_scale_insider` is new math.

```python
class Normalizer:
    @staticmethod
    def winsorize(series: pd.Series, limits: tuple[float, float] = (0.01, 0.99)) -> pd.Series
    @staticmethod
    def log_scale_insider(amount: float, market_cap: float, tier: str = "large") -> float
    @staticmethod
    def cross_sectional_norm(series: pd.Series) -> pd.Series
```

### `winsorize`
Delegates to `regime_trader.scoring.normalize.winsorize`. Uniform call signature only.

### `log_scale_insider(amount, market_cap, tier)`

Log-scale conviction signal with tier-aware dynamic ceiling:

| Tier | Ceiling (% of market cap) | Rationale |
|------|---------------------------|-----------|
| `small` | 2.0% | High sensitivity — $500k at $25M company is a massive signal |
| `mid` | 1.0% | Balanced |
| `large` | 0.5% | Prevents flatlining — $1B buy at $3T company still scores meaningfully |

Formula:
```python
ceiling_ratio = {"small": 0.02, "mid": 0.01, "large": 0.005}[tier]
score = log(1 + amount / market_cap) / log(1 + ceiling_ratio)
return min(score, 1.0)
```

Guards: returns `float("nan")` if `amount` is NaN/negative/zero, `market_cap` is NaN/zero/negative, or `tier` is unrecognized.

### `cross_sectional_norm`
Delegates to `_cross_sectional_normalize` in `backend/market_intel/generate_top_lists.py`.

---

## Section 3 — `detect_anomalies(rows, run_id, log_dir)` + Anomaly Report

### Circuit Breakers

| Flag | Trigger condition | Action |
|------|------------------|--------|
| `VOLUME_SPIKE` | `volume_spike > 10 × universe mean volume_spike` | `flag_only` |
| `INSIDER_CAP_LIMIT` | `insider_usd / market_cap > tier ceiling` (same tiers as log_scale_insider) | `flag_only` |
| `CONGRESS_CLUSTER` | ≥ 3 congress trades on same ticker within 7-day window | `flag_only` |
| `SENTIMENT_EXTREME` | `news_score > 0.95` or `news_score < 0.05` | `flag_only` |
| `STALE_DATA` | Row age > `max_age_days` (5d default) | `flag_only` |
| `STALE_SOURCE` | Source `last_updated` age > 48h | `quarantine` |

`action` semantics:
- `flag_only`: ticker stays in pipeline; Discord shows ⚠️ next to ticker name
- `quarantine`: ticker excluded from `top_buys` (sets `_validation_failed=True`)

### `AnomalyRecord` schema

```json
{
  "run_id": "edgar-2026-05-21-091500",
  "ticker": "BA",
  "timestamp": "2026-05-21T12:00:00Z",
  "flag": "INSIDER_CAP_LIMIT",
  "value": 0.0085,
  "threshold": 0.005,
  "action": "flag_only",
  "source": "quiver"
}
```

### File strategy

- `anomaly_report_{run_id}.json` — one file per run, **never overwritten**. Permanent audit trail.
- `anomaly_report_latest.json` — always a copy of the most recent run. Read by `send_toplists_discord.py`.

Both written atomically (write to `.tmp`, then rename).

### Return value

`detect_anomalies()` returns `list[AnomalyRecord]`. Empty list if no anomalies detected (no file written).

---

## Section 4 — Discord Integration

`send_toplists_discord.py` reads `anomaly_report_latest.json` at send time. For each ticker in `top_buys` that appears in the report:

- `flag_only` → append ` ⚠️` to ticker name in the conviction field
- `quarantine` → ticker already excluded from `top_buys` by `generate_top_lists.py`; no change needed in Discord

If the anomaly report contains any `STALE_SOURCE` entries, append a `diff` code block alert to the embed description (same pattern as kill-switch alerts).

---

## Section 5 — `test_validator.py`

### `TestValidation`

Non-regression tests using fixed mock rows (no network calls):

- `test_valid_row_passes_all_checks` — clean row → no issues
- `test_zero_amount_sets_nan_and_fails` — `insider_usd=0` → field becomes `NaN`, `_validation_failed=True`
- `test_negative_amount_sets_nan` — `market_cap=-1` → `NaN`
- `test_future_date_quarantines_ticker` — timestamp 1 hour ahead → `FUTURE_DATE`, quarantined
- `test_stale_row_flag_only` — row 6 days old → `STALE_DATA`, stays in pipeline
- `test_stale_source_quarantines_all_source_rows` — Quiver `last_updated` 61h ago → all Quiver rows `_validation_failed=True`
- `test_invalid_ticker_quarantined` — `ticker="123X!"` → quarantined
- `test_failure_threshold_raises_integrity_error` — 5/5 tickers fail → `PipelineIntegrityError`
- `test_integrity_error_message_contains_summary` — error message includes failure counts by category
- `test_below_threshold_continues_on_clean_subset` — 1/5 fails → pipeline continues with 4 clean rows

### `TestAnomalyDetection`

Inject absurd values and verify flags:

- `test_volume_spike_flagged` — `volume_spike=150` vs mean of 5 → `VOLUME_SPIKE`
- `test_insider_cap_limit_large` — `insider_usd=0.9% of market_cap` for large-cap → `INSIDER_CAP_LIMIT` (threshold 0.5%)
- `test_insider_cap_limit_small` — same ratio but small-cap ceiling is 2% → no flag
- `test_congress_cluster_flagged` — 3 trades within 7 days → `CONGRESS_CLUSTER`
- `test_sentiment_extreme_high` — `news_score=0.97` → `SENTIMENT_EXTREME`
- `test_sentiment_extreme_low` — `news_score=0.02` → `SENTIMENT_EXTREME`
- `test_anomaly_report_written_to_disk` — verify `anomaly_report_{run_id}.json` created
- `test_anomaly_report_latest_written` — verify `anomaly_report_latest.json` created
- `test_anomaly_report_schema_valid` — all required keys present in every record
- `test_no_file_written_when_no_anomalies` — clean data → no file written

### `TestNormalizer`

- `test_log_scale_insider_large_cap` — 0.5% of market_cap → score ~1.0
- `test_log_scale_insider_small_cap_tighter_ceiling` — same ratio, small tier → score ~1.0 but at lower absolute USD
- `test_log_scale_insider_nan_on_zero_amount` — returns `NaN`
- `test_log_scale_insider_nan_on_zero_cap` — returns `NaN`
- `test_log_scale_insider_nan_on_negative` — returns `NaN`
- `test_log_scale_insider_monotone` — larger purchase = higher score
- `test_winsorize_caps_outliers` — 99th percentile capped
- `test_cross_sectional_norm_zero_mean` — output mean ≈ 0

### CI isolation

All fixtures use in-memory mock rows. No Quiver/FMP/EDGAR HTTP calls. Enforced by existing `conftest.py` `requests.Session.send` block when `CI=1`.

---

## Constraints

- No external dependencies beyond `pandas`, `numpy`, `python-dateutil` (already in requirements)
- All public functions fully type-annotated (`mypy` compatible)
- `validator.py` has no import from `scripts/` — one-way dependency only
- `PipelineIntegrityError` reuses the existing exception from `generate_top_lists.py` (no new exception class)
- `anomaly_report_*.json` files never deleted by the pipeline; log rotation is an ops concern

---

## What This Is NOT

- Not a replacement for `regime_trader/scoring/normalize.py` — `Normalizer` delegates to it
- Not a re-implementation of `_schema_gate()` in `generate_top_lists.py` — the schema gate checks factor completeness post-scoring; this checks raw data quality pre-scoring
- Not a new scoring model — `log_scale_insider` is a normalization transform, not a new weight
