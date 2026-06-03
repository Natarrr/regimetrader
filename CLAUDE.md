## PROJECT CONTEXT: regime_trader

- **Goal:** Institutional-grade automated quantitative trading pipeline.
- **Environment:** Python-based, orchestrated via GitHub Actions.
- **Data Sources:** Financial Modeling Prep (FMP) Ultimate tier.
- **Core Philosophy:** Safety-first, evidence-based alpha, and strict orthogonality.
- **Key Constraints:**
  - Do not introduce dependencies outside the `requirements.txt` environment.
  - All logic must support CI/CD pipeline compatibility (no interactive/blocking code).
  - Pipeline status is determined by artifact state, not live web scraping.

---

## Architecture Overview

```text
scripts/run_pipeline.py               ← EDGAR + FMP fetch + per-ticker scoring
backend/market_intel/
  generate_top_lists.py               ← cross-sectional normalisation → top_lists.json
  satellite_factors.py                ← macro / commodity overlays
  validator.py                        ← Stage 1 gate (quarantines bad rows)
regime_trader/
  weights.py                          ← canonical 12-factor WEIGHTS (single source of truth)
  scoring/
    market_config.py                  ← per-market factor availability + weight renorm
    insider_signals.py
    momentum_signals.py
    news_signals.py
    neutralization.py
  services/fmp_client.py              ← all FMP stable/ API calls
  utils/io.py                         ← save_json_atomic()
scripts/fmp_bulk_prefetch.py          ← FMP bulk endpoint pre-fetcher (TTL cache)
monitoring/
  check_metrics.py
  metrics_exporter.py
  minsky_alert.py
tests/                                ← 887 tests; must all pass before any commit
.github/workflows/                    ← edgar_3x, canary, nightly_edgar,
                                         hybrid_pipeline, weekly_backtest
```

---

## Critical Invariants — Never Break These

### Weight schema (`regime_trader/weights.py`)

`WEIGHTS` is a **12-factor dict** and the **single source of truth**. Both `run_pipeline.py` and
`generate_top_lists.py` import from it. Never define WEIGHTS anywhere else (not in
`config/weights.py` or inline).

Tests enforce all of the following simultaneously — violating any one breaks CI:

- `sum(WEIGHTS.values()) == 1.0` (tolerance 1e-6)
- `momentum_long` must be the **largest** single weight
- `congress <= 0.10` (sparse US-only binary signal)
- `volume_attention <= 0.05` (attention tilt, not alpha)
- All 12 values strictly positive (no zeros)

Current distribution (do not redistribute without regenerating the golden record):

```python
{
    "momentum_long":       0.21,   # must remain the max
    "insider_conviction":  0.20,
    "insider_breadth":     0.10,
    "news_sentiment":      0.10,
    "analyst_consensus":   0.08,
    "congress":            0.08,   # must remain <= 0.10
    "quality_piotroski":   0.06,
    "news_buzz":           0.05,
    "analyst_revision":    0.04,
    "price_target_upside": 0.03,
    "volume_attention":    0.03,   # must remain <= 0.05
    "transcript_tone":     0.02,
}
```

### `FACTOR_FIELDS` in `generate_top_lists.py`

Must contain exactly the same 12 keys as `WEIGHTS`. A weight key change requires updating
`FACTOR_FIELDS` in the same commit.

### `MARKET_FACTORS` in `regime_trader/scoring/market_config.py`

- `Market.US` → exactly **12 factors** (WEIGHTS keys + `_score` suffix)
- `Market.EUROPE` and `Market.ASIA` → exactly **4 factors**:
  `momentum_long_score`, `volume_attention_score`, `quality_piotroski_score`, `price_target_upside_score`
- Structurally absent for EU/Asia (must be `None`, not `0.0`): `insider_conviction_score`,
  `insider_breadth_score`, `congress_score`, `news_sentiment_score`, `news_buzz_score`,
  `analyst_consensus_score`, `analyst_revision_score`, `transcript_tone_score`

### Dead-signal convention

- `0.0` = feed alive but no data for this ticker (penalised in cross-sectional normaliser)
- `None` = structurally absent for this market (weight excluded from renorm denominator)
- Never return `0.5` for a missing feed — it silently wastes the factor's weight at neutral

### Golden record

`tests/fixtures/golden_top_lists_2026_05_16.json` pins expected scores. Any weight or
normalisation change requires regeneration:

```python
from backend.market_intel.generate_top_lists import generate
from unittest.mock import patch
result = generate(golden_input, run_id="golden-regression", log_dir=tmp)
# overwrite the fixture file with json.dumps(result)
```

### FMP API routing

Always use `https://financialmodelingprep.com/stable/` routes via `FMPClient`. Never use
`/api/v3/` or `/api/v4/` directly — those are retired. `FMPClient.health_report()` tracks
endpoint failures; a non-zero count means a factor is silently zeroed. Swallowing connection
drops by zero-filling is forbidden — raise `FMPEndpointError` so the circuit breaker fires.

---

## Signal Quality Rules (from Layer 1 audit)

### IC ranges by factor (reference only — do not hard-code these)

| Factor | Empirical IC range | Signal half-life |
| ------ | ------------------ | ---------------- |
| `insider_conviction` | 0.03–0.06 | ~63 trading days |
| `insider_breadth` | 0.03–0.06 | ~63 trading days |
| `congress` | 0.02–0.05 | ~42–63 trading days |
| `news_sentiment` | 0.04–0.08 | 1–3 trading days |
| `news_buzz` | 0.02–0.04 | 5–10 trading days |
| `momentum_long` | 0.02–0.05 | 126–252 trading days |
| `volume_attention` | 0.02–0.04 | 5–10 trading days |
| `analyst_consensus` | 0.03–0.05 | 21–42 trading days |
| `analyst_revision` | 0.03–0.05 | 21–42 trading days |
| `quality_piotroski` | 0.03–0.06 | 63–252 trading days |
| `price_target_upside` | 0.01–0.03 | 30–90 days (stale >90d → 0.0) |
| `transcript_tone` | 0.02–0.04 | ~5–21 trading days |

### Regime-sensitivity rules

- `momentum_long` crashes during sharp VIX spikes. The `_MOMENTUM_REGIME_MULTIPLIERS`
  dampener in `generate_top_lists.py` handles this — do not add a separate dampener.
- `quality_piotroski` and `analyst_consensus` are defensive: their IC rises in bear regimes.
- During VIX ≥ 30 all stock-specific signals are overwhelmed by macro beta. The VIX
  kill-switch (×0.50 at VIX ≥ 30, ×0.20 at VIX ≥ 40) handles this — do not double-damp.

### Cross-sectional normalisation rules

- US and EU/Asia assets are **never pooled** into the same normalisation vector. EU/Asia tickers
  have structural zeros for insider/congress, which would distort the US peer group.
- Normalisation runs inside `generate_top_lists.py` via
  `regime_trader/scoring/neutralization.py`; raw scores from `run_pipeline.py` are unnormalised.
- Winsorization is applied per-factor (5th/95th percentile by default; disabled for factors
  where > 90% of values are exactly 0.0 to preserve sparse signal).
- `congress_score` for non-US tickers = `0.0` (hard zero, never `0.5`).

### Factor correlation / orthogonality

- `insider_conviction` and `insider_breadth` share the same FMP endpoint — they are
  intentionally orthogonal decompositions (dollar conviction vs. distinct-insider breadth),
  not redundant. Do not merge them.
- `news_sentiment` and `news_buzz` share the same corpus — treat as a correlated pair;
  their combined weight budget (currently 0.15) should not grow above ~0.15–0.18.
- Run `log_conviction_breadth_correlation()` and the `_pearson()` diagnostics in
  `run_pipeline.py` after any factor change; warn if ρ exceeds threshold.
- The orthogonality report (`factor_orthogonality` key in `intel_source_status.json`) is
  written every run — check it before increasing correlated factor weights.

### Timing / look-ahead bias rules

- Anchor EPS surprise signals on the public `filingDate`, not `transactionDate` or
  fiscal-period-end date.
- PEAD boost decays with a 20-day half-life (see `score_news_sentiment_combined` in
  `run_pipeline.py`). Do not use a flat 90-day window — it overstates old surprises.
- IC backtests must de-overlap snapshots to horizon spacing (e.g. every 21 trading days
  for a 21d horizon). Daily snapshots + 21d forward return inflates significance
  (López de Prado leakage). See `tests/test_pipeline_integrity.py::TestFetchEdgarDataRecency`.

---

## Scoring Pipeline Rules (from Layer 2–4 audit)

### VIX overlay (`_apply_vix_overlay` in `generate_top_lists.py`)

```python
VIX >= 40  →  score × 0.20   # crash — strips all BUY badges
VIX >= 30  →  score × 0.50   # panic kill-switch
VIX >= 25  →  score × 0.80   # bear caution
VIX  < 25  →  score × 1.00   # normal
```

SELL signals pass through unaffected at all VIX levels.

### Momentum regime dampener

Applied before the VIX overlay when SPY 63d return < -10% (BEAR_MOMENTUM) or < -20%
(BEAR_CRASH). If VIX overlay is already active (VIX ≥ 30), the momentum dampener is skipped
to prevent double-dampening. Values live in `_MOMENTUM_REGIME_MULTIPLIERS` in
`generate_top_lists.py`.

### Piotroski gate (multiplicative, applied after linear combination)

- F-Score < 3 → BUY ×0.0 (suppressed)
- F-Score 3–5 → BUY ×0.6 (discounted)
- F-Score 6–9 → BUY ×1.0 (full)
- SELL signals: gate does NOT apply (asymmetric protection)
- Missing data: treat as F-Score = 3 (discounted, not suppressed)
- Gate applies equally to EU/Asia (accounting identities are exchange-agnostic)

### Analyst consensus scoring

Mapped from `consensusRating` field in `upgrades-downgrades-consensus-bulk`:

```python
"Strong Buy" / "strongBuy"   → 1.00
"Buy"        / "buy"         → 0.75
"Hold"       / "hold"        → 0.50
"Sell"       / "sell"        → 0.25
"Strong Sell"/ "strongSell"  → 0.00
No coverage                  → 0.50  (neutral — not penalised like congress)
```

EU/Asia `analyst_consensus_score` = `None` (structurally absent; FMP grades-consensus is US-only).

### Price target upside scoring

Stale targets (> 90 days old) must return `0.0`, not a cached value.
Clip raw upside to [-0.5, +1.5] to prevent outlier distortion.

### Position concentration

When multiple correlated factors fire on the same ticker, the composite score inflates.
Any position-sizing layer must enforce a hard cap (e.g. 5% max portfolio weight per asset).
The `_correlated_signal_flag` and `_correlated_signal_discount_advisory` fields in pipeline
output are diagnostic — they do not auto-apply a discount; that is a human decision.

---

## Data Flow and Bulk Cache

```text
fmp_bulk_prefetch.py       ← 7 bulk endpoints → .cache/bulk_snapshots/
        ↓
run_pipeline.py            ← bulk indexes + EDGAR + per-ticker scoring
        ↓ writes
logs/intel_source_status.json
        ↓
generate_top_lists.py      ← cross-sectional normalise → top_lists.json
        ↓
send_toplists_discord.py   ← Discord webhook
```

Always pass `--bulk-cache .cache/bulk_snapshots` to both `run_pipeline.py` and
`generate_top_lists.py` in CI. Bulk endpoints replace ~1,280 per-ticker FMP calls with 7 calls.
Adding new bulk endpoints adds +0 incremental live HTTP calls per run (cache is shared across
the three intraday edgar_3x runs via GitHub Actions cache).

### FMP endpoint health matrix

| Endpoint (stable/ route) | Used for | Bulk cached |
| ------------------------- | -------- | ----------- |
| `historical-price-eod/full` | price history, momentum, volume | No (per-ticker) |
| `insider-trading/search` | insider conviction + breadth | No (per-ticker) |
| `news/stock` | news sentiment + buzz | No (per-ticker) |
| `financial-scores-bulk` | quality_piotroski | Yes |
| `upgrades-downgrades-consensus-bulk` | analyst_consensus | Yes |
| `earnings-surprises-bulk` | PEAD boost | Yes |
| `price-target-summary-bulk` | price_target_upside | Yes |
| `ratios-ttm-bulk` | Piotroski verification | Yes |
| `key-metrics-ttm-bulk` | fundamental context | Yes |
| `batch-eod-prices` (eod-bulk) | EOD price data | Yes |

All endpoints must raise `FMPEndpointError` on HTTP 4xx. Never zero-fill on auth failure.

### EU/Asia data sourcing

- News for EU/Asia: FMP `news/stock` returns empty — use `0.0` (dead signal), not a fallback.
- yfinance remains legitimate only for: EU/Asia price history, SPY/EZU/AAXJ benchmark returns.
- International tickers require exact exchange suffixes (`.L`, `.DE`, `.T`, `.PA`) for correct
  FMP ingestion.
- 13F filings require `year` + `quarter` parameters, not a date range.
- Congressional trades: S3 Stock Watcher feeds are primary; FMP senate/house is the fallback.

---

## Claude NLP Layer

`ClaudeClient` is a data-isolated Anthropic wrapper with zero internal FMP dependencies.
It operates on pre-fetched transcript text passed in from `run_pipeline.py`.

Token cost model (claude-sonnet-4-6, 12:30 UTC run, 20-ticker shortlist):

- Input: 20 symbols × 8,600 tokens × $3/Mtoken = $0.516
- Output: 20 symbols × 450 tokens × $15/Mtoken = $0.135
- Total: ~$0.65 per run (well within the $2.00 safety gate)

The `preflight_cost_gate` job in `hybrid_pipeline.yml` enforces this cap before any API call.
Use `prompt_version` to track schema changes in `ClaudeClient.analyze()`.

Transcript prompts must request a strict JSON response schema with `sentiment_score`,
`confidence`, `key_signals`, and `tone`. The `cross_check_citations` function validates that
quantitative claims in the Claude output are supported by the pipeline's quant data.

---

## Tests

Run: `python -m pytest` (887 tests, ~25s).

Key test files that enforce hard invariants:

| File | What it locks |
| ---- | ------------- |
| `tests/test_weights_consistency.py` | weight sum, max factor, congress/volume thresholds |
| `tests/scoring/test_market_parity.py` | 12 US factors, 4 EU/Asia factors, None vs 0.0 |
| `tests/test_pipeline_integrity.py` | 12-key `factors` dict, scoring function contracts |
| `tests/test_golden_record.py` | exact scores, ranking order, badges |

Test helpers live in `tests/conftest.py`. Key fixtures: `_raw_row()`, `_make_eu_entry()`,
`_bind_st` autouse.

Mock pattern for `@st.cache_data`:

```python
with patch("streamlit.cache_data", lambda f: f):
    ...
```

---

## CI/CD

GitHub Actions workflows:

- **edgar_3x.yml** — 3×/day full pipeline (00:00, 08:00, 16:00 UTC). Timeout: 25 min.
- **canary.yml** — 3×/day 10-ticker health check (06:00, 12:00, 18:00 UTC). Timeout: 10 min.
- **nightly_edgar.yml** — manual backfill only (`workflow_dispatch`).
- **hybrid_pipeline.yml** — weekdays 12:30 UTC, Claude analysis on shortlist.
- **weekly_backtest.yml** — Fridays 21:00 UTC, per-factor IC report.

The `hybrid_pipeline` at 12:30 UTC consumes the frozen artifact from the 08:00 UTC
`edgar_3x` run — never reads live files.

GitHub Actions repository variable required: `FMP_MAX_RPS = 50` (FMP Ultimate = 3,000 req/min).
