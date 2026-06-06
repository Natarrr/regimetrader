# Design Spec: qlib Research Sandbox Integration
*Date: 2026-06-06 | Author: Nathan T + Claude*

## Problem Statement

regime_trader's 9-factor composite uses academically-justified weights, but has no empirical proof those weights carry forward IC on the actual 160-ticker universe. The backtest ledger is empty and the archive is only 17 days old. Weights may be misallocated (e.g., insider_conviction at 0.30 may underperform on large-cap S&P 500 tickers where most insider trades are non-informational).

## Goal

Validate → Calibrate → Portfolio-optimize, sequenced across three phases, using qlib as a local research sandbox. Production pipeline stays clean — qlib is never imported in production code.

---

## Architecture & Boundaries

```
regime_trader/
├── research/                          ← NEW (gitignored from CI, local only)
│   ├── .gitignore                     # __pycache__, qlib data, model artifacts
│   ├── requirements-research.txt      # qlib, lightgbm, jupyter, plotly, shap
│   ├── data/
│   │   ├── backfill/                  # FMP historical factor scores (NDJSON)
│   │   └── qlib_data/                 # qlib binary dataset (built from backfill)
│   ├── notebooks/
│   │   ├── 01_ic_analysis.ipynb
│   │   ├── 02_weight_calibration.ipynb
│   │   └── 03_portfolio_construction.ipynb
│   └── scripts/
│       ├── backfill_factors.py        # FMP historical endpoints → factor scores
│       ├── build_qlib_dataset.py      # backfill NDJSON → qlib binary format
│       ├── train_lgbm.py              # LightGBM training + SHAP weight export
│       └── export_weights.py          # optimal_weights.json → config/weights.py
│
├── regime_trader/config/weights.py    ← updated only by export_weights.py
├── backend/market_intel/generate_top_lists.py  ← Phase 4 MVO added natively
└── logs/archive/                      ← live daily snapshots continue unchanged
```

**Production boundary rule**: Nothing in `research/` is ever imported by production code. The only artifact that crosses the boundary is the output of `export_weights.py` writing a validated dict into `config/weights.py`.

**Data flow**:
```
FMP historical API
  → backfill_factors.py → data/backfill/factor_scores.ndjson
  → build_qlib_dataset.py → data/qlib_data/
  → train_lgbm.py → research/optimal_weights.json
  → export_weights.py → regime_trader/config/weights.py
```

---

## Phase 1: Historical Backfill

**Goal**: Reconstruct factor scores for 160 tickers at weekly granularity, 52 weeks back. Produces 8,320 (ticker, snapshot_date) data points — sufficient for non-overlapping 21-day IC computation.

**Sampling**: Every Friday close for 52 weeks. Avoids day-of-week bias and keeps API calls manageable on FMP Ultimate (50 req/s).

**Per-factor reconstruction**:

| Factor | Endpoint | Method |
|--------|----------|--------|
| `momentum_long` | `stable/historical-price-eod/full` | Filter prices ≤ D, compute 12-1m SPY-relative return |
| `insider_conviction` | `stable/insider-trading/search` | Filter `transactionDate ≤ D`, 90-day lookback |
| `insider_breadth` | same | P-code consensus among filings in [D-90, D] |
| `congress` | S3 Stock Watcher all_transactions.json | Filter `transaction_date ≤ D` |
| `news_sentiment` | `stable/news/stock?from=&to=` | Slice [D-30, D], apply existing decay formula |
| `news_buzz` | same | Article count in [D-7, D] |
| `analyst_consensus` | `stable/upgrades-downgrades-consensus-bulk` | Treat as slow-moving, use current snapshot ⚠️ |
| `quality_piotroski` | `stable/ratios-ttm` anchored to `filingDate ≤ D` | Most recent filing before D |

⚠️ **Known limitation**: `analyst_consensus` historical reconstruction uses the current bulk snapshot as a proxy for all past dates. Analyst ratings are slow-moving (median revision frequency ~6 weeks), so this is conservative but not point-in-time accurate. IC results for this factor should be treated as indicative only; they will tighten once the live archive accumulates 90+ days of real snapshots.

**Forward return label**: `(price_D+21 - price_D) / price_D` minus SPY return over same window. SPY-relative is consistent with the momentum signal construction.

**Output**: `data/backfill/factor_scores.ndjson` — one record per (ticker, snapshot_date):
```json
{
  "ticker": "AAPL",
  "snapshot_date": "2025-08-01",
  "insider_conviction": 0.72,
  "insider_breadth": 0.45,
  "congress": 0.30,
  "news_sentiment": 0.61,
  "news_buzz": 0.38,
  "momentum_long": 0.84,
  "volume_attention": 0.22,
  "analyst_consensus": 0.70,
  "quality_piotroski": 0.80,
  "forward_return_21d": 0.034,
  "spy_return_21d": 0.018
}
```

---

## Phase 2: IC Validation (Notebook 01)

**Goal**: Per-factor rank IC against 21d forward return. Surface which factors are earning their weight.

**Metrics**:

| Metric | Formula |
|--------|---------|
| Rank IC | Spearman(factor_score, fwd_return_21d) per snapshot |
| Mean IC | Average rank IC across 52 snapshots |
| IC IR | Mean IC / Std IC |
| IC > 0 rate | % snapshots where IC > 0 |
| Monthly IC heatmap | IC by calendar month (seasonality detection) |

**`weight_recommendation` logic** (bridges IC → Phase 3):

```python
if mean_ic < 0:
    recommendation = "investigate"       # signal may be inverting
elif ic_ir > 0.5 and ic_positive_rate >= 0.60:
    recommendation = "increase"
elif ic_ir >= 0.3:
    recommendation = "hold"
else:
    recommendation = "decrease"
```

**`research/ic_report.json` schema per factor**:
```json
{
  "insider_conviction": {
    "mean_ic": 0.042,
    "ic_ir": 0.61,
    "ic_positive_rate": 0.73,
    "monthly_ic": {"2025-06": 0.05, "2025-07": 0.03},
    "weight_recommendation": "increase"
  }
}
```

**Notebook charts**:
1. Monthly IC bar chart per factor (positive green, negative red)
2. Factor IC IR bar chart with current weight overlay
3. Cumulative IC over time (signal decay detection)

---

## Phase 3: LightGBM Weight Calibration (Notebook 02)

**Goal**: Train LightGBM on historical factor scores → returns. Use SHAP importances, stability-adjusted and blended with academic priors, to derive validated weights.

**Training setup**:
```
Features:  9 factor scores (factors flagged "investigate" excluded)
Label:     SPY-relative 21d forward return
Model:     LightGBM regression (LGBMRegressor)
Folds:     Walk-forward expanding window, 2 folds
           Fold 1: train weeks 1-17  → validate weeks 18-34
           Fold 2: train weeks 1-34  → validate weeks 35-52
```

**Monotonicity constraints** (enforced in LightGBM natively):
```python
MONOTONE_CONSTRAINTS = {
    "insider_conviction": +1,
    "insider_breadth":    +1,
    "congress":           +1,
    "news_sentiment":     +1,
    "news_buzz":           0,   # attention can be contrarian
    "momentum_long":      +1,
    "volume_attention":    0,
    "analyst_consensus":  +1,
    "quality_piotroski":  +1,
}
```

**SHAP stability check** (prevents over-reacting to noise):
```python
shap_per_fold = [shap_fold1, shap_fold2]
shap_mean = mean(shap_per_fold, axis=0)
shap_cv   = std(shap_per_fold, axis=0) / shap_mean   # coefficient of variation
stability_multiplier = clip(1 - shap_cv, 0.3, 1.0)
shap_weights_stable  = shap_mean * stability_multiplier
shap_weights_stable /= sum(shap_weights_stable)       # re-normalize
```

Factors with CV > 0.5 are penalized before the academic blend.

**Blending rule** (academic weights as Bayesian prior):
```python
BLEND_ALPHA = 0.6   # trust data 60%, academic prior 40%
final_weight = BLEND_ALPHA * shap_weights_stable + (1 - BLEND_ALPHA) * academic_weight
```

**`"investigate"` factor handling**: Excluded from LightGBM training entirely. Their weight in `optimal_weights.json` is set to their academic weight unchanged — the model has no data-driven opinion on them. Flagged with `"shap_cv": null` and `"stability_multiplier": null` in the output JSON.

**Hard constraints** (enforced post-blend):
```python
WEIGHT_FLOOR = 0.05   # no factor collapses to zero due to noise
WEIGHT_CAP_MULTIPLIER = 2.0  # no factor exceeds 2× its academic weight

for factor, academic_w in ACADEMIC_WEIGHTS.items():
    if factor in INVESTIGATE_FACTORS:
        final_weights[factor] = academic_w  # unchanged
        continue
    w = final_weights[factor]
    w = max(w, WEIGHT_FLOOR)                        # floor
    w = min(w, WEIGHT_CAP_MULTIPLIER * academic_w)  # cap
    final_weights[factor] = w

# Re-normalize after floor + cap so sum = 1.0
total = sum(final_weights.values())
final_weights = {k: v / total for k, v in final_weights.items()}
assert abs(sum(final_weights.values()) - 1.0) < 1e-6
```

**`research/optimal_weights.json` schema**:
```json
{
  "generated_at": "2026-06-06",
  "blend_alpha": 0.6,
  "weight_floor": 0.05,
  "lgbm_val_ic_per_fold": [0.038, 0.044],
  "lgbm_val_ic_mean": 0.041,
  "weights": {
    "insider_conviction": {
      "academic": 0.30,
      "shap_per_fold": [0.36, 0.40],
      "shap_cv": 0.08,
      "stability_multiplier": 0.92,
      "shap_stable": 0.38,
      "final": 0.348,
      "weight_recommendation": "increase"
    }
  }
}
```

**`export_weights.py`** reads `optimal_weights.json` and rewrites `config/weights.py`, producing a git-committable diff showing what changed and why.

---

## Phase 4: Portfolio Construction (Notebook 03 + native in generate_top_lists.py)

**Goal**: Replace score-ranking-only output with formal portfolio weights. The optimizer lives natively in `generate_top_lists.py` — no qlib dependency.

**Expected return proxy** (Grinold & Kahn):
```python
z_scores = (composite_scores - mean) / std
E_returns = IC_estimate * z_scores   # IC from ic_report.json mean_ic
```

**Covariance matrix** — Ledoit-Wolf shrinkage (robust for 160 assets, 252 observations):
```python
from sklearn.covariance import LedoitWolf
lw = LedoitWolf().fit(returns_matrix)   # 12m daily returns
cov = lw.covariance_
```

**MVO optimizer** via `scipy.optimize.minimize` (SLSQP), constraints:
- `sum(w) = 1.0` — fully invested
- `w_i >= 0` — long-only
- `w_i <= 0.10` — max 10% per position
- `sector_weight <= 0.30` — sector concentration cap
- `sum(|w - w_prev|) <= 0.20` — max 20% turnover vs previous

**VIX vol-targeting** (reduces target vol, does not override optimizer):
```python
TARGET_VOL = {
    "normal": 0.15,   # VIX < 25
    "bear":   0.10,   # VIX 25-30
    "panic":  0.05,   # VIX >= 30
}
# Scale weights by (target_vol / portfolio_vol) after optimization
```

**Fallback chain** (always yields a valid weight vector):
```
MVO converges        → use MVO weights
MVO fails            → risk parity (inverse volatility weighting)
Risk parity fails    → score-proportional weights (current behavior)
```

**Output** — additive field on existing top_lists.json entries:
```json
{
  "ticker": "AAPL",
  "final_score": 0.84,
  "badge": "HIGH BUY",
  "portfolio_weight": 0.067,
  "portfolio_weight_method": "MVO",
  "sector_weight_contribution": 0.18
}
```

Tickers outside the top 20 by composite_score receive `portfolio_weight: 0.0`. Ties at position 20 are broken alphabetically by ticker. Badge/score/ranking logic is **unchanged**.

---

## Testing Strategy

### Research scripts (local only, not in CI)

| Test | Validates |
|------|-----------|
| `test_backfill_factors.py` | Factor scores in [0,1]; no future data leaks into snapshot (filingDate anchor) |
| `test_build_qlib_dataset.py` | NDJSON → qlib binary round-trip is lossless |
| `test_ic_report_schema.py` | All 9 factors present; `weight_recommendation` is one of 4 valid values |
| `test_train_lgbm.py` | `optimal_weights.json` sums to 1.0; all weights ≥ 0.05; no factor exceeds 2× academic |
| `test_export_weights.py` | Written `config/weights.py` passes `assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6` |

### Production additions (added to existing CI suite)

| Test | Validates |
|------|-----------|
| `test_mvo_optimizer.py` | Weights sum to 1.0, all ≥ 0, max ≤ 0.10, sector ≤ 0.30 |
| `test_mvo_determinism.py` | Same inputs → identical weights across 10 consecutive runs |
| `test_mvo_fallback.py` | Full fallback chain always yields valid weight vector summing to 1.0 |
| `test_vix_vol_scaling.py` | At VIX ≥ 30, scaled weights produce portfolio vol ≤ 0.05 |
| `test_vix_monotonicity.py` | As VIX increases across thresholds, portfolio leverage never increases |
| `test_sector_exposure.py` | Every ticker in top_lists.json has valid non-null sector mapping |
| `test_portfolio_weight_schema.py` | `portfolio_weight` present in all entries, non-negative, top-20 sum ≤ 1.0 |

All existing tests (weight sum assertion, golden record, stress test, circuit breaker) remain unchanged.

---

## Sequencing

```
Week 1:  Phase 1 — backfill_factors.py + build_qlib_dataset.py
Week 2:  Phase 2 — IC analysis notebook + ic_report.json + weight_recommendation
Week 3:  Phase 3 — LightGBM training + SHAP stability + export_weights.py
Week 4:  Phase 4 — MVO optimizer native in generate_top_lists.py + full test suite
```

Each phase is independently mergeable and does not break production.

---

## Non-Goals

- qlib is never imported in production code
- No changes to Discord embed format (portfolio_weight is metadata only)
- No changes to existing badge logic, VIX overlay, or circuit breaker
- No new GitHub Actions workflows — research is local-only
