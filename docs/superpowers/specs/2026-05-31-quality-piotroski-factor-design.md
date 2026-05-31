# Design: `quality_piotroski` Factor

**Date:** 2026-05-31  
**Status:** Approved

---

## Problem

The current factor model has no quality gate: a high-scoring insider-conviction + momentum ticker could be a value trap — deteriorating fundamentals, rising leverage, margin compression. Ilmanen ("Expected Returns", 2011) and Novy-Marx (2013 JF) show that quality companies outperform across regimes, while distressed companies drag down any momentum signal regardless of insider activity. A simplified Piotroski F-score from already-cached `ratios-ttm` data adds this gate at zero additional API cost.

---

## Scope

Five changes across four files. No new endpoints, no new cache buckets, no new dependencies.

---

## Changes

### 1. `score_quality_piotroski` — `regime_trader/scoring/momentum_signals.py`

New function added to `momentum_signals.py`. Quality is a cross-sectional fundamental screen that belongs alongside the return/attention signals; it is not Form 4 specific (which would belong in `insider_signals.py`).

```python
def score_quality_piotroski(ratios: dict) -> float:
```

**8-point simplified F-score:**

| Point | Condition | Field |
|---|---|---|
| 1 | `returnOnAssetsTTM > 0` | Profitable at all |
| 2 | `returnOnAssetsTTM > 0.05` | Strong ROA (>5%) |
| 3 | `operatingProfitMarginTTM > 0` | Positive operating income (OCF proxy) |
| 4 | `debtEquityRatioTTM < 1.0` | Manageable leverage |
| 5 | `debtEquityRatioTTM < 0.5` | Low leverage (bonus) |
| 6 | `currentRatioTTM > 1.5` | Liquid (current assets > 1.5× liabilities) |
| 7 | `grossProfitMarginTTM > 0.30` | 30%+ gross margin = pricing power |
| 8 | `netProfitMarginTTM > 0.05` | Profitable after all costs |

`score = round(points_earned / 8.0, 4)` → `[0, 1]`

**Partial-data handling:** Missing individual fields score 0 for that point but do not collapse the whole score. This means a company with 6 of 8 fields available and 5 passing still scores 5/8 = 0.625.

**Guard:** If `ratios` is `None`, not a dict, or every relevant field is `None` → return `0.0` (dead signal). This is the correct treatment for missing ratios data: the cross-sectional normalizer penalizes 0.0 rather than granting a neutral pass.

**Negative D/E handling:** `debtEquityRatioTTM` can be negative for companies with negative book equity (heavily leveraged or loss-making). Any negative value fails both leverage points (4 and 5) — it is worse than D/E > 1.0, not a sentinel for "no debt."

**References:** Piotroski (2000), "Value Investing: The Use of Historical Financial Statement Information to Separate Winners from Losers", JAR; Novy-Marx (2013), "The Other Side of Value", JFE; Ilmanen (2011), "Expected Returns", Wiley.

---

### 2. `get_quality_score` — `regime_trader/services/fmp_client.py`

New method on `FMPClient`:

```python
def get_quality_score(self, ticker: str) -> float:
```

- Calls `self.get_ratios_ttm(ticker)` — already cached in "ratios" bucket (24h TTL). Zero additional API calls.
- Calls `score_quality_piotroski(ratios)` and returns the result.
- Returns `float` (not `Optional[float]`) — dead signal = 0.0. This differs from `get_upside_to_target` (which returns `None` for missing coverage) because quality is universally available for any company in the ratios-ttm endpoint. A missing ratios response means a broken API, not "no quality data for this ticker."
- Catches exceptions → returns 0.0 (soft-fail, consistent with the dead-signal convention).

---

### 3+4. WEIGHTS + FACTOR_FIELDS — two files

**Both** `scripts/run_pipeline.py` and `backend/market_intel/generate_top_lists.py` must be updated identically.

**WEIGHTS change:**
```python
"insider_breadth":     0.09,   # reduced 0.12→0.09 to fund quality_piotroski
"congress":            0.10,   # reduced 0.13→0.10 (structurally sparse, ~5% density)
"quality_piotroski":   0.06,   # Piotroski (2000) / Novy-Marx (2013) quality gate
```

**Sum verification:**
```
0.30 (insider_conviction)
+ 0.09 (insider_breadth)   ← was 0.12
+ 0.10 (congress)          ← was 0.13
+ 0.10 (news_sentiment)
+ 0.03 (news_buzz)
+ 0.15 (momentum_long)
+ 0.03 (volume_attention)
+ 0.04 (analyst_consensus)
+ 0.06 (analyst_revision)
+ 0.04 (price_target_upside)
+ 0.06 (quality_piotroski)  ← new
= 1.00 ✓
```

Both files have `assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6` — catches arithmetic errors at import time.

**FACTOR_FIELDS change** (both files):
```python
"quality_piotroski": "quality_piotroski_score",
```

**Weight reduction rationale:** Congress has ~5% non-zero density in the S&P 500 universe (most tickers are never traded by congress). Reducing it from 13% → 10% and insider_breadth from 12% → 9% funds quality_piotroski (6%), which fires for all tickers with ratios data (>95% density). This improves the information-per-weight-dollar ratio substantially.

---

### 5. `_score_ticker()` — `scripts/run_pipeline.py`

One new call in the main try block:

```python
# quality_piotroski: Piotroski (2000) / Novy-Marx (2013) fundamental quality gate
quality_piotroski_score = _fmp_client.get_quality_score(ticker)
```

No `or 0.0` needed — `get_quality_score` already returns `float`.

Result dict (success path): `"quality_piotroski_score": quality_piotroski_score`

Fallback dict (except branch): `"quality_piotroski_score": 0.0`

`_score_ticker_international()`: `"quality_piotroski_score": None` (structurally absent — FMP returns 403 for EU/Asia ratios-ttm as confirmed in Phase-0 smoke-test; `None` signals "weight excluded from renormalization" per the existing EU/Asia architecture).

---

## CI Validation

After wiring, run:

```bash
pytest tests/monitoring/test_factor_orthogonality.py -v
```

This confirms the orthogonality monitoring framework still functions (uses synthetic factors — does not check live ρ). The framework is agnostic to new factor names.

Live ρ check (requires a real pipeline run): verify `quality_piotroski_score` vs `insider_conviction_score` Pearson ρ < 0.4 in the factor orthogonality report. The hypothesis: insider conviction (Form 4 purchases) and fundamental quality (balance sheet ratios) are measured from completely different data sources and should be largely orthogonal. A company's CFO buying shares has low correlation with its ROA or current ratio.

---

## What does not change

- `ClaudeClient`, `validate_analysis_schema` — untouched
- `test_factor_orthogonality.py` — untouched (synthetic factors, not sensitive to new real factors)
- `_score_ticker_international` core logic beyond adding the `None` sentinel
- The `_SCHEMA_MISSING_THRESHOLD` (4) — with 11 factors now, threshold still appropriate
