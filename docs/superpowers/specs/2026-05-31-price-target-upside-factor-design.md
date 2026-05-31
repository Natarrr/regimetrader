# Design: `price_target_upside` Factor

**Date:** 2026-05-31  
**Status:** Approved

---

## Problem

`FMPClient.get_price_target_consensus()` already fetches and caches analyst consensus price targets, but the data is unused. Analyst price targets are a forward-looking signal — they capture where sell-side analysts expect the stock to go, orthogonal to price momentum (backward-looking, Jegadeesh-Titman 1993). Adding "upside to target" as a scored factor captures this signal at zero additional API cost.

---

## Scope

Five changes across four files. No new endpoints, no new cache buckets, no new dependencies.

---

## Changes

### 1. `score_price_target_upside` — `regime_trader/scoring/momentum_signals.py`

New scoring function added to `momentum_signals.py`. This file owns all return-signal scorers (backward-looking momentum and attention); price target upside is the forward-looking complement and belongs here.

```python
def score_price_target_upside(target_price: float, current_price: float) -> float:
```

**Formula:**
```
upside  = (target_price - current_price) / current_price
clipped = max(-0.50, min(+0.50, upside))
score   = round((clipped + 0.50) / 1.00, 4)   # linear map → [0, 1]
```

**Score semantics:**
| upside | score |
|---|---|
| +50% or more | 1.00 |
| +25% | 0.75 |
| 0% (at target) | 0.50 |
| −25% | 0.25 |
| −50% or worse | 0.00 |

**Guard:** Either arg is `None`, falsy (0), or non-numeric → return `0.0` (dead signal, not neutral — consistent with `score_momentum_long` and all other scorers in the codebase).

**Docstring must explain:** forward-looking vs backward-looking distinction — price momentum (Jegadeesh-Titman 1993) measures past 12-1m returns; price target upside measures where analysts expect the price to go, which is an independent alpha source.

---

### 2. `get_upside_to_target` — `regime_trader/services/fmp_client.py`

New method on `FMPClient`:

```python
def get_upside_to_target(self, ticker: str) -> Optional[float]:
```

- Calls `self.get_price_target_consensus(ticker)` → reads `data.get("targetConsensus")`
- Calls `self.get_quote(ticker)` → reads `data.get("price")`
- Returns `score_price_target_upside(target, current)` on success
- Returns `None` when either value is missing, zero, or non-numeric
- **Writes nothing to cache** — delegates entirely to `get_price_target_consensus()` (bucket: "ratings", 6h TTL) and `get_quote()` (bucket: "quote", 5min TTL), both of which handle caching internally. Zero additional API calls.
- Catches any exception and returns `None` (soft-fail)

Import: `from regime_trader.scoring.momentum_signals import score_price_target_upside` added inside the method (or at module level — follow the existing pattern for scoring imports in `fmp_client.py`).

---

### 3+4. WEIGHTS + FACTOR_FIELDS — two files, identical change

**Both** `scripts/run_pipeline.py` and `backend/market_intel/generate_top_lists.py` maintain their own copies of `WEIGHTS` and `FACTOR_FIELDS`. Both must be updated identically.

**WEIGHTS change:**
```python
"congress":            0.13,   # reduced 0.17→0.13 to fund price_target_upside
"price_target_upside": 0.04,   # forward-looking analyst target signal
```

**Sum verification (must hold):**
```
0.30 (insider_conviction)
+ 0.12 (insider_breadth)
+ 0.13 (congress)          ← was 0.17
+ 0.10 (news_sentiment)
+ 0.03 (news_buzz)
+ 0.15 (momentum_long)
+ 0.03 (volume_attention)
+ 0.04 (analyst_consensus)
+ 0.06 (analyst_revision)
+ 0.04 (price_target_upside)  ← new
= 1.00 ✓
```

Both files already have `assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6` — this will catch any arithmetic error at import time.

**FACTOR_FIELDS change** (both files):
```python
"price_target_upside": "price_target_upside_score",
```

---

### 5. `_score_ticker()` — `scripts/run_pipeline.py`

Inside the `_score_ticker` function's main scoring block, add one call:

```python
# None = no analyst coverage / data missing → dead signal (penalised by normalizer).
# Not the same as a ticker at-target, which would score 0.50.
price_target_upside_score = _fmp_client.get_upside_to_target(ticker) or 0.0
```

The `or 0.0` converts `None` (no coverage) to a dead signal that the cross-sectional normalizer penalizes — the intended behavior. **This is not the same as 0.50 (at-target).** The comment is mandatory so future developers don't confuse "no data" with "no upside expectation."

Add to **result dict**:
```python
"price_target_upside_score": price_target_upside_score,
```

Add to **fallback dict** (the `except Exception` branch):
```python
"price_target_upside_score": 0.0,
```

Add to **`_score_ticker_international`** return dict and exception dict with `None` (structurally absent for EU/Asia — same pattern as `congress_score`, `insider_conviction_score`, etc.):
```python
"price_target_upside_score": None,
```

---

## What does not change

- No new FMP endpoint (reuses `stable/price-target-consensus` and `stable/quote`)
- No new cache bucket
- `validate_analysis_schema` / `ClaudeClient` — untouched
- `_score_ticker_international` factors beyond adding the `None` sentinel
- The `_SCHEMA_MISSING_THRESHOLD` (4) — with 10 factors now, a threshold of 4 still gives appropriate circuit-breaker sensitivity

---

## API cost

Zero. Both `get_price_target_consensus` and `get_quote` are already called per-ticker in every pipeline run. `get_upside_to_target` computes from cached results.

---

## Weight reduction rationale

Congress data (S3 Stock Watcher feeds + FMP fallback) is structurally sparse: most tickers score 0.0 because congress members rarely trade individual stocks. A dead signal at 17% weight wastes model capacity. Reducing to 13% and funding a 4% forward-looking analyst target signal improves information density without touching the high-conviction factors (insider_conviction 30%, momentum_long 15%, analyst_revision 6%).
