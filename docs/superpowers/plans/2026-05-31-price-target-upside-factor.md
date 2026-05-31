# `price_target_upside` Factor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `price_target_upside` as a 4% weighted factor that scores analyst consensus price target upside vs current price, at zero additional API cost.

**Architecture:** New `score_price_target_upside()` scorer in `momentum_signals.py`; new `get_upside_to_target()` method on `FMPClient` that computes the score from two already-cached calls; `WEIGHTS` and `FACTOR_FIELDS` updated in both `run_pipeline.py` and `generate_top_lists.py`; `_score_ticker()` wired up with the correct `None`-vs-`0.0` semantics.

**Tech Stack:** Python 3.11, `pytest`, existing `FMPClient` cache infrastructure, `regime_trader/scoring/momentum_signals.py`.

---

## File Map

| File | Change |
|---|---|
| `regime_trader/scoring/momentum_signals.py` | Add `score_price_target_upside()` |
| `regime_trader/services/fmp_client.py` | Add `get_upside_to_target()` |
| `scripts/run_pipeline.py` | Update `WEIGHTS`, `FACTOR_FIELDS`, `_score_ticker()`, `_score_ticker_international()` |
| `backend/market_intel/generate_top_lists.py` | Update `WEIGHTS`, `FACTOR_FIELDS` (must mirror `run_pipeline.py`) |
| `tests/scoring/test_momentum_news_signals.py` | Add `TestPriceTargetUpside` class |
| `tests/test_fmp_client.py` | Add `TestGetUpsideToTarget` class |

---

## Task 1: `score_price_target_upside` — tests first

**Files:**
- Test: `tests/scoring/test_momentum_news_signals.py`

- [ ] **Step 1: Append the failing test class**

Open `tests/scoring/test_momentum_news_signals.py`. The file currently imports only `score_momentum_long` and `score_volume_attention` from `regime_trader.scoring.momentum_signals`. Add `score_price_target_upside` to that import line, then append this class at the bottom of the file:

```python
class TestPriceTargetUpside:
    """Forward-looking analyst price target signal in [0, 1].

    Semantics:
        0.50 = target == current price (no upside/downside)
        0.75 = 25% upside to target
        0.25 = 25% downside to target
        1.00 = 50%+ upside (clipped)
        0.00 = 50%+ downside (clipped) OR dead signal (None/zero input)
    """

    def test_at_target_scores_neutral(self):
        """Target == current → exactly 0.50 (no upside/downside)."""
        assert score_price_target_upside(100.0, 100.0) == 0.5000

    def test_25pct_upside_scores_0_75(self):
        """25% upside → 0.75."""
        assert score_price_target_upside(125.0, 100.0) == 0.7500

    def test_25pct_downside_scores_0_25(self):
        """25% downside → 0.25."""
        assert score_price_target_upside(75.0, 100.0) == 0.2500

    def test_clips_at_50pct_upside(self):
        """70% upside clipped to 50% → 1.00."""
        assert score_price_target_upside(170.0, 100.0) == 1.0000

    def test_clips_at_50pct_downside(self):
        """-70% downside clipped to -50% → 0.00."""
        assert score_price_target_upside(30.0, 100.0) == 0.0000

    def test_exact_50pct_upside_scores_1(self):
        assert score_price_target_upside(150.0, 100.0) == 1.0000

    def test_exact_50pct_downside_scores_0(self):
        assert score_price_target_upside(50.0, 100.0) == 0.0000

    def test_none_target_returns_dead_signal(self):
        assert score_price_target_upside(None, 100.0) == 0.0

    def test_none_current_returns_dead_signal(self):
        assert score_price_target_upside(100.0, None) == 0.0

    def test_zero_current_price_returns_dead_signal(self):
        """Zero current price → division guard → 0.0."""
        assert score_price_target_upside(100.0, 0.0) == 0.0

    def test_zero_target_returns_dead_signal(self):
        """Zero target is a data error → 0.0."""
        assert score_price_target_upside(0.0, 100.0) == 0.0

    def test_returns_float_rounded_to_4dp(self):
        result = score_price_target_upside(110.0, 100.0)
        assert isinstance(result, float)
        assert result == round(result, 4)

    def test_small_upside(self):
        """5% upside → (0.05 + 0.50) / 1.00 = 0.55."""
        assert score_price_target_upside(105.0, 100.0) == 0.5500
```

Update the import line at the top of the file:

```python
from regime_trader.scoring.momentum_signals import score_momentum_long, score_volume_attention, score_price_target_upside
```

- [ ] **Step 2: Run to confirm failure**

```
pytest tests/scoring/test_momentum_news_signals.py::TestPriceTargetUpside -v
```

Expected: `ImportError: cannot import name 'score_price_target_upside'`

---

## Task 2: Implement `score_price_target_upside`

**Files:**
- Modify: `regime_trader/scoring/momentum_signals.py`

- [ ] **Step 3: Add the function**

Append to `regime_trader/scoring/momentum_signals.py` after `score_volume_attention`:

```python


def score_price_target_upside(
    target_price: float | None,
    current_price: float | None,
) -> float:
    """Analyst consensus price target upside, in [0, 1].

    Captures the forward-looking dimension that backward-looking price momentum
    (Jegadeesh-Titman 1993, 12-1m returns) cannot: where sell-side analysts
    collectively expect the price to go. These two signals are orthogonal —
    a stock can have strong past momentum and low analyst upside (priced in)
    or weak momentum and high analyst upside (re-rating candidate).

    Formula:
        upside  = (target_price - current_price) / current_price
        clipped = max(-0.50, min(+0.50, upside))   # ±50% practical bounds
        score   = round((clipped + 0.50) / 1.00, 4) # linear map → [0, 1]

    Score semantics:
        1.00 = 50%+ upside to target
        0.75 = 25% upside
        0.50 = at target (no upside/downside)
        0.25 = 25% downside
        0.00 = 50%+ downside OR dead signal

    Returns 0.0 (dead signal) when either argument is None, zero, or
    non-numeric. Consistent with score_momentum_long and score_volume_attention:
    a missing/zero input is penalised rather than granted a neutral pass.

    Source: FMPClient.get_price_target_consensus() → stable/price-target-consensus.
    """
    try:
        t = float(target_price)
        c = float(current_price)
    except (TypeError, ValueError):
        return 0.0
    if not t or not c:
        return 0.0
    upside  = (t - c) / c
    clipped = max(-0.50, min(0.50, upside))
    return round((clipped + 0.50) / 1.00, 4)
```

- [ ] **Step 4: Run tests to confirm pass**

```
pytest tests/scoring/test_momentum_news_signals.py::TestPriceTargetUpside -v
```

Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add regime_trader/scoring/momentum_signals.py tests/scoring/test_momentum_news_signals.py
git commit -m "feat(scoring): add score_price_target_upside — forward-looking analyst target signal"
```

---

## Task 3: `FMPClient.get_upside_to_target` — tests first

**Files:**
- Test: `tests/test_fmp_client.py`

- [ ] **Step 6: Append the failing test class**

Open `tests/test_fmp_client.py`. Read the existing `_ok_resp` helper and `client` fixture (already defined at the top). Append this class at the bottom:

```python
class TestGetUpsideToTarget:
    """get_upside_to_target computes score from two already-cached calls.

    Delegates entirely to get_price_target_consensus() and get_quote().
    Writes nothing to cache itself. Returns None on missing/zero data.
    """

    def test_returns_score_when_both_values_present(self, client):
        """25% upside → score 0.75."""
        with patch.object(client, "get_price_target_consensus",
                          return_value={"targetConsensus": 125.0}):
            with patch.object(client, "get_quote",
                              return_value={"price": 100.0}):
                result = client.get_upside_to_target("AAPL")
        assert result == 0.75

    def test_returns_none_when_no_api_key(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FMP_API_KEY", raising=False)
        c = FMPClient(api_key="", cache_root=tmp_path / "fmp")
        result = c.get_upside_to_target("AAPL")
        assert result is None

    def test_returns_none_when_target_missing(self, client):
        with patch.object(client, "get_price_target_consensus", return_value={}):
            with patch.object(client, "get_quote", return_value={"price": 100.0}):
                result = client.get_upside_to_target("AAPL")
        assert result is None

    def test_returns_none_when_price_missing(self, client):
        with patch.object(client, "get_price_target_consensus",
                          return_value={"targetConsensus": 125.0}):
            with patch.object(client, "get_quote", return_value={}):
                result = client.get_upside_to_target("AAPL")
        assert result is None

    def test_returns_none_when_target_is_zero(self, client):
        with patch.object(client, "get_price_target_consensus",
                          return_value={"targetConsensus": 0.0}):
            with patch.object(client, "get_quote", return_value={"price": 100.0}):
                result = client.get_upside_to_target("AAPL")
        assert result is None

    def test_returns_none_when_price_is_zero(self, client):
        with patch.object(client, "get_price_target_consensus",
                          return_value={"targetConsensus": 125.0}):
            with patch.object(client, "get_quote", return_value={"price": 0.0}):
                result = client.get_upside_to_target("AAPL")
        assert result is None

    def test_returns_none_on_exception(self, client):
        with patch.object(client, "get_price_target_consensus",
                          side_effect=RuntimeError("network error")):
            result = client.get_upside_to_target("AAPL")
        assert result is None

    def test_does_not_write_to_cache(self, client, tmp_path):
        """get_upside_to_target is a pure computation wrapper — writes nothing."""
        with patch.object(client, "get_price_target_consensus",
                          return_value={"targetConsensus": 125.0}):
            with patch.object(client, "get_quote", return_value={"price": 100.0}):
                with patch.object(client, "_cache_write") as mock_write:
                    client.get_upside_to_target("AAPL")
        mock_write.assert_not_called()

    def test_at_target_scores_0_50(self, client):
        with patch.object(client, "get_price_target_consensus",
                          return_value={"targetConsensus": 100.0}):
            with patch.object(client, "get_quote", return_value={"price": 100.0}):
                result = client.get_upside_to_target("AAPL")
        assert result == 0.50
```

- [ ] **Step 7: Run to confirm failure**

```
pytest tests/test_fmp_client.py::TestGetUpsideToTarget -v
```

Expected: `AttributeError: 'FMPClient' object has no attribute 'get_upside_to_target'`

---

## Task 4: Implement `FMPClient.get_upside_to_target`

**Files:**
- Modify: `regime_trader/services/fmp_client.py`

- [ ] **Step 8: Add the method**

In `regime_trader/services/fmp_client.py`, insert after `get_price_target_consensus` (around line 691) and before `get_batch_quotes`:

```python
    def get_upside_to_target(self, ticker: str) -> Optional[float]:
        """Analyst consensus price target upside score in [0, 1], or None.

        Computes score_price_target_upside(targetConsensus, currentPrice)
        using two already-cached FMP calls:
          - get_price_target_consensus() → stable/price-target-consensus (ratings bucket, 6h TTL)
          - get_quote()                  → stable/quote (quote bucket, 5min TTL)

        Writes nothing to cache — delegates entirely to those two methods.
        Zero additional API calls: both results are cached from earlier in the
        pipeline run.

        Returns None when:
          - No API key
          - targetConsensus or price is missing, zero, or non-numeric
          - Either delegated call raises an exception

        None signals "no analyst coverage / data missing" — the caller converts
        this to 0.0 (dead signal) via `or 0.0`, which the cross-sectional
        normalizer penalizes. This is distinct from 0.50 (at-target, valid data).
        """
        if not self._api_key:
            return None
        try:
            from regime_trader.scoring.momentum_signals import score_price_target_upside  # noqa: PLC0415
            target_data = self.get_price_target_consensus(ticker)
            quote_data  = self.get_quote(ticker)
            target = target_data.get("targetConsensus")
            price  = quote_data.get("price")
            if not target or not price:
                return None
            return score_price_target_upside(float(target), float(price))
        except Exception as exc:
            log.debug("get_upside_to_target %s failed: %s", ticker, exc)
            return None
```

- [ ] **Step 9: Run tests to confirm pass**

```
pytest tests/test_fmp_client.py::TestGetUpsideToTarget -v
```

Expected: 9 passed.

- [ ] **Step 10: Commit**

```bash
git add regime_trader/services/fmp_client.py tests/test_fmp_client.py
git commit -m "feat(fmp): add get_upside_to_target() — zero-cost analyst target upside score"
```

---

## Task 5: Update `WEIGHTS` and `FACTOR_FIELDS` in `run_pipeline.py`

**Files:**
- Modify: `scripts/run_pipeline.py`

- [ ] **Step 11: Update `WEIGHTS`**

In `scripts/run_pipeline.py`, find the `WEIGHTS` dict (around line 47). Replace the `congress` line and add `price_target_upside`:

```python
WEIGHTS = {
    "insider_conviction":  0.30,
    "insider_breadth":     0.12,
    "congress":            0.13,   # reduced 0.17→0.13 to fund price_target_upside
    "news_sentiment":      0.10,
    "news_buzz":           0.03,
    "momentum_long":       0.15,
    "volume_attention":    0.03,
    "analyst_consensus":   0.04,
    "analyst_revision":    0.06,
    "price_target_upside": 0.04,   # forward-looking analyst target signal (Womack-adjacent)
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, f"WEIGHTS must sum to 1, got {sum(WEIGHTS.values())}"
```

- [ ] **Step 12: Verify the assert passes immediately**

```
python -c "
import sys; sys.path.insert(0, '.')
# Import just the WEIGHTS dict — triggers the assert
exec(open('scripts/run_pipeline.py').read().split('_WEIGHTS_MIGRATION_NOTE')[0])
print('WEIGHTS sum OK:', sum(WEIGHTS.values()))
"
```

Expected output: `WEIGHTS sum OK: 1.0` (or `0.9999...` within 1e-6).

Actually, simpler — just run:

```
python -c "import scripts.run_pipeline" 2>&1 | head -5
```

Expected: no AssertionError.

- [ ] **Step 13: Update `FACTOR_FIELDS` in `run_pipeline.py`**

Find the `FACTOR_FIELDS` dict — note that in `run_pipeline.py` there is no standalone `FACTOR_FIELDS` dict; `FACTOR_FIELDS` only exists in `generate_top_lists.py`. Confirm this is the case before proceeding:

```
grep -n "FACTOR_FIELDS" scripts/run_pipeline.py
```

Expected: no matches (skip this step if confirmed absent).

---

## Task 6: Update `WEIGHTS` and `FACTOR_FIELDS` in `generate_top_lists.py`

**Files:**
- Modify: `backend/market_intel/generate_top_lists.py`

- [ ] **Step 14: Update `WEIGHTS`**

In `backend/market_intel/generate_top_lists.py`, find the `WEIGHTS` dict (around line 55). Apply the identical change:

```python
WEIGHTS: Dict[str, float] = {
    "insider_conviction": 0.30,
    "insider_breadth":    0.12,
    "congress":           0.13,   # reduced 0.17→0.13 to fund price_target_upside
    "news_sentiment":     0.10,
    "news_buzz":          0.03,
    "momentum_long":      0.15,
    "volume_attention":   0.03,
    "analyst_consensus":  0.04,
    "analyst_revision":   0.06,
    "price_target_upside": 0.04,  # forward-looking analyst target signal
}
```

Note: `generate_top_lists.py` does NOT have its own `assert` on the sum — the sum is verified by the test suite (`test_cross_sectional.py` or `test_pipeline_integrity.py`). No assert needed here.

- [ ] **Step 15: Update `FACTOR_FIELDS`**

In `backend/market_intel/generate_top_lists.py`, find the `FACTOR_FIELDS` dict (around line 68). Add the new entry:

```python
FACTOR_FIELDS: Dict[str, str] = {
    "insider_conviction":  "insider_conviction_score",
    "insider_breadth":     "insider_breadth_score",
    "congress":            "congress_score",
    "news_sentiment":      "news_sentiment_score",
    "news_buzz":           "news_buzz_score",
    "momentum_long":       "momentum_long_score",
    "volume_attention":    "volume_attention_score",
    "analyst_consensus":   "analyst_consensus_score",
    "analyst_revision":    "analyst_revision_score",
    "price_target_upside": "price_target_upside_score",
}
```

- [ ] **Step 16: Run the generate_top_lists import check**

```
python -c "from backend.market_intel.generate_top_lists import WEIGHTS, FACTOR_FIELDS; print('WEIGHTS keys:', list(WEIGHTS)); print('sum:', sum(WEIGHTS.values())); print('FACTOR_FIELDS keys:', list(FACTOR_FIELDS))"
```

Expected: both dicts contain `price_target_upside`, sum is 1.0.

- [ ] **Step 17: Commit**

```bash
git add scripts/run_pipeline.py backend/market_intel/generate_top_lists.py
git commit -m "feat(weights): add price_target_upside 0.04, reduce congress 0.17→0.13"
```

---

## Task 7: Wire `_score_ticker()` in `run_pipeline.py`

**Files:**
- Modify: `scripts/run_pipeline.py`

- [ ] **Step 18: Add the call inside `_score_ticker()`**

In `scripts/run_pipeline.py`, find `_score_ticker()`. Inside the main `try:` block, after the `analyst_revision_score` line (around line 1345), add:

```python
            # ── Price target upside (Womack-adjacent: forward-looking target signal) ─
            # None = no analyst coverage / data missing → dead signal (penalised).
            # Not the same as 0.50 (at-target with valid data).
            price_target_upside_score = _fmp_client.get_upside_to_target(ticker) or 0.0
```

- [ ] **Step 19: Add to the result dict**

In the `return { ... }` dict inside `_score_ticker()` (the success path, around line 1393), add after `"analyst_revision_score"`:

```python
                "price_target_upside_score": price_target_upside_score,
```

- [ ] **Step 20: Add to the fallback dict**

In the `except Exception:` branch return dict (around line 1438), add after `"analyst_revision_score": 0.0`:

```python
                "price_target_upside_score": 0.0,
```

- [ ] **Step 21: Add `None` sentinel to `_score_ticker_international()`**

In `_score_ticker_international()` (around line 1136), in the return dict under "Structurally absent — None (not 0.0)", add:

```python
            "price_target_upside_score": None,
```

The existing comment block already explains that `None` means "weight excluded from renormalization denominator" for non-US markets. No new comment needed.

- [ ] **Step 22: Verify the pipeline module imports cleanly**

```
python -c "import scripts.run_pipeline; print('OK')"
```

Expected: `OK`

- [ ] **Step 23: Commit**

```bash
git add scripts/run_pipeline.py
git commit -m "feat(pipeline): wire price_target_upside_score into _score_ticker and _score_ticker_international"
```

---

## Task 8: Full suite verification

- [ ] **Step 24: Run all affected test files**

```
pytest tests/scoring/test_momentum_news_signals.py tests/test_fmp_client.py -v --tb=short 2>&1 | tail -15
```

Expected: all pass, 0 failures.

- [ ] **Step 25: Run the full test suite**

```
pytest tests/ -q --tb=short 2>&1 | tail -20
```

Expected: same number of failures as before this feature (21 pre-existing failures in `test_cross_sectional`, `test_golden_record`, etc. from unstaged changes — confirmed pre-existing). No new failures.

- [ ] **Step 26: Smoke-check the full score path**

```
python -c "
from regime_trader.scoring.momentum_signals import score_price_target_upside
from regime_trader.services.fmp_client import FMPClient

# Score check
print('at_target:', score_price_target_upside(100.0, 100.0))   # 0.5
print('25pct_up:', score_price_target_upside(125.0, 100.0))    # 0.75
print('none_guard:', score_price_target_upside(None, 100.0))   # 0.0

# FMPClient method present
c = FMPClient(api_key='')
print('method callable:', callable(c.get_upside_to_target))
print('no_key_returns_none:', c.get_upside_to_target('AAPL'))  # None

# WEIGHTS and FACTOR_FIELDS
from scripts.run_pipeline import WEIGHTS as W1
from backend.market_intel.generate_top_lists import WEIGHTS as W2, FACTOR_FIELDS
print('run_pipeline sum:', round(sum(W1.values()), 10))
print('generate_top_lists sum:', round(sum(W2.values()), 10))
print('FACTOR_FIELDS has price_target_upside:', 'price_target_upside' in FACTOR_FIELDS)
print('field name:', FACTOR_FIELDS.get('price_target_upside'))
"
```

Expected output:
```
at_target: 0.5
25pct_up: 0.75
none_guard: 0.0
method callable: True
no_key_returns_none: None
run_pipeline sum: 1.0
generate_top_lists sum: 1.0
FACTOR_FIELDS has price_target_upside: True
field name: price_target_upside_score
```

---

## Self-Review Checklist

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| `score_price_target_upside(target, current)` with ±50% clip, linear map | Tasks 1–2 |
| Guard: None/zero/non-numeric → 0.0 | Tasks 1–2 |
| Forward-looking vs backward-looking docstring | Task 2 Step 3 |
| `get_upside_to_target()` on `FMPClient` | Tasks 3–4 |
| Delegates to `get_price_target_consensus()` and `get_quote()` | Tasks 3–4 |
| Returns `None` on missing/zero data | Tasks 3–4 |
| Writes nothing to cache | Task 3 Step 6 (test), Task 4 Step 8 (impl) |
| `WEIGHTS`: congress 0.17→0.13, price_target_upside 0.04 | Tasks 5–6 |
| Sum == 1.0 (assert) | Task 5 Step 12 |
| `FACTOR_FIELDS`: `"price_target_upside" → "price_target_upside_score"` | Task 6 Step 15 |
| Both files updated identically | Tasks 5–6 |
| `_score_ticker()` call with `or 0.0` and mandatory comment | Task 7 Step 18 |
| Result dict includes `price_target_upside_score` | Task 7 Step 19 |
| Fallback dict includes `price_target_upside_score: 0.0` | Task 7 Step 20 |
| `_score_ticker_international()` includes `price_target_upside_score: None` | Task 7 Step 21 |

All requirements covered.
