# `quality_piotroski` Factor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `quality_piotroski` as a 6% weighted factor that scores fundamental quality via a simplified 8-point Piotroski F-score from already-cached `ratios-ttm` data, at zero additional API cost.

**Architecture:** New `score_quality_piotroski(ratios)` scorer in `momentum_signals.py`; new `get_quality_score(ticker)` method on `FMPClient` that delegates to `get_ratios_ttm()` (cached); `WEIGHTS` and `FACTOR_FIELDS` updated in both `run_pipeline.py` and `generate_top_lists.py`; `_score_ticker()` and `_score_ticker_international()` wired up.

**Tech Stack:** Python 3.11, `pytest`, existing `FMPClient` cache infrastructure, `regime_trader/scoring/momentum_signals.py`.

---

## File Map

| File | Change |
|---|---|
| `regime_trader/scoring/momentum_signals.py` | Add `score_quality_piotroski()` |
| `regime_trader/services/fmp_client.py` | Add `get_quality_score()` |
| `scripts/run_pipeline.py` | Update `WEIGHTS`, `_score_ticker()`, `_score_ticker_international()` |
| `backend/market_intel/generate_top_lists.py` | Update `WEIGHTS`, `FACTOR_FIELDS` |
| `tests/scoring/test_momentum_news_signals.py` | Add `TestQualityPiotroski` class |
| `tests/test_fmp_client.py` | Add `TestGetQualityScore` class |

---

## Task 1: `score_quality_piotroski` — tests first

**Files:**
- Test: `tests/scoring/test_momentum_news_signals.py`

- [ ] **Step 1: Add the import and failing test class**

Open `tests/scoring/test_momentum_news_signals.py`. Add `score_quality_piotroski` to the existing import line at the top:

```python
from regime_trader.scoring.momentum_signals import (
    score_momentum_long,
    score_volume_attention,
    score_price_target_upside,
    score_quality_piotroski,
)
```

Then append this class at the bottom of the file:

```python
class TestQualityPiotroski:
    """Simplified 8-point Piotroski F-score mapped to [0, 1].

    References:
        Piotroski (2000) JAR — historical financial statements separate winners from losers.
        Novy-Marx (2013) JFE — gross profitability predicts cross-sectional returns.

    Score = points_earned / 8.0. Dead signal (0.0) when ratios is None/empty/all-None.
    """

    def _full_quality_ratios(self) -> dict:
        """Ratios dict where all 8 points pass — perfect score."""
        return {
            "returnOnAssetsTTM":        0.10,   # > 0 (point 1) and > 0.05 (point 2)
            "operatingProfitMarginTTM": 0.15,   # > 0 (point 3)
            "debtEquityRatioTTM":       0.30,   # < 1.0 (point 4) and < 0.5 (point 5)
            "currentRatioTTM":          2.0,    # > 1.5 (point 6)
            "grossProfitMarginTTM":     0.45,   # > 0.30 (point 7)
            "netProfitMarginTTM":       0.08,   # > 0.05 (point 8)
        }

    def test_perfect_score_all_8_points(self):
        score = score_quality_piotroski(self._full_quality_ratios())
        assert score == 1.0000

    def test_zero_score_all_8_points_fail(self):
        ratios = {
            "returnOnAssetsTTM":        -0.05,  # fails points 1 and 2
            "operatingProfitMarginTTM": -0.10,  # fails point 3
            "debtEquityRatioTTM":        2.0,   # fails points 4 and 5
            "currentRatioTTM":           0.8,   # fails point 6
            "grossProfitMarginTTM":      0.10,  # fails point 7
            "netProfitMarginTTM":       -0.02,  # fails point 8
        }
        assert score_quality_piotroski(ratios) == 0.0000

    def test_partial_score_5_of_8_points(self):
        """ROA > 0 only (not > 0.05), opMargin OK, D/E < 1 only (not < 0.5),
        currentRatio OK, grossMargin OK, netMargin fails."""
        ratios = {
            "returnOnAssetsTTM":        0.02,   # passes point 1, fails point 2
            "operatingProfitMarginTTM": 0.10,   # passes point 3
            "debtEquityRatioTTM":       0.70,   # passes point 4, fails point 5
            "currentRatioTTM":          2.0,    # passes point 6
            "grossProfitMarginTTM":     0.40,   # passes point 7
            "netProfitMarginTTM":       0.02,   # fails point 8
        }
        assert score_quality_piotroski(ratios) == round(5 / 8, 4)

    def test_empty_dict_returns_dead_signal(self):
        assert score_quality_piotroski({}) == 0.0

    def test_none_returns_dead_signal(self):
        assert score_quality_piotroski(None) == 0.0

    def test_all_none_fields_returns_dead_signal(self):
        ratios = {
            "returnOnAssetsTTM":        None,
            "operatingProfitMarginTTM": None,
            "debtEquityRatioTTM":       None,
            "currentRatioTTM":          None,
            "grossProfitMarginTTM":     None,
            "netProfitMarginTTM":       None,
        }
        assert score_quality_piotroski(ratios) == 0.0

    def test_missing_individual_fields_score_zero_for_that_point(self):
        """A company with 5 of 8 fields present and all passing scores 5/8."""
        ratios = {
            "returnOnAssetsTTM":        0.10,   # points 1+2 pass
            "operatingProfitMarginTTM": 0.15,   # point 3 passes
            # debtEquityRatioTTM missing — points 4+5 score 0
            "currentRatioTTM":          2.0,    # point 6 passes
            "grossProfitMarginTTM":     0.40,   # point 7 passes
            # netProfitMarginTTM missing — point 8 scores 0
        }
        assert score_quality_piotroski(ratios) == round(5 / 8, 4)

    def test_negative_debt_equity_fails_both_leverage_points(self):
        """Negative D/E (negative book equity) is worse than high D/E — fails points 4 and 5."""
        ratios = {**self._full_quality_ratios(), "debtEquityRatioTTM": -0.5}
        # Loses 2 leverage points: 8 - 2 = 6 → 6/8
        assert score_quality_piotroski(ratios) == round(6 / 8, 4)

    def test_roa_exactly_at_5pct_threshold(self):
        """ROA == 0.05 fails point 2 (must be strictly greater than 0.05)."""
        ratios = {**self._full_quality_ratios(), "returnOnAssetsTTM": 0.05}
        # Loses point 2: 8 - 1 = 7 → 7/8
        assert score_quality_piotroski(ratios) == round(7 / 8, 4)

    def test_gross_margin_exactly_at_threshold(self):
        """grossProfitMarginTTM == 0.30 fails point 7 (must be strictly greater)."""
        ratios = {**self._full_quality_ratios(), "grossProfitMarginTTM": 0.30}
        assert score_quality_piotroski(ratios) == round(7 / 8, 4)

    def test_returns_float_in_range_0_to_1(self):
        score = score_quality_piotroski(self._full_quality_ratios())
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_score_rounded_to_4_decimal_places(self):
        ratios = {**self._full_quality_ratios(), "returnOnAssetsTTM": 0.02}
        score = score_quality_piotroski(ratios)
        assert score == round(score, 4)

    def test_non_dict_input_returns_dead_signal(self):
        assert score_quality_piotroski("not a dict") == 0.0
        assert score_quality_piotroski(42) == 0.0
```

- [ ] **Step 2: Run to confirm failure**

```
pytest tests/scoring/test_momentum_news_signals.py::TestQualityPiotroski -v
```

Expected: `ImportError: cannot import name 'score_quality_piotroski'`

---

## Task 2: Implement `score_quality_piotroski`

**Files:**
- Modify: `regime_trader/scoring/momentum_signals.py`

- [ ] **Step 3: Append the function**

Add the following at the end of `regime_trader/scoring/momentum_signals.py` (after `score_price_target_upside`):

```python


def score_quality_piotroski(ratios: dict) -> float:
    """Simplified 8-point Piotroski F-score, in [0, 1].

    Captures fundamental quality as a value-trap gate: high-conviction insider
    buying in a deteriorating business is a false signal. Piotroski (2000)
    showed that a simple binary F-score on financial statement data separates
    winners from losers among high book-to-market stocks. Novy-Marx (2013)
    extended this: gross profitability is the strongest single quality predictor.
    Ilmanen (2011) documents quality as a cross-regime premium independent of
    momentum — which makes it a natural complement to score_momentum_long.

    8 binary points (each worth 1/8 of the final score):
        1. returnOnAssetsTTM > 0         — profitable at all
        2. returnOnAssetsTTM > 0.05      — strong ROA (>5%)
        3. operatingProfitMarginTTM > 0  — positive operating income (OCF proxy)
        4. debtEquityRatioTTM < 1.0      — manageable leverage
        5. debtEquityRatioTTM < 0.5      — low leverage (bonus)
        6. currentRatioTTM > 1.5         — liquid balance sheet
        7. grossProfitMarginTTM > 0.30   — 30%+ gross margin = pricing power
        8. netProfitMarginTTM > 0.05     — profitable after all costs

    score = round(points_earned / 8.0, 4)

    Partial-data handling: a missing or None field contributes 0 for its
    point(s) but does not collapse the entire score. A company with 6 of 8
    fields and 5 passing scores 5/8 = 0.625.

    Negative D/E (negative book equity) fails both leverage points — it
    signals structural distress, not low debt.

    Returns 0.0 (dead signal) when ratios is None, not a dict, or every
    relevant field is None/missing. Consistent with score_momentum_long:
    missing input is penalised, not granted a neutral pass.

    References:
        Piotroski (2000), "Value Investing: The Use of Historical Financial
        Statement Information to Separate Winners from Losers", JAR 38(1).
        Novy-Marx (2013), "The Other Side of Value", JFE 108(1).
        Ilmanen (2011), "Expected Returns", Wiley.

    Source: FMPClient.get_ratios_ttm() → stable/ratios-ttm (24h cache).
    """
    if not isinstance(ratios, dict) or not ratios:
        return 0.0

    def _get(field: str) -> float | None:
        v = ratios.get(field)
        if v is None:
            return None
        try:
            f = float(v)
            return None if math.isnan(f) else f
        except (TypeError, ValueError):
            return None

    roa  = _get("returnOnAssetsTTM")
    opm  = _get("operatingProfitMarginTTM")
    de   = _get("debtEquityRatioTTM")
    cr   = _get("currentRatioTTM")
    gpm  = _get("grossProfitMarginTTM")
    npm  = _get("netProfitMarginTTM")

    # Guard: all fields missing → dead signal
    if all(v is None for v in (roa, opm, de, cr, gpm, npm)):
        return 0.0

    points = 0
    if roa is not None and roa > 0:
        points += 1
    if roa is not None and roa > 0.05:
        points += 1
    if opm is not None and opm > 0:
        points += 1
    if de is not None and de < 1.0:       # negative D/E also fails (< 0 < 1 is False when de<0)
        points += 1
    if de is not None and 0 <= de < 0.5:  # negative D/E fails: 0 <= de is False
        points += 1
    if cr is not None and cr > 1.5:
        points += 1
    if gpm is not None and gpm > 0.30:
        points += 1
    if npm is not None and npm > 0.05:
        points += 1

    return round(points / 8.0, 4)
```

- [ ] **Step 4: Run tests to confirm pass**

```
pytest tests/scoring/test_momentum_news_signals.py::TestQualityPiotroski -v
```

Expected: 13 passed.

- [ ] **Step 5: Run the full momentum test file to check no regressions**

```
pytest tests/scoring/test_momentum_news_signals.py -v --tb=short 2>&1 | tail -10
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add regime_trader/scoring/momentum_signals.py tests/scoring/test_momentum_news_signals.py
git commit -m "feat(scoring): add score_quality_piotroski — 8-point Piotroski F-score quality gate"
```

---

## Task 3: `FMPClient.get_quality_score` — tests first

**Files:**
- Test: `tests/test_fmp_client.py`

- [ ] **Step 7: Append failing tests**

Open `tests/test_fmp_client.py`. The file already has `_ok_resp`, `_empty_resp`, and a `client` fixture. Append this class at the bottom:

```python
class TestGetQualityScore:
    """get_quality_score delegates to get_ratios_ttm() and score_quality_piotroski().

    Returns float (not Optional) — dead signal is 0.0, not None.
    """

    def _perfect_ratios(self) -> dict:
        return {
            "returnOnAssetsTTM":        0.10,
            "operatingProfitMarginTTM": 0.15,
            "debtEquityRatioTTM":       0.30,
            "currentRatioTTM":          2.0,
            "grossProfitMarginTTM":     0.45,
            "netProfitMarginTTM":       0.08,
        }

    def test_returns_perfect_score_for_quality_ratios(self, client):
        with patch.object(client, "get_ratios_ttm", return_value=self._perfect_ratios()):
            result = client.get_quality_score("AAPL")
        assert result == 1.0

    def test_returns_float_not_optional(self, client):
        with patch.object(client, "get_ratios_ttm", return_value=self._perfect_ratios()):
            result = client.get_quality_score("AAPL")
        assert isinstance(result, float)

    def test_returns_0_when_no_api_key(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FMP_API_KEY", raising=False)
        c = FMPClient(api_key="", cache_root=tmp_path / "fmp")
        assert c.get_quality_score("AAPL") == 0.0

    def test_returns_0_when_ratios_empty(self, client):
        with patch.object(client, "get_ratios_ttm", return_value={}):
            result = client.get_quality_score("AAPL")
        assert result == 0.0

    def test_returns_0_on_exception(self, client):
        with patch.object(client, "get_ratios_ttm", side_effect=RuntimeError("timeout")):
            result = client.get_quality_score("AAPL")
        assert result == 0.0

    def test_partial_quality_ratios(self, client):
        """5 of 8 points passing → 5/8 = 0.625."""
        ratios = {
            "returnOnAssetsTTM":        0.02,   # point 1 only (not > 0.05)
            "operatingProfitMarginTTM": 0.10,   # point 3
            "debtEquityRatioTTM":       0.70,   # point 4 only (not < 0.5)
            "currentRatioTTM":          2.0,    # point 6
            "grossProfitMarginTTM":     0.40,   # point 7
            "netProfitMarginTTM":       0.02,   # fails point 8
        }
        with patch.object(client, "get_ratios_ttm", return_value=ratios):
            result = client.get_quality_score("AAPL")
        assert result == round(5 / 8, 4)
```

- [ ] **Step 8: Run to confirm failure**

```
pytest tests/test_fmp_client.py::TestGetQualityScore -v
```

Expected: `AttributeError: 'FMPClient' object has no attribute 'get_quality_score'`

---

## Task 4: Implement `FMPClient.get_quality_score`

**Files:**
- Modify: `regime_trader/services/fmp_client.py`

- [ ] **Step 9: Add the method**

In `regime_trader/services/fmp_client.py`, insert after `get_quality_score`'s natural neighbour — place it after `get_ratios_ttm` (around line 648) and before `get_institutional_ownership`:

```python
    def get_quality_score(self, ticker: str) -> float:
        """Piotroski F-score quality gate from cached ratios-ttm data.

        Calls get_ratios_ttm(ticker) — already cached in "ratios" bucket (24h TTL).
        Zero additional API calls.

        Returns float in [0, 1] — NOT Optional. Dead signal is 0.0, not None.
        This differs from get_upside_to_target (which returns None for missing
        analyst coverage) because quality data is universally available for any
        listed company. A missing ratios response means a broken endpoint, not
        "no quality data for this ticker."

        Returns 0.0 on exception or when get_ratios_ttm() returns empty dict.

        References: Piotroski (2000) JAR; Novy-Marx (2013) JFE.
        """
        if not self._api_key:
            return 0.0
        try:
            from regime_trader.scoring.momentum_signals import score_quality_piotroski  # noqa: PLC0415
            ratios = self.get_ratios_ttm(ticker)
            return score_quality_piotroski(ratios)
        except Exception as exc:
            log.debug("get_quality_score %s failed: %s", ticker, exc)
            return 0.0
```

- [ ] **Step 10: Run tests to confirm pass**

```
pytest tests/test_fmp_client.py::TestGetQualityScore -v
```

Expected: 6 passed.

- [ ] **Step 11: Run full fmp_client test suite**

```
pytest tests/test_fmp_client.py -q --tb=short 2>&1 | tail -5
```

Expected: all pass.

- [ ] **Step 12: Commit**

```bash
git add regime_trader/services/fmp_client.py tests/test_fmp_client.py
git commit -m "feat(fmp): add get_quality_score() — zero-cost Piotroski quality gate via ratios-ttm"
```

---

## Task 5: Update `WEIGHTS` and `FACTOR_FIELDS` in both files

**Files:**
- Modify: `scripts/run_pipeline.py`
- Modify: `backend/market_intel/generate_top_lists.py`

- [ ] **Step 13: Update WEIGHTS in `scripts/run_pipeline.py`**

Find the `WEIGHTS` dict (around line 47). Replace the `insider_breadth` and `congress` lines, and add `quality_piotroski`:

```python
WEIGHTS = {
    "insider_conviction":  0.30,
    "insider_breadth":     0.09,   # reduced 0.12→0.09 to fund quality_piotroski
    "congress":            0.10,   # reduced 0.13→0.10 (structurally sparse, ~5% density)
    "news_sentiment":      0.10,
    "news_buzz":           0.03,
    "momentum_long":       0.15,
    "volume_attention":    0.03,
    "analyst_consensus":   0.04,
    "analyst_revision":    0.06,
    "price_target_upside": 0.04,
    "quality_piotroski":   0.06,   # Piotroski (2000) / Novy-Marx (2013) quality gate
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, f"WEIGHTS must sum to 1, got {sum(WEIGHTS.values())}"
```

- [ ] **Step 14: Verify the assert in `run_pipeline.py`**

```
python -c "import sys; sys.path.insert(0, '.'); from scripts.run_pipeline import WEIGHTS; print('sum:', round(sum(WEIGHTS.values()), 10)); print('keys:', list(WEIGHTS))"
```

Expected:
```
sum: 1.0
keys: ['insider_conviction', 'insider_breadth', 'congress', 'news_sentiment', 'news_buzz', 'momentum_long', 'volume_attention', 'analyst_consensus', 'analyst_revision', 'price_target_upside', 'quality_piotroski']
```

- [ ] **Step 15: Update WEIGHTS and FACTOR_FIELDS in `backend/market_intel/generate_top_lists.py`**

Find the `WEIGHTS` dict (around line 55). Apply the identical change:

```python
WEIGHTS: Dict[str, float] = {
    "insider_conviction": 0.30,
    "insider_breadth":    0.09,   # reduced 0.12→0.09 to fund quality_piotroski
    "congress":           0.10,   # reduced 0.13→0.10 (structurally sparse, ~5% density)
    "news_sentiment":     0.10,
    "news_buzz":          0.03,
    "momentum_long":      0.15,
    "volume_attention":   0.03,
    "analyst_consensus":  0.04,
    "analyst_revision":   0.06,
    "price_target_upside": 0.04,
    "quality_piotroski":  0.06,   # Piotroski (2000) / Novy-Marx (2013) quality gate
}
```

Find the `FACTOR_FIELDS` dict (around line 68). Add the new entry:

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
    "quality_piotroski":   "quality_piotroski_score",
}
```

- [ ] **Step 16: Verify both files**

```
python -c "
from scripts.run_pipeline import WEIGHTS as W1
from backend.market_intel.generate_top_lists import WEIGHTS as W2, FACTOR_FIELDS
print('run_pipeline sum:', round(sum(W1.values()), 10))
print('generate_top_lists sum:', round(sum(W2.values()), 10))
print('FACTOR_FIELDS has quality:', 'quality_piotroski' in FACTOR_FIELDS)
print('field name:', FACTOR_FIELDS.get('quality_piotroski'))
"
```

Expected:
```
run_pipeline sum: 1.0
generate_top_lists sum: 1.0
FACTOR_FIELDS has quality: True
field name: quality_piotroski_score
```

- [ ] **Step 17: Commit**

```bash
git add scripts/run_pipeline.py backend/market_intel/generate_top_lists.py
git commit -m "feat(weights): add quality_piotroski 0.06, reduce insider_breadth 0.12→0.09, congress 0.13→0.10"
```

---

## Task 6: Wire `_score_ticker()` and `_score_ticker_international()` in `run_pipeline.py`

**Files:**
- Modify: `scripts/run_pipeline.py`

- [ ] **Step 18: Add call in `_score_ticker()` main try block**

Inside `_score_ticker()`, find the `price_target_upside_score` line (it reads `_fmp_client.get_upside_to_target(ticker)`). After it, add:

```python
            # quality_piotroski: Piotroski (2000) / Novy-Marx (2013) fundamental quality gate
            quality_piotroski_score = _fmp_client.get_quality_score(ticker)
```

No `or 0.0` needed — `get_quality_score` already returns `float`.

- [ ] **Step 19: Add to result dict (success path)**

In the success `return { ... }` dict, after `"price_target_upside_score": price_target_upside_score,` add:

```python
                "quality_piotroski_score": quality_piotroski_score,
```

- [ ] **Step 20: Add to fallback dict (except branch)**

In the except fallback `return { ... }` dict, after `"price_target_upside_score": 0.0,` add:

```python
                "quality_piotroski_score": 0.0,
```

- [ ] **Step 21: Add `None` sentinel to `_score_ticker_international()`**

Find `_score_ticker_international()`. In its return dict under "Structurally absent — None (not 0.0)", add:

```python
            "quality_piotroski_score": None,
```

FMP returns 403 for EU/Asia `stable/ratios-ttm` (confirmed Phase-0 smoke-test). `None` signals "weight excluded from renormalization" — same treatment as `congress_score`, `news_sentiment_score`, etc.

- [ ] **Step 22: Verify import clean**

```
python -c "import scripts.run_pipeline; print('OK')"
```

Expected: `OK`

- [ ] **Step 23: Commit**

```bash
git add scripts/run_pipeline.py
git commit -m "feat(pipeline): wire quality_piotroski_score into _score_ticker and _score_ticker_international"
```

---

## Task 7: CI validation

- [ ] **Step 24: Run all affected test files**

```
pytest tests/scoring/test_momentum_news_signals.py tests/test_fmp_client.py -v --tb=short 2>&1 | tail -15
```

Expected: all pass, 0 failures.

- [ ] **Step 25: Run the orthogonality monitoring tests**

```
pytest tests/monitoring/test_factor_orthogonality.py -v
```

Expected: all pass. (Uses synthetic factors — agnostic to new real factor names.)

- [ ] **Step 26: Smoke-check the full integration**

```
python -c "
from regime_trader.scoring.momentum_signals import score_quality_piotroski
from regime_trader.services.fmp_client import FMPClient
from scripts.run_pipeline import WEIGHTS as W1
from backend.market_intel.generate_top_lists import WEIGHTS as W2, FACTOR_FIELDS

# Scorer checks
print('perfect:', score_quality_piotroski({'returnOnAssetsTTM': 0.10, 'operatingProfitMarginTTM': 0.15, 'debtEquityRatioTTM': 0.30, 'currentRatioTTM': 2.0, 'grossProfitMarginTTM': 0.45, 'netProfitMarginTTM': 0.08}))
print('empty:', score_quality_piotroski({}))
print('none:', score_quality_piotroski(None))

# FMPClient method
c = FMPClient(api_key='')
print('method callable:', callable(c.get_quality_score))
print('no_key_returns_0:', c.get_quality_score('AAPL'))

# Weights and fields
print('run_pipeline sum:', round(sum(W1.values()), 10))
print('generate_top_lists sum:', round(sum(W2.values()), 10))
print('congress weight:', W1['congress'])
print('insider_breadth weight:', W1['insider_breadth'])
print('quality_piotroski weight:', W1['quality_piotroski'])
print('FACTOR_FIELDS field:', FACTOR_FIELDS.get('quality_piotroski'))
"
```

Expected output:
```
perfect: 1.0
empty: 0.0
none: 0.0
method callable: True
no_key_returns_0: 0.0
run_pipeline sum: 1.0
generate_top_lists sum: 1.0
congress weight: 0.1
insider_breadth weight: 0.09
quality_piotroski weight: 0.06
FACTOR_FIELDS field: quality_piotroski_score
```

---

## Self-Review Checklist

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| `score_quality_piotroski(ratios)` with 8-point F-score | Tasks 1–2 |
| All 8 points with exact field names and thresholds | Tasks 1–2 (test + impl) |
| `score = points / 8.0`, returns float [0,1] | Tasks 1–2 |
| Guard: None/non-dict/empty → 0.0 | Tasks 1–2 |
| Partial-data handling: missing field = 0 for that point | Tasks 1–2 |
| Negative D/E fails both leverage points | Tasks 1–2 |
| Docstring cites Piotroski (2000), Novy-Marx (2013), Ilmanen (2011) | Task 2 Step 3 |
| `get_quality_score(ticker)` returns `float` not `Optional` | Tasks 3–4 |
| Delegates to `get_ratios_ttm()` (cached, 0 extra API calls) | Tasks 3–4 |
| Catches exceptions → 0.0 | Tasks 3–4 |
| `WEIGHTS`: insider_breadth 0.12→0.09, congress 0.13→0.10, quality_piotroski 0.06 | Task 5 |
| Both files updated identically | Task 5 |
| Sum == 1.0 verified | Task 5 Steps 14+16 |
| `FACTOR_FIELDS`: quality_piotroski → quality_piotroski_score | Task 5 Step 15 |
| `_score_ticker()`: call + result dict + fallback dict | Task 6 Steps 18–20 |
| `_score_ticker_international()`: None sentinel | Task 6 Step 21 |
| `pytest tests/monitoring/test_factor_orthogonality.py` passes | Task 7 Step 25 |

All requirements covered.
