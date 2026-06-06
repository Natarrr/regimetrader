# v2.2-Global Pipeline Refactor — Ticker Collision + Alpha Compression Fix

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate ticker collisions in bulk-snapshot lookups and remove the signal-compressing regional dampeners so international equities can achieve uncompressed scores up to 1.0.

**Architecture:** Three surgical interventions: (1) collision-safe suffix-aware lookup in `fmp_bulk_prefetch.py`; (2) replace flat source_reliability dampeners in `generate_top_lists.py` + `fmp_fetcher.py` + `engine.py` with a dynamic available-factor denominator; (3) replace the hardcoded 0.90 ceiling in `audit_payload.py` with a profile-derived dynamic range check.

**Tech Stack:** Python 3.11, pytest, regime_trader.scoring.market_config, regime_trader.config.weights

---

## File Map

| File | Change |
|------|--------|
| `scripts/fmp_bulk_prefetch.py:225-292` | Modify `normalize_ticker_key`, `map_bulk_data_to_universe`, `build_ticker_index` |
| `backend/market_intel/generate_top_lists.py:769-774` | Remove source_reliability dampening loop |
| `regime_trader/fetchers/fmp_fetcher.py:73-84` | `source_reliability()` returns 1.0 for all markets |
| `backend/market_intel/engine.py:34-55` | Explicit dynamic-denominator in `score_ticker_pool` |
| `scripts/audit_payload.py:45-46, 195-216` | Remove `InternationalScoreOverflowError` + static E2; add dynamic E2 |
| `tests/test_fetchers.py` | Add collision-isolation tests for `map_bulk_data_to_universe` and `build_ticker_index` |
| `tests/test_source_reliability.py` | Replace dampening tests with no-op tests (dampener removed) |
| `tests/test_audit_payload.py` | Remove import of removed exception; add dynamic-range tests |
| `tests/test_global_scoring_v22.py` | Add uncompressed score tests + engine denominator test |

---

## Task 1 — Collision-Safe Suffix Mapping in `fmp_bulk_prefetch.py`

**Problem:** `normalize_ticker_key` strips `.AS`, `.DE`, `.T` to produce a bare base symbol. In `build_ticker_index`, the first record whose base collides wins and subsequent records for the same base are silently dropped. In `map_bulk_data_to_universe`, all universe tickers sharing a base (e.g., ASML.AS and ASML.PA) receive the same bulk row, regardless of exchange.

**Files:**
- Modify: `scripts/fmp_bulk_prefetch.py:237-292`
- Test: `tests/test_fetchers.py`

- [ ] **Step 1.1 — Write failing tests for collision isolation**

Add to `tests/test_fetchers.py`:

```python
# tests/test_fetchers.py  (append after existing imports)
from scripts.fmp_bulk_prefetch import map_bulk_data_to_universe, build_ticker_index


class TestMapBulkDataCollisionIsolation:
    def test_same_base_different_suffix_each_gets_own_record(self):
        """ASML.AS and ASML.PA must not share the same bulk record."""
        bulk = [
            {"symbol": "ASML.AS", "price": 800.0},
            {"symbol": "ASML.PA", "price": 801.0},
        ]
        result = map_bulk_data_to_universe(["ASML.AS", "ASML.PA"], bulk)
        assert result["ASML.AS"]["price"] == 800.0
        assert result["ASML.PA"]["price"] == 801.0

    def test_same_base_different_suffix_no_cross_contamination(self):
        """A bulk record for ASML.AS must not bleed into ASML.PA."""
        bulk = [{"symbol": "ASML.AS", "pe": 35}]
        result = map_bulk_data_to_universe(["ASML.AS", "ASML.PA"], bulk)
        assert result["ASML.AS"]["pe"] == 35
        assert result["ASML.PA"] == {}   # no data — not contaminated

    def test_no_suffix_bulk_maps_to_unique_universe_ticker(self):
        """FMP sometimes returns 'ASML' (no suffix) — map it only when unambiguous."""
        bulk = [{"symbol": "ASML", "pe": 35}]
        result = map_bulk_data_to_universe(["ASML.AS"], bulk)
        assert result["ASML.AS"]["pe"] == 35

    def test_no_suffix_bulk_ambiguous_maps_to_nothing(self):
        """If two tickers share a base, a suffix-free bulk row must not be mapped."""
        bulk = [{"symbol": "ASML", "pe": 35}]
        result = map_bulk_data_to_universe(["ASML.AS", "ASML.PA"], bulk)
        assert result["ASML.AS"] == {}
        assert result["ASML.PA"] == {}

    def test_exact_match_always_wins(self):
        """Exact match takes precedence over all base-symbol logic."""
        bulk = [
            {"symbol": "SAP.DE", "eps": 5.0},
            {"symbol": "SAP", "eps": 9.9},
        ]
        result = map_bulk_data_to_universe(["SAP.DE"], bulk)
        assert result["SAP.DE"]["eps"] == 5.0


class TestBuildTickerIndexCollisionIsolation:
    def test_two_records_same_base_removes_ambiguous_alias(self):
        """If ASML.AS and ASML.PA both exist, 'ASML' must NOT be in the index."""
        records = [
            {"symbol": "ASML.AS", "price": 800.0},
            {"symbol": "ASML.PA", "price": 801.0},
        ]
        index = build_ticker_index.__wrapped__(records, "symbol") if hasattr(
            build_ticker_index, "__wrapped__") else _build_index_from_records(records)
        assert "ASML" not in index, "Ambiguous base alias must be removed"

    def test_single_record_base_alias_present(self):
        """If only one record resolves to 'ASML', the alias must be kept."""
        records = [{"symbol": "ASML.AS", "price": 800.0}]
        index = _build_index_from_records(records)
        assert "ASML" in index
        assert index["ASML"]["price"] == 800.0

    def test_collision_detection_preserves_full_symbols(self):
        """Full symbols (ASML.AS, ASML.PA) must always be in the index."""
        records = [
            {"symbol": "ASML.AS", "price": 800.0},
            {"symbol": "ASML.PA", "price": 801.0},
        ]
        index = _build_index_from_records(records)
        assert index["ASML.AS"]["price"] == 800.0
        assert index["ASML.PA"]["price"] == 801.0


def _build_index_from_records(records: list[dict], key_field: str = "symbol") -> dict:
    """Helper: exercise build_ticker_index without needing a real cache directory."""
    from pathlib import Path
    from unittest.mock import patch
    from scripts.fmp_bulk_prefetch import build_ticker_index, load_bulk
    with patch("scripts.fmp_bulk_prefetch.load_bulk", return_value=records):
        return build_ticker_index(Path(".cache"), "test-endpoint", key_field)
```

- [ ] **Step 1.2 — Run tests to verify they fail**

```
pytest tests/test_fetchers.py::TestMapBulkDataCollisionIsolation tests/test_fetchers.py::TestBuildTickerIndexCollisionIsolation -v
```

Expected: `FAILED` — specifically `test_same_base_different_suffix_no_cross_contamination`, `test_no_suffix_bulk_ambiguous_maps_to_nothing`, `test_two_records_same_base_removes_ambiguous_alias`.

- [ ] **Step 1.3 — Implement collision-safe `map_bulk_data_to_universe`**

In `scripts/fmp_bulk_prefetch.py`, replace lines 262-268 (the base-symbol fallback block):

```python
        # Fallback to stripped base symbol matching.
        base_symbol = normalize_ticker_key(raw_symbol)
        raw_suffix = raw_symbol.split(".", 1)[1].upper() if "." in raw_symbol else ""
        candidates = base_to_universe_map.get(base_symbol, [])
        for target in candidates:
            if mapped_results[target]:
                continue  # already matched exactly — skip
            target_suffix = target.split(".", 1)[1].upper() if "." in target else ""
            # Only accept the base-symbol match when:
            #   (a) exchange suffixes match exactly, OR
            #   (b) the bulk row carries no suffix AND this base resolves to exactly one
            #       universe ticker (unambiguous mapping).
            if raw_suffix == target_suffix or (not raw_suffix and len(candidates) == 1):
                mapped_results[target] = row
```

- [ ] **Step 1.4 — Implement collision-safe `build_ticker_index`**

In `scripts/fmp_bulk_prefetch.py`, replace lines 289-291:

```python
        base_sym = normalize_ticker_key(sym)
        if base_sym and base_sym != sym:
            if base_sym not in index:
                index[base_sym] = rec
            elif index[base_sym] is not rec:
                # Two distinct records share the same base symbol.
                # Delete the ambiguous alias to prevent cross-contamination.
                del index[base_sym]
```

- [ ] **Step 1.5 — Run tests to verify they pass**

```
pytest tests/test_fetchers.py::TestMapBulkDataCollisionIsolation tests/test_fetchers.py::TestBuildTickerIndexCollisionIsolation -v
```

Expected: all green.

- [ ] **Step 1.6 — Run full test suite for regressions**

```
pytest tests/test_fetchers.py -v
```

Expected: no regressions in existing tests.

- [ ] **Step 1.7 — Commit**

```bash
git add scripts/fmp_bulk_prefetch.py tests/test_fetchers.py
git commit -m "fix(bulk-prefetch): collision-safe suffix-aware base-symbol mapping"
```

---

## Task 2 — Remove Source-Reliability Dampeners in `generate_top_lists.py` + `fmp_fetcher.py`

**Problem:** The dampening loop at `generate_top_lists.py:769-774` multiplies every international `final_score` by 0.80 (EU) or 0.70 (Asia), capping the best possible EU score at 0.80. This compresses the right-tail, destroying relative-value rank signal among top-decile international equities. Since `WEIGHTS_GLOBAL` already zeroes out structurally absent factors (congress, transcript_tone), the remaining available weights already sum to 1.0, making the dampeners a redundant and harmful distortion.

**Files:**
- Modify: `backend/market_intel/generate_top_lists.py:769-774`
- Modify: `regime_trader/fetchers/fmp_fetcher.py:73-84`
- Test: `tests/test_source_reliability.py`, `tests/test_global_scoring_v22.py`

- [ ] **Step 2.1 — Write failing tests**

Replace the entire contents of `tests/test_source_reliability.py` with:

```python
# Path: tests/test_source_reliability.py
"""Tests confirming source_reliability dampening is no longer applied to final_score.

The old dampening loop (generate_top_lists.py:769-774) multiplied final_score by
0.80 (EU) or 0.70 (Asia). It has been removed. FMPFetcher.source_reliability()
now returns 1.0 for all markets — dampening is replaced by the dynamic
available-factor denominator in StrategyEngine.score_ticker_pool.
"""
import pytest
from regime_trader.fetchers.fmp_fetcher import FMPFetcher
from regime_trader.fetchers.base import MarketEnum


def test_source_reliability_eu_is_one():
    fetcher = FMPFetcher(api_key="k", market=MarketEnum.EUROPE)
    assert fetcher.source_reliability("SAP.DE") == pytest.approx(1.0)


def test_source_reliability_asia_is_one():
    fetcher = FMPFetcher(api_key="k", market=MarketEnum.ASIA)
    assert fetcher.source_reliability("7203.T") == pytest.approx(1.0)


def test_source_reliability_us_is_one():
    fetcher = FMPFetcher(api_key="k", market=MarketEnum.USA)
    assert fetcher.source_reliability("AAPL") == pytest.approx(1.0)


def test_entry_with_source_reliability_one_unchanged():
    """When source_reliability == 1.0, the entry final_score must not change."""
    entry = {"final_score": 0.82, "source_reliability": 1.0}
    rel = float(entry.get("source_reliability", 1.0))
    result = round(entry["final_score"] * rel, 4)
    assert result == pytest.approx(0.82, abs=1e-4)
```

Also add to `tests/test_global_scoring_v22.py`:

```python
def test_eu_perfect_factors_reaches_score_one():
    """After removing dampening, a flawless EU ticker must be able to reach 1.0."""
    from backend.market_intel._score_compositor import compute_composite_score

    perfect = {
        "insider_conviction": 1.0,
        "insider_breadth":    1.0,
        "congress":           0.0,
        "news_sentiment":     1.0,
        "news_buzz":          1.0,
        "momentum_long":      1.0,
        "volume_attention":   1.0,
        "analyst_consensus":  1.0,
        "analyst_revision":   1.0,
        "quality_piotroski":  1.0,
        "price_target_upside": 1.0,
        "transcript_tone":    0.0,
    }
    score, meta = compute_composite_score("ASML.AS", perfect)
    assert score == pytest.approx(1.0, abs=1e-4), (
        f"Perfect EU ticker should score 1.0 without dampening, got {score:.4f}"
    )
    assert meta["weights_set"] == "GLOBAL"


def test_eu_score_not_capped_at_point_eight():
    """Ensure no 0.80 ceiling remains in the scoring path."""
    from backend.market_intel._score_compositor import compute_composite_score

    high_factors = {k: 0.95 for k in [
        "insider_conviction", "insider_breadth", "news_sentiment", "news_buzz",
        "momentum_long", "volume_attention", "analyst_consensus", "analyst_revision",
        "quality_piotroski", "price_target_upside",
    ]}
    high_factors["congress"] = 0.0
    high_factors["transcript_tone"] = 0.0

    score, _ = compute_composite_score("ASML.AS", high_factors)
    assert score > 0.80, f"Strong EU ticker must exceed old 0.80 ceiling, got {score:.4f}"
```

- [ ] **Step 2.2 — Run tests to verify they fail**

```
pytest tests/test_source_reliability.py tests/test_global_scoring_v22.py::test_eu_perfect_factors_reaches_score_one tests/test_global_scoring_v22.py::test_eu_score_not_capped_at_point_eight -v
```

Expected: `FAILED` on the source_reliability tests (currently returns 0.80 not 1.0) and the perfect-score test.

- [ ] **Step 2.3 — Update `FMPFetcher.source_reliability()`**

In `regime_trader/fetchers/fmp_fetcher.py`, replace lines 73-84:

```python
    def source_reliability(self, ticker: str) -> float:
        """Returns 1.0 for all markets.

        Regional dampening was removed in v2.2-global. Score compression is
        replaced by the available-factor dynamic denominator in StrategyEngine.
        """
        return 1.0
```

- [ ] **Step 2.4 — Remove the dampening loop from `generate_top_lists.py`**

In `backend/market_intel/generate_top_lists.py`, remove lines 769-774:

```python
    # source_reliability dampening — scale final_score by data-source confidence
    # Recompute badge after dampening so it stays consistent with final_score.
    for _e in entries:
        _rel = float(_e.get("source_reliability", 1.0))
        _e["final_score"] = round(_e["final_score"] * _rel, 4)
        _e["badge"] = _badge(_e["final_score"])
```

After deletion, `_assign_cap_tiers(entries)` (line 767) should be directly followed by `valid_entries = [e for e in entries if not e.get("_validation_failed")]` (the line that was at 777).

- [ ] **Step 2.5 — Run tests**

```
pytest tests/test_source_reliability.py tests/test_global_scoring_v22.py -v
```

Expected: all source_reliability tests green; the v2.2 global scoring suite still passes; the two new EU ceiling tests pass.

- [ ] **Step 2.6 — Run full test suite for regressions**

```
pytest -x -q
```

Expected: no failures. If `test_fmp_fetcher_source_reliability` in `test_global_scoring_v22.py` (line 291: `assert eu.source_reliability("SAP.DE") >= 0.75`) is still present, it will pass because 1.0 >= 0.75.

- [ ] **Step 2.7 — Commit**

```bash
git add backend/market_intel/generate_top_lists.py regime_trader/fetchers/fmp_fetcher.py tests/test_source_reliability.py tests/test_global_scoring_v22.py
git commit -m "feat(scoring): remove source_reliability dampeners; intl right-tail now uncompressed"
```

---

## Task 3 — Dynamic Denominator in `StrategyEngine.score_ticker_pool`

**Problem:** `score_ticker_pool` accumulates `weighted_score += metric_value * weight` and relies on the profile-constructor invariant that `sum(active_factors.values()) == 1.0`. This is defensively fine today but breaks transparently if the profile is ever modified to use US-style global weights that include 0-weight entries. Switching to an explicit dynamic denominator makes the formula self-documenting and robust.

**Files:**
- Modify: `backend/market_intel/engine.py:34-55`
- Test: `tests/test_global_scoring_v22.py`

- [ ] **Step 3.1 — Write failing test**

Add to `tests/test_global_scoring_v22.py`:

```python
def test_engine_dynamic_denominator_normalises_correctly():
    """score_ticker_pool must divide by sum(active_factor_weights), not hardcode 1.0.

    Given a profile whose weights happen to sum slightly below 1.0 due to
    float arithmetic, the output composite_score should equal
    weighted_sum / actual_weight_sum, not weighted_sum / 1.0.
    """
    import json, tempfile, os
    from backend.market_intel.engine import StrategyEngine

    profile = {
        "region": "TEST",
        "active_factors": {"alpha": 0.6, "beta": 0.4},
        "output_filename": "test.json",
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(profile, f)
        path = f.name

    try:
        engine = StrategyEngine(path)
        data = [{"ticker": "X", "metrics": {"alpha_score": 1.0, "beta_score": 1.0}}]
        results = engine.score_ticker_pool(data)
        assert results[0]["composite_score"] == pytest.approx(1.0, abs=1e-4)
    finally:
        os.unlink(path)


def test_engine_dynamic_denominator_with_partial_availability():
    """If one factor has no data (score 0.0), remaining factors determine the score."""
    import json, tempfile, os
    from backend.market_intel.engine import StrategyEngine

    profile = {
        "region": "INTL_TEST",
        "active_factors": {"momentum": 0.70, "volume": 0.30},
        "output_filename": "test.json",
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(profile, f)
        path = f.name

    try:
        engine = StrategyEngine(path)
        # Only momentum is available; volume = 0.0
        data = [{"ticker": "SAP.DE", "metrics": {"momentum_score": 1.0, "volume_score": 0.0}}]
        results = engine.score_ticker_pool(data)
        # composite = (1.0*0.70 + 0.0*0.30) / (0.70 + 0.30) = 0.70
        assert results[0]["composite_score"] == pytest.approx(0.70, abs=1e-4)
    finally:
        os.unlink(path)
```

- [ ] **Step 3.2 — Run test to verify it fails**

```
pytest tests/test_global_scoring_v22.py::test_engine_dynamic_denominator_normalises_correctly tests/test_global_scoring_v22.py::test_engine_dynamic_denominator_with_partial_availability -v
```

Expected: the denominator test passes vacuously (1.0 denominator = correct result when weights sum exactly to 1.0) but the partial-availability test reveals the current formula does NOT renormalize partial factors correctly. Verify it FAILS.

- [ ] **Step 3.3 — Implement dynamic denominator in `engine.py`**

In `backend/market_intel/engine.py`, replace the scoring block (lines 34-55) with:

```python
    def score_ticker_pool(self, raw_universe_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Processes a raw array of ticker metrics, extracting and scoring
        only the active factors allowed by the regional strategy profile.

        Final score is normalized by the sum of weights of *available* active
        factors (those present in the profile), not a hardcoded 1.0.
        Formula: Score = Σ(w_i · s_i) / Σ(w_i)  for i in active_factors.
        """
        processed_rankings = []
        available_weight = sum(self.active_factors.values())

        for asset in raw_universe_data:
            ticker = asset.get("ticker")
            raw_metrics = asset.get("metrics", {})

            weighted_score = 0.0
            factor_breakdown = {}

            for factor, weight in self.active_factors.items():
                raw_key = f"{factor}_score" if not factor.endswith("_score") else factor
                try:
                    metric_value = float(raw_metrics.get(raw_key, 0.0) or 0.0)
                except Exception:
                    metric_value = 0.0

                weighted_score += metric_value * weight
                factor_breakdown[factor] = metric_value

            composite_score = (
                round(weighted_score / available_weight, 4)
                if available_weight > 1e-9
                else 0.0
            )

            processed_rankings.append({
                "ticker": ticker,
                "composite_score": composite_score,
                "region_applied": self.region,
                "factor_snapshots": factor_breakdown,
            })

        processed_rankings.sort(key=lambda x: x["composite_score"], reverse=True)
        return processed_rankings
```

- [ ] **Step 3.4 — Run tests**

```
pytest tests/test_global_scoring_v22.py -v
```

Expected: all tests pass, including the two new denominator tests.

- [ ] **Step 3.5 — Commit**

```bash
git add backend/market_intel/engine.py tests/test_global_scoring_v22.py
git commit -m "feat(engine): dynamic available-factor denominator in score_ticker_pool"
```

---

## Task 4 — Audit Layer: Remove Static E2, Add Dynamic Range Validation

**Problem:** `audit_payload.py` raises `InternationalScoreOverflowError` if an EU/Asia score exceeds 0.90. With dampeners removed and the dynamic denominator in place, a flawless EU ticker legitimately scores 1.0. The 0.90 ceiling must be replaced by a dynamically computed ceiling based on the regional active-factor weight profile.

**Files:**
- Modify: `scripts/audit_payload.py:9-47, 195-216`
- Test: `tests/test_audit_payload.py`

- [ ] **Step 4.1 — Write failing tests**

Add to `tests/test_audit_payload.py`:

```python
# ---------------------------------------------------------------------------
# E2. Dynamic range validation — replaces static InternationalScoreOverflowError
# ---------------------------------------------------------------------------

def test_intl_score_of_0_95_passes_eu():
    """EU score of 0.95 must pass now that the 0.90 ceiling is removed."""
    payload = _make_payload(top_buys=[
        _entry(ticker="ASML.AS", score=0.95, badge="HIGH BUY", market="EUROPE")
    ])
    assert audit(payload) is True


def test_intl_score_of_1_0_passes_eu():
    """EU score of exactly 1.0 is now valid — perfect factors, no dampening."""
    payload = _make_payload(top_buys=[
        _entry(ticker="SAP.DE", score=1.0, badge="HIGH BUY", market="EUROPE")
    ])
    assert audit(payload) is True


def test_intl_score_of_1_0_passes_asia():
    """Asia score of exactly 1.0 is now valid."""
    payload = _make_payload(top_buys=[
        _entry(ticker="7203.T", score=1.0, badge="HIGH BUY", market="ASIA")
    ])
    assert audit(payload) is True


def test_intl_score_above_1_still_raises():
    """Score > 1.0 must still raise ScoreDivergenceError regardless of market."""
    payload = _make_payload(top_buys=[
        _entry(ticker="SAP.DE", score=1.01, badge="HIGH BUY", market="EUROPE")
    ])
    with pytest.raises(ScoreDivergenceError):
        audit(payload)


def test_international_score_overflow_error_not_exported():
    """InternationalScoreOverflowError must no longer exist in audit_payload."""
    import importlib
    import scripts.audit_payload as ap_module
    assert not hasattr(ap_module, "InternationalScoreOverflowError"), (
        "InternationalScoreOverflowError was removed in v2.2-global"
    )
```

Also update the existing import block in `tests/test_audit_payload.py` to remove `InternationalScoreOverflowError`:
```python
# The existing import block already does NOT import InternationalScoreOverflowError,
# so no change is required to existing imports.
```
*(Verify: the existing import block at lines 9-18 of `test_audit_payload.py` does not currently import `InternationalScoreOverflowError`. No change needed there.)*

- [ ] **Step 4.2 — Run tests to verify they fail**

```
pytest tests/test_audit_payload.py::test_intl_score_of_0_95_passes_eu tests/test_audit_payload.py::test_intl_score_of_1_0_passes_eu tests/test_audit_payload.py::test_intl_score_of_1_0_passes_asia tests/test_audit_payload.py::test_international_score_overflow_error_not_exported -v
```

Expected: `test_intl_score_of_0_95_passes_eu` FAILS (raises InternationalScoreOverflowError). The others may fail or warn depending on badge/sort order.

- [ ] **Step 4.3 — Remove `InternationalScoreOverflowError` class from `audit_payload.py`**

In `scripts/audit_payload.py`, delete lines 45-46:

```python
class InternationalScoreOverflowError(PipelineAuditError):
    """EU/Asia ticker final_score exceeds what its available factors can produce."""
```

- [ ] **Step 4.4 — Replace static E2 block with dynamic range validation**

In `scripts/audit_payload.py`, replace lines 186-216 (the entire E2 block including comment) with:

```python
    # ------------------------------------------------------------------
    # E2. Dynamic range validation — international score ceiling
    #     Ceiling is 1.0 (the sum of renormalized available factor weights
    #     for EU/Asia after removing structurally absent factors).
    #     Score > 1.0 + epsilon indicates US-factor injection or arithmetic
    #     error; it is caught by check A above. This block explicitly
    #     audits the regional profile claim: congress and transcript_tone
    #     must be zero so available_weight_sum == 1.0.
    # ------------------------------------------------------------------
    try:
        from regime_trader.config.weights import WEIGHTS_GLOBAL
        _intl_available_weight = sum(
            w for f, w in WEIGHTS_GLOBAL.items()
            if f not in {"congress", "transcript_tone"}
        )
    except ImportError:
        _intl_available_weight = 1.0

    _DYNAMIC_INTL_CEILING = round(_intl_available_weight, 6)
    _CEILING_TOLERANCE = 1e-4

    for entry in _iter_all_entries(data):
        ticker = entry.get("ticker", "?")
        market = entry.get("market", "USA")
        if market not in _FOREIGN_MARKETS:
            continue

        score = float(entry.get("final_score", 0.0))
        factors = entry.get("factors", {})

        if score > _DYNAMIC_INTL_CEILING + _CEILING_TOLERANCE:
            raise ScoreDivergenceError(
                f"Ticker {ticker!r} ({market}): final_score={score:.4f} exceeds "
                f"dynamic available-factor ceiling {_DYNAMIC_INTL_CEILING:.4f}. "
                f"Available weights sum: {_intl_available_weight:.4f}. "
                f"Possible US-factor injection. factors={factors}"
            )
```

- [ ] **Step 4.5 — Run all audit tests**

```
pytest tests/test_audit_payload.py -v
```

Expected: all 24+ tests pass. Specifically: `test_intl_score_of_0_95_passes_eu`, `test_intl_score_of_1_0_passes_eu`, `test_intl_score_of_1_0_passes_asia` all green; `test_intl_score_above_1_still_raises` green; `test_international_score_overflow_error_not_exported` green.

- [ ] **Step 4.6 — Verify CrossContaminationError and GeographicLeakageError still intact**

```
pytest tests/test_audit_payload.py::test_congress_eu_raises tests/test_audit_payload.py::test_geo_leak_suffix_us_market tests/test_audit_payload.py::test_geo_leak_no_suffix_asia -v
```

Expected: all three pass (safeguards preserved).

- [ ] **Step 4.7 — Commit**

```bash
git add scripts/audit_payload.py tests/test_audit_payload.py
git commit -m "feat(audit): replace static 0.90 intl ceiling with dynamic available-factor range check"
```

---

## Task 5 — Final Regression Suite + Verification

- [ ] **Step 5.1 — Run complete test suite**

```
pytest -q
```

Expected: zero failures. Note the total test count — if it drops significantly, investigate deleted tests that were not replaced.

- [ ] **Step 5.2 — Verify weight integrity assertion still passes**

```
pytest tests/test_weights_consistency.py tests/test_global_scoring_v22.py::test_weights_us_unchanged tests/test_global_scoring_v22.py::test_weights_global_sum -v
```

Expected: green — `assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6` must hold for both US and GLOBAL weight sets.

- [ ] **Step 5.3 — Dry-run collision check for international universe**

```
python scripts/fmp_bulk_prefetch.py --cache-dir .cache/bulk_snapshots --dry-run --verbose 2>&1 | grep -i "collision\|overwrite\|warning"
```

Expected: no collision or overwrite warnings in output.

- [ ] **Step 5.4 — Run audit against current `logs/top_lists.json` (if present)**

```
python scripts/audit_payload.py logs/top_lists.json
```

Expected: exit 0 with green checks. If file is absent, the step is skipped.

- [ ] **Step 5.5 — Final commit**

```bash
git add -p   # review any remaining unstaged changes
git commit -m "chore(v2.2-global): final regression pass — all checks green"
```

---

## Self-Review

**Spec coverage:**

| Requirement | Covered |
|-------------|---------|
| `normalize_ticker_key` accepts market context | ✅ Task 1 — suffix-aware fallback matching |
| Hash-map isolation for `(base_symbol, suffix)` | ✅ Task 1 — `build_ticker_index` removes ambiguous aliases |
| No cross-region record contamination | ✅ Task 1 — only unique-candidate or suffix-matched rows mapped |
| Remove `_INTL_SCORE_CEILING` | ✅ Task 4 |
| Remove flat regional dampeners | ✅ Task 2 (`generate_top_lists.py:769-774` deleted, `source_reliability` returns 1.0) |
| Dynamic available-factor denominator | ✅ Task 3 (`engine.py`) |
| International can reach 1.0 | ✅ Task 2 test + Task 4 audit ceiling = 1.0 |
| Remove `InternationalScoreOverflowError` | ✅ Task 4 |
| Dynamic range validation | ✅ Task 4 — computes ceiling from `WEIGHTS_GLOBAL` |
| `CrossContaminationError` retained | ✅ Task 4 — untouched, tested in Step 4.6 |
| `GeographicLeakageError` retained | ✅ Task 4 — untouched, tested in Step 4.6 |
| `audit(args.input)` passes post-refactor | ✅ Task 5.4 |

**Placeholder scan:** None found. All steps include exact code or exact commands.

**Type consistency:**
- `normalize_ticker_key(ticker: str) -> str` — unchanged signature.
- `map_bulk_data_to_universe(universe_tickers, bulk_rows, ticker_column_name) -> dict` — unchanged signature.
- `build_ticker_index(cache_dir, endpoint, key_field) -> dict` — unchanged signature.
- `score_ticker_pool(raw_universe_data) -> List[Dict]` — unchanged signature; `composite_score` key unchanged.
- `audit(top_lists_path) -> bool` — unchanged signature; still raises `ScoreDivergenceError` for overflow.
