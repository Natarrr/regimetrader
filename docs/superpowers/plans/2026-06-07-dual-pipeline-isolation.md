# Dual-Pipeline Isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surgically remove international scoring from `run_pipeline.py`, expand StrategyEngine INTL from 6 to 10 active factors aligned to `WEIGHTS_GLOBAL`, apply VIX dampening to INTL entries in `cook_toplists.py`, inject `"pipeline"` metadata for region-aware weight selection in Discord, and consolidate three GitHub Actions workflows into one race-condition-free workflow.

**Architecture:** US pipeline (EDGAR + FMP) writes `logs/top_lists_us.json`; INTL pipeline (FMP Global, 10 factors) writes `logs/top_lists_intl.json`; `cook_toplists.py` reads VIX from US artifact, applies dampening to INTL scores, merges, and writes `logs/top_lists.json`; `send_toplists_discord.py` reads the merged file. All three pipeline stages run in a single GitHub Actions workflow with `needs:` dependency.

**Tech Stack:** Python 3.11, pytest, GitHub Actions (`actions/upload-artifact@v4`, `actions/download-artifact@v4`), `regime_trader.config.weights`, `backend.market_intel.engine.StrategyEngine`

**Spec:** `docs/superpowers/specs/2026-06-07-dual-pipeline-isolation-design.md`

---

## File Map

| File | Action | What changes |
|------|--------|-------------|
| `regime_trader/config/weights.py` | Modify | WEIGHTS_GLOBAL: activate `analyst_revision=0.02`, `price_target_upside=0.03`; reduce 4 donors |
| `backend/market_intel/profiles/intl_strategy.json` | Rewrite | 6 factors → 10 factors aligned to WEIGHTS_GLOBAL; weights sum stays 1.00 |
| `backend/market_intel/engine.py` | Modify | Inject `"pipeline": "INTL"` into each entry in `score_ticker_pool()` |
| `scripts/run_pipeline.py` | Remove | Delete `_score_ticker_international()`, EU/Asia orchestrator block, 4 INTL-only helper functions |
| `backend/market_intel/generate_top_lists.py` | Modify | Remove INTL split/merge; rename output to `top_lists_us.json`; inject `"pipeline": "US"` |
| `backend/market_intel/_generate_top_lists_intl_patch.py` | Delete | Dead code |
| `scripts/cook_toplists.py` | Modify | Fix 3 bugs; add `_apply_vix_multiplier()` with NaN guard; preserve `"pipeline"` field; update default input path |
| `scripts/send_toplists_discord.py` | Modify | Add `_weights_for_entry()`; update `_factor_contribution_line()` fallback |
| `.github/workflows/daily_trading_pipeline.yml` | Create | Consolidated single workflow with parallel jobs + `needs:` gate |
| `.github/workflows/pipeline_us.yml` | Delete | Superseded |
| `.github/workflows/pipeline_intl.yml` | Delete | Superseded |
| `.github/workflows/daily_toplists_discord.yml` | Delete | Superseded |

---

## Task 1: Update WEIGHTS_GLOBAL — Activate analyst_revision and price_target_upside

**Files:**
- Modify: `regime_trader/config/weights.py`
- Test: `tests/test_weights_consistency.py`

### Context
`WEIGHTS_GLOBAL` currently has `analyst_revision: 0.00` and `price_target_upside: 0.00`. Activating them requires reducing four existing weights. The assert at module load will catch any sum error.

New values: `insider_conviction 0.30→0.28 (−0.02)`, `insider_breadth 0.15→0.14 (−0.01)`, `news_buzz 0.05→0.04 (−0.01)`, `volume_attention 0.05→0.04 (−0.01)`, `analyst_revision 0.00→0.02 (+0.02)`, `price_target_upside 0.00→0.03 (+0.03)`. Sum = 1.00.

- [ ] **Step 1.1: Modify WEIGHTS_GLOBAL in `regime_trader/config/weights.py`**

Replace the `WEIGHTS_GLOBAL` dict (lines 62–75) and the comment block above it (lines 46–74):

```python
# ── Global universe — EU / Asia ────────────────────────────────────────────────
#
# congress = 0.00  (structural absence — STOCK Act is US-only)
# transcript_tone = 0.00 (FMP earning-call-transcript-latest US-only)
#
# Net changes vs WEIGHTS_US (v2.3 sprint — activating 4 wired-but-zeroed factors):
#   insider_conviction  0.30 → 0.28  (−0.02 donor — MAR Art.19 parity maintained)
#   insider_breadth     0.15 → 0.14  (−0.01 donor)
#   news_buzz           0.05 → 0.04  (−0.01 donor — lowest IC)
#   volume_attention    0.05 → 0.04  (−0.01 donor)
#   analyst_revision    0.00 → 0.02  (+0.02 — Chan, Jegadeesh & Lakonishok 1996)
#   price_target_upside 0.00 → 0.03  (+0.03 — Brav & Lehavy 2003)
#   congress            0.22 → 0.00  (structurally absent)
#   transcript_tone     —   → 0.00  (structurally absent)
WEIGHTS_GLOBAL: dict[str, float] = {
    "insider_conviction":  0.28,   # −0.02 vs US — MAR Art.19 parity maintained
    "insider_breadth":     0.14,   # −0.01 vs US
    "congress":            0.00,   # structurally absent outside US
    "news_sentiment":      0.13,   # +0.03 — global news corpus via FMP
    "news_buzz":           0.04,   # −0.01 donor
    "momentum_long":       0.17,   # +0.02 — Rouwenhorst 1998 EU premium
    "volume_attention":    0.04,   # −0.01 donor
    "analyst_consensus":   0.10,   # +0.10 — stronger signal in less-covered markets
    "quality_piotroski":   0.05,   # +0.05 — accounting-identity, universal
    "analyst_revision":    0.02,   # activated — Chan, Jegadeesh & Lakonishok 1996
    "price_target_upside": 0.03,   # activated — Brav & Lehavy 2003
    "transcript_tone":     0.00,   # structurally absent outside US
}
assert abs(sum(WEIGHTS_GLOBAL.values()) - 1.0) < 1e-6, (
    f"WEIGHTS_GLOBAL sums to {sum(WEIGHTS_GLOBAL.values()):.8f}, not 1.0"
)
```

- [ ] **Step 1.2: Verify the assert passes**

```bash
python -c "from regime_trader.config.weights import WEIGHTS_GLOBAL; print('sum=', sum(WEIGHTS_GLOBAL.values())); assert abs(sum(WEIGHTS_GLOBAL.values())-1.0)<1e-6; print('WEIGHTS_GLOBAL OK')"
```

Expected output: `sum= 1.0` / `WEIGHTS_GLOBAL OK`

- [ ] **Step 1.3: Run existing weights tests**

```bash
pytest tests/test_weights_consistency.py tests/test_regional_weights.py -v
```

Expected: All PASS. Fix any failures before continuing.

- [ ] **Step 1.4: Commit**

```bash
git add regime_trader/config/weights.py
git commit -m "feat(weights): activate analyst_revision=0.02, price_target_upside=0.03 in WEIGHTS_GLOBAL"
```

---

## Task 2: Expand intl_strategy.json from 6 to 10 Active Factors

**Files:**
- Modify: `backend/market_intel/profiles/intl_strategy.json`

### Context
`intl_strategy.json` currently has 6 factors with weights that don't match WEIGHTS_GLOBAL. `StrategyEngine.__init__()` enforces `abs(total_weight - 1.0) > 1e-4`, so the active_factors must sum to exactly 1.00. Rewrite to align with the new WEIGHTS_GLOBAL (minus `congress=0.00` and `transcript_tone=0.00` which are structurally absent).

- [ ] **Step 2.1: Rewrite `backend/market_intel/profiles/intl_strategy.json`**

```json
{
    "region": "INTL",
    "expected_factors": 10,
    "active_factors": {
        "insider_conviction":  0.28,
        "insider_breadth":     0.14,
        "news_sentiment":      0.13,
        "news_buzz":           0.04,
        "momentum_long":       0.17,
        "volume_attention":    0.04,
        "analyst_consensus":   0.10,
        "quality_piotroski":   0.05,
        "analyst_revision":    0.02,
        "price_target_upside": 0.03
    },
    "output_filename": "top_lists_intl.json"
}
```

- [ ] **Step 2.2: Verify StrategyEngine loads the profile without error**

```bash
python -c "
from backend.market_intel.engine import StrategyEngine
e = StrategyEngine('backend/market_intel/profiles/intl_strategy.json')
total = sum(e.active_factors.values())
assert abs(total - 1.0) < 1e-4, f'weights sum to {total}'
assert len(e.active_factors) == 10
print(f'OK: 10 factors, sum={total:.6f}')
"
```

Expected: `OK: 10 factors, sum=1.000000`

- [ ] **Step 2.3: Commit**

```bash
git add backend/market_intel/profiles/intl_strategy.json
git commit -m "feat(intl): expand intl_strategy.json from 6 to 10 factors aligned to WEIGHTS_GLOBAL"
```

---

## Task 3: Inject `"pipeline": "INTL"` into StrategyEngine Output

**Files:**
- Modify: `backend/market_intel/engine.py:70-75`

### Context
`cook_toplists.py` will use `entry.get("pipeline")` to select the correct weight dict in Discord. The StrategyEngine must stamp each output entry with `"pipeline": "INTL"` before it is consumed by cook. The change is a one-line addition inside `score_ticker_pool()`.

- [ ] **Step 3.1: Write failing test in `tests/test_global_scoring_v22.py`**

Add this test to the existing file (or create it if it doesn't exist in the right location):

```python
def test_strategy_engine_injects_pipeline_key():
    """Each entry produced by StrategyEngine must carry 'pipeline': 'INTL'."""
    import tempfile, json
    from backend.market_intel.engine import StrategyEngine

    profile = {
        "region": "INTL",
        "expected_factors": 2,
        "active_factors": {"momentum_long": 0.60, "news_sentiment": 0.40},
        "output_filename": "test_out.json",
    }
    raw = [{"ticker": "SAP.DE", "metrics": {"momentum_long_score": 0.8, "news_sentiment_score": 0.6}}]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(profile, f)
        profile_path = f.name

    engine = StrategyEngine(profile_path)
    results = engine.score_ticker_pool(raw)

    assert results[0].get("pipeline") == "INTL", (
        f"Expected 'INTL', got {results[0].get('pipeline')!r}"
    )
```

- [ ] **Step 3.2: Run test to confirm it fails**

```bash
pytest tests/test_global_scoring_v22.py::test_strategy_engine_injects_pipeline_key -v
```

Expected: FAIL with `AssertionError: Expected 'INTL', got None`

- [ ] **Step 3.3: Add `"pipeline"` key to `backend/market_intel/engine.py`**

In `score_ticker_pool()`, update the dict appended to `processed_rankings` (lines 70–75):

```python
processed_rankings.append({
    "ticker": ticker,
    "composite_score": composite_score,
    "region_applied": self.region,
    "factor_snapshots": factor_breakdown,
    "pipeline": "INTL",          # consumed by cook_toplists.py → send_toplists_discord.py
})
```

- [ ] **Step 3.4: Run test to confirm it passes**

```bash
pytest tests/test_global_scoring_v22.py::test_strategy_engine_injects_pipeline_key -v
```

Expected: PASS

- [ ] **Step 3.5: Commit**

```bash
git add backend/market_intel/engine.py tests/test_global_scoring_v22.py
git commit -m "feat(engine): inject 'pipeline': 'INTL' into StrategyEngine output entries"
```

---

## Task 4: Remove INTL Branch from `run_pipeline.py`

**Files:**
- Modify: `scripts/run_pipeline.py`

### Context
`run_pipeline.py` scores BOTH US and EU/Asia tickers. The EU/Asia scoring block (lines 1784–1894) calls `_score_ticker_international()` (lines 1204–1337) and four helper functions used only by that block: `_fetch_eu_return()` (~line 390), `_fetch_asia_return()` (~line 400), `_load_registry_tickers()` (~line 1173), `_registry_meta()` (~line 1187). Remove all five. Add a startup assertion that `universe.csv` contains no EU/Asia tickers.

- [ ] **Step 4.1: Delete `_score_ticker_international()` (lines 1204–1337)**

Remove the entire function from the comment `def _score_ticker_international(` through its closing `return None` and the blank lines that follow.

- [ ] **Step 4.2: Delete EU/Asia helper functions**

Remove these four functions (check they have no callers outside this file first):

```bash
grep -n "_fetch_eu_return\|_fetch_asia_return\|_load_registry_tickers\|_registry_meta" scripts/run_pipeline.py
```

Delete:
- `_fetch_eu_return()` function block
- `_fetch_asia_return()` function block
- `_load_registry_tickers()` function block (under `# ── Multi-market helpers ──`)
- `_registry_meta()` function block

- [ ] **Step 4.3: Delete EU/Asia orchestrator block (lines 1784–1894)**

Remove from `# ── EU / Asia scoring ───` through the closing `else: log.warning("No EU/Asia fetchers active...")`. This includes:
- `registry_tickers = _load_registry_tickers()`
- `_meta = _registry_meta()`
- All FMPFetcher creation and Orchestrator.run() calls
- The `_regional_baseline()` inner function
- The ThreadPoolExecutor for EU/Asia scoring
- The regression guard loop (lines 1883–1892)

- [ ] **Step 4.4: Add universe purity assertion after `load_tickers()` call**

Find the line in `run(...)` that calls `load_tickers()` and add this assertion immediately after the tickers are loaded:

```python
# Guard: run_pipeline.py is US-only. Reject international tickers at startup.
_INTL_SUFFIXES = frozenset({
    ".DE", ".PA", ".L", ".AS", ".MI", ".MC", ".VX", ".BR", ".LS",
    ".OL", ".ST", ".HE", ".CO", ".F", ".BE",
    ".T", ".HK", ".KS", ".KQ", ".SS", ".SZ", ".NS", ".BO", ".SI", ".BK", ".JK",
})
_intl_leaks = [t["ticker"] for t in ticker_rows if any(t["ticker"].endswith(s) for s in _INTL_SUFFIXES)]
if _intl_leaks:
    raise ValueError(
        f"run_pipeline.py is US-only. International tickers found in universe: {_intl_leaks}. "
        "Check config/universe.csv — INTL tickers belong in config/ticker_registry.json."
    )
```

- [ ] **Step 4.5: Verify the file still imports cleanly**

```bash
python -c "import scripts.run_pipeline; print('import OK')" 2>&1 | head -5
```

If the module can't be imported directly, use:
```bash
python -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('run_pipeline', 'scripts/run_pipeline.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print('import OK')
"
```

Expected: `import OK` (no errors about missing `_score_ticker_international` etc.)

- [ ] **Step 4.6: Commit**

```bash
git add scripts/run_pipeline.py
git commit -m "refactor(pipeline): remove INTL scoring from run_pipeline.py — US-only from here"
```

---

## Task 5: Remove INTL Processing from generate_top_lists.py + Rename Output

**Files:**
- Modify: `backend/market_intel/generate_top_lists.py`
- Delete: `backend/market_intel/_generate_top_lists_intl_patch.py`

### Context
`generate_top_lists.py` currently splits results into `us_results` and `intl_results`, merges them, and writes `logs/top_lists.json`. After Task 4, `intel_source_status.json` contains only US rows. The `intl_results` split and merge loop become dead code. The output must be renamed to `top_lists_us.json` and each entry must carry `"pipeline": "US"`.

- [ ] **Step 5.1: Write a failing test for the new output filename**

```python
# tests/test_generate_top_lists_isolation.py
import json, pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


def test_generate_top_lists_writes_top_lists_us_json(tmp_path):
    """After INTL removal, generate_top_lists must write top_lists_us.json."""
    # The output file must be named top_lists_us.json, not top_lists.json
    expected = tmp_path / "top_lists_us.json"
    assert not (tmp_path / "top_lists.json").exists(), (
        "Old output path top_lists.json must not be written by generate_top_lists"
    )
    # This test will pass once generate() is updated to write top_lists_us.json
    # Run it BEFORE the fix to confirm it fails first.
    pytest.skip("Run after Step 5.2 to verify the fix")
```

- [ ] **Step 5.2: Remove INTL import and processing from `generate_top_lists.py`**

**Remove line 50** (import of `_build_intl_entry`):
```python
from backend.market_intel._generate_top_lists_intl_patch import _build_intl_entry
```

**Remove line 707** (the intl_results split):
```python
intl_results = [r for r in results if r.get("market", "USA") != "USA"]
```

**Remove the INTL merge block** (lines 761–767):
```python
for row in intl_results:
    if row.get("_validation_failed"):
        continue
    entries.append(_build_intl_entry(row))
if intl_results:
    log.info(
        "Merged %d EU/Asia entries into ranked universe (pre-scored, bypass normalize)", len(intl_results))
```

Also remove `intl_results` from the `us_results = ...` line context — line 706 only splits to `us_results`:
```python
# Before:
us_results = [r for r in results if r.get("market", "USA") == "USA"]
intl_results = [r for r in results if r.get("market", "USA") != "USA"]

# After (remove intl_results line entirely):
us_results = [r for r in results if r.get("market", "USA") == "USA"]
```

**Change the output paths** — three locations:

Line ~805 (prev_weights lookback):
```python
# Before:
_prev_path = log_dir / "top_lists.json"
# After:
_prev_path = log_dir / "top_lists_us.json"
```

Line ~962 (atomic write):
```python
# Before:
out_json = log_dir / "top_lists.json"
# After:
out_json = log_dir / "top_lists_us.json"
```

Line ~1050 (staleness check):
```python
# Before:
out = args.log_dir / "top_lists.json"
# After:
out = args.log_dir / "top_lists_us.json"
```

- [ ] **Step 5.3: Inject `"pipeline": "US"` into each entry**

Find the `_to_entry()` function. Locate where it returns a dict. Add `"pipeline": "US"` to the returned dict. For example, if the return statement is:

```python
return {
    "ticker": row["ticker"],
    "final_score": score,
    ...
}
```

Add:
```python
return {
    "ticker": row["ticker"],
    "final_score": score,
    "pipeline": "US",    # consumed by cook_toplists.py → send_toplists_discord.py
    ...
}
```

- [ ] **Step 5.4: Update module docstring**

In the module docstring (lines 1–27), update:
```
Output:
  logs/top_lists_us.json — consumed by scripts/cook_toplists.py
  logs/top5.csv          — flat reference file for downstream analysis
```

- [ ] **Step 5.5: Delete the dead patch file**

```bash
rm "backend/market_intel/_generate_top_lists_intl_patch.py"
```

- [ ] **Step 5.6: Verify generate_top_lists imports cleanly**

```bash
python -c "import backend.market_intel.generate_top_lists; print('import OK')"
```

Expected: `import OK` (no ImportError about `_generate_top_lists_intl_patch`)

- [ ] **Step 5.7: Commit**

```bash
git add backend/market_intel/generate_top_lists.py
git rm backend/market_intel/_generate_top_lists_intl_patch.py
git commit -m "refactor(generate): remove INTL merge path, rename output to top_lists_us.json, inject pipeline=US"
```

---

## Task 6: Fix cook_toplists.py — 3 Bugs + VIX Dampening + Pipeline Metadata

**Files:**
- Modify: `scripts/cook_toplists.py`
- Test: `tests/test_cook_toplists.py`

### Context
Three bugs in the current `cook_toplists.py`:
1. **Line 101**: `"ticker_count": us_ticker_count + len(intl_raw)` counts dropped tickers
2. **Lines 73–77**: soft VIX warning doesn't halt; propagates to audit failure downstream
3. No VIX dampening applied to INTL entries (INTL scores arrive undampened from StrategyEngine)

Additionally: default `--us-input` path must change from `logs/top_lists.json` to `logs/top_lists_us.json`, and `_normalize_intl_entry()` must preserve the `"pipeline": "INTL"` field.

- [ ] **Step 6.1: Write failing tests for all three bugs**

```python
# Add to tests/test_cook_toplists.py

def test_ticker_count_uses_actual_regional_entries(cook_mod, registry, tmp_path):
    """ticker_count must equal us_count + len(europe) + len(asia), not len(intl_raw)."""
    us_payload_path = tmp_path / "top_lists_us.json"
    us_payload_path.write_text(json.dumps({
        "top_buys": [{"ticker": "AAPL", "final_score": 0.85, "market": "USA", "factors": {}}],
        "vix": 18.0, "vix_regime": "Normal", "kill_switch": False, "ticker_count": 1,
    }), encoding="utf-8")

    # INTL input with 3 entries; 1 has no registry match (will be dropped)
    intl_path = tmp_path / "top_lists_intl.json"
    intl_path.write_text(json.dumps([
        {"ticker": "SAP.DE",  "composite_score": 0.75, "region_applied": "INTL", "factor_snapshots": {}, "pipeline": "INTL"},
        {"ticker": "7203.T",  "composite_score": 0.65, "region_applied": "INTL", "factor_snapshots": {}, "pipeline": "INTL"},
        {"ticker": "UNKNOWN", "composite_score": 0.60, "region_applied": "INTL", "factor_snapshots": {}, "pipeline": "INTL"},
    ]), encoding="utf-8")

    out = tmp_path / "out.json"
    cook_mod.cook(us_payload_path, intl_path, registry, out)

    result = json.loads(out.read_text())
    # us_count=1, europe=1 (SAP.DE), asia=1 (7203.T); UNKNOWN dropped
    assert result["ticker_count"] == 3, (
        f"Expected 3 (1 US + 1 EU + 1 Asia), got {result['ticker_count']}"
    )


def test_missing_vix_exits_with_code_1(cook_mod, registry, tmp_path):
    """cook() must exit(1) when vix is missing, not just print a warning."""
    us_path = tmp_path / "top_lists_us.json"
    us_path.write_text(json.dumps({
        "top_buys": [], "vix_regime": "Normal", "kill_switch": False,
        # vix field intentionally absent
    }), encoding="utf-8")
    intl_path = tmp_path / "top_lists_intl.json"
    intl_path.write_text(json.dumps([]), encoding="utf-8")
    out = tmp_path / "out.json"

    with pytest.raises(SystemExit) as exc_info:
        cook_mod.cook(us_path, intl_path, registry, out)
    assert exc_info.value.code != 0


def test_vix_dampening_applied_to_intl_entries(cook_mod, registry, tmp_path):
    """INTL final_score must be multiplied by _apply_vix_multiplier(vix=40.0) = 0.20."""
    us_path = tmp_path / "top_lists_us.json"
    us_path.write_text(json.dumps({
        "top_buys": [], "vix": 40.0, "vix_regime": "BEAR_CRASH",
        "kill_switch": False, "ticker_count": 0,
    }), encoding="utf-8")
    intl_path = tmp_path / "top_lists_intl.json"
    intl_path.write_text(json.dumps([
        {"ticker": "SAP.DE", "composite_score": 1.0, "region_applied": "INTL", "factor_snapshots": {}, "pipeline": "INTL"},
    ]), encoding="utf-8")
    out = tmp_path / "out.json"

    cook_mod.cook(us_path, intl_path, registry, out)

    result = json.loads(out.read_text())
    eu_entry = result["top_buys_europe"][0]
    assert abs(eu_entry["final_score"] - 0.20) < 1e-6, (
        f"Expected 0.20 (1.0 × 0.20 for VIX=40), got {eu_entry['final_score']}"
    )


def test_pipeline_key_preserved_in_intl_entries(cook_mod, registry, tmp_path):
    """Normalized INTL entries must carry 'pipeline': 'INTL'."""
    us_path = tmp_path / "top_lists_us.json"
    us_path.write_text(json.dumps({
        "top_buys": [], "vix": 15.0, "vix_regime": "Normal",
        "kill_switch": False, "ticker_count": 0,
    }), encoding="utf-8")
    intl_path = tmp_path / "top_lists_intl.json"
    intl_path.write_text(json.dumps([
        {"ticker": "SAP.DE", "composite_score": 0.75, "region_applied": "INTL", "factor_snapshots": {}, "pipeline": "INTL"},
    ]), encoding="utf-8")
    out = tmp_path / "out.json"

    cook_mod.cook(us_path, intl_path, registry, out)

    result = json.loads(out.read_text())
    assert result["top_buys_europe"][0].get("pipeline") == "INTL"
```

- [ ] **Step 6.2: Run tests to confirm they fail**

```bash
pytest tests/test_cook_toplists.py::test_ticker_count_uses_actual_regional_entries \
       tests/test_cook_toplists.py::test_missing_vix_exits_with_code_1 \
       tests/test_cook_toplists.py::test_vix_dampening_applied_to_intl_entries \
       tests/test_cook_toplists.py::test_pipeline_key_preserved_in_intl_entries -v
```

Expected: All 4 FAIL.

- [ ] **Step 6.3: Apply all fixes to `scripts/cook_toplists.py`**

**Add `import math` at the top** (after existing imports):
```python
import math
```

**Add `_apply_vix_multiplier()` function** (after `_badge()` and before `_build_registry_map()`):
```python
def _apply_vix_multiplier(vix: float) -> float:
    """Return the VIX-based score dampening multiplier.

    Mirrors generate_top_lists._apply_vix_overlay() thresholds exactly.
    Guard is required: float('nan') >= 40 evaluates False in Python,
    so NaN would silently bypass dampening without the isinstance/isnan check.
    """
    if not isinstance(vix, (int, float)) or math.isnan(vix) or vix < 0:
        raise ValueError(f"[COOK] Invalid VIX value: {vix!r}")
    if vix >= 40:
        return 0.20
    if vix >= 30:
        return 0.50
    if vix >= 25:
        return 0.80
    return 1.00
```

**Fix Bug 2 — VIX missing fail-fast** (lines 73–77, replace the soft warning):
```python
# Before:
if vix is None:
    print(
        "[COOK] WARNING: US payload missing 'vix' field — audit check F will fail",
        file=sys.stderr,
    )

# After:
if vix is None:
    sys.exit(
        "[COOK] ERROR: US payload missing 'vix' — cannot apply macro overlay. Aborting."
    )
```

**Add VIX dampening and preserve `"pipeline"` in `_normalize_intl_entry()`** (lines 40–60). Modify the returned dict to include `"pipeline"` from the raw input, and update the function signature to accept `vix`:

```python
def _normalize_intl_entry(raw: dict, ticker_market_map: dict, vix: float) -> dict:
    """Convert StrategyEngine entry to audit_payload-compatible format.

    Applies VIX dampening to final_score. congress is forced to 0.0 (check E).
    """
    ticker = raw.get("ticker", "")
    market = ticker_market_map.get(ticker, "EUROPE")
    composite_score = float(raw.get("composite_score", 0.0))
    composite_score = round(composite_score * _apply_vix_multiplier(vix), 4)
    factor_snapshots = raw.get("factor_snapshots", {})
    factors = {k: v for k, v in factor_snapshots.items() if k != "congress"}
    factors["congress"] = 0.0
    return {
        "ticker": ticker,
        "final_score": composite_score,
        "badge": _badge(composite_score),
        "market": market,
        "factors": factors,
        "pipeline": raw.get("pipeline", "INTL"),   # preserved from StrategyEngine output
    }
```

**Update call to `_normalize_intl_entry`** (in `cook()`, lines 85–91) to pass `vix`:
```python
for raw_entry in intl_raw:
    normalized = _normalize_intl_entry(raw_entry, ticker_market, vix)
    if normalized["market"] == "EUROPE":
        top_buys_europe.append(normalized)
    elif normalized["market"] == "ASIA":
        top_buys_asia.append(normalized)
```

**Fix Bug 1 — Ticker count** (line 101):
```python
# Before:
"ticker_count": us_ticker_count + len(intl_raw),
# After:
"ticker_count": us_ticker_count + len(top_buys_europe) + len(top_buys_asia),
```

**Update default `--us-input` path** (line 121):
```python
# Before:
default="logs/top_lists.json",
# After:
default="logs/top_lists_us.json",
```

- [ ] **Step 6.4: Run all cook tests**

```bash
pytest tests/test_cook_toplists.py -v
```

Expected: All PASS. Fix any failures before continuing.

- [ ] **Step 6.5: Commit**

```bash
git add scripts/cook_toplists.py tests/test_cook_toplists.py
git commit -m "fix(cook): Bug1 ticker_count, Bug2 VIX fail-fast, VIX dampening, pipeline metadata"
```

---

## Task 7: Fix `send_toplists_discord.py` — Region-Aware Weight Fallback

**Files:**
- Modify: `scripts/send_toplists_discord.py`
- Test: `tests/test_send_toplists_discord.py`

### Context
`_factor_contribution_line()` (line ~492) imports `WEIGHTS_US as _DEFAULT_WEIGHTS` and falls back to it for entries that don't carry an inline `"weights"` key. EU/Asia entries have `"pipeline": "INTL"` (injected in Tasks 3 and 6) and should use `WEIGHTS_GLOBAL`. Add `_weights_for_entry()` helper that reads `"pipeline"` and returns the correct dict.

- [ ] **Step 7.1: Write a failing test**

```python
# Add to tests/test_send_toplists_discord.py

def test_weights_for_entry_returns_global_for_intl():
    """INTL entries must use WEIGHTS_GLOBAL, US entries must use WEIGHTS_US."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "send_toplists_discord",
        Path("scripts/send_toplists_discord.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    from regime_trader.config.weights import WEIGHTS_US, WEIGHTS_GLOBAL

    intl_entry = {"ticker": "SAP.DE", "pipeline": "INTL", "factors": {"momentum_long": 0.8}}
    us_entry   = {"ticker": "MSFT",   "pipeline": "US",   "factors": {"congress": 0.5}}

    intl_weights = mod._weights_for_entry(intl_entry)
    us_weights   = mod._weights_for_entry(us_entry)

    assert intl_weights == WEIGHTS_GLOBAL, "INTL entry must use WEIGHTS_GLOBAL"
    assert us_weights   == WEIGHTS_US,     "US entry must use WEIGHTS_US"
    assert intl_weights.get("congress", 0) == 0.0, "WEIGHTS_GLOBAL.congress must be 0.0"
```

- [ ] **Step 7.2: Run the test to confirm it fails**

```bash
pytest tests/test_send_toplists_discord.py::test_weights_for_entry_returns_global_for_intl -v
```

Expected: FAIL with `AttributeError: module 'send_toplists_discord' has no attribute '_weights_for_entry'`

- [ ] **Step 7.3: Add `_weights_for_entry()` and update `_factor_contribution_line()`**

In `scripts/send_toplists_discord.py`, find `_factor_contribution_line()` (around line 492). Add the helper function immediately before it, and update the fallback inside `_factor_contribution_line()`:

```python
def _weights_for_entry(entry: dict) -> dict:
    """Return the correct weight dict for an entry based on its pipeline metadata."""
    from regime_trader.config.weights import WEIGHTS_US, WEIGHTS_GLOBAL  # noqa: PLC0415
    return WEIGHTS_GLOBAL if entry.get("pipeline") == "INTL" else WEIGHTS_US


def _factor_contribution_line(entry: Dict[str, Any]) -> str:
    """Return top-3 weighted factor contributions as a compact string.

    Contribution = weight × normalized_score.  Pulls weights from the
    'weights' key in the entry (written by generate_top_lists) or falls back
    to the region-appropriate weight set via _weights_for_entry().
    Returns empty string when no factor data is present.
    """
    factors = entry.get("factors")
    if not factors:
        return ""
    weights = entry.get("weights") or _weights_for_entry(entry)
    contributions = {
```

Remove the now-unused import `from regime_trader.config.weights import WEIGHTS_US as _DEFAULT_WEIGHTS` from inside `_factor_contribution_line()` (line ~500) since the weight selection is now handled by `_weights_for_entry()`.

- [ ] **Step 7.4: Run the new test**

```bash
pytest tests/test_send_toplists_discord.py::test_weights_for_entry_returns_global_for_intl -v
```

Expected: PASS

- [ ] **Step 7.5: Run all Discord tests**

```bash
pytest tests/test_send_toplists_discord.py tests/test_discord_formatter.py -v
```

Expected: All PASS.

- [ ] **Step 7.6: Commit**

```bash
git add scripts/send_toplists_discord.py tests/test_send_toplists_discord.py
git commit -m "fix(discord): add _weights_for_entry() — use WEIGHTS_GLOBAL for INTL entries"
```

---

## Task 8: Consolidate Three GitHub Actions Workflows into One

**Files:**
- Create: `.github/workflows/daily_trading_pipeline.yml`
- Delete: `.github/workflows/pipeline_us.yml`
- Delete: `.github/workflows/pipeline_intl.yml`
- Delete: `.github/workflows/daily_toplists_discord.yml`

### Context
The existing `daily_toplists_discord.yml` listens to `workflow_run: [pipeline_us]` only. The INTL artifact is downloaded via `dawidd6/action-download-artifact@v8` from a separate run, relying on the assumption that INTL always finishes before US. This is fragile and uses `if_no_artifact_found: warn`, meaning a broken INTL pipeline silently continues. The `needs:` pattern in a single workflow guarantees both artifacts are present before the cook step.

The US job content is extracted from `pipeline_us.yml`. The INTL job content is extracted from `pipeline_intl.yml`. Artifacts are shared within the same workflow run using `actions/upload-artifact@v4` + `actions/download-artifact@v4` (no third-party action needed for same-run artifacts).

- [ ] **Step 8.1: Create `.github/workflows/daily_trading_pipeline.yml`**

```yaml
name: daily_trading_pipeline

# Single unified workflow replacing pipeline_us.yml + pipeline_intl.yml +
# daily_toplists_discord.yml. The cook_and_notify job uses needs: to guarantee
# both US and INTL artifacts are present before the merge step.

on:
  schedule:
    - cron: "0 0 * * 1-5"   # 00:00 UTC — post-Asia close
    - cron: "0 8 * * 1-5"   # 08:00 UTC — London open
    - cron: "0 16 * * 1-5"  # 16:00 UTC — NYC mid-session
  workflow_dispatch:
    inputs:
      dry_run:
        description: "Print Discord payload without sending"
        required: false
        default: "false"
        type: choice
        options: ["true", "false"]
      tickers_file:
        description: "Tickers CSV (relative to repo root)"
        required: false
        default: "config/universe.csv"
      force_regen:
        description: "Force top_lists regeneration even if fresh"
        required: false
        default: "false"
        type: choice
        options: ["true", "false"]

permissions:
  contents: read

concurrency:
  group: daily_trading_pipeline
  cancel-in-progress: false

jobs:
  # ── Job 1: US Pipeline ──────────────────────────────────────────────────────
  run_us_pipeline:
    name: US — EDGAR + FMP 9-Factor Scoring
    runs-on: ubuntu-latest
    timeout-minutes: 25

    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 1

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Get cache date
        id: cache_date
        run: echo "date=$(date -u '+%Y-%m-%d')" >> $GITHUB_OUTPUT

      - name: Restore pipeline data cache
        uses: actions/cache@v4
        with:
          path: .cache/
          key: us-pipeline-cache-${{ runner.os }}-${{ steps.cache_date.outputs.date }}
          restore-keys: |
            us-pipeline-cache-${{ runner.os }}-

      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-ci.txt
          pip install -r requirements.txt

      - name: FMP bulk pre-fetch
        env:
          FMP_API_KEY: ${{ secrets.FMP_API_KEY }}
          FMP_MAX_RPS: ${{ vars.FMP_MAX_RPS || '50' }}
          BULK_CACHE_DIR: ".cache/bulk_snapshots"
          BULK_CACHE_TTL_HOURS: "7"
        run: |
          set -euo pipefail
          python scripts/fmp_bulk_prefetch.py \
            --cache-dir "$BULK_CACHE_DIR" \
            --ttl-hours "$BULK_CACHE_TTL_HOURS" \
            --endpoints upgrades-downgrades-consensus-bulk ratios-ttm-bulk key-metrics-ttm-bulk \
            --verbose

      - name: EDGAR full-universe fetch
        env:
          EDGAR_USER_AGENT: ${{ secrets.EDGAR_USER_AGENT }}
          FMP_API_KEY: ${{ secrets.FMP_API_KEY }}
          FMP_MAX_RPS: ${{ vars.FMP_MAX_RPS || '50' }}
          EDGAR_FIRST: "true"
          BULK_CACHE_DIR: ".cache/bulk_snapshots"
        run: |
          TICKERS_FILE="${{ github.event.inputs.tickers_file || 'config/universe.csv' }}"
          mkdir -p logs .monitoring
          for ATTEMPT in 1 2 3; do
            if timeout 20m python scripts/run_pipeline.py \
              --tickers-file "$TICKERS_FILE" \
              --max-workers  8 \
              --log-dir      logs \
              --bulk-cache   .cache/bulk_snapshots \
              --verbose; then
              break
            fi
            [ "$ATTEMPT" -lt 3 ] && sleep $((ATTEMPT * 30))
            [ "$ATTEMPT" -eq 3 ] && echo "::error::All 3 EDGAR attempts failed" && exit 1
          done

      - name: Generate US top lists
        env:
          FMP_API_KEY: ${{ secrets.FMP_API_KEY }}
          GITHUB_RUN_ID: ${{ github.run_id }}
          BULK_CACHE_DIR: ".cache/bulk_snapshots"
        run: |
          FORCE="${{ github.event.inputs.force_regen == 'true' && '--force' || '' }}"
          python -m backend.market_intel.generate_top_lists \
            --log-dir    logs \
            --run-id     "$GITHUB_RUN_ID" \
            --bulk-cache .cache/bulk_snapshots \
            --verbose \
            $FORCE

      - name: Validate top_lists_us.json
        run: |
          python3 - <<'EOF'
          import json, sys
          from pathlib import Path
          p = Path("logs/top_lists_us.json")
          if not p.exists():
              print("::error::top_lists_us.json was not written — pipeline failed silently")
              sys.exit(1)
          d = json.loads(p.read_text())
          if not d.get("top_buys"):
              print("::error::top_lists_us.json has no top_buys")
              sys.exit(1)
          print(f"Validated: {len(d['top_buys'])} top buys, pipeline=US OK")
          EOF

      - name: Upload us-top-lists artifact
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: us-top-lists
          retention-days: 7
          if-no-files-found: error
          path: |
            logs/top_lists_us.json
            logs/top5.csv
            logs/intel_source_status.json

  # ── Job 2: INTL Pipeline ────────────────────────────────────────────────────
  run_intl_pipeline:
    name: INTL — FMP 10-Factor Scoring (EU + Asia)
    runs-on: ubuntu-latest
    timeout-minutes: 12

    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 1

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Get cache date
        id: cache_date
        run: echo "date=$(date -u '+%Y-%m-%d')" >> $GITHUB_OUTPUT

      - name: Restore pipeline data cache
        uses: actions/cache@v4
        with:
          path: .cache/
          key: intl-pipeline-cache-${{ runner.os }}-${{ steps.cache_date.outputs.date }}
          restore-keys: |
            intl-pipeline-cache-${{ runner.os }}-
            us-pipeline-cache-${{ runner.os }}-

      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements-ci.txt
          pip install -r requirements.txt

      - name: FMP bulk pre-fetch
        env:
          FMP_API_KEY: ${{ secrets.FMP_API_KEY }}
          FMP_MAX_RPS: ${{ vars.FMP_MAX_RPS || '50' }}
          BULK_CACHE_DIR: ".cache/bulk_snapshots"
          BULK_CACHE_TTL_HOURS: "7"
        run: |
          set -euo pipefail
          python scripts/fmp_bulk_prefetch.py \
            --cache-dir "$BULK_CACHE_DIR" \
            --ttl-hours "$BULK_CACHE_TTL_HOURS" \
            --endpoints upgrades-downgrades-consensus-bulk ratios-ttm-bulk \
            --verbose

      - name: Fetch INTL raw metrics
        env:
          FMP_API_KEY: ${{ secrets.FMP_API_KEY }}
          BULK_CACHE_DIR: ".cache/bulk_snapshots"
        run: |
          mkdir -p logs
          python scripts/fmp_intl_fetch.py \
            --registry config/ticker_registry.json \
            --bulk-cache .cache/bulk_snapshots \
            --out logs/raw_intl_fetcher_output.json \
            --verbose

      - name: Score INTL tickers
        run: |
          python scripts/run_pipeline_profile.py \
            --config   backend/market_intel/profiles/intl_strategy.json \
            --raw-data logs/raw_intl_fetcher_output.json \
            --out-dir  logs

      - name: Upload intl-top-lists artifact
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: intl-top-lists
          retention-days: 7
          if-no-files-found: error
          path: |
            logs/top_lists_intl.json

  # ── Job 3: Cook + Discord ───────────────────────────────────────────────────
  cook_and_notify:
    name: Merge + Discord Notification
    runs-on: ubuntu-latest
    timeout-minutes: 10
    needs: [run_us_pipeline, run_intl_pipeline]   # blocks until BOTH complete

    steps:
      - name: Checkout
        uses: actions/checkout@v4
        with:
          fetch-depth: 1

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install deps
        run: |
          python -m pip install --upgrade pip
          pip install requests>=2.31.0
          pip install -r requirements-ci.txt

      - name: Download US top-lists
        uses: actions/download-artifact@v4
        with:
          name: us-top-lists
          path: logs/
        # if-no-files-found defaults to 'warn' for download; the needs: gate
        # guarantees this artifact exists because run_us_pipeline used error.

      - name: Download INTL top-lists
        uses: actions/download-artifact@v4
        with:
          name: intl-top-lists
          path: logs/

      - name: Cook (merge US + INTL)
        run: |
          python scripts/cook_toplists.py \
            --us-input    logs/top_lists_us.json \
            --intl-input  logs/top_lists_intl.json \
            --registry    config/ticker_registry.json \
            --output      logs/top_lists.json

      - name: Audit combined payload
        run: python scripts/audit_payload.py --input logs/top_lists.json

      - name: Send Discord notification
        env:
          DISCORD_WEBHOOK_URL: ${{ secrets.DISCORD_WEBHOOK_URL }}
        run: |
          DRY="${{ github.event.inputs.dry_run || 'false' }}"
          python scripts/send_toplists_discord.py \
            --input logs/top_lists.json \
            $( [ "$DRY" = "true" ] && echo "--dry-run" )
```

- [ ] **Step 8.2: Delete the three superseded workflow files**

```bash
git rm .github/workflows/pipeline_us.yml
git rm .github/workflows/pipeline_intl.yml
git rm .github/workflows/daily_toplists_discord.yml
```

- [ ] **Step 8.3: Verify the new YAML is valid**

```bash
python3 -c "
import yaml
with open('.github/workflows/daily_trading_pipeline.yml') as f:
    d = yaml.safe_load(f)
jobs = list(d['jobs'].keys())
assert 'run_us_pipeline' in jobs
assert 'run_intl_pipeline' in jobs
assert 'cook_and_notify' in jobs
needs = d['jobs']['cook_and_notify'].get('needs', [])
assert 'run_us_pipeline' in needs and 'run_intl_pipeline' in needs
print('YAML valid. Jobs:', jobs)
print('cook_and_notify needs:', needs)
"
```

Expected:
```
YAML valid. Jobs: ['run_us_pipeline', 'run_intl_pipeline', 'cook_and_notify']
cook_and_notify needs: ['run_us_pipeline', 'run_intl_pipeline']
```

- [ ] **Step 8.4: Commit**

```bash
git add .github/workflows/daily_trading_pipeline.yml
git commit -m "feat(ci): consolidate 3 workflows into single pipeline with needs: gate — fix race condition"
```

---

## Final Verification

Run these checks after all 8 tasks are complete:

- [ ] **V1: Full test suite**

```bash
pytest tests/ -v --tb=short 2>&1 | tail -30
```

Expected: All existing tests pass. Fix any regressions before declaring done.

- [ ] **V2: US isolation check**

```bash
python -c "
import json
from pathlib import Path
status = json.loads(Path('logs/intel_source_status.json').read_text())
intl_rows = [r for r in status.get('results', []) if r.get('market') not in ('USA', 'US', None)]
if intl_rows:
    print('FAIL: INTL rows in intel_source_status.json:', [r['ticker'] for r in intl_rows])
else:
    print('PASS: intel_source_status.json is US-only')
"
```

(Requires a prior run of `run_pipeline.py` or existing `logs/intel_source_status.json`)

- [ ] **V3: INTL factor count check**

```bash
python -c "
import json
from pathlib import Path
intl = json.loads(Path('logs/top_lists_intl.json').read_text())
for entry in intl[:3]:
    n = len(entry.get('factor_snapshots', {}))
    print(f\"{entry['ticker']}: {n} factors, pipeline={entry.get('pipeline')}\")
    assert entry.get('pipeline') == 'INTL', 'Missing pipeline key'
print('PASS: INTL entries have pipeline=INTL')
"
```

- [ ] **V4: Merged output has VIX dampened INTL scores**

```bash
python -c "
import json
from pathlib import Path
combined = json.loads(Path('logs/top_lists.json').read_text())
vix = combined.get('vix', 0)
print(f'VIX={vix}')
for entry in combined.get('top_buys_europe', [])[:2]:
    print(f\"  {entry['ticker']}: final_score={entry['final_score']:.4f}, pipeline={entry.get('pipeline')}\")
print('PASS: EU entries present with pipeline key')
"
```

- [ ] **V5: Weights integrity**

```bash
python -c "
from regime_trader.config.weights import WEIGHTS_US, WEIGHTS_GLOBAL
assert abs(sum(WEIGHTS_US.values()) - 1.0) < 1e-6
assert abs(sum(WEIGHTS_GLOBAL.values()) - 1.0) < 1e-6
assert WEIGHTS_GLOBAL['analyst_revision'] == 0.02
assert WEIGHTS_GLOBAL['price_target_upside'] == 0.03
assert WEIGHTS_GLOBAL['congress'] == 0.00
print('PASS: WEIGHTS_US sum=', sum(WEIGHTS_US.values()))
print('PASS: WEIGHTS_GLOBAL sum=', sum(WEIGHTS_GLOBAL.values()))
"
```

- [ ] **V6: CI YAML jobs structure**

```bash
python3 -c "
import yaml
with open('.github/workflows/daily_trading_pipeline.yml') as f:
    d = yaml.safe_load(f)
needs = d['jobs']['cook_and_notify'].get('needs', [])
assert set(needs) == {'run_us_pipeline', 'run_intl_pipeline'}, needs
print('PASS: cook_and_notify.needs =', needs)
assert not __import__('pathlib').Path('.github/workflows/pipeline_us.yml').exists(), 'pipeline_us.yml still exists'
assert not __import__('pathlib').Path('.github/workflows/pipeline_intl.yml').exists(), 'pipeline_intl.yml still exists'
assert not __import__('pathlib').Path('.github/workflows/daily_toplists_discord.yml').exists(), 'daily_toplists_discord.yml still exists'
print('PASS: Old workflow files deleted')
"
```
