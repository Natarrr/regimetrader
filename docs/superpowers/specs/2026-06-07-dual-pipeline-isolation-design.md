# Design: Dual-Pipeline Isolation — US (EDGAR/SEC) vs EU/Asia (FMP)

## Context

The `regime_trader` codebase has two parallel development tracks that both score international tickers, creating competing outputs, a VIX overlay gap, and cross-sectional normalization contamination risk:

- **Procedural track** (`run_pipeline.py`): Scores US tickers via EDGAR + FMP AND EU/Asia tickers via `_score_ticker_international()` → writes both into `intel_source_status.json`
- **StrategyEngine track** (`run_pipeline_profile.py`): Scores the same EU/Asia tickers via FMP → writes `top_lists_intl.json`

INTL tickers are double-scored. The StrategyEngine output is never VIX-dampened. `generate_top_lists.py` reads a mixed-market `intel_source_status.json`, where INTL rows inflate the universe count used by the schema-gate circuit-breaker. `cook_toplists.py` has three bugs that inflate ticker counts, fail silently on missing VIX, and apply `WEIGHTS_US` (with `congress=0.22`) to EU/Asia entries in Discord.

**Goal:** Two fully isolated, independently normalizing pipelines — US via EDGAR/SEC, EU/Asia via FMP — with VIX/SPY as a shared global macro overlay, and a unified Discord notification after both finish.

---

## Chosen Approach: Surgical Removal + StrategyEngine Expansion

Remove INTL from `run_pipeline.py`. Expand StrategyEngine INTL from 6 to 8 active factors. Apply VIX dampening to INTL entries in `cook_toplists.py`. Fix 3 bugs. Both CI pipelines run in parallel; Discord waits for both artifacts.

---

## Architecture

### Target Data Flow

```
pipeline_us.yml (GitHub Actions)        pipeline_intl.yml (GitHub Actions)
      |                                           |
run_pipeline.py                         run_pipeline_profile.py
  US tickers only (EDGAR + FMP)           EU/Asia tickers only (FMP Global)
  Publishes: us-top-lists artifact         Publishes: intl-top-lists artifact
      |                                           |
intel_source_status.json                top_lists_intl.json
  (US tickers only, post-fix)             (8 active factors per ticker)
      |                                           |
generate_top_lists.py                   [VIX applied in cook step]
  US-only normalization + MVO                    |
  VIX overlay + Congress boost                   |
      |                                           |
logs/top_lists_us.json                          ...
      |                                           |
      └─────────────────┬─────────────────────────┘
                        |
               cook_toplists.py
               ├─ Read VIX from top_lists_us.json
               ├─ Apply _apply_vix_multiplier() to INTL final_score
               ├─ Split INTL by registry → europe / asia
               └─ Bug fixes: ticker_count, VIX fail-fast, weight fallback
                        |
               logs/top_lists.json (unified SSOT)
                        |
               send_toplists_discord.py
               (US + EU + Asia sections, global VIX)
```

### Macro Regime: VIX/SPY Shared Signal

The US pipeline computes `vix` and `spy_momentum_regime` and writes them into `logs/top_lists_us.json`. `cook_toplists.py` reads these values and applies the same dampening multiplier to INTL `final_score` entries before writing the merged output. INTL pipeline has zero dependency on US pipeline at scoring time — the dependency is deferred to the cook step only.

---

## Component Changes

### 1. `scripts/run_pipeline.py` — Remove INTL Branch (US-Only)

**Remove:**
- `_score_ticker_international()` function (lines 1204–1337)
- EU/Asia pipeline branch orchestrator (lines 1784–1894)
- Regression guard asserting `congress_score=0.0` for INTL markets (lines 1883–1892) — no longer needed

**Preserve unchanged:**
- All EDGAR/SEC functions: `_sec_get`, `_load_cik_map`, `_parse_form4_xml`, `fetch_edgar_data`
- `_score_ticker()` US scorer (lines 1463–1700)
- `load_tickers()` — reads from `config/universe.csv` (US only, no change)
- All FMP US fetchers, Congress feed, transcript tone

**Guard added** after ticker load: assert no ticker suffix belongs to EU/Asia suffix sets (`.DE`, `.L`, `.T`, `.HK`, etc.) to catch accidental registry contamination at startup.

---

### 2. `backend/market_intel/generate_top_lists.py` — Remove INTL Processing Path

**Remove:**
- `intl_clean = [r for r in clean if r.get("market") not in ("USA", "US")]` split (line ~2049)
- Any separate INTL ranking path using this split

**Preserve:**
- `_cross_sectional_normalize()` — now runs on US universe only (no contamination)
- `_schema_gate()` circuit-breaker — counts US tickers only (40% threshold now correct)
- `_apply_vix_overlay()` — US only, unchanged
- `_apply_congress_boost()` — US only, unchanged
- `run_optimizer()` MVO — US portfolio only, unchanged

**Output path**: `logs/top_lists.json` → **`logs/top_lists_us.json`** (unambiguous data contract).

---

### 3. `backend/market_intel/_generate_top_lists_intl_patch.py` — Delete

Becomes dead code once `generate_top_lists.py` no longer imports `_build_intl_entry()`. Delete file entirely.

---

### 4. `backend/market_intel/engine.py` + `run_pipeline_profile.py` — Expand INTL to 8 Factors

**Activate in `backend/market_intel/profiles/intl_strategy.json` (6 → 8 active factors):**
- `insider_conviction` — MAR Art. 19 equivalent; FMP `insider-trading/search` works for EU/Asia; already fetched in `FMPFetcher`, currently zeroed in profile
- `insider_breadth` — same endpoint, breadth from participant count; already in `FMPFetcher`
- `analyst_revision` — `analyst-estimates` endpoint; already wired in `FMPFetcher`, weight=0.00
- `price_target_upside` — `price-target-consensus` endpoint; already wired, weight=0.00

**Update `regime_trader/config/weights.py` → `WEIGHTS_GLOBAL`:**

| Factor | Old | New | Delta | Citation |
|--------|-----|-----|-------|----------|
| `insider_conviction` | 0.30 | 0.28 | −0.02 | MAR Art. 19 parity |
| `insider_breadth` | 0.15 | 0.14 | −0.01 | |
| `news_sentiment` | 0.13 | 0.13 | — | Tetlock (2007) |
| `news_buzz` | 0.05 | 0.04 | −0.01 | donor |
| `momentum_long` | 0.17 | 0.17 | — | Rouwenhorst (1998) |
| `volume_attention` | 0.05 | 0.04 | −0.01 | donor |
| `analyst_consensus` | 0.10 | 0.10 | — | Givoly & Lakonishok (1979) |
| `quality_piotroski` | 0.05 | 0.05 | — | Piotroski (2000) |
| `analyst_revision` | 0.00 | **0.02** | +0.02 | Chan, Jegadeesh & Lakonishok (1996) |
| `price_target_upside` | 0.00 | **0.03** | +0.03 | Brav & Lehavy (2003) |
| `congress` | 0.00 | 0.00 | — | structurally absent |
| `transcript_tone` | 0.00 | 0.00 | — | FMP US-only |

**Sum = 1.00** — `assert abs(sum(WEIGHTS_GLOBAL.values()) - 1.0) < 1e-6` enforced at module load.

StrategyEngine dynamic normalization (`score = weighted_sum / available_weight`) already handles missing factors gracefully; activating new factors increases `available_weight` when data is present.

---

### 5. `scripts/cook_toplists.py` — VIX Dampening + 3 Bug Fixes

**Bug 1 — Ticker count inflation:**
```python
# Before: counts dropped/unknown tickers
"ticker_count": us_ticker_count + len(intl_raw),
# After: counts only entries that passed registry lookup
"ticker_count": us_ticker_count + len(top_buys_europe) + len(top_buys_asia),
```

**Bug 2 — VIX missing must fail fast:**
```python
# Before: soft warning, continues to audit failure
if vix is None:
    print("[COOK] WARNING: missing 'vix'", file=sys.stderr)
# After: fail with non-zero exit code
if vix is None:
    sys.exit("[COOK] ERROR: US payload missing 'vix' — cannot apply macro overlay. Aborting.")
```

**VIX dampening for INTL entries (new):**
```python
import math

def _apply_vix_multiplier(vix: float) -> float:
    if not isinstance(vix, (int, float)) or math.isnan(vix) or vix < 0:
        raise ValueError(f"[COOK] Invalid VIX value: {vix!r}")
    if vix >= 40: return 0.20
    if vix >= 30: return 0.50
    if vix >= 25: return 0.80
    return 1.00
```
The guard is required because `float('nan') >= 40` evaluates `False` in Python — NaN would silently fall through to `return 1.00`, bypassing dampening entirely. Raising here propagates to `cook_toplists.py`'s main error handler, which already exits with code 1.

Apply `entry["final_score"] *= _apply_vix_multiplier(vix)` to each normalized INTL entry before the europe/asia split. Mirrors the thresholds in `generate_top_lists.py:_apply_vix_overlay()` exactly.

**Input path**: `logs/top_lists.json` → **`logs/top_lists_us.json`** (match new generate_top_lists output name).

---

### 6. `scripts/send_toplists_discord.py` — Bug 8: Region-Aware Weight Fallback

The `_factor_contribution_line()` function falls back to `WEIGHTS_US` (which carries `congress=0.22`) when `entry.get("weights")` is absent. For EU/Asia entries this produces incorrect contribution percentages.

**Fix — inject `"pipeline"` metadata upstream, not suffix-parse downstream:**

In `generate_top_lists.py`, add `"pipeline": "US"` to each entry in `logs/top_lists_us.json`. In `run_pipeline_profile.py` / StrategyEngine, add `"pipeline": "INTL"` to each entry in `top_lists_intl.json`. `cook_toplists.py` preserves this field in the merged output.

`send_toplists_discord.py` then reads the metadata directly:
```python
def _weights_for_entry(entry: dict) -> dict:
    return WEIGHTS_GLOBAL if entry.get("pipeline") == "INTL" else WEIGHTS_US
```
Use `_weights_for_entry(entry)` as the fallback in `_factor_contribution_line()`. This removes the fragile suffix-parsing and makes the weight selection explicit and exchange-agnostic — adding `.TO` or `.AX` tickers in the future requires zero changes in the Discord layer.

---

### 7. `.github/workflows/daily_trading_pipeline.yml` — Consolidate into Single Workflow

**Replace the three-file structure** (`pipeline_us.yml` + `pipeline_intl.yml` + `daily_toplists_discord.yml`) with a single `daily_trading_pipeline.yml`. The `workflow_run` trigger fires once per completed workflow — if the Discord workflow listens to both upstream workflows, it fires twice: once when US finishes (INTL artifact not yet available) and once when INTL finishes (stale US artifact). The `needs:` dependency in a single workflow eliminates this race condition entirely.

```yaml
jobs:
  run_us_pipeline:
    runs-on: ubuntu-latest
    # ... EDGAR + FMP US steps ...
    # Uploads: us-top-lists artifact

  run_intl_pipeline:
    runs-on: ubuntu-latest
    # ... FMP Global INTL steps ...
    # Uploads: intl-top-lists artifact

  cook_and_notify:
    runs-on: ubuntu-latest
    needs: [run_us_pipeline, run_intl_pipeline]  # blocks until both complete
    steps:
      - uses: actions/download-artifact@v4
        with: { name: us-top-lists, if-no-files-found: error }
      - uses: actions/download-artifact@v4
        with: { name: intl-top-lists, if-no-files-found: error }
      - run: python scripts/cook_toplists.py
      - run: python scripts/audit_payload.py
      - run: python scripts/send_toplists_discord.py
```

The cron schedule moves to the top-level `on: schedule:` trigger. Both pipeline jobs run concurrently within the same workflow run, and `cook_and_notify` is guaranteed to have both artifacts present when it starts.

---

## Error Handling

| Scenario | Behavior |
|----------|----------|
| INTL artifact missing at cook time | `cook_toplists.py` exits code 1; Discord step skipped |
| US artifact missing at cook time | `cook_toplists.py` exits code 1 |
| VIX absent from US artifact | `cook_toplists.py` exits code 1 (fail-fast) |
| Ticker absent from `ticker_registry.json` | Logged WARNING, silently dropped (unchanged) |
| INTL score > dynamic ceiling | `audit_payload.py` check E2 catches it (unchanged) |

---

## Files to Modify

| File | Change | Summary |
|------|--------|---------|
| `scripts/run_pipeline.py` | Remove | Delete `_score_ticker_international()` + EU/Asia orchestrator + regression guard |
| `backend/market_intel/generate_top_lists.py` | Remove + Modify | Remove `intl_clean` split; write `logs/top_lists_us.json`; inject `"pipeline": "US"` into each entry |
| `backend/market_intel/_generate_top_lists_intl_patch.py` | Delete | Dead code after removal |
| `backend/market_intel/profiles/intl_strategy.json` | Modify | Add 4 factors to `active_factors` |
| `run_pipeline_profile.py` / StrategyEngine | Modify | Inject `"pipeline": "INTL"` into each entry in `top_lists_intl.json` |
| `regime_trader/config/weights.py` | Modify WEIGHTS_GLOBAL | Activate analyst_revision (0.02) and price_target_upside (0.03) |
| `scripts/cook_toplists.py` | Modify | Read `top_lists_us.json`; VIX dampening with NaN guard; fix Bugs 1, 2, count; preserve `"pipeline"` field |
| `scripts/send_toplists_discord.py` | Modify | Replace suffix-parsing with `_weights_for_entry()` reading `entry["pipeline"]` |
| `.github/workflows/daily_trading_pipeline.yml` | Create (replace 3 files) | Single workflow: parallel US + INTL jobs, `cook_and_notify` with `needs:` |
| `.github/workflows/pipeline_us.yml` | Delete | Superseded by consolidated workflow |
| `.github/workflows/pipeline_intl.yml` | Delete | Superseded by consolidated workflow |
| `.github/workflows/daily_toplists_discord.yml` | Delete | Superseded by consolidated workflow |

---

## Verification

1. **US isolation**: `python scripts/run_pipeline.py --dry-run` → `intel_source_status.json` has zero rows with `market ∈ {EUROPE, ASIA}`
2. **INTL factors**: `python scripts/run_pipeline_profile.py` → each entry in `top_lists_intl.json` has 8 non-zero `factor_snapshots` keys
3. **VIX dampening**: `python scripts/cook_toplists.py` with injected `vix=40.0` → all INTL `final_score ≤ 0.20 × pre-dampened score`
4. **Audit gate**: `python scripts/audit_payload.py` → all 7 checks A-G pass
5. **Ticker count**: merged `ticker_count == len(top_buys_usa) + len(top_buys_europe) + len(top_buys_asia)`
6. **Weights integrity**: `assert abs(sum(WEIGHTS_GLOBAL.values()) - 1.0) < 1e-6` passes at module import
7. **CI dry run**: `daily_toplists_discord.yml` with `dry_run=true` → Discord renders all three regional sections with correct badge labels and factor matrices
