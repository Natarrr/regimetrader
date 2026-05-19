# System-Wide Consistency Fixes — Design Spec

**Date:** 2026-05-19  
**Status:** Approved

---

## Goal

Fix four inter-system discrepancies that cause stale Discord data, empty backtest results,
a broken hybrid pipeline, and a dead code loader in the Streamlit UI.

## Architecture Overview

The system has two distinct scoring pipelines that serve different purposes and must not
be conflated:

| Pipeline | Weights | Output | Consumers |
|---|---|---|---|
| **edgar_3x** (5-factor) | edgar 28%, insider 23%, congress 22%, news 15%, momentum 12% | `logs/top_lists.json` → `top-lists` artifact | Discord, Stock Picker, hybrid_pipeline (after fix), backtest |
| **engine_worker** (3-factor) | insider 45%, institutional 35%, momentum 20% | `data/market_state.json` | Market Intel dashboard tab |

Both pipelines are kept. The 3-factor model is a fast smart-money pre-screen; the
5-factor model is the full institutional-grade ranking. They serve different UI surfaces
and should not be merged.

---

## Change 1: Fix `hybrid_pipeline.yml` — consume edgar_3x artifact

### Problem

The `quant` job re-runs `discovery_scanner.get_top_alpha_picks_sync()` which uses the
3-weight model and re-hits the SEC. This is wrong: `hybrid_pipeline` is meant to be a
Claude enrichment layer on top of the 5-factor `edgar_3x` output, not an independent
pipeline. Additionally:
- It references `secrets.SEC_USER_AGENT` but the codebase reads `EDGAR_USER_AGENT` → 403
- The `edgar_3x` and `hybrid_pipeline` quant jobs run independently, producing different
  rankings fed to Claude — the Claude analysis is therefore inconsistent with what the
  Discord message reports

### Fix

**`quant` job — replace entirely:**

1. Download the `top-lists` artifact from the latest successful `edgar_3x` run using
   `dawidd6/action-download-artifact@v8` (same action used by `daily_toplists_discord`)
2. Load `logs/top_lists.json` and reshape `top_buys` entries into the `shortlist` and
   `candidates` format the `claude` job already expects:
   - `shortlist`: list of ticker strings from `top_buys`
   - `candidates`: list of dicts with fields `symbol`, `smart_money_score` (= `final_score`),
     `insider_score`, `institutional_score` (= `congress` factor), `momentum_score`,
     `edgar_score`
3. Read `regime` from `top_lists.json` fields: if `kill_switch == true` → "Panic/Crash";
   else derive from `vix`: ≥25 → "Bear", <25 → "Normal". No yfinance re-fetch.
4. Write `data/pipeline/shortlist.json` in the same schema as before so the `claude` job
   needs zero changes
5. Remove `secrets.SEC_USER_AGENT` env var — not needed (no SEC calls in quant job)
6. Remove `from regime_trader.scanners.discovery_scanner import get_top_alpha_picks_sync`
   inline Python block
7. Remove `from analysis.earnings_analyzer import build_shortlist` inline Python block —
   replaced by inline reshaping of `top_lists.json`

**`detect regime` step — replace:**
- Remove yfinance VIX download; derive regime string from `top_lists.json` directly
- Output `regime=` to `$GITHUB_OUTPUT` as before so the `claude` job still works

**`preflight_cost_gate` job — no change.**

**`claude` job — no change.** It already consumes `data/pipeline/shortlist.json` and calls
`analysis/earnings_analyzer.build_prompt(symbol, quant_data, [], regime)` which accepts
any dict for `quant_data`. The field names need to match what `build_prompt` reads:
- `quant_data.get("smart_money_score", 0) * 100` → use `final_score * 100`
- `quant_data.get("insider_score", 0)` → from reshaped entry
- `quant_data.get("inst_score", 0)` → map from `congress` factor (closest institutional proxy)
- `quant_data.get("momentum_score", 0)` → from `momentum` factor

**Schedule dependency note** (add as comment):  
`edgar_3x` 08:00 UTC run takes ~40 min → completes ~08:40. `hybrid_pipeline` fires at
12:30 UTC — safe margin. The `dawidd6` action always fetches the most recent successful
`edgar_3x` artifact, so even if a run is delayed, it gets the latest available data.

**Files changed:**
- `.github/workflows/hybrid_pipeline.yml` — rewrite `quant` job + `detect regime` step

---

## Change 2: Add archive accumulation to `edgar_3x.yml`

### Problem

`weekly_backtest.yml` reads `logs/archive/` for historical snapshots but no mechanism
populates it. The directory exists locally but is empty. The backtest runs every Friday
and silently reports 0 signals.

### Fix

Add an `archive-snapshot` job to `edgar_3x.yml` that runs after `fetch-and-rank`:

```yaml
archive-snapshot:
  name: Archive Daily Snapshot
  needs: fetch-and-rank
  runs-on: ubuntu-latest
  permissions:
    contents: write
  steps:
    - Checkout (with token: ${{ secrets.GITHUB_TOKEN }})
    - Download top-lists artifact (dawidd6/action-download-artifact@v8)
    - Check if logs/archive/YYYY-MM-DD_top_lists.json already exists
      → if yes: skip (only first run of the day archives)
    - Copy logs/top_lists.json → logs/archive/YYYY-MM-DD_top_lists.json
    - git add + git commit "chore(archive): snapshot YYYY-MM-DD" + git push
```

**`.gitignore` update:**  
`logs/` is currently ignored. Add `!logs/archive/` exemption so the archive files are
tracked by git. This is the only change to `.gitignore`.

**Guard against duplicate daily snapshots:**  
The `edgar_3x` workflow runs 3× per day. Only the first successful run of each calendar
day (UTC) should write to the archive. The step checks `git ls-files logs/archive/YYYY-MM-DD_top_lists.json` — if the file is already tracked, it skips silently with exit 0.

**Files changed:**
- `.github/workflows/edgar_3x.yml` — add `archive-snapshot` job
- `.gitignore` — add `!logs/archive/` exemption

---

## Change 3: Add "Sync from GitHub" button to Stock Picker

### Problem

`pages/6_Stock_Picker.py` reads `logs/top_lists.json` from disk. This file is only
updated when you run `edgar_3x` locally or manually copy the artifact. The GitHub Actions
version is always fresh but the local file can be days stale.

### Fix

Add a **"⬇ Sync from GitHub"** button next to the existing "↻ Refresh" button.

**Behaviour:**
1. Find the latest `top-lists` artifact ID:  
   `GET /repos/{owner}/{repo}/actions/artifacts?name=top-lists&per_page=5`  
   → pick the first non-expired entry
2. Download the artifact zip:  
   `GET /repos/{owner}/{repo}/actions/artifacts/{id}/zip`  
   with `Authorization: Bearer {GH_PAT}`, follow redirects
3. Extract `top_lists.json` from the zip using `zipfile` (stdlib)
4. Compare `generated_at` in the downloaded file vs the current local file:
   - If downloaded is newer → write atomically to `logs/top_lists.json`, clear
     `_load_top_lists` cache, `st.rerun()`
   - If local is already newer or same → `st.toast("Already up to date")`
5. On any HTTP/parse error → `st.error(f"Sync failed: {msg}")`, no file written

**Visibility gate:** only shown when `GH_PAT` is set (inside the existing `if _gh_pat:` block, alongside the pipeline trigger button).

**New helper function:** `_sync_from_github(pat: str) -> tuple[bool, str]` — returns
`(success, message)`. Self-contained, uses `requests` (already imported) and `zipfile`
(stdlib). No new imports, no new dependencies.

**Files changed:**
- `pages/6_Stock_Picker.py` — add `_sync_from_github()` and "⬇ Sync" button

---

## Change 4: Remove dead `_load_discovery` from streamlit_app

### Problem

`_load_discovery()` in `regime_trader/ui/streamlit_app.py` (lines ~206–223) imports
`get_top_alpha_picks_sync` and wraps it in a `@st.cache_data` decorator. It is never
called to render anything — the Market Intel tab reads exclusively from `_load_market_state()`
which loads `data/market_state.json`. The only reference to `_load_discovery` is a
`.clear()` call in the sidebar cache controls, which would then also be removed.

### Fix

- Remove `_load_discovery()` function definition (~18 lines)
- Remove the `_load_discovery.clear()` line in the "Clear engine state cache" button handler
  (the `_load_market_state.clear()` call on the same button stays)

**What is NOT removed:**
- `discovery_scanner.py` module — still used by `engine_worker.py`
- `engine_worker.py` — still the data source for the Market Intel tab
- All tests — `test_discovery_scanner.py` etc. remain valid

**Files changed:**
- `regime_trader/ui/streamlit_app.py` — remove `_load_discovery` function + `.clear()` call

---

## Data Flow After All Fixes

```
edgar_3x (00/08/16 UTC)
  └─ run_pipeline.py → intel_source_status.json
  └─ generate_top_lists.py → top_lists.json → artifact "top-lists"
  └─ archive-snapshot job → logs/archive/YYYY-MM-DD_top_lists.json (git commit)

daily_toplists_discord (18:30 UTC)
  └─ downloads "top-lists" artifact → sends Discord embed

hybrid_pipeline (12:30 UTC weekdays)
  └─ downloads "top-lists" artifact → reshapes to shortlist
  └─ claude job → enriched analysis (audit artifact)

weekly_backtest (Friday 21:00 UTC)
  └─ reads logs/archive/*.json (now populated by archive-snapshot)
  └─ produces backtest_report_latest.json

Stock Picker page (local)
  └─ reads logs/top_lists.json
  └─ "⬇ Sync" button → downloads latest artifact → writes logs/top_lists.json

engine_worker (local / separate)
  └─ discovery_scanner (3-weight) → data/market_state.json
  └─ Market Intel tab reads market_state.json
```

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| `edgar_3x` artifact not found when `hybrid_pipeline` runs | `dawidd6` warns (not fails); `shortlist.json` → empty list; `claude` job skips gracefully |
| `archive-snapshot` git push conflict (two concurrent runs) | Second run's `git ls-files` check finds file already committed → skips silently |
| "Sync from GitHub" when PAT lacks `actions:read` | `st.error()` with 403 message; local file unchanged |
| `top_lists.json` missing from artifact zip | `st.error()` with filename; local file unchanged |

---

## Testing

- `CI` workflow: no new test files needed. Existing `test_streamlit_app_smoke.py` imports `streamlit_app` — removing `_load_discovery` must not break it (confirm `_load_discovery` is not referenced in test fixtures)
- `hybrid_pipeline` changes verified by running `workflow_dispatch` with `dry_run=true` after implementation
- `archive-snapshot` verified by checking `git log --oneline logs/archive/` after first `edgar_3x` run
- "Sync" button verified locally by deleting `logs/top_lists.json` and clicking Sync
