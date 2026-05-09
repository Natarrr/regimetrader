# PR: Prune unused folders (2026-05-09)

**Branch:** `chore/prune-unused-folders-20260509` → `main`  
**Archive commit:** `1e08acf96e6a3ab45a1aa8b0018b9c9497c41ed3`  
**Archive path:** `archive/prune-chore-prune-unused-folders-20260509/`

---

## Rationale

Four top-level folders accumulated in the repo with **zero Python runtime callers**
and no CI pipeline or test suite references. They were contributing dead weight
(326+ MB for `frontend/` alone) and would grow harder to reason about with time.

This PR moves them to `archive/` with full git rename history so they are
permanently recoverable with a single `git mv`.

---

## Archived folders

| Folder | Tracked size | Last commit | Justification |
| ------ | ------------ | ----------- | ------------- |
| `frontend/` | 141,506 B (tracked; node_modules gitignored) | 2026-05-07 (`8f60009`) | Next.js / React app. Zero Python imports. No CI job runs it. Production tool is Streamlit-only. |
| `intelligence/` | 356,461 B | 2026-05-07 (`8f60009`) | Self-contained LLM/news module. Zero external Python callers. Zero test references. |
| `log_manager/` | 9,719 B | 2026-05-07 (`8f60009`) | Only depended on by `intelligence/` (also archived). No independent callers. |
| `infra/` | 4,499 B | 2026-05-07 (`8f60009`) | Single GCF deploy shell script (`gcf_deploy.sh`). Referenced in docs prose only — no CI execution, no imports. |

---

## Validation methodology

1. **Static import scan** — `grep`/`rg` across all `*.py` files (`.venv/` and `archive/` excluded):
   - `from intelligence`, `import intelligence` → 0 hits outside archive
   - `from log_manager`, `import log_manager` → 0 hits outside archive
2. **CI workflow scan** — `*.yml` files searched for each folder name → 0 hits
3. **Test coverage check** — `tests/` directory grep for each candidate → 0 hits
4. **Audit report** — `prune/candidates-20260509.json` (machine-readable, with risk ratings)
5. **Dynamic validation** — full pytest run post-archive

**Test baseline:** `539 passed in 91.75s` (all unit tests; live-network screener tests excluded)

---

## What was NOT touched

All live production folders confirmed kept with active references:

| Folder | Kept because |
| ------ | ------------ |
| `regime_trader/` | Primary Python package |
| `backend/` | FastAPI application entrypoint |
| `pages/` | Streamlit multi-page app |
| `cloud/` | `tests/test_gcf_scheduler.py` imports it |
| `config/` | Referenced by `.github/workflows/canary.yml` |
| `hmm_engine/` | Imported by `backend/routers/regime.py` |
| `feature_engineering/` | Imported by `pages/5_Regime_Prediction.py` |
| `valuation/` | Imported by `backend/main.py` (26 refs) |
| `analysis/`, `core/`, `data/` | 100–177 refs each |

---

## How to restore a folder from archive

```bash
# Restore one folder
git mv archive/prune-chore-prune-unused-folders-20260509/frontend frontend
git commit -m "restore: frontend from archive"

# Restore all archived folders at once
for dir in frontend infra intelligence log_manager; do
  git mv "archive/prune-chore-prune-unused-folders-20260509/$dir" "$dir"
done
git commit -m "restore: all pruned folders from archive"
```

Alternatively, revert the archive commit entirely:

```bash
git revert 1e08acf96e6a3ab45a1aa8b0018b9c9497c41ed3
```

---

## Running scripts/verify_prune.sh

```bash
bash scripts/verify_prune.sh
```

The script performs all six checks (absence, archive presence, static scan,
CI-file scan, `check_imports.py`, `pytest`) and prints a JSON summary:

```json
{
  "archived_folders": ["frontend", "infra", "intelligence", "log_manager"],
  "archive_path": "archive/prune-chore-prune-unused-folders-20260509",
  "static_refs_found": 0,
  "imports_ok": true,
  "tests_ok": true,
  "checks_passed": 14,
  "checks_failed": 0,
  "timestamp": "2026-05-09T15:57:16Z"
}
```

Exit code 0 = clean. Exit code 1 = at least one check failed.

---

## Also included in this branch

| Area | Change |
| ---- | ------ |
| FMP API migration | `fmp_screener`, `fmp_insider_buys`, `fmp_institutional_accumulation` → yfinance (FMP v3/v4 retired Aug 2025) |
| Streamlit hardening | Shim calls `main()` explicitly; `width="stretch"` (Streamlit 1.57); thread-pool timeouts; `_safe_payload()` crash guard |
| Logging security | `SecretMaskFilter` redacts live env-var values from log records; 15 new tests in `test_logging_cfg.py` |
| Test suite | `TestFmpScreener`: 4 FMP-API tests rewritten for yfinance mock pattern |
