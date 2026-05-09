# PR: Prune unused folders (2026-05-09)

## Branch
`chore/prune-unused-folders-20260509` → `main`

## Summary

Removes four top-level folders that have **zero Python runtime callers** and are not referenced by any CI pipeline or test suite. All content is preserved in `archive/prune-chore-prune-unused-folders-20260509/` with full git rename history.

| Folder | Size | Reason archived |
|--------|------|-----------------|
| `frontend/` | 326 MB | Next.js / React app, 0 Python imports, 0 CI/test refs |
| `intelligence/` | 0.3 MB | Self-contained LLM/news module, 0 external callers |
| `log_manager/` | 0.01 MB | Only depended on by `intelligence/` (archived) |
| `infra/` | 0.01 MB | Single GCF deploy script, docs-only references |

**Total recovered from tracked content:** ~327 MB

## What was checked

1. **Static analysis** — `grep`/`ast` scan of all `*.py` files (`.venv/` excluded)
2. **CI workflow scan** — `*.yml`, `*.sh` files searched for folder names
3. **Test coverage check** — `tests/` directory grep for each candidate
4. **Audit report** — `prune/candidates-20260509.json`

## What was NOT touched

All live production folders are confirmed kept:

- `regime_trader/` — primary Python package
- `backend/` — FastAPI application  
- `pages/` — Streamlit multi-page app
- `cloud/` — active CI tests (`tests/test_gcf_scheduler.py`)
- `config/` — referenced by `.github/workflows/canary.yml`
- `hmm_engine/` — imported by `backend/routers/regime.py`
- `feature_engineering/` — imported by `pages/5_Regime_Prediction.py`
- `valuation/` — imported by `backend/main.py`
- `analysis/`, `core/`, `data/` — 100+ references each

## Also included in this branch

### FMP API migration (endpoints retired Aug 2025)
- `fmp_screener()` → yfinance batch OHLCV on curated 130-ticker watchlist
- `fmp_insider_buys()` → yfinance `insider_transactions` with exec-role filter
- `fmp_institutional_accumulation()` → yfinance `institutional_holders`
- `fmp_profile_batch()` → FMP stable API (`stable/profile`) parallel calls

### Streamlit hardening
- Shim `streamlit_app.py` calls `main()` explicitly (fixes blank re-runs)
- All `use_container_width=True` → `width="stretch"` (Streamlit 1.57)
- `ThreadPoolExecutor` loops use `as_completed(timeout=…)` with `FutureTimeoutError`
- `_safe_payload()` fallback prevents UI crash on scanner failure

### Logging security
- `SecretMaskFilter` redacts live env-var values from `record.msg` and `record.args`
- `configure_logging(mask_env=True)` is the default; no handler stacking on repeat calls
- `tests/test_logging_cfg.py`: 15 tests

### Test suite
- `TestFmpScreener`: 4 old FMP-API tests → yfinance mock pattern (pass rate: 4/4)

## Rollback instructions

The archived content is recoverable at any time:

```bash
# Restore a single folder
git mv archive/prune-chore-prune-unused-folders-20260509/frontend frontend
git commit -m "restore: frontend from archive"

# Restore all archived folders
for dir in frontend infra intelligence log_manager; do
  git mv "archive/prune-chore-prune-unused-folders-20260509/$dir" "$dir"
done
git commit -m "restore: all pruned folders from archive"
```

## Test results (post-prune)

```
539 passed in 91.75s
```

Run `bash scripts/verify_prune.sh` to reproduce the post-prune validation.
