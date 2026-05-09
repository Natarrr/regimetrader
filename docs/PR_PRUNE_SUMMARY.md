# Prune PR — reviewer one-pager (2026-05-09)

## What changed

- **4 unused top-level folders moved to `archive/prune-chore-prune-unused-folders-20260509/`**
  — `frontend/` (326 MB Next.js app), `intelligence/`, `log_manager/`, `infra/`
- All content preserved in git with full rename history; recoverable with one `git mv`
- No production code deleted; no schemas, interfaces, or CI pipelines modified

## Why each folder was archived

- **`frontend/`** — Next.js / React app with zero Python imports and no CI job that builds or tests it. The live tool runs on Streamlit, not this frontend.
- **`intelligence/`** — Self-contained LLM/news module. Static scan found 0 external Python callers and 0 test references.
- **`log_manager/`** — Only caller was `intelligence/` (also archived). No independent dependents.
- **`infra/`** — One GCF deploy shell script mentioned in docs prose; never executed by CI.

## Evidence

- Static import scan: `from intelligence`, `import intelligence`, `from log_manager`, `import log_manager` → **0 hits** outside archive
- CI workflow grep for each folder name → **0 hits**
- `pytest` post-prune: **539/539 passed**
- `python scripts/check_imports.py` → exit 0

## Verification

```bash
bash scripts/verify_prune.sh   # runs all 6 checks + prints JSON summary
```

## Rollback (one command per folder)

```bash
git mv archive/prune-chore-prune-unused-folders-20260509/frontend   frontend   && git commit -m "restore: frontend"
git mv archive/prune-chore-prune-unused-folders-20260509/intelligence intelligence && git commit -m "restore: intelligence"
git mv archive/prune-chore-prune-unused-folders-20260509/log_manager log_manager && git commit -m "restore: log_manager"
git mv archive/prune-chore-prune-unused-folders-20260509/infra       infra       && git commit -m "restore: infra"
```

## Audit artifacts

- `prune/candidates-20260509.json` — machine-readable report with risk rating per folder
- `docs/PR_PRUNE.md` — full rationale, restore commands, and verification guide
- `scripts/verify_prune.sh` — reproducible post-prune validation script
