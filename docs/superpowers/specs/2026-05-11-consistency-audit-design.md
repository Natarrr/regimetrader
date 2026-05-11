# Consistency Audit Design
**Date:** 2026-05-11
**Scope:** Full project — structural deduplication, orphan/broken-import removal, dead-doc cleanup, naming/config uniformity
**Target:** 95% consistency across the `regime_trader` codebase

---

## Goals

1. **Structural deduplication (B):** Eliminate competing module pairs by auto-scoring each side and keeping the winner. Merge any unique symbols before deleting the loser.
2. **Dead-code and broken-import removal (C):** Delete `.py` files that are never imported and not entry points; fix or delete files with broken import references.
3. **Dead and duplicate doc removal (D):** Delete `.md` files where >50% of referenced code symbols no longer exist, and collapse pairs with >70% content overlap.
4. **Naming and config uniformity (D):** Enforce `snake_case` file names, consistent internal import style, and single-source-of-truth config keys.

---

## Phase 1 — Discovery (no deletions)

Produces `docs/superpowers/specs/audit-report.json` and `audit-report.md`. Three lists:

### 1a — Never-imported `.py` files
- Build full AST import graph across all `.py` files.
- Flag any file with **zero inbound edges** that is not an entry point.
- Entry points (excluded from flagging): `pages/*.py`, `scripts/*.py`, `cloud/**/*.py`, any top-level `app.py` / `main.py`.

### 1b — Broken-import `.py` files
- For every `import X` / `from X import Y` in each file, resolve the target path within the repo.
- Flag any file containing at least one import that resolves to a non-existent path.

### 1c — Dead and duplicate documentation
- Grep each `.md` for code symbols (`class Foo`, `def bar`, module paths).
- Flag docs where **>50% of referenced symbols** do not exist in the current codebase.
- Compute pairwise Jaccard similarity on headings + code blocks; flag pairs with **>70% overlap**.

---

## Phase 2 — Structural Overlap Auto-Scoring

Seven competing module pairs are audited in order:

| # | Left | Right |
|---|------|-------|
| 1 | `backend/market_intel/` | `regime_trader/services/` |
| 2 | `regime/` | `regime_trader/` |
| 3 | `utils/` (root) | `regime_trader/utils/` |
| 4 | `utils/` (root) | `backend/utils/` |
| 5 | `backend/quant_models/` | `hmm_engine/` |
| 6 | `analysis/` (root) | `feature_engineering/` (root) |
| 7 | `data/` (root) | `backend/data/` |

### Scoring criteria (per side)

| Criterion | Weight | Measurement |
|-----------|--------|-------------|
| Inbound import count | 30% | Files that `import` from this module |
| Test coverage | 25% | Tests in `pytest.ini` testpaths referencing this module |
| Implementation completeness | 25% | Count of non-empty functions/classes |
| Recency | 10% | Latest `git log` modification date |
| CLAUDE.md compliance | 10% | % of functions with Nobel-laureate docstrings |

**Decision rule:** Higher total score → **KEEP**. Lower score → **DELETE**, with a safety check: any function/class in the DELETE side not present in the KEEP side is flagged as a **unique symbol** requiring manual merge before deletion.

Results are written to `docs/superpowers/specs/2026-05-11-consistency-audit-decisions.md`.

---

## Phase 3 — Execution (three commits)

### Commit 1 — Structural cleanup
- Merge any flagged unique symbols into the KEEP module.
- Delete the losing module directory.
- Update all inbound imports across the codebase to point to the winning path.

### Commit 2 — Orphan and dead-doc cleanup
- Delete never-imported `.py` files identified in Phase 1a.
- Delete or fix broken-import `.py` files from Phase 1b (fix if the correct target is unambiguous).
- Delete dead and duplicate `.md` files from Phase 1c.

### Commit 3 — Naming and config uniformity
- Rename any non-`snake_case` `.py` module files.
- Standardise internal imports: `from x import y` over bare `import x.y.z`; remove `sys.path` hacks.
- Audit config keys duplicated across `pytest.ini`, `.streamlit/config.toml`, `.vscode/settings.json`, `requirements.txt`, `requirements-ci.txt` — flag duplicates and consolidate.

---

## Validation (after Commit 3)

```bash
# Full test suite must pass
python -m pytest tests/ backend/tests/ -q --tb=short

# Smoke imports must succeed
python -c "import regime_trader; import hmm_engine; import monitoring; print('OK')"
```

Both must pass before the audit is considered complete.

---

## Out of scope
- CLAUDE.md compliance (Nobel docstrings, LaTeX comments) — separate concern, not part of this audit.
- Adding new tests or features.
- Changing business logic in any module.
