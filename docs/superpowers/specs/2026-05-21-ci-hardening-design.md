# CI Hardening — Option A Implementation Design

**Date:** 2026-05-21
**Author:** Nathan T (via Claude Code brainstorming)
**Status:** Approved

---

## Goal

Close the four reliability and safety gaps identified in the codebase audit without
introducing new infrastructure, rewriting tests, or forcing formatting changes.
Every change is a pure workflow or config delta. The full diff touches four files:
`requirements-ci.txt`, `requirements.txt`, `.github/workflows/ci.yml`, and
`.github/workflows/edgar_3x.yml`.

---

## Architecture

No new services, no new Python modules. All changes live in:

| File | Change type |
|------|-------------|
| `requirements-ci.txt` | Add `pytest-timeout`, `pytest-cov`; pin `anthropic` |
| `requirements.txt` | Pin `anthropic` upper bound |
| `.github/workflows/ci.yml` | Add `--timeout`, `--cov`, ruff lint step |
| `.github/workflows/edgar_3x.yml` | Subprocess timeout, output gate, summary safety |

---

## Section 1 — pytest hardening

### 1a. New packages in `requirements-ci.txt`

```text
pytest-timeout>=2.3.0
pytest-cov>=5.0.0
```

`pytest-timeout` kills any test that exceeds its per-test time limit, preventing a
leaked network call or blocking I/O from consuming the entire job timeout silently.

`pytest-cov` measures source coverage during the full test run and can fail the job
when coverage drops below a threshold, preventing silent regression of test quality.

### 1b. `--timeout=60` on every pytest invocation

Applied to: both smoke steps, the full `tests/` step, and the `backend/tests/`
conditional step. All four pytest invocations get `--timeout=60`.

Before:

```yaml
pytest tests/test_token_bucket.py ... -q --tb=short -m "not slow"
```

After:

```yaml
pytest tests/test_token_bucket.py ... -q --tb=short -m "not slow" --timeout=60
```

### 1c. Coverage gate on full test run only

The full `tests/` step gains:

```text
--cov=regime_trader --cov=scripts --cov=backend --cov=monitoring \
--cov-report=term-missing --cov-fail-under=60
```

Threshold is **60%** on first landing. Rationale: starting at 70% risks immediately
breaking CI if actual coverage is 62%. We measure actual coverage from the first
passing run, then ratchet the threshold upward in a follow-up commit.

Coverage report is printed to the job log (`term-missing`) — no HTML artifact needed
until threshold is ≥80% and the report becomes actionable.

---

## Section 2 — Ruff lint step

### Placement

New step in the `sanity` job, immediately after `Sanity import check`. Lint runs
before any test job begins. An `F821` (undefined name) or `F401` (unused import)
caught here saves the full smoke + test runtime.

### Installation

`ruff` is installed inline in the step (`pip install ruff>=0.4.0`), not added to
`requirements-ci.txt`. Rationale: lint tooling is not a test dependency; keeping it
out of the requirements file means it never accidentally gets imported in tests.

### Rule selection

```bash
ruff check scripts/ regime_trader/ backend/ monitoring/ \
  --select E,F,W --ignore E501
```

| Ruleset | What it catches |
|---------|----------------|
| `E` | pycodestyle errors (syntax-adjacent issues) |
| `F` | pyflakes — undefined names, unused imports, redefined variables |
| `W` | pycodestyle warnings |
| `E501` | **excluded** — line-length; codebase has intentional long lines in financial formulas |

No `ruff.toml` or `pyproject.toml` config file is created. The inline flags are the
single source of truth, visible directly in the workflow step.

---

## Section 3 — `edgar_3x.yml` hardening

### 3a. Subprocess timeout on `run_pipeline.py`

The EDGAR fetch step wraps the Python call in `timeout 55m`:

```bash
if timeout 55m python scripts/run_pipeline.py \
  --tickers-file "$TICKERS_FILE" \
  --max-workers  4 \
  --log-dir      "$LOG_DIR" \
  --verbose; then
```

The job-level `timeout-minutes: 60` is the hard wall. `timeout 55m` fires 5 minutes
before the wall, allowing the retry loop to log a failure, the metrics exporter to
run, and artifacts to upload. Without this, a hung ticker fetch silently consumes
the entire 60 minutes and produces zero artifacts.

`timeout` exits with code 124 on expiry. The `if` condition treats 124 as failure,
which the existing retry loop handles correctly (increments ATTEMPT, sleeps, retries).

### 3b. Hard gate — validate `top_lists.json` before Discord fires

New step inserted after `Generate top lists`, before `Generate satellite insights`:

```yaml
- name: Validate top_lists.json
  run: |
    python3 - <<'EOF'
    import json, sys
    from pathlib import Path
    p = Path("logs/top_lists.json")
    if not p.exists():
        print("::error::top_lists.json was not written — pipeline failed silently")
        sys.exit(1)
    d = json.loads(p.read_text())
    if not d.get("top_buys"):
        print("::error::top_lists.json exists but top_buys is empty")
        sys.exit(1)
    count = d.get("ticker_count", 0)
    print(f"Validated: {len(d['top_buys'])} top buys, {count} tickers scored")
    EOF
```

This step is not `if: always()` — it blocks the pipeline. If top_buys is empty,
the Discord message would contain nothing useful. Failing here gives a clear signal
to investigate before the Discord step runs.

### 3c. Wrap inline summary Python in try/except

The `Generate Run Execution Summary` step currently crashes if `top_lists.json`
contains malformed JSON (e.g. truncated by disk-full). This is a summary step — it
should never fail the job. Wrap the `json.load` in a broad try/except:

```python
try:
    with open("logs/top_lists.json") as f:
        d = json.load(f)
    for i, t in enumerate(d.get('top_buys', [])[:5], 1):
        f = t.get('factors', {})
        print(f"| {i} | **{t['ticker']}** | {t['final_score']:.4f} | ...")
except Exception as e:
    print(f"| — | _summary unavailable: {e}_ | | | | | | | |")
```

---

## Section 4 — Anthropic SDK version pin

One-line change in both `requirements-ci.txt` and `requirements.txt`:

```text
# Before
anthropic>=0.28.0

# After
anthropic>=0.28.0,<2.0.0
```

The Anthropic Python SDK has historically introduced breaking API surface changes on
major version bumps. Without an upper bound, a future `pip install` could silently
pull `anthropic==2.0.0` and break the install with an obscure error. The `<2.0.0`
pin is safe: there is no `2.x` release as of this writing. When Anthropic ships v2,
we upgrade explicitly with a tested PR.

---

## Error Handling

| Scenario | Behaviour after this change |
|----------|-----------------------------|
| Test hangs (leaked network call) | `pytest-timeout` kills it after 60 s, test fails with `TIMEOUT` |
| Coverage drops below 60% | `--cov-fail-under=60` exits non-zero, full test job fails |
| Undefined name / unused import | `ruff` exits non-zero in sanity job, downstream jobs never run |
| `run_pipeline.py` hangs > 55 min | `timeout 55m` sends SIGTERM, retry loop logs and retries |
| `top_lists.json` missing or empty | Validate step exits 1, satellite/Discord steps never run |
| `top_lists.json` malformed in summary | try/except prints placeholder row, summary exits 0 |
| `anthropic>=2.0.0` released | `pip install` rejects it; install fails with clear version conflict |

---

## Testing

No new test files are needed. The changes are all in workflow configuration and
dependency pins.

Manual validation after merge:

- Trigger `edgar_3x.yml` via workflow_dispatch — verify Validate step passes.
- Check CI run on this PR — verify ruff step runs, coverage number appears in logs.
- Confirm `--timeout=60` appears in smoke step logs.

---

## Files Changed

| File | Added | Changed | Removed |
| ---- | ----- | ------- | ------- |
| `requirements-ci.txt` | 3 lines | 1 line | 0 |
| `requirements.txt` | 0 | 1 line | 0 |
| `.github/workflows/ci.yml` | ~15 lines | ~6 lines | 0 |
| `.github/workflows/edgar_3x.yml` | ~20 lines | ~3 lines | 0 |

Total diff: approximately 45 lines added, 10 changed, 0 deleted.
