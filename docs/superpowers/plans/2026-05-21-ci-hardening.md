# CI Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply four targeted CI reliability improvements (ruff lint gate, pytest-timeout, coverage gate, edgar_3x hardening) without introducing new infrastructure or rewriting tests.

**Architecture:** Pure workflow + config delta. Four commits, each touching a specific layer: (1) fix ruff violations in source, (2) add packages to requirements, (3) harden ci.yml, (4) harden edgar_3x.yml.

**Tech Stack:** GitHub Actions, pytest, pytest-timeout, pytest-cov, ruff, bash `timeout`

---

## File Map

| File | Change |
|------|--------|
| `backend/data/schemas.py` | Remove unused `Optional` import |
| `backend/engine_worker.py` | Add `# noqa: E402` to 3 intentional post-path imports |
| `backend/quant_models/monetary_pulse.py` | Remove unused `numpy`, convert docstring strings to raw strings |
| `backend/quant_models/volatility_brain.py` | Remove unused `pandas` import |
| `backend/tests/quant_models/test_monetary_pulse.py` | Remove unused `pytest`, fix 2 unused `spread` vars |
| `backend/tests/quant_models/test_prediction_controller.py` | Remove unused `pytest` |
| `backend/tests/quant_models/test_valuation_radar.py` | Remove unused `pytest`, `numpy` |
| `backend/tests/quant_models/test_volatility_brain.py` | Remove unused `pytest` |
| `regime_trader/models/regime_detector.py` | Remove unused `dataclass`, `field`; remove unused `n` var |
| `regime_trader/scanners/discovery_scanner.py` | Remove unused `timedelta`, `urljoin` |
| `regime_trader/scanners/market_intel_macro.py` | Remove unused `brent` var |
| `regime_trader/tools/backtest.py` | Remove 6 unused imports |
| `regime_trader/utils/io.py` | Remove unused `Optional` |
| `scripts/audit/audit_structure.py` | Rename ambiguous `l` → `line` |
| `scripts/backtest_signals.py` | Remove unused `asdict` |
| `scripts/run_pipeline.py` | Add `# noqa: E402` to 2 intentional post-path imports |
| `scripts/send_toplists_discord.py` | Remove unused `math` import |
| `requirements-ci.txt` | Add `pytest-timeout>=2.3.0`, `pytest-cov>=5.0.0`; pin `anthropic` upper bound |
| `requirements.txt` | Pin `anthropic` upper bound |
| `.github/workflows/ci.yml` | Ruff lint step in sanity; `--timeout=60` on all pytest calls; `--cov` on full test run |
| `.github/workflows/edgar_3x.yml` | `timeout 55m` wrapper; validate top_lists.json step; try/except in summary |

---

### Task 1: Fix all ruff violations

**Files:** all source files listed in file map above (17 files)

**Context:** Running `ruff check scripts/ regime_trader/ backend/ monitoring/ --select E,F,W --ignore E501` currently reports 40 errors. 28 are auto-fixable with `--fix`. The remaining 12 require manual fixes:
- `backend/engine_worker.py:47,51,56` — E402: intentional imports after `sys.path.insert`. Suppress with `# noqa: E402`, do NOT move them.
- `scripts/run_pipeline.py:37,38` — E402: same pattern. Suppress with `# noqa: E402`.
- `scripts/audit/audit_structure.py:122` — E741: ambiguous variable `l`. Rename to `line`.
- `backend/tests/quant_models/test_monetary_pulse.py:43,103` — F841: unused `spread`. Delete the assignments.
- `regime_trader/models/regime_detector.py:193` — F841: unused `n`. Delete the assignment.
- `regime_trader/scanners/market_intel_macro.py:596` — F841: unused `brent`. Delete the assignment.

- [ ] **Step 1: Run auto-fix**

```bash
cd "c:\Users\ntard\Projects\Trading dashboard\regime_trader"
python -m ruff check scripts/ regime_trader/ backend/ monitoring/ --select E,F,W --ignore E501 --fix
```

Expected: 28 files auto-fixed, 12 errors remain (the E402, E741, and F841 ones listed above).

- [ ] **Step 2: Fix E402 — engine_worker.py (3 imports after sys.path.insert)**

Read `backend/engine_worker.py` lines 47–56. Add `# noqa: E402` to the end of the first line of each multi-line import block and to the single-line import:

```python
from regime_trader.scanners.discovery_scanner import (  # noqa: E402
    get_top_alpha_picks_sync,
    force_refresh_sync,
)
from regime_trader.scanners.market_intel_macro import (  # noqa: E402
    COMMODITY_UNIVERSE,
    fetch_commodity_prices,
    calc_macro_conviction,
)
from regime_trader.utils.logging_cfg import configure_logging  # noqa: E402
```

- [ ] **Step 3: Fix E402 — run_pipeline.py (2 imports after sys.path.insert)**

Read `scripts/run_pipeline.py` lines 35–40. The two lines at 37 and 38 are the first import lines after the `sys.path.insert` call. Add `# noqa: E402` to each:

```python
from regime_trader.pipeline.orchestrator import run_full_pipeline  # noqa: E402
from regime_trader.utils.logging_cfg import configure_logging  # noqa: E402
```

(The exact import names — read the file to confirm before editing.)

- [ ] **Step 4: Fix E741 — audit_structure.py line 122**

Read `scripts/audit/audit_structure.py` around line 122. Find the variable named `l` and rename it to `line` in both the assignment and all usages within that scope.

- [ ] **Step 5: Fix F841 — delete 4 unused variable assignments**

For each file, read the lines indicated, then delete the assignment line entirely:

- `backend/tests/quant_models/test_monetary_pulse.py:43` — `spread = ...`
- `backend/tests/quant_models/test_monetary_pulse.py:103` — `spread = ...`
- `regime_trader/models/regime_detector.py:193` — `n = ...`
- `regime_trader/scanners/market_intel_macro.py:596` — `brent = ...`

- [ ] **Step 6: Verify zero violations remain**

```bash
cd "c:\Users\ntard\Projects\Trading dashboard\regime_trader"
python -m ruff check scripts/ regime_trader/ backend/ monitoring/ --select E,F,W --ignore E501
```

Expected output: no errors (exit code 0). If any remain, fix them before proceeding.

- [ ] **Step 7: Run smoke tests to confirm no regressions**

```bash
cd "c:\Users\ntard\Projects\Trading dashboard\regime_trader"
pytest tests/test_cross_sectional.py tests/test_quiver_client.py tests/test_normalize.py -q --tb=short -m "not slow"
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add backend/data/schemas.py backend/engine_worker.py backend/quant_models/monetary_pulse.py backend/quant_models/volatility_brain.py backend/tests/quant_models/test_monetary_pulse.py backend/tests/quant_models/test_prediction_controller.py backend/tests/quant_models/test_valuation_radar.py backend/tests/quant_models/test_volatility_brain.py regime_trader/models/regime_detector.py regime_trader/scanners/discovery_scanner.py regime_trader/scanners/market_intel_macro.py regime_trader/tools/backtest.py regime_trader/utils/io.py scripts/audit/audit_structure.py scripts/backtest_signals.py scripts/run_pipeline.py scripts/send_toplists_discord.py
git commit -m "$(cat <<'EOF'
fix(lint): resolve all 40 ruff violations (E/F/W rules)

Auto-fixed 28 with ruff --fix; manually suppressed 2 E402 groups
with noqa (intentional sys.path-first imports); renamed ambiguous l→line;
deleted 4 unused variable assignments.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Add packages to requirements files

**Files:** `requirements-ci.txt`, `requirements.txt`

- [ ] **Step 1: Add pytest-timeout and pytest-cov to requirements-ci.txt**

Read `requirements-ci.txt`. Under the `# ── Test framework ────` section, add two lines after `pytest-asyncio>=0.23.0`:

```text
pytest-timeout>=2.3.0
pytest-cov>=5.0.0
```

Also change:
```text
anthropic>=0.28.0
```
to:
```text
anthropic>=0.28.0,<2.0.0
```

- [ ] **Step 2: Pin anthropic in requirements.txt**

Read `requirements.txt`. Find `anthropic>=0.28.0` and change it to:
```text
anthropic>=0.28.0,<2.0.0
```

- [ ] **Step 3: Verify pip can install the updated requirements-ci.txt**

```bash
pip install -r requirements-ci.txt --dry-run 2>&1 | tail -5
```

Expected: no errors. If anthropic version conflict appears, the existing installed version is already within range — that's fine.

- [ ] **Step 4: Commit**

```bash
git add requirements-ci.txt requirements.txt
git commit -m "$(cat <<'EOF'
chore(deps): add pytest-timeout + pytest-cov; pin anthropic<2.0.0

Prevents hanging tests from silently consuming the job timeout.
Enables coverage gate in CI. anthropic upper bound guards against
breaking API changes on a future major-version release.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Harden ci.yml (ruff step + timeout + coverage)

**Files:** `.github/workflows/ci.yml`

**Context:** The workflow has 3 jobs: `sanity`, `smoke`, `test`. Changes needed:
1. New ruff lint step in `sanity` job, after "Sanity import check"
2. `--timeout=60` added to all 4 pytest invocations (2 in smoke, 2 in test)
3. `--cov` flags added to the full `tests/` step in the `test` job

- [ ] **Step 1: Read the current ci.yml**

Read `.github/workflows/ci.yml` in full to confirm exact line positions before editing.

- [ ] **Step 2: Add ruff lint step to sanity job**

After the `Sanity import check` step and before `Secrets presence check`, insert this new step:

```yaml
      - name: Lint (ruff)
        run: |
          pip install ruff>=0.4.0
          ruff check scripts/ regime_trader/ backend/ monitoring/ \
            --select E,F,W --ignore E501
```

- [ ] **Step 3: Add --timeout=60 to smoke step 1**

Find the first pytest invocation in the smoke job:
```yaml
          pytest tests/test_streamlit_app_smoke.py \
                 tests/test_logging_cfg.py \
                 tests/test_check_secrets.py \
                 tests/test_fmt_insider_pct.py \
                 tests/test_ci_environment.py \
                 -q --tb=short -m "not slow"
```
Change to:
```yaml
          pytest tests/test_streamlit_app_smoke.py \
                 tests/test_logging_cfg.py \
                 tests/test_check_secrets.py \
                 tests/test_fmt_insider_pct.py \
                 tests/test_ci_environment.py \
                 -q --tb=short -m "not slow" --timeout=60
```

- [ ] **Step 4: Add --timeout=60 to smoke step 2**

Find the second pytest invocation in the smoke job:
```yaml
          pytest tests/test_token_bucket.py \
                 tests/test_normalize.py \
                 tests/test_backtest.py \
                 tests/test_quiver_client.py \
                 tests/test_congress_fetcher.py \
                 tests/test_cross_sectional.py \
                 -q --tb=short -m "not slow"
```
Change to:
```yaml
          pytest tests/test_token_bucket.py \
                 tests/test_normalize.py \
                 tests/test_backtest.py \
                 tests/test_quiver_client.py \
                 tests/test_congress_fetcher.py \
                 tests/test_cross_sectional.py \
                 -q --tb=short -m "not slow" --timeout=60
```

- [ ] **Step 5: Add --timeout=60 and --cov flags to full test suite step**

Find the `Run test suite — tests/` step:
```yaml
          pytest tests/ -q --tb=short -m "not slow"
```
Change to:
```yaml
          pytest tests/ -q --tb=short -m "not slow" --timeout=60 \
            --cov=regime_trader --cov=scripts --cov=backend --cov=monitoring \
            --cov-report=term-missing --cov-fail-under=60
```

- [ ] **Step 6: Add --timeout=60 to backend/tests/ conditional step**

Find:
```yaml
            pytest backend/tests/ -q --tb=short -m "not slow"
```
Change to:
```yaml
            pytest backend/tests/ -q --tb=short -m "not slow" --timeout=60
```

- [ ] **Step 7: Verify YAML syntax**

```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))" && echo "YAML OK"
```

Expected: `YAML OK`

- [ ] **Step 8: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "$(cat <<'EOF'
ci: add ruff lint gate, --timeout=60, coverage gate at 60%

Ruff step in sanity catches undefined names before tests run.
--timeout=60 prevents a hanging test from silently consuming the job.
--cov-fail-under=60 gates on coverage floor; threshold starts low
and will be ratcheted up once baseline is measured.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Harden edgar_3x.yml (subprocess timeout + validate gate + summary safety)

**Files:** `.github/workflows/edgar_3x.yml`

**Context:** Three independent changes to the `fetch-and-rank` job:
1. Wrap `python scripts/run_pipeline.py` in `timeout 55m` inside the retry loop
2. New step between "Generate top lists" (step 6) and "Generate satellite insights" (step 7): validates `logs/top_lists.json` exists and has non-empty `top_buys`
3. Wrap the inline Python in "Generate Run Execution Summary" (step 9) in try/except

- [ ] **Step 1: Read edgar_3x.yml around the retry loop (lines 85–110)**

Confirm the exact indentation and `if python scripts/run_pipeline.py` block before editing.

- [ ] **Step 2: Add timeout 55m wrapper to run_pipeline.py call**

Inside the retry loop, change:
```yaml
            if python scripts/run_pipeline.py \
              --tickers-file "$TICKERS_FILE" \
              --max-workers  4 \
              --log-dir      "$LOG_DIR" \
              --verbose; then
```
to:
```yaml
            if timeout 55m python scripts/run_pipeline.py \
              --tickers-file "$TICKERS_FILE" \
              --max-workers  4 \
              --log-dir      "$LOG_DIR" \
              --verbose; then
```

- [ ] **Step 3: Add validate top_lists.json step after "Generate top lists"**

After the `Generate top lists` step (step 6, the one running `python -m backend.market_intel.generate_top_lists`) and before `Generate satellite insights` (step 7), insert:

```yaml
      # ── 6b. Gate — validate top_lists.json before Discord fires ─────────────
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

- [ ] **Step 4: Wrap inline summary Python in try/except**

In the `Generate Run Execution Summary` step, find the heredoc Python block:
```python
          import json
          with open("logs/top_lists.json") as f:
              d = json.load(f)
          for i, t in enumerate(d.get('top_buys', [])[:5], 1):
              f = t.get('factors', {})
              print(f"| {i} | **{t['ticker']}** | {t['final_score']:.4f} | {t['badge']} | "
                    f"{f.get('edgar', 0):.2f} | {f.get('insider', 0):.2f} | "
                    f"{f.get('congress', 0):.2f} | {f.get('news', 0):.2f} | {f.get('momentum', 0):.2f} |")
```

Replace with:
```python
          import json
          try:
              with open("logs/top_lists.json") as f:
                  d = json.load(f)
              for i, t in enumerate(d.get('top_buys', [])[:5], 1):
                  fac = t.get('factors', {})
                  print(f"| {i} | **{t['ticker']}** | {t['final_score']:.4f} | {t['badge']} | "
                        f"{fac.get('edgar', 0):.2f} | {fac.get('insider', 0):.2f} | "
                        f"{fac.get('congress', 0):.2f} | {fac.get('news', 0):.2f} | {fac.get('momentum', 0):.2f} |")
          except Exception as e:
              print(f"| — | _summary unavailable: {e}_ | | | | | | | |")
```

Note: the inner loop variable was renamed from `f` to `fac` to avoid shadowing the file handle `f`.

- [ ] **Step 5: Verify YAML syntax**

```bash
python -c "import yaml; yaml.safe_load(open('.github/workflows/edgar_3x.yml'))" && echo "YAML OK"
```

Expected: `YAML OK`

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/edgar_3x.yml
git commit -m "$(cat <<'EOF'
ci(edgar_3x): subprocess timeout, validate gate, summary safety

timeout 55m fires before the 60-min job wall, allowing retry loop
and artifact upload to complete. Validate step blocks Discord if
top_buys is empty. Summary try/except prevents malformed JSON from
failing the always() step.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- Section 1a (pytest-timeout, pytest-cov packages) → Task 2 ✓
- Section 1b (--timeout=60 on all pytest calls) → Task 3 steps 3–6 ✓
- Section 1c (coverage gate on full test run only) → Task 3 step 5 ✓
- Section 2 (ruff lint step in sanity, placement, rule selection) → Task 3 step 2; Task 1 clears violations first ✓
- Section 3a (timeout 55m wrapper) → Task 4 step 2 ✓
- Section 3b (validate top_lists.json gate) → Task 4 step 3 ✓
- Section 3c (try/except in summary) → Task 4 step 4 ✓
- Section 4 (anthropic pin) → Task 2 steps 1–2 ✓

**Placeholder scan:** None found. All code blocks are complete and exact.

**Type consistency:** No shared types across tasks.

**Order rationale:** Task 1 (fix violations) must precede Task 3 (add ruff step) — if ruff step lands before violations are fixed, the ruff step itself would fail CI immediately. Tasks 2, 3, 4 are independent of each other but all depend on Task 1 being merged first.
