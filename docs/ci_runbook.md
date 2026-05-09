# CI Runbook — Regime Trader Hybrid

## Overview

The CI system has two layers:

| Workflow | File | Trigger | Purpose |
| -------- | ---- | ------- | ------- |
| **CI** | `ci.yml` | push / PR / dispatch | Full test gate (654 tests) |
| **Market Intel** | `market_intel.yml` | schedule + path filter | Data fetch + quant suite |
| **Hybrid Pipeline** | `hybrid_pipeline.yml` | schedule 08:30 ET | Quant → Claude → Gate |

```
push / PR
   │
   ▼
ci.yml ─── sanity job ────── 11 import checks (< 30 s)
              │
              ▼ (needs: sanity)
           test job ────────── pytest tests/ + backend/tests/
```

---

## Requirements Files

| File | Used by | Contains |
| ---- | ------- | -------- |
| `requirements.txt` | Production, canary, fetch jobs | Full stack (ML, dashboard, broker) |
| `requirements-ci.txt` | All CI test jobs | Test deps only — no dashboard/broker |

### Adding a new dependency

**Step 1** — add to `requirements.txt` (always):

```
my-package>=1.0.0   # purpose
```

**Step 2** — if any test file imports the package, also add to `requirements-ci.txt`:

```
my-package>=1.0.0
```

**Step 3** — add an import assertion in [tests/test_imports.py](../tests/test_imports.py):

```python
def test_my_package():
    """my-package: used by <module> for <purpose>."""
    import my_package  # noqa: F401
```

**Step 4** — run locally and confirm no regression:

```bash
pytest tests/test_imports.py tests/test_ci_environment.py -v
```

---

## Running Tests Locally

### Mirror CI exactly

```bash
# 1. Create a fresh venv (optional but recommended)
python -m venv .venv-ci
source .venv-ci/bin/activate   # Windows: .venv-ci\Scripts\activate

# 2. Install CI requirements only
pip install -r requirements-ci.txt

# 3. Run the full suite
pytest tests/ backend/tests/ -v --tb=short
```

### Quick run (uses your project venv)

```bash
pytest tests/ backend/tests/ -q
```

### Run a single test module

```bash
pytest tests/test_regime_detector.py -v
pytest backend/tests/quant_models/test_volatility_brain.py -v
```

---

## Diagnosing Import Errors

### Symptom: `ModuleNotFoundError: No module named 'X'` during collection

**Cause**: Package `X` is used in source or test code but missing from `requirements-ci.txt`.

**Fix**:
```bash
# 1. Find which package provides the module
pip show X 2>/dev/null || pip index versions X

# 2. Add to requirements-ci.txt
echo "X>=<version>" >> requirements-ci.txt

# 3. Add to test_imports.py
# 4. Commit both files together
```

### Symptom: `ModuleNotFoundError: No module named 'pydantic'`

Pydantic is used in `backend/data/schemas.py`, which is imported by all backend quant model
tests. It is now declared in `requirements-ci.txt`. If this error reappears, the file was
probably modified without the `pydantic` line.

**Quick check**:
```bash
grep pydantic requirements-ci.txt   # must return a line
python -c "import pydantic; print(pydantic.__version__)"
```

### Symptom: `ModuleNotFoundError: No module named 'requests'`

`requests` is in `requirements-ci.txt`. If it fails after a clean install, check for a
version conflict with another package (e.g., `urllib3`):

```bash
pip check   # reports dependency conflicts
```

### Symptom: Tests pass locally but fail in CI

1. Run locally with only CI requirements:

   ```bash
   pip install -r requirements-ci.txt
   pytest tests/ backend/tests/ -v
   ```

2. Check Python version (`python --version` must be ≥ 3.11).

3. Check `pytest.ini` is detected:

   ```bash
   pytest --co -q 2>&1 | head -5   # rootdir line should show project root
   ```

---

## Diagnosing Collection Errors

### Symptom: `ERROR collecting tests/test_X.py`

```bash
pytest tests/test_X.py --collect-only 2>&1
```

Common causes:
- Import error in the test file itself or a module it imports
- Syntax error in production code
- Missing `__init__.py` in a package directory

### Symptom: `No tests ran`

```bash
pytest tests/ --collect-only -q | head -20
```

If the list is empty, check `pytest.ini`:

```ini
[pytest]
pythonpath = .
testpaths = tests
```

---

## Environment Checks at Runtime

The test file [tests/test_ci_environment.py](../tests/test_ci_environment.py) enforces:

| Check | Assertion |
| ----- | --------- |
| Python version | `>= 3.11` |
| `pytest.ini` present | at repo root |
| `testpaths` declared | in `pytest.ini` |
| `backend/__init__.py` | exists |
| `backend/tests/__init__.py` | exists |
| `pydantic` in `requirements-ci.txt` | always |
| `anthropic` in `requirements-ci.txt` | always |
| `hmmlearn` in `requirements-ci.txt` | always |
| `ci.yml` exists | `.github/workflows/ci.yml` |

---

## Workflow Reference

### ci.yml — Full test gate

```
Trigger: push to main / any PR / workflow_dispatch
Jobs:
  sanity  → imports 11 packages, fails fast if env is broken
  test    → pytest tests/ + backend/tests/ (needs: sanity)
Timeout: 10 min (sanity) + 30 min (test)
```

### market_intel.yml — Data + quant tests

```
Trigger: schedule (3×/day) + path filters + dispatch
Jobs:
  test       → pip install -r requirements-ci.txt
               pytest backend/tests/market_intel/ tests/
  test-quant → pip install -r requirements-ci.txt
               pytest backend/tests/quant_models/
  fetch      → pip install -r requirements.txt (full stack)
               python -m backend.market_intel.run_pipeline
```

### hybrid_pipeline.yml — Trading pipeline

```
Trigger: schedule 08:30 ET weekdays + dispatch
Jobs:
  quant  → requirements-ci.txt + hmmlearn scikit-learn → EDGAR scoring
  claude → anthropic + requests → LLM analysis (gated by secret)
  gate   → schema validation + auto-exec gate summary
```

---

## Package × Test Matrix

| Package | Source module | Test file |
| ------- | ------------- | --------- |
| `pydantic` | `backend/data/schemas.py` | `backend/tests/quant_models/*` |
| `requests` | `backend/quant_models/valuation_radar.py` | `test_valuation_radar.py` |
| `hmmlearn` | `regime/regime_detector.py` | `test_regime_detector.py` |
| `scikit-learn` | `regime/regime_detector.py` | `test_regime_detector.py` |
| `scipy` | `backend/quant_models/volatility_brain.py` | `test_volatility_brain.py` |
| `statsmodels` | `backend/quant_models/monetary_pulse.py` | `test_monetary_pulse.py` |
| `arch` | `backend/quant_models/volatility_brain.py` | `test_volatility_brain.py` |
| `anthropic` | `analysis/claude_client.py` | `test_claude_client.py` |
| `yfinance` | `regime/credit_regime_detector.py` | `test_credit_regime_detector.py` |
