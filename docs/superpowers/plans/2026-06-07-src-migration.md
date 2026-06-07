# src/ Architecture Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize 10 source files from `backend/`, `regime_trader/fetchers/`, and `scripts/` into a unified `src/` workflow-aligned package while keeping every pytest passing and zero business logic altered.

**Architecture:** Files migrate strictly in dependency order (core → engine → ingestion → risk → delivery). Each move is immediately followed by a global import sweep and a pytest green-bar check before committing. The old file is deleted only after all consumers reference the new path and tests pass.

**Tech Stack:** Python 3.11, pytest, GitHub Actions YAML (8 workflow files)

---

## File Map: Before → After

| Old path | New path | Notes |
|---|---|---|
| `regime_trader/fetchers/base.py` | `src/core/fetchers_base.py` | ABC + dataclasses |
| `backend/market_intel/engine.py` | `src/engine/engine.py` | StrategyEngine |
| `scripts/fmp_bulk_prefetch.py` | `src/ingestion/fmp_bulk_prefetch.py` | bulk cache |
| `scripts/run_pipeline.py` | `src/ingestion/run_pipeline.py` | EDGAR+FMP entrypoint |
| `regime_trader/fetchers/fmp_fetcher.py` | `src/ingestion/fmp_fetcher.py` | EU/Asia fetcher |
| `scripts/run_pipeline_profile.py` | `src/engine/profile_runner.py` | INTL StrategyEngine runner |
| `regime_trader/risk/regime.py` | `src/risk/regime.py` | RiskRegime SSOT |
| `scripts/cook_toplists.py` | `src/delivery/cook_toplists.py` | US+INTL merge |
| `scripts/audit_payload.py` | `src/delivery/audit_payload.py` | RT-QA-2026-REV5 |
| `scripts/send_toplists_discord.py` | `src/delivery/send_discord.py` | Discord notifier |

**Files that stay (NOT migrated):** `regime_trader/services/`, `regime_trader/scoring/`, `regime_trader/utils/`, `regime_trader/config/`, `backend/market_intel/generate_top_lists.py`, `backend/market_intel/portfolio_optimizer.py`, `backend/market_intel/satellite_factors.py`, `backend/market_intel/validator.py`, `monitoring/`, `analysis/`, `regime_trader/fetchers/orchestrator.py`, `regime_trader/risk/exit_rules.py`, `scripts/backtest_signals.py`, `scripts/check_imports.py`, `scripts/check_secrets.py`, `scripts/mock_webhook_server.py`

---

## Task 0: Establish baseline

**Files:** none

- [ ] **Step 0.1: Record passing test count**

Run:
```bash
cd "c:/Users/ntard/Projects/Trading dashboard/regime_trader"
python -m pytest tests/ -q --tb=short -m "not slow" --timeout=60 2>&1 | tail -5
```
Expected: all tests green. Note the exact count for comparison after each phase.

---

## Task 1: Scaffold src/ package skeleton

**Files:**
- Create: `src/__init__.py`
- Create: `src/core/__init__.py`
- Create: `src/ingestion/__init__.py`
- Create: `src/engine/__init__.py`
- Create: `src/risk/__init__.py`
- Create: `src/delivery/__init__.py`

- [ ] **Step 1.1: Create directories and empty __init__.py files**

```bash
mkdir -p src/core src/ingestion src/engine src/risk src/delivery
```

Then write each `__init__.py` with empty content (just a newline):

`src/__init__.py`:
```python
```

`src/core/__init__.py`:
```python
```

`src/ingestion/__init__.py`:
```python
```

`src/engine/__init__.py`:
```python
```

`src/risk/__init__.py`:
```python
```

`src/delivery/__init__.py`:
```python
```

- [ ] **Step 1.2: Verify tree**

Run:
```bash
find src -name "__init__.py" | sort
```
Expected:
```
src/__init__.py
src/core/__init__.py
src/delivery/__init__.py
src/engine/__init__.py
src/ingestion/__init__.py
src/risk/__init__.py
```

- [ ] **Step 1.3: Run pytest (baseline still holds)**

```bash
python -m pytest tests/ -q --tb=short -m "not slow" --timeout=60 2>&1 | tail -5
```
Expected: same green count as Task 0.

- [ ] **Step 1.4: Commit**

```bash
git add src/
git commit -m "chore: scaffold src/ package skeleton with __init__.py anchors"
```

---

## Task 2: Move src/core/fetchers_base.py

Source: `regime_trader/fetchers/base.py`
No internal imports (stdlib only: `abc`, `dataclasses`, `enum`).

**Files:**
- Create: `src/core/fetchers_base.py` (copy of `regime_trader/fetchers/base.py`)
- Modify: `regime_trader/fetchers/__init__.py` (1 import line)
- Modify: `regime_trader/fetchers/orchestrator.py` (1 import line)
- Modify: `tests/test_fetchers.py` (2 import lines)
- Modify: `tests/test_global_scoring_v22.py` (1 import line)
- Modify: `tests/test_source_reliability.py` (1 import line)
- Delete: `regime_trader/fetchers/base.py`

- [ ] **Step 2.1: Copy file to new location**

```bash
cp regime_trader/fetchers/base.py src/core/fetchers_base.py
```

No import changes needed inside `src/core/fetchers_base.py` — it only uses stdlib.

- [ ] **Step 2.2: Update regime_trader/fetchers/__init__.py**

Old line 1:
```python
from .base import BaseMarketFetcher, MarketEnum
```
New line 1:
```python
from src.core.fetchers_base import BaseMarketFetcher, MarketEnum
```

Full updated file:
```python
from src.core.fetchers_base import BaseMarketFetcher, MarketEnum

try:
    from .orchestrator import Orchestrator
    __all__ = ["BaseMarketFetcher", "MarketEnum", "Orchestrator"]
except ImportError:
    __all__ = ["BaseMarketFetcher", "MarketEnum"]
```

- [ ] **Step 2.3: Update regime_trader/fetchers/orchestrator.py**

Find the line:
```python
from .base import BaseMarketFetcher, MarketEnum, TickerEntry
```
Replace with:
```python
from src.core.fetchers_base import BaseMarketFetcher, MarketEnum, TickerEntry
```

- [ ] **Step 2.4: Update tests/test_fetchers.py**

Line 1:
```python
from regime_trader.fetchers.base import TickerEntry
```
→
```python
from src.core.fetchers_base import TickerEntry
```

Line 9:
```python
from regime_trader.fetchers.base import BaseMarketFetcher, MarketEnum
```
→
```python
from src.core.fetchers_base import BaseMarketFetcher, MarketEnum
```

- [ ] **Step 2.5: Update tests/test_global_scoring_v22.py**

Line 164:
```python
from regime_trader.fetchers.base import MarketEnum
```
→
```python
from src.core.fetchers_base import MarketEnum
```

- [ ] **Step 2.6: Update tests/test_source_reliability.py**

Line 11:
```python
from regime_trader.fetchers.base import MarketEnum
```
→
```python
from src.core.fetchers_base import MarketEnum
```

- [ ] **Step 2.7: Run pytest (before delete — all consumers updated)**

```bash
python -m pytest tests/ -q --tb=short -m "not slow" --timeout=60 2>&1 | tail -5
```
Expected: same green count. If any test fails with `ModuleNotFoundError: regime_trader.fetchers.base`, find the missed import and update it.

- [ ] **Step 2.8: Delete old file**

```bash
rm regime_trader/fetchers/base.py
```

- [ ] **Step 2.9: Run pytest (post-delete)**

```bash
python -m pytest tests/ -q --tb=short -m "not slow" --timeout=60 2>&1 | tail -5
```
Expected: same green count.

- [ ] **Step 2.10: Commit**

```bash
git add src/core/fetchers_base.py \
        regime_trader/fetchers/__init__.py \
        regime_trader/fetchers/orchestrator.py \
        tests/test_fetchers.py \
        tests/test_global_scoring_v22.py \
        tests/test_source_reliability.py
git commit -m "chore: move fetchers/base.py → src/core/fetchers_base.py; update all importers"
```

---

## Task 3: Move src/engine/engine.py

Source: `backend/market_intel/engine.py`
No internal imports (stdlib: `os`, `json`, `logging`).

**Files:**
- Create: `src/engine/engine.py` (copy of `backend/market_intel/engine.py`)
- Modify: `scripts/run_pipeline_profile.py` (1 import line)
- Modify: `tests/test_global_scoring_v22.py` (4 import lines, same pattern)
- Delete: `backend/market_intel/engine.py`

Note: `generate_top_lists.py`, `portfolio_optimizer.py`, `satellite_factors.py` do NOT import from `engine.py` (confirmed by grep).

- [ ] **Step 3.1: Copy file**

```bash
cp backend/market_intel/engine.py src/engine/engine.py
```

- [ ] **Step 3.2: Update scripts/run_pipeline_profile.py line 6**

Old:
```python
from backend.market_intel.engine import StrategyEngine
```
New:
```python
from src.engine.engine import StrategyEngine
```

- [ ] **Step 3.3: Update tests/test_global_scoring_v22.py (4 occurrences)**

Each occurrence (lines 222, 250, 279, 308) reads:
```python
from backend.market_intel.engine import StrategyEngine
```
Replace all 4 with:
```python
from src.engine.engine import StrategyEngine
```

Use replace_all=True on Edit tool since the pattern is identical on all 4 lines.

- [ ] **Step 3.4: Run pytest (before delete)**

```bash
python -m pytest tests/ -q --tb=short -m "not slow" --timeout=60 2>&1 | tail -5
```

- [ ] **Step 3.5: Delete old file**

```bash
rm backend/market_intel/engine.py
```

- [ ] **Step 3.6: Run pytest (post-delete)**

```bash
python -m pytest tests/ -q --tb=short -m "not slow" --timeout=60 2>&1 | tail -5
```

- [ ] **Step 3.7: Commit**

```bash
git add src/engine/engine.py scripts/run_pipeline_profile.py tests/test_global_scoring_v22.py
git commit -m "chore: move backend/market_intel/engine.py → src/engine/engine.py"
```

---

## Task 4: Move src/ingestion/fmp_bulk_prefetch.py

Source: `scripts/fmp_bulk_prefetch.py`
No internal imports (stdlib + `requests`).

**Files:**
- Create: `src/ingestion/fmp_bulk_prefetch.py` (copy of `scripts/fmp_bulk_prefetch.py`)
- Modify: `tests/test_fetchers.py` (2 import lines)
- Note: `scripts/run_pipeline.py` also imports from `scripts.fmp_bulk_prefetch` — handled in Task 5 when run_pipeline.py moves.
- Delete: `scripts/fmp_bulk_prefetch.py`

- [ ] **Step 4.1: Copy file**

```bash
cp scripts/fmp_bulk_prefetch.py src/ingestion/fmp_bulk_prefetch.py
```

No changes needed inside the file (no internal imports).

- [ ] **Step 4.2: Update tests/test_fetchers.py line 3**

Old:
```python
from scripts.fmp_bulk_prefetch import build_ticker_index, map_bulk_data_to_universe, normalize_ticker_key
```
New:
```python
from src.ingestion.fmp_bulk_prefetch import build_ticker_index, map_bulk_data_to_universe, normalize_ticker_key
```

- [ ] **Step 4.3: Update tests/test_fetchers.py line 291**

Old:
```python
from scripts.fmp_bulk_prefetch import build_ticker_index
```
New:
```python
from src.ingestion.fmp_bulk_prefetch import build_ticker_index
```

- [ ] **Step 4.4: Run pytest (before delete)**

```bash
python -m pytest tests/test_fetchers.py -q --tb=short --timeout=60 2>&1 | tail -5
```

- [ ] **Step 4.5: Delete old file**

```bash
rm scripts/fmp_bulk_prefetch.py
```

- [ ] **Step 4.6: Run pytest (full suite, post-delete)**

```bash
python -m pytest tests/ -q --tb=short -m "not slow" --timeout=60 2>&1 | tail -5
```
Expected: same green count. Note: `scripts/run_pipeline.py` still has `from scripts.fmp_bulk_prefetch import ...` which will now fail — this is expected and will be fixed in Task 5.

**STOP if `scripts/run_pipeline.py` itself is imported by tests and causes a cascade failure.** In that case, proceed to Task 5 immediately without committing Task 4 separately — commit both as one atomic unit.

- [ ] **Step 4.7: Commit (or defer to Task 5 if tests fail)**

```bash
git add src/ingestion/fmp_bulk_prefetch.py tests/test_fetchers.py
git commit -m "chore: move scripts/fmp_bulk_prefetch.py → src/ingestion/fmp_bulk_prefetch.py"
```

---

## Task 5: Move src/ingestion/run_pipeline.py

Source: `scripts/run_pipeline.py`

Internal imports to update inside the moved file:
- Remove `ROOT = Path(...).resolve().parent.parent` + `sys.path.insert(0, str(ROOT))`
- Remove the `# noqa: E402` comments on the lines after the sys.path hack (they no longer apply)
- Change deferred import `from scripts.fmp_bulk_prefetch import build_ticker_index as _bti` → `from src.ingestion.fmp_bulk_prefetch import build_ticker_index as _bti`
- Keep: `from regime_trader.utils.io import save_json_atomic` (not moved)
- Keep: `from regime_trader.services.fmp_client import FMPClient as _FMPClient, FMPEndpointError` (not moved)
- Keep: `from backend.market_intel.validator import validate_raw` (not moved)

External files to update (tests that import from `scripts.run_pipeline`):
- `tests/test_congress_fetcher.py`
- `tests/test_cross_sectional.py`
- `tests/test_discord_pipeline_audit.py`
- `tests/test_fmp_client.py`
- `tests/test_pipeline_integrity.py`
- `tests/test_pipeline_momentum.py`
- `tests/test_scoring_signals.py`
- `tests/test_stress_pipeline.py`

**Files:**
- Create: `src/ingestion/run_pipeline.py`
- Modify: (inside moved file) remove sys.path hack, update fmp_bulk_prefetch import
- Modify: 8 test files (change `scripts.run_pipeline` → `src.ingestion.run_pipeline` everywhere)
- Delete: `scripts/run_pipeline.py`

- [ ] **Step 5.1: Copy file**

```bash
cp scripts/run_pipeline.py src/ingestion/run_pipeline.py
```

- [ ] **Step 5.2: Remove sys.path hack from src/ingestion/run_pipeline.py**

Locate and remove these lines (near top of file, around line 40-44):
```python
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

Also remove the `# noqa: E402` comments on the three import lines that follow them:
```python
from regime_trader.utils.io import save_json_atomic  # noqa: E402
from regime_trader.services.fmp_client import FMPClient as _FMPClient, FMPEndpointError  # noqa: E402
from backend.market_intel.validator import validate_raw  # noqa: E402
```
→
```python
from regime_trader.utils.io import save_json_atomic
from regime_trader.services.fmp_client import FMPClient as _FMPClient, FMPEndpointError
from backend.market_intel.validator import validate_raw
```

- [ ] **Step 5.3: Update deferred fmp_bulk_prefetch import inside src/ingestion/run_pipeline.py**

Find (around line 1160 of the original, now inside src/ingestion/run_pipeline.py):
```python
from scripts.fmp_bulk_prefetch import build_ticker_index as _bti  # noqa: PLC0415
```
Replace with:
```python
from src.ingestion.fmp_bulk_prefetch import build_ticker_index as _bti  # noqa: PLC0415
```

- [ ] **Step 5.4: Update tests/test_congress_fetcher.py**

Every occurrence of `scripts.run_pipeline` → `src.ingestion.run_pipeline`:

Line 14:
```python
from scripts.run_pipeline import fetch_congress_buys, score_congress
```
→
```python
from src.ingestion.run_pipeline import fetch_congress_buys, score_congress
```

Lines 89, 101, 109, 117, 131, 154, 161 — all `monkeypatch.setattr("scripts.run_pipeline.CONGRESS_CACHE_PATH", ...)`:
```python
monkeypatch.setattr("scripts.run_pipeline.CONGRESS_CACHE_PATH", ...)
```
→
```python
monkeypatch.setattr("src.ingestion.run_pipeline.CONGRESS_CACHE_PATH", ...)
```

- [ ] **Step 5.5: Update tests/test_cross_sectional.py line 22**

```python
from scripts.run_pipeline import score_congress
```
→
```python
from src.ingestion.run_pipeline import score_congress
```

- [ ] **Step 5.6: Update tests/test_discord_pipeline_audit.py**

All `scripts.run_pipeline` occurrences → `src.ingestion.run_pipeline`:

Lines 297, 308, 324, 338, 361, 405, 406, 460, 461.

Special case — lines 405-406 do a `__file__` lookup:
```python
import scripts.run_pipeline
src = Path(scripts.run_pipeline.__file__).read_text(encoding="utf-8")
```
→
```python
import src.ingestion.run_pipeline as run_pipeline_mod
file_src = Path(run_pipeline_mod.__file__).read_text(encoding="utf-8")
```
(Rename `src` → `file_src` to avoid shadowing the `src` package name. Update any subsequent use of the `src` variable in those same test methods to `file_src`.)

Same rename needed for lines 460-461:
```python
import scripts.run_pipeline
src = Path(scripts.run_pipeline.__file__).read_text(encoding="utf-8")
```
→
```python
import src.ingestion.run_pipeline as run_pipeline_mod
file_src = Path(run_pipeline_mod.__file__).read_text(encoding="utf-8")
```

All `patch("scripts.run_pipeline.` → `patch("src.ingestion.run_pipeline.` in this file.

- [ ] **Step 5.7: Update tests/test_fmp_client.py**

Lines 271, 282, 288:
```python
from scripts.run_pipeline import fetch_fmp_insider_all
```
→
```python
from src.ingestion.run_pipeline import fetch_fmp_insider_all
```

- [ ] **Step 5.8: Update tests/test_pipeline_integrity.py**

Lines 78, 109, 294: `from scripts.run_pipeline import ...` → `from src.ingestion.run_pipeline import ...`

Lines 323-325, 342-344, 356: `patch("scripts.run_pipeline.` → `patch("src.ingestion.run_pipeline.`

Line 374: `import scripts.run_pipeline as rp` → `import src.ingestion.run_pipeline as rp`

- [ ] **Step 5.9: Update tests/test_pipeline_momentum.py line 14**

```python
from scripts.run_pipeline import (
```
→
```python
from src.ingestion.run_pipeline import (
```

- [ ] **Step 5.10: Update tests/test_scoring_signals.py**

Line 13: `from scripts.run_pipeline import (` → `from src.ingestion.run_pipeline import (`

Lines 84, 105, 114, 128, 137, 178, 229, 236, 245, 251: `from scripts.run_pipeline import ...` → `from src.ingestion.run_pipeline import ...`

Lines 160-165, 203-208: `patch("scripts.run_pipeline.` → `patch("src.ingestion.run_pipeline.`

- [ ] **Step 5.11: Update tests/test_stress_pipeline.py**

Lines 30, 32, 34, 36, 38, 40, 42, 44: `patch("scripts.run_pipeline.` → `patch("src.ingestion.run_pipeline.`

Lines 56, 67, 77, 87: `from scripts.run_pipeline import ...` → `from src.ingestion.run_pipeline import ...`

- [ ] **Step 5.12: Run pytest (before deleting old file)**

```bash
python -m pytest tests/ -q --tb=short -m "not slow" --timeout=60 2>&1 | tail -10
```
Expected: same green count. Fix any remaining `scripts.run_pipeline` references before continuing.

- [ ] **Step 5.13: Delete old file**

```bash
rm scripts/run_pipeline.py
```

- [ ] **Step 5.14: Run pytest (post-delete)**

```bash
python -m pytest tests/ -q --tb=short -m "not slow" --timeout=60 2>&1 | tail -5
```

- [ ] **Step 5.15: Commit**

```bash
git add src/ingestion/run_pipeline.py \
        tests/test_congress_fetcher.py \
        tests/test_cross_sectional.py \
        tests/test_discord_pipeline_audit.py \
        tests/test_fmp_client.py \
        tests/test_pipeline_integrity.py \
        tests/test_pipeline_momentum.py \
        tests/test_scoring_signals.py \
        tests/test_stress_pipeline.py
git commit -m "chore: move scripts/run_pipeline.py → src/ingestion/run_pipeline.py; update all test imports and patch targets"
```

---

## Task 6: Move src/ingestion/fmp_fetcher.py

Source: `regime_trader/fetchers/fmp_fetcher.py`

Internal imports to update inside the moved file:
- Relative imports like `from .base import ...` → `from src.core.fetchers_base import ...`
- Deferred import `from scripts.run_pipeline import score_analyst_revision` → `from src.ingestion.run_pipeline import score_analyst_revision`
- Keep: `from regime_trader.services.fmp_client import FMPClient` (not moved)

External consumers to update:
- `tests/test_fetchers.py` line 4
- `tests/test_global_scoring_v22.py` line 163
- `tests/test_source_reliability.py` line 10
- `.github/workflows/daily_trading_pipeline.yml` line 403 (inline Python)

**Files:**
- Create: `src/ingestion/fmp_fetcher.py`
- Modify: (inside moved file) update relative/deferred imports
- Modify: `tests/test_fetchers.py`
- Modify: `tests/test_global_scoring_v22.py`
- Modify: `tests/test_source_reliability.py`
- Modify: `.github/workflows/daily_trading_pipeline.yml`
- Delete: `regime_trader/fetchers/fmp_fetcher.py`

- [ ] **Step 6.1: Copy file**

```bash
cp regime_trader/fetchers/fmp_fetcher.py src/ingestion/fmp_fetcher.py
```

- [ ] **Step 6.2: Update relative base import inside src/ingestion/fmp_fetcher.py**

Find (relative import at top of file — exact form may vary):
```python
from .base import TickerEntry, MarketEnum, BaseMarketFetcher
```
or
```python
from regime_trader.fetchers.base import TickerEntry, MarketEnum, BaseMarketFetcher
```
Replace with:
```python
from src.core.fetchers_base import TickerEntry, MarketEnum, BaseMarketFetcher
```

- [ ] **Step 6.3: Update deferred run_pipeline import inside src/ingestion/fmp_fetcher.py**

Find (around line 251, inside a function body):
```python
from scripts.run_pipeline import score_analyst_revision  # noqa: PLC0415
```
Replace with:
```python
from src.ingestion.run_pipeline import score_analyst_revision  # noqa: PLC0415
```

- [ ] **Step 6.4: Update tests/test_fetchers.py line 4**

```python
from regime_trader.fetchers.fmp_fetcher import FMPFetcher
```
→
```python
from src.ingestion.fmp_fetcher import FMPFetcher
```

- [ ] **Step 6.5: Update tests/test_global_scoring_v22.py line 163**

```python
from regime_trader.fetchers.fmp_fetcher import FMPFetcher
```
→
```python
from src.ingestion.fmp_fetcher import FMPFetcher
```

- [ ] **Step 6.6: Update tests/test_source_reliability.py line 10**

```python
from regime_trader.fetchers.fmp_fetcher import FMPFetcher
```
→
```python
from src.ingestion.fmp_fetcher import FMPFetcher
```

- [ ] **Step 6.7: Update .github/workflows/daily_trading_pipeline.yml inline Python (INTL fetch step)**

Find the inline Python block in the "Fetch INTL raw metrics" step (~line 403-404):
```python
          from regime_trader.fetchers.fmp_fetcher import FMPFetcher
          from regime_trader.fetchers.base import MarketEnum
```
Replace with:
```python
          from src.ingestion.fmp_fetcher import FMPFetcher
          from src.core.fetchers_base import MarketEnum
```

Also in the same step, find the deferred bulk index import (~line 412):
```python
              from scripts.fmp_bulk_prefetch import build_ticker_index
```
Replace with:
```python
              from src.ingestion.fmp_bulk_prefetch import build_ticker_index
```

- [ ] **Step 6.8: Run pytest (before delete)**

```bash
python -m pytest tests/ -q --tb=short -m "not slow" --timeout=60 2>&1 | tail -5
```

- [ ] **Step 6.9: Delete old file**

```bash
rm regime_trader/fetchers/fmp_fetcher.py
```

- [ ] **Step 6.10: Run pytest (post-delete)**

```bash
python -m pytest tests/ -q --tb=short -m "not slow" --timeout=60 2>&1 | tail -5
```

- [ ] **Step 6.11: Commit**

```bash
git add src/ingestion/fmp_fetcher.py \
        tests/test_fetchers.py \
        tests/test_global_scoring_v22.py \
        tests/test_source_reliability.py \
        .github/workflows/daily_trading_pipeline.yml
git commit -m "chore: move regime_trader/fetchers/fmp_fetcher.py → src/ingestion/fmp_fetcher.py"
```

---

## Task 7: Move src/engine/profile_runner.py

Source: `scripts/run_pipeline_profile.py`
Internal imports to update: `from backend.market_intel.engine import StrategyEngine` → `from src.engine.engine import StrategyEngine` (already updated in Task 3 on the original file; copy after that change).

No sys.path hack in this file (it was run from repo root in CI).

**Files:**
- Create: `src/engine/profile_runner.py`
- Delete: `scripts/run_pipeline_profile.py`
- No test files import from `scripts.run_pipeline_profile` (confirmed by grep)

- [ ] **Step 7.1: Copy updated scripts/run_pipeline_profile.py to new location**

```bash
cp scripts/run_pipeline_profile.py src/engine/profile_runner.py
```

The import is already `from src.engine.engine import StrategyEngine` (updated in Task 3).

- [ ] **Step 7.2: Run pytest**

```bash
python -m pytest tests/ -q --tb=short -m "not slow" --timeout=60 2>&1 | tail -5
```

- [ ] **Step 7.3: Delete old file**

```bash
rm scripts/run_pipeline_profile.py
```

- [ ] **Step 7.4: Run pytest (post-delete)**

```bash
python -m pytest tests/ -q --tb=short -m "not slow" --timeout=60 2>&1 | tail -5
```

- [ ] **Step 7.5: Commit**

```bash
git add src/engine/profile_runner.py
git commit -m "chore: move scripts/run_pipeline_profile.py → src/engine/profile_runner.py"
```

---

## Task 8: Move src/risk/regime.py

Source: `regime_trader/risk/regime.py`
No internal imports (stdlib only: `math`, `enum`, `typing`).

External consumers:
- `scripts/cook_toplists.py` (also moving to src/delivery in Task 9 — handle there)
- `tests/test_risk_regime.py`

Note: `regime_trader/risk/exit_rules.py` does NOT import from `regime.py` (confirmed by reading its imports — only stdlib + `typing`).

**Files:**
- Create: `src/risk/regime.py`
- Modify: `tests/test_risk_regime.py`
- Note: `regime_trader/risk/__init__.py` is empty — no change needed
- Delete: `regime_trader/risk/regime.py`

- [ ] **Step 8.1: Copy file**

```bash
cp regime_trader/risk/regime.py src/risk/regime.py
```

No internal import changes needed.

- [ ] **Step 8.2: Update tests/test_risk_regime.py line 4**

```python
from regime_trader.risk.regime import (
```
→
```python
from src.risk.regime import (
```

- [ ] **Step 8.3: Run pytest (before delete)**

```bash
python -m pytest tests/test_risk_regime.py -v --tb=short --timeout=60 2>&1 | tail -10
```

- [ ] **Step 8.4: Delete old file**

```bash
rm regime_trader/risk/regime.py
```

- [ ] **Step 8.5: Run pytest (full suite, post-delete)**

```bash
python -m pytest tests/ -q --tb=short -m "not slow" --timeout=60 2>&1 | tail -5
```

- [ ] **Step 8.6: Commit**

```bash
git add src/risk/regime.py tests/test_risk_regime.py
git commit -m "chore: move regime_trader/risk/regime.py → src/risk/regime.py"
```

---

## Task 9: Move src/delivery/cook_toplists.py

Source: `scripts/cook_toplists.py`

Internal imports to update inside the moved file:
```python
from regime_trader.risk.regime import (
    RiskRegime, apply_capitulation_filter, get_regime, score_multiplier,
)
```
→
```python
from src.risk.regime import (
    RiskRegime, apply_capitulation_filter, get_regime, score_multiplier,
)
```

Keep unchanged:
- `from backend.market_intel.portfolio_optimizer import run_optimizer, build_large_cap_anchors` (not moved)
- `from regime_trader.risk.exit_rules import enrich_with_exit_anchors` (not moved)

External consumers:
- `tests/test_cook_toplists.py` — uses `importlib.util.spec_from_file_location` pointing to `scripts/cook_toplists.py`

**Files:**
- Create: `src/delivery/cook_toplists.py`
- Modify: (inside moved file) update `from regime_trader.risk.regime import ...`
- Modify: `tests/test_cook_toplists.py`
- Delete: `scripts/cook_toplists.py`

- [ ] **Step 9.1: Copy file**

```bash
cp scripts/cook_toplists.py src/delivery/cook_toplists.py
```

- [ ] **Step 9.2: Update regime import inside src/delivery/cook_toplists.py**

Old:
```python
from regime_trader.risk.regime import (
    RiskRegime,
    apply_capitulation_filter,
    get_regime,
    score_multiplier,
)
```
New:
```python
from src.risk.regime import (
    RiskRegime,
    apply_capitulation_filter,
    get_regime,
    score_multiplier,
)
```

- [ ] **Step 9.3: Update tests/test_cook_toplists.py**

The test uses `importlib.util.spec_from_file_location` to load the module dynamically. Update the path:

Old helper function:
```python
def _load_cook():
    """Dynamically import cook_toplists from the scripts/ directory."""
    spec = importlib.util.spec_from_file_location(
        "cook_toplists",
        Path(__file__).parents[1] / "scripts" / "cook_toplists.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
```
New:
```python
def _load_cook():
    """Dynamically import cook_toplists from src/delivery/."""
    spec = importlib.util.spec_from_file_location(
        "cook_toplists",
        Path(__file__).parents[1] / "src" / "delivery" / "cook_toplists.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
```

- [ ] **Step 9.4: Run pytest (before delete)**

```bash
python -m pytest tests/test_cook_toplists.py -v --tb=short --timeout=60 2>&1 | tail -10
```

- [ ] **Step 9.5: Delete old file**

```bash
rm scripts/cook_toplists.py
```

- [ ] **Step 9.6: Run pytest (full suite)**

```bash
python -m pytest tests/ -q --tb=short -m "not slow" --timeout=60 2>&1 | tail -5
```

- [ ] **Step 9.7: Commit**

```bash
git add src/delivery/cook_toplists.py tests/test_cook_toplists.py
git commit -m "chore: move scripts/cook_toplists.py → src/delivery/cook_toplists.py; update regime import"
```

---

## Task 10: Move src/delivery/audit_payload.py

Source: `scripts/audit_payload.py`
No internal imports.

External consumers:
- `tests/test_audit_payload.py` — uses `sys.path.insert(0, .../scripts)` + bare `from audit_payload import ...`
- `tests/test_audit_payload.py` line 266 — `import scripts.audit_payload as ap_module`

**Files:**
- Create: `src/delivery/audit_payload.py`
- Modify: `tests/test_audit_payload.py` (remove sys.path hack, update imports)
- Delete: `scripts/audit_payload.py`

- [ ] **Step 10.1: Copy file**

```bash
cp scripts/audit_payload.py src/delivery/audit_payload.py
```

No internal imports to change.

- [ ] **Step 10.2: Update tests/test_audit_payload.py**

Remove the sys.path manipulation block at the top:
```python
import sys
import os
# Allow importing from scripts/ directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
```

Change the bare import:
```python
from audit_payload import (
    audit,
    ScoreDivergenceError,
    BadgeMismatchError,
    SortingError,
    CrossContaminationError,
    GeographicLeakageError,
    StructuralIntegrityError,
    VIXCoherenceError,
)
```
→
```python
from src.delivery.audit_payload import (
    audit,
    ScoreDivergenceError,
    BadgeMismatchError,
    SortingError,
    CrossContaminationError,
    GeographicLeakageError,
    StructuralIntegrityError,
    VIXCoherenceError,
)
```

Line 266:
```python
import scripts.audit_payload as ap_module
```
→
```python
import src.delivery.audit_payload as ap_module
```

- [ ] **Step 10.3: Run pytest (before delete)**

```bash
python -m pytest tests/test_audit_payload.py -v --tb=short --timeout=60 2>&1 | tail -10
```

- [ ] **Step 10.4: Delete old file**

```bash
rm scripts/audit_payload.py
```

- [ ] **Step 10.5: Run pytest (full suite)**

```bash
python -m pytest tests/ -q --tb=short -m "not slow" --timeout=60 2>&1 | tail -5
```

- [ ] **Step 10.6: Commit**

```bash
git add src/delivery/audit_payload.py tests/test_audit_payload.py
git commit -m "chore: move scripts/audit_payload.py → src/delivery/audit_payload.py"
```

---

## Task 11: Move src/delivery/send_discord.py

Source: `scripts/send_toplists_discord.py`

Internal changes to moved file:
- Remove `sys.path.insert(0, str(Path(__file__).resolve().parent.parent))` line
- Keep: `from regime_trader.utils.formatting import score_bar as _score_bar_util` (not moved)

External consumers (tests):
- `tests/test_discord_formatter.py` — 15+ occurrences of `from scripts.send_toplists_discord import ...`
- `tests/test_discord_pipeline_audit.py` — 10+ occurrences of `from scripts.send_toplists_discord import ...`
- `tests/test_send_toplists_discord.py` line 10 — `from scripts.send_toplists_discord import ...`

**Files:**
- Create: `src/delivery/send_discord.py`
- Modify: (inside moved file) remove sys.path hack
- Modify: `tests/test_discord_formatter.py`
- Modify: `tests/test_discord_pipeline_audit.py`
- Modify: `tests/test_send_toplists_discord.py`
- Delete: `scripts/send_toplists_discord.py`

- [ ] **Step 11.1: Copy file**

```bash
cp scripts/send_toplists_discord.py src/delivery/send_discord.py
```

- [ ] **Step 11.2: Remove sys.path hack inside src/delivery/send_discord.py**

Find and remove:
```python
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

- [ ] **Step 11.3: Update tests/test_discord_formatter.py (all occurrences)**

Replace ALL occurrences of:
```python
from scripts.send_toplists_discord import
```
with:
```python
from src.delivery.send_discord import
```

Use `replace_all=True` on the Edit tool. This covers lines 62, 68, 74, 87, 96, 103, 109, 116, 127, 132, 138, 146, 158, 164, 175, 181, 188, 208, 216, 225, 270 and any others in the file.

- [ ] **Step 11.4: Update tests/test_discord_pipeline_audit.py (all occurrences)**

Replace ALL occurrences of:
```python
from scripts.send_toplists_discord import
```
with:
```python
from src.delivery.send_discord import
```

Also replace all patch targets:
```python
patch("scripts.send_toplists_discord.
```
→
```python
patch("src.delivery.send_discord.
```

- [ ] **Step 11.5: Update tests/test_send_toplists_discord.py line 10**

```python
from scripts.send_toplists_discord import _load_satellite, build_payload
```
→
```python
from src.delivery.send_discord import _load_satellite, build_payload
```

- [ ] **Step 11.6: Run pytest (before delete)**

```bash
python -m pytest tests/test_discord_formatter.py tests/test_discord_pipeline_audit.py tests/test_send_toplists_discord.py -v --tb=short --timeout=60 2>&1 | tail -15
```

- [ ] **Step 11.7: Delete old file**

```bash
rm scripts/send_toplists_discord.py
```

- [ ] **Step 11.8: Run pytest (full suite)**

```bash
python -m pytest tests/ -q --tb=short -m "not slow" --timeout=60 2>&1 | tail -5
```
Expected: same green count as Task 0.

- [ ] **Step 11.9: Commit**

```bash
git add src/delivery/send_discord.py \
        tests/test_discord_formatter.py \
        tests/test_discord_pipeline_audit.py \
        tests/test_send_toplists_discord.py
git commit -m "chore: move scripts/send_toplists_discord.py → src/delivery/send_discord.py"
```

---

## Task 12: Update CI/CD GitHub Actions Workflows

All 8 workflow files in `.github/workflows/` need execution paths updated.

### 12a: canary.yml

- [ ] **Step 12a.1: Update FMP bulk pre-fetch step**

Find:
```yaml
          python scripts/fmp_bulk_prefetch.py \
```
Replace with:
```yaml
          python src/ingestion/fmp_bulk_prefetch.py \
```

- [ ] **Step 12a.2: Update EDGAR pipeline step**

Find:
```yaml
          python scripts/run_pipeline.py \
```
Replace with:
```yaml
          python src/ingestion/run_pipeline.py \
```

### 12b: edgar_3x.yml

- [ ] **Step 12b.1: Update FMP bulk pre-fetch step**

```yaml
          python scripts/fmp_bulk_prefetch.py \
```
→
```yaml
          python src/ingestion/fmp_bulk_prefetch.py \
```

- [ ] **Step 12b.2: Update EDGAR full-universe fetch step (timeout wrapper)**

```yaml
            if timeout 20m python scripts/run_pipeline.py \
```
→
```yaml
            if timeout 20m python src/ingestion/run_pipeline.py \
```

- [ ] **Step 12b.3: Update Generate top lists step**

```yaml
          python -m backend.market_intel.generate_top_lists \
```
→
```yaml
          python -m backend.market_intel.generate_top_lists \
```
NOTE: `generate_top_lists.py` is NOT moving. This command stays unchanged.

- [ ] **Step 12b.4: Update Audit payload step**

```yaml
          python scripts/audit_payload.py --input logs/top_lists.json
```
→
```yaml
          python src/delivery/audit_payload.py --input logs/top_lists.json
```

- [ ] **Step 12b.5: satellite_factors stays unchanged**

`python -m backend.market_intel.satellite_factors` — NOT moving. No change needed.

### 12c: daily_trading_pipeline.yml

- [ ] **Step 12c.1: Update FMP bulk pre-fetch in run_us_pipeline job**

```yaml
          python scripts/fmp_bulk_prefetch.py \
```
→
```yaml
          python src/ingestion/fmp_bulk_prefetch.py \
```
(Appears twice — once in run_us_pipeline and once in run_intl_pipeline)

- [ ] **Step 12c.2: Update EDGAR fetch in run_us_pipeline job**

```yaml
            if timeout 20m python scripts/run_pipeline.py \
```
→
```yaml
            if timeout 20m python src/ingestion/run_pipeline.py \
```

- [ ] **Step 12c.3: Update generate_top_lists in run_us_pipeline job**

```yaml
          python -m backend.market_intel.generate_top_lists \
```
NOTE: stays unchanged (not moving).

- [ ] **Step 12c.4: Update audit_payload in run_us_pipeline job**

```yaml
          python scripts/audit_payload.py --input logs/top_lists_us.json
```
→
```yaml
          python src/delivery/audit_payload.py --input logs/top_lists_us.json
```

- [ ] **Step 12c.5: satellite_factors stays unchanged in run_us_pipeline**

- [ ] **Step 12c.6: Update run_intl_pipeline Score INTL universe step**

```yaml
          python scripts/run_pipeline_profile.py \
```
→
```yaml
          python src/engine/profile_runner.py \
```

- [ ] **Step 12c.7: cook_and_notify — Cook (merge US + INTL) step**

```yaml
          python scripts/cook_toplists.py \
```
→
```yaml
          python src/delivery/cook_toplists.py \
```

- [ ] **Step 12c.8: cook_and_notify — Audit combined payload step**

```yaml
          python scripts/audit_payload.py --input logs/top_lists.json
```
→
```yaml
          python src/delivery/audit_payload.py --input logs/top_lists.json
```

- [ ] **Step 12c.9: cook_and_notify — Send daily market checkup step**

```yaml
          python scripts/send_toplists_discord.py \
```
→
```yaml
          python src/delivery/send_discord.py \
```

### 12d: ci.yml

- [ ] **Step 12d.1: Update Lint (ruff) step to include src/**

```yaml
          ruff check scripts/ regime_trader/ backend/ monitoring/ \
            --select E,F,W --ignore E501
```
→
```yaml
          ruff check scripts/ src/ regime_trader/ backend/ monitoring/ \
            --select E,F,W --ignore E501
```

Note: `python scripts/check_imports.py` stays (check_imports.py not moving). `python scripts/check_secrets.py` stays.

The `Run test suite — backend/tests/` step checks `if [ -d backend/tests ]` — stays unchanged (no backend/tests dir).

### 12e: nightly_edgar.yml

- [ ] **Step 12e.1: Update FMP bulk pre-fetch step**

```yaml
          python scripts/fmp_bulk_prefetch.py \
```
→
```yaml
          python src/ingestion/fmp_bulk_prefetch.py \
```

- [ ] **Step 12e.2: Update EDGAR full-universe fetch step**

```yaml
            if python scripts/run_pipeline.py \
```
→
```yaml
            if python src/ingestion/run_pipeline.py \
```

### 12f: weekly_backtest.yml

No changes needed — `scripts/backtest_signals.py` is NOT in the migration.

### 12g: test_daily_toplists_absence.yml

- [ ] **Step 12g.1: Update send script path in Trigger alert step**

```yaml
          python scripts/send_toplists_discord.py \
```
→
```yaml
          python src/delivery/send_discord.py \
```

- [ ] **Step 12g.2: Update push trigger paths**

```yaml
    paths:
      - "scripts/send_toplists_discord.py"
      - "scripts/mock_webhook_server.py"
```
→
```yaml
    paths:
      - "src/delivery/send_discord.py"
      - "scripts/mock_webhook_server.py"
```

### 12h: hybrid_pipeline.yml

No changes needed — inline Python imports from `analysis.*` which is not moving.

- [ ] **Step 12i: Run pytest after all workflow edits**

```bash
python -m pytest tests/ -q --tb=short -m "not slow" --timeout=60 2>&1 | tail -5
```
Expected: same green count (CI yaml changes don't affect local pytest).

- [ ] **Step 12j: Commit all workflow changes**

```bash
git add .github/workflows/canary.yml \
        .github/workflows/edgar_3x.yml \
        .github/workflows/daily_trading_pipeline.yml \
        .github/workflows/ci.yml \
        .github/workflows/nightly_edgar.yml \
        .github/workflows/test_daily_toplists_absence.yml
git commit -m "chore: update CI/CD workflow paths for src/ migration"
```

---

## Task 13: Final verification

- [ ] **Step 13.1: Full pytest run**

```bash
python -m pytest tests/ -v --tb=short -m "not slow" --timeout=60 2>&1 | tail -30
```
Expected: same count as Task 0, all green.

- [ ] **Step 13.2: Confirm no stale scripts.* / backend.market_intel.engine / regime_trader.fetchers.base / regime_trader.risk.regime imports remain**

```bash
grep -rn "from scripts\.run_pipeline\|from scripts\.send_toplists_discord\|from scripts\.fmp_bulk_prefetch\|from backend\.market_intel\.engine import\|from regime_trader\.fetchers\.base\|from regime_trader\.risk\.regime" \
    tests/ scripts/ src/ regime_trader/ backend/ monitoring/ \
    --include="*.py" | grep -v "__pycache__" | grep -v "docs/"
```
Expected: no output (all old paths replaced).

- [ ] **Step 13.3: Confirm src/ package imports cleanly**

```bash
python -c "
import src.core.fetchers_base
import src.ingestion.fmp_bulk_prefetch
import src.ingestion.run_pipeline
import src.ingestion.fmp_fetcher
import src.engine.engine
import src.engine.profile_runner
import src.risk.regime
import src.delivery.cook_toplists
import src.delivery.audit_payload
import src.delivery.send_discord
print('All src/ imports OK')
"
```
Expected: `All src/ imports OK`

- [ ] **Step 13.4: Final commit (if any cleanup needed)**

```bash
git add -A
git commit -m "chore: src/ migration complete — all 10 modules in workflow-aligned domains"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] Step 2.1: Initialize the New Package Space → Task 1
- [x] Step 2.2: Isolate and Move Source Files → Tasks 2–11
- [x] Step 2.3: Global Internal Import Repair → covered per-task + Task 12 workflow fixes
- [x] Step 2.4: Update Test Suite Framework → covered in Tasks 2–11
- [x] Step 2.5: Re-map CI/CD GitHub Actions Workflows → Task 12

**Files NOT in target spec that need import updates due to moved dependencies:**
- `regime_trader/fetchers/__init__.py` — updated in Task 2 (re-exports base)
- `regime_trader/fetchers/orchestrator.py` — updated in Task 2 (relative base import)
- `scripts/run_pipeline_profile.py` — updated in Task 3 (engine import), then moved in Task 7
- `tests/test_global_scoring_v22.py` — updated in Tasks 2, 3, 6 (three different moved modules)

**Immutability guard:** No business math, VIX thresholds, ATR multipliers, or weight dictionaries are touched. Only import statements and sys.path hacks are modified inside moved files.
