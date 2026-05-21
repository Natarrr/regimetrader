# Pipeline Validator & Anomaly Detection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `backend/market_intel/validator.py` — a two-stage data quality gate that catches bad raw data before scoring and flags statistical anomalies after scoring, with a permanent JSON audit trail.

**Architecture:** Stage 1 (`validate_raw`) runs before `generate_top_lists.py` and quarantines bad rows; Stage 2 (`detect_anomalies`) runs after scoring and writes `anomaly_report_{run_id}.json` + `anomaly_report_latest.json`. The `Normalizer` class wraps existing `winsorize`/`cross_sectional_normalize` and adds the new `log_scale_insider` transform. Discord formatter reads `anomaly_report_latest.json` to append ⚠️ to flagged tickers.

**Tech Stack:** Python 3.11+, pandas, numpy, python-dateutil, pytest. No new dependencies.

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `backend/market_intel/validator.py` | **Create** | All validation, normalization, anomaly detection |
| `tests/test_validator.py` | **Create** | Full test suite — 28 tests across 3 classes |
| `backend/market_intel/generate_top_lists.py` | **Modify** | Exclude `_validation_failed` tickers from `top_buys` |
| `scripts/send_toplists_discord.py` | **Modify** | Load `anomaly_report_latest.json`, append ⚠️ |
| `scripts/run_pipeline.py` | **Modify** | Call `validate_raw()` after writing `intel_source_status.json` |

---

## Task 1: `Normalizer` class — `log_scale_insider` + wrappers

**Files:**
- Create: `backend/market_intel/validator.py`
- Test: `tests/test_validator.py`

- [ ] **Step 1: Write the failing tests for `Normalizer`**

Create `tests/test_validator.py`:

```python
"""tests/test_validator.py
Validator, Normalizer, and anomaly detection tests.
All fixtures are in-memory — no network calls.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

import pytest


# ── Helpers ────────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _ago_iso(hours: float = 0, days: float = 0) -> str:
    delta = timedelta(hours=hours, days=days)
    return (datetime.now(timezone.utc) - delta).isoformat()

def _future_iso(hours: float = 1) -> str:
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).isoformat()

def _row(
    ticker: str = "AAPL",
    insider_usd: float = 50_000.0,
    market_cap: float = 3e12,
    cap_tier: str = "large",
    news_score: float = 0.65,
    volume_spike: float = 1.5,
    timestamp: str | None = None,
    source: str = "quiver",
) -> Dict[str, Any]:
    return {
        "ticker": ticker,
        "insider_usd": insider_usd,
        "market_cap": market_cap,
        "cap_tier": cap_tier,
        "news_score": news_score,
        "volume_spike": volume_spike,
        "computed_at": timestamp or _now_iso(),
        "insider_source": source,
    }

def _source_meta(quiver_age_hours: float = 1.0) -> Dict[str, Any]:
    return {
        "quiver": {"last_updated": _ago_iso(hours=quiver_age_hours)},
        "fmp":    {"last_updated": _ago_iso(hours=1.0)},
        "edgar":  {"last_updated": _ago_iso(hours=1.0)},
    }


# ── TestNormalizer ─────────────────────────────────────────────────────────────

class TestNormalizer:
    def _norm(self):
        from backend.market_intel.validator import Normalizer
        return Normalizer

    def test_log_scale_insider_large_cap_at_ceiling(self):
        # 0.5% of market_cap for large tier → score should be 1.0
        N = self._norm()
        cap = 1e12
        amount = cap * 0.005   # exactly at ceiling
        score = N.log_scale_insider(amount, cap, tier="large")
        assert abs(score - 1.0) < 1e-9

    def test_log_scale_insider_small_cap_tighter_ceiling(self):
        # Same 0.5% ratio, small tier (ceiling=2%) → score below 1.0
        N = self._norm()
        cap = 50_000_000.0
        amount = cap * 0.005   # 0.5% — well below 2% small ceiling
        score = N.log_scale_insider(amount, cap, tier="small")
        assert 0.0 < score < 1.0

    def test_log_scale_insider_nan_on_zero_amount(self):
        N = self._norm()
        assert math.isnan(N.log_scale_insider(0.0, 1e9, tier="large"))

    def test_log_scale_insider_nan_on_negative_amount(self):
        N = self._norm()
        assert math.isnan(N.log_scale_insider(-100.0, 1e9, tier="large"))

    def test_log_scale_insider_nan_on_zero_cap(self):
        N = self._norm()
        assert math.isnan(N.log_scale_insider(50_000.0, 0.0, tier="large"))

    def test_log_scale_insider_nan_on_nan_amount(self):
        N = self._norm()
        assert math.isnan(N.log_scale_insider(float("nan"), 1e9, tier="large"))

    def test_log_scale_insider_monotone(self):
        N = self._norm()
        cap = 1e10
        scores = [N.log_scale_insider(amt, cap, tier="mid") for amt in (1_000, 50_000, 500_000, 5_000_000)]
        assert scores == sorted(scores), "larger purchase must produce higher score"

    def test_winsorize_caps_outliers(self):
        import numpy as np
        from backend.market_intel.validator import Normalizer
        series = np.array([0.1] * 98 + [1000.0, -1000.0])
        result = Normalizer.winsorize(series)
        assert result.max() <= 1.0 + 1e-6
        assert result.min() >= 0.1 - 1e-6

    def test_cross_sectional_norm_mean_near_neutral(self):
        import numpy as np
        from backend.market_intel.validator import Normalizer
        series = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        result = Normalizer.cross_sectional_norm(series)
        # min-max scaled to [0,1] — mean ≈ 0.5 for symmetric input
        assert abs(float(result.mean()) - 0.5) < 0.1
```

- [ ] **Step 2: Run tests to verify they fail**

```
cd "c:/Users/ntard/Projects/Trading dashboard/regime_trader"
python -m pytest tests/test_validator.py::TestNormalizer -v
```

Expected: `ERROR` — `ModuleNotFoundError: No module named 'backend.market_intel.validator'`

- [ ] **Step 3: Create `backend/market_intel/validator.py` with `Normalizer` class**

```python
"""backend/market_intel/validator.py
Two-stage data quality gate for the regime_trader pipeline.

Stage 1 — validate_raw():  pre-scoring checks on raw rows
Stage 2 — detect_anomalies(): post-scoring circuit breakers

Normalizer: thin wrappers + log_scale_insider (new math only here).
"""
from __future__ import annotations

import json
import logging
import math
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from backend.market_intel.generate_top_lists import PipelineIntegrityError
from regime_trader.scoring.normalize import (
    normalize_score,
    winsorize as _winsorize_np,
)

log = logging.getLogger("validator")

# ── Tier ceilings for log_scale_insider ───────────────────────────────────────
_TIER_CEILING: Dict[str, float] = {
    "small": 0.02,
    "mid":   0.01,
    "large": 0.005,
}


# ── Normalizer ────────────────────────────────────────────────────────────────

class Normalizer:
    """Thin delegation layer + log_scale_insider.

    All methods are static — no state, no instantiation required.
    """

    @staticmethod
    def winsorize(
        series: np.ndarray,
        limits: Tuple[float, float] = (0.01, 0.99),
    ) -> np.ndarray:
        """Winsorize series at [lo, hi] fractional limits (0.01 = 1st pct)."""
        lo_pct = limits[0] * 100
        hi_pct = limits[1] * 100
        return _winsorize_np(np.asarray(series, dtype=np.float64), lo=lo_pct, hi=hi_pct)

    @staticmethod
    def log_scale_insider(
        amount: float,
        market_cap: float,
        tier: Literal["small", "mid", "large"] = "large",
    ) -> float:
        """Log-scale insider conviction signal with tier-aware ceiling.

        Formula:  min( log(1 + amount/cap) / log(1 + ceiling), 1.0 )

        Returns float("nan") on any invalid input.
        """
        try:
            if math.isnan(amount) or math.isnan(market_cap):
                return float("nan")
        except (TypeError, ValueError):
            return float("nan")
        if amount <= 0 or market_cap <= 0:
            return float("nan")
        ceiling = _TIER_CEILING.get(tier)
        if ceiling is None:
            return float("nan")
        ratio = amount / market_cap
        score = math.log1p(ratio) / math.log1p(ceiling)
        return min(score, 1.0)

    @staticmethod
    def cross_sectional_norm(series: np.ndarray) -> np.ndarray:
        """Min-max scale series to [0, 1]. Delegates to normalize_score."""
        arr = np.asarray(series, dtype=np.float64)
        if arr.size == 0:
            return arr
        return normalize_score(arr, lo_pct=0, hi_pct=100, out_min=0.0, out_max=1.0)
```

- [ ] **Step 4: Run tests to verify they pass**

```
python -m pytest tests/test_validator.py::TestNormalizer -v
```

Expected: `9 passed`

- [ ] **Step 5: Commit**

```
git add backend/market_intel/validator.py tests/test_validator.py
git commit -m "feat(validator): Normalizer class with log_scale_insider + wrappers"
```

---

## Task 2: Validation data structures + `validate_tickers`

**Files:**
- Modify: `backend/market_intel/validator.py`
- Modify: `tests/test_validator.py`

- [ ] **Step 1: Write the failing tests for `validate_tickers`**

Append to `tests/test_validator.py`:

```python
# ── TestValidation ─────────────────────────────────────────────────────────────

class TestValidation:
    def _validate_tickers(self, rows):
        from backend.market_intel.validator import validate_tickers
        return validate_tickers(rows)

    def test_valid_ticker_passes(self):
        ok, issues = self._validate_tickers([_row("AAPL")])
        assert ok is True
        assert issues == []

    def test_empty_ticker_quarantined(self):
        row = _row("")
        ok, issues = self._validate_tickers([row])
        assert ok is False
        assert row.get("_validation_failed") is True

    def test_numeric_ticker_quarantined(self):
        row = _row("123AB")
        ok, issues = self._validate_tickers([row])
        assert ok is False
        assert row.get("_validation_failed") is True

    def test_lowercase_ticker_quarantined(self):
        row = _row("aapl")
        ok, issues = self._validate_tickers([row])
        assert ok is False
        assert row.get("_validation_failed") is True

    def test_too_long_ticker_quarantined(self):
        row = _row("TOOLONG")
        ok, issues = self._validate_tickers([row])
        assert ok is False
        assert row.get("_validation_failed") is True
```

- [ ] **Step 2: Run tests to verify they fail**

```
python -m pytest tests/test_validator.py::TestValidation::test_valid_ticker_passes -v
```

Expected: `ERROR` — `ImportError: cannot import name 'validate_tickers'`

- [ ] **Step 3: Add data structures + `validate_tickers` to `validator.py`**

Append after the `Normalizer` class:

```python
# ── Validation data structures ────────────────────────────────────────────────

import re as _re

_TICKER_RE = _re.compile(r"^[A-Z]{1,5}$")


class ValidationIssue:
    __slots__ = ("ticker", "field", "code", "original_value")

    def __init__(self, ticker: str, field: str, code: str, original_value: Any = None) -> None:
        self.ticker = ticker
        self.field = field
        self.code = code
        self.original_value = original_value

    def __repr__(self) -> str:  # pragma: no cover
        return f"ValidationIssue({self.code} ticker={self.ticker} field={self.field})"


# ── validate_tickers ──────────────────────────────────────────────────────────

def validate_tickers(rows: List[Dict[str, Any]]) -> Tuple[bool, List[ValidationIssue]]:
    """Check each ticker matches ^[A-Z]{1,5}$.

    Mutates rows in-place: sets _validation_failed=True on bad rows.
    Never raises.
    """
    issues: List[ValidationIssue] = []
    for row in rows:
        t = row.get("ticker", "")
        if not isinstance(t, str) or not _TICKER_RE.match(t):
            row["_validation_failed"] = True
            issues.append(ValidationIssue(
                ticker=str(t), field="ticker",
                code="INVALID_TICKER", original_value=t,
            ))
    return (len(issues) == 0, issues)
```

- [ ] **Step 4: Run tests to verify they pass**

```
python -m pytest tests/test_validator.py::TestValidation -k "ticker" -v
```

Expected: `5 passed`

- [ ] **Step 5: Commit**

```
git add backend/market_intel/validator.py tests/test_validator.py
git commit -m "feat(validator): ValidationIssue dataclass + validate_tickers"
```

---

## Task 3: `validate_amounts`

**Files:**
- Modify: `backend/market_intel/validator.py`
- Modify: `tests/test_validator.py`

- [ ] **Step 1: Write the failing tests**

Append to `TestValidation` in `tests/test_validator.py`:

```python
    def test_zero_insider_usd_sets_nan_and_fails(self):
        from backend.market_intel.validator import validate_amounts
        row = _row(insider_usd=0.0)
        ok, issues = validate_amounts([row])
        assert ok is False
        assert math.isnan(row["insider_usd"])
        assert row.get("_validation_failed") is True

    def test_negative_market_cap_sets_nan_and_fails(self):
        from backend.market_intel.validator import validate_amounts
        row = _row(market_cap=-1.0)
        ok, issues = validate_amounts([row])
        assert ok is False
        assert math.isnan(row["market_cap"])
        assert row.get("_validation_failed") is True

    def test_none_insider_usd_sets_nan(self):
        from backend.market_intel.validator import validate_amounts
        row = _row()
        row["insider_usd"] = None
        ok, issues = validate_amounts([row])
        assert ok is False
        assert math.isnan(row["insider_usd"])

    def test_valid_amounts_pass(self):
        from backend.market_intel.validator import validate_amounts
        row = _row(insider_usd=50_000.0, market_cap=3e12)
        ok, issues = validate_amounts([row])
        assert ok is True
        assert issues == []
```

- [ ] **Step 2: Run tests to verify they fail**

```
python -m pytest tests/test_validator.py::TestValidation -k "amount" -v
```

Expected: `ERROR` — `ImportError: cannot import name 'validate_amounts'`

- [ ] **Step 3: Add `validate_amounts` to `validator.py`**

```python
# ── validate_amounts ──────────────────────────────────────────────────────────

_AMOUNT_FIELDS = ("insider_usd", "market_cap")


def validate_amounts(rows: List[Dict[str, Any]]) -> Tuple[bool, List[ValidationIssue]]:
    """Check insider_usd and market_cap are positive finite numbers.

    Invalid values are set to float("nan") in-place.
    Row is tagged _validation_failed=True on any failure.
    Never raises.
    """
    issues: List[ValidationIssue] = []
    for row in rows:
        ticker = row.get("ticker", "?")
        for field in _AMOUNT_FIELDS:
            val = row.get(field)
            bad = False
            if val is None:
                bad = True
            else:
                try:
                    fval = float(val)
                    if math.isnan(fval) or math.isinf(fval) or fval <= 0:
                        bad = True
                except (TypeError, ValueError):
                    bad = True
            if bad:
                row[field] = float("nan")
                row["_validation_failed"] = True
                issues.append(ValidationIssue(
                    ticker=ticker, field=field,
                    code="MISSING_AMOUNT", original_value=val,
                ))
    return (len(issues) == 0, issues)
```

- [ ] **Step 4: Run tests to verify they pass**

```
python -m pytest tests/test_validator.py::TestValidation -k "amount" -v
```

Expected: `4 passed`

- [ ] **Step 5: Commit**

```
git add backend/market_intel/validator.py tests/test_validator.py
git commit -m "feat(validator): validate_amounts — NaN quarantine for bad financials"
```

---

## Task 4: `validate_dates` — row-level + source-level staleness

**Files:**
- Modify: `backend/market_intel/validator.py`
- Modify: `tests/test_validator.py`

- [ ] **Step 1: Write the failing tests**

Append to `TestValidation`:

```python
    def test_unparseable_date_quarantines_ticker(self):
        from backend.market_intel.validator import validate_dates
        row = _row()
        row["computed_at"] = "not-a-date"
        ok, issues = validate_dates([row], _source_meta())
        assert ok is False
        assert row.get("_validation_failed") is True
        assert any(i.code == "INVALID_DATE" for i in issues)

    def test_future_date_quarantines_ticker(self):
        from backend.market_intel.validator import validate_dates
        row = _row(timestamp=_future_iso(hours=2))
        ok, issues = validate_dates([row], _source_meta())
        assert ok is False
        assert row.get("_validation_failed") is True
        assert any(i.code == "FUTURE_DATE" for i in issues)

    def test_stale_row_flag_only(self):
        from backend.market_intel.validator import validate_dates
        row = _row(timestamp=_ago_iso(days=6))
        ok, issues = validate_dates([row], _source_meta(), max_age_days=5)
        # flag_only — row stays, but _stale_data=True, NOT _validation_failed
        assert row.get("_stale_data") is True
        assert row.get("_validation_failed") is not True

    def test_fresh_row_passes(self):
        from backend.market_intel.validator import validate_dates
        row = _row(timestamp=_ago_iso(hours=1))
        ok, issues = validate_dates([row], _source_meta())
        assert ok is True
        assert issues == []
        assert row.get("_validation_failed") is not True

    def test_stale_source_quarantines_all_source_rows(self):
        from backend.market_intel.validator import validate_dates
        rows = [_row(source="quiver") for _ in range(3)]
        meta = _source_meta(quiver_age_hours=61)  # > 48h threshold
        ok, issues = validate_dates(rows, meta, source_stale_hours=48)
        assert ok is False
        for row in rows:
            assert row.get("_validation_failed") is True
            assert row.get("_stale_source") is True
        assert any(i.code == "STALE_SOURCE" for i in issues)

    def test_fresh_source_does_not_quarantine(self):
        from backend.market_intel.validator import validate_dates
        rows = [_row(source="quiver") for _ in range(3)]
        meta = _source_meta(quiver_age_hours=2)   # well within 48h
        ok, issues = validate_dates(rows, meta, source_stale_hours=48)
        assert ok is True
        for row in rows:
            assert row.get("_validation_failed") is not True
```

- [ ] **Step 2: Run tests to verify they fail**

```
python -m pytest tests/test_validator.py::TestValidation -k "date or stale or fresh" -v
```

Expected: `ERROR` — `ImportError: cannot import name 'validate_dates'`

- [ ] **Step 3: Add `validate_dates` to `validator.py`**

```python
# ── validate_dates ────────────────────────────────────────────────────────────

_CLOCK_SKEW_SECONDS = 60.0


def _parse_dt(ts: Any) -> Optional[datetime]:
    """Parse ISO-8601 string to UTC-aware datetime. Returns None on failure."""
    if not isinstance(ts, str):
        return None
    try:
        from dateutil.parser import isoparse
        dt = isoparse(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def validate_dates(
    rows: List[Dict[str, Any]],
    source_meta: Dict[str, Dict[str, Any]],
    max_age_days: int = 5,
    source_stale_hours: float = 48.0,
) -> Tuple[bool, List[ValidationIssue]]:
    """Check computed_at timestamps and source freshness.

    Row-level checks (per ticker):
      - INVALID_DATE:  unparseable → _validation_failed=True
      - FUTURE_DATE:   > now + 60s → _validation_failed=True
      - STALE_DATA:    age > max_age_days → _stale_data=True (flag_only, NOT failed)

    Source-level check (quarantines all rows from that source):
      - STALE_SOURCE:  source last_updated > source_stale_hours → _validation_failed=True

    Never raises.
    """
    issues: List[ValidationIssue] = []
    now = datetime.now(timezone.utc)

    # ── Source-level staleness check first ─────────────────────────────────────
    stale_sources: set[str] = set()
    for source_name, meta in source_meta.items():
        lu_str = meta.get("last_updated")
        lu_dt = _parse_dt(lu_str)
        if lu_dt is None:
            continue
        age_h = (now - lu_dt).total_seconds() / 3600.0
        if age_h > source_stale_hours:
            stale_sources.add(source_name)
            issues.append(ValidationIssue(
                ticker="__SOURCE__", field="last_updated",
                code="STALE_SOURCE",
                original_value=f"{source_name} age={age_h:.1f}h",
            ))

    # ── Per-row checks ─────────────────────────────────────────────────────────
    for row in rows:
        ticker = row.get("ticker", "?")

        # Source quarantine
        row_source = row.get("insider_source", "")
        if row_source in stale_sources:
            row["_validation_failed"] = True
            row["_stale_source"] = True
            continue  # no need to check individual timestamps

        ts_str = row.get("computed_at")
        dt = _parse_dt(ts_str)

        if dt is None:
            row["_validation_failed"] = True
            issues.append(ValidationIssue(
                ticker=ticker, field="computed_at",
                code="INVALID_DATE", original_value=ts_str,
            ))
            continue

        if dt > now + __import__("datetime").timedelta(seconds=_CLOCK_SKEW_SECONDS):
            row["_validation_failed"] = True
            issues.append(ValidationIssue(
                ticker=ticker, field="computed_at",
                code="FUTURE_DATE", original_value=ts_str,
            ))
            continue

        age_days = (now - dt).total_seconds() / 86400.0
        if age_days > max_age_days:
            row["_stale_data"] = True   # flag_only — not _validation_failed
            issues.append(ValidationIssue(
                ticker=ticker, field="computed_at",
                code="STALE_DATA", original_value=ts_str,
            ))

    all_ok = all(i.code not in ("INVALID_DATE", "FUTURE_DATE", "STALE_SOURCE") for i in issues)
    return (all_ok, issues)
```

- [ ] **Step 4: Run tests to verify they pass**

```
python -m pytest tests/test_validator.py::TestValidation -k "date or stale or fresh" -v
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```
git add backend/market_intel/validator.py tests/test_validator.py
git commit -m "feat(validator): validate_dates — row + source-level staleness checks"
```

---

## Task 5: `validate_raw` — orchestrator + `PipelineIntegrityError`

**Files:**
- Modify: `backend/market_intel/validator.py`
- Modify: `tests/test_validator.py`

- [ ] **Step 1: Write the failing tests**

Append to `TestValidation`:

```python
    def test_failure_threshold_raises_integrity_error(self):
        from backend.market_intel.validator import validate_raw
        # All 5 tickers have zero market_cap → all fail
        rows = [_row(f"T{i}", market_cap=0.0) for i in range(5)]
        with pytest.raises(Exception) as exc_info:
            validate_raw(rows, _source_meta(), failure_threshold=0.20)
        assert "failed validation" in str(exc_info.value).lower()

    def test_integrity_error_message_contains_summary(self):
        from backend.market_intel.validator import validate_raw
        rows = [_row(f"T{i}", market_cap=0.0) for i in range(5)]
        with pytest.raises(Exception) as exc_info:
            validate_raw(rows, _source_meta(), failure_threshold=0.20)
        msg = str(exc_info.value)
        assert "MISSING_AMOUNT" in msg
        assert "tickers" in msg.lower()

    def test_below_threshold_continues_on_clean_subset(self):
        from backend.market_intel.validator import validate_raw
        rows = [_row("GOOD")] * 4 + [_row("BAD", market_cap=0.0)]
        clean, quarantined, issues = validate_raw(rows, _source_meta(), failure_threshold=0.30)
        assert len(clean) == 4
        assert len(quarantined) == 1
        assert quarantined[0]["ticker"] == "BAD"

    def test_all_clean_returns_all_rows(self):
        from backend.market_intel.validator import validate_raw
        rows = [_row(f"T{i}") for i in range(5)]
        clean, quarantined, issues = validate_raw(rows, _source_meta())
        assert len(clean) == 5
        assert quarantined == []
```

- [ ] **Step 2: Run tests to verify they fail**

```
python -m pytest tests/test_validator.py::TestValidation -k "threshold or integrity or clean_subset or all_clean" -v
```

Expected: `ERROR` — `ImportError: cannot import name 'validate_raw'`

- [ ] **Step 3: Add `validate_raw` to `validator.py`**

```python
# ── validate_raw ──────────────────────────────────────────────────────────────

def validate_raw(
    rows: List[Dict[str, Any]],
    source_meta: Dict[str, Dict[str, Any]],
    failure_threshold: float = 0.20,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[ValidationIssue]]:
    """Run all Stage 1 validators. Raise PipelineIntegrityError if too many fail.

    Returns (clean_rows, quarantined_rows, all_issues).
    Raises PipelineIntegrityError with structured summary if failed/total > threshold.
    """
    all_issues: List[ValidationIssue] = []

    _, issues_t = validate_tickers(rows)
    _, issues_a = validate_amounts(rows)
    _, issues_d = validate_dates(rows, source_meta)
    all_issues.extend(issues_t)
    all_issues.extend(issues_a)
    all_issues.extend(issues_d)

    failed_rows  = [r for r in rows if r.get("_validation_failed")]
    clean_rows   = [r for r in rows if not r.get("_validation_failed")]
    total        = len(rows)
    failed_count = len(failed_rows)

    if total > 0 and (failed_count / total) > failure_threshold:
        # Build human-readable summary grouped by issue code
        counts: Dict[str, int] = defaultdict(int)
        for issue in all_issues:
            if issue.code != "STALE_DATA":  # STALE_DATA is flag_only, not failure
                counts[issue.code] += 1

        summary_lines = [
            f"  {code:<20} {count} tickers"
            for code, count in sorted(counts.items(), key=lambda x: -x[1])
        ]
        pct = failed_count / total * 100
        msg = (
            f"{failed_count}/{total} tickers failed validation "
            f"({pct:.1f}% > {failure_threshold * 100:.1f}% threshold)\n"
            + "\n".join(summary_lines)
            + "\nAborting pipeline to prevent degenerate top_lists.json."
        )
        raise PipelineIntegrityError(msg)

    return (clean_rows, failed_rows, all_issues)
```

- [ ] **Step 4: Run all validation tests**

```
python -m pytest tests/test_validator.py::TestValidation -v
```

Expected: All `TestValidation` tests pass (≥15 tests)

- [ ] **Step 5: Commit**

```
git add backend/market_intel/validator.py tests/test_validator.py
git commit -m "feat(validator): validate_raw orchestrator with PipelineIntegrityError summary"
```

---

## Task 6: `detect_anomalies` — circuit breakers + JSON audit report

**Files:**
- Modify: `backend/market_intel/validator.py`
- Modify: `tests/test_validator.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_validator.py`:

```python
# ── TestAnomalyDetection ───────────────────────────────────────────────────────

class TestAnomalyDetection:
    def _detect(self, rows, run_id="test-run", tmp_path=None):
        from backend.market_intel.validator import detect_anomalies
        if tmp_path is None:
            import tempfile, pathlib
            tmp_path = pathlib.Path(tempfile.mkdtemp())
        return detect_anomalies(rows, run_id=run_id, log_dir=tmp_path), tmp_path

    def test_volume_spike_flagged(self):
        # mean = 5.0, spike = 150 → 30× mean > 10× threshold
        rows = [_row(f"T{i}", volume_spike=5.0) for i in range(4)]
        rows.append(_row("SPIKE", volume_spike=150.0))
        records, _ = self._detect(rows)
        assert any(r["ticker"] == "SPIKE" and r["flag"] == "VOLUME_SPIKE" for r in records)

    def test_volume_below_threshold_not_flagged(self):
        rows = [_row(f"T{i}", volume_spike=5.0) for i in range(5)]
        records, _ = self._detect(rows)
        assert not any(r["flag"] == "VOLUME_SPIKE" for r in records)

    def test_insider_cap_limit_large(self):
        # 0.9% > 0.5% large ceiling
        cap = 1e12
        row = _row("BA", insider_usd=cap * 0.009, market_cap=cap, cap_tier="large")
        records, _ = self._detect([row])
        assert any(r["ticker"] == "BA" and r["flag"] == "INSIDER_CAP_LIMIT" for r in records)

    def test_insider_cap_limit_small_not_triggered(self):
        # 0.9% < 2.0% small ceiling → no flag
        cap = 50_000_000.0
        row = _row("SMALL", insider_usd=cap * 0.009, market_cap=cap, cap_tier="small")
        records, _ = self._detect([row])
        assert not any(r["ticker"] == "SMALL" and r["flag"] == "INSIDER_CAP_LIMIT" for r in records)

    def test_sentiment_extreme_high(self):
        row = _row("HYPE", news_score=0.97)
        records, _ = self._detect([row])
        assert any(r["ticker"] == "HYPE" and r["flag"] == "SENTIMENT_EXTREME" for r in records)

    def test_sentiment_extreme_low(self):
        row = _row("DOOM", news_score=0.02)
        records, _ = self._detect([row])
        assert any(r["ticker"] == "DOOM" and r["flag"] == "SENTIMENT_EXTREME" for r in records)

    def test_normal_sentiment_not_flagged(self):
        row = _row("NORM", news_score=0.65)
        records, _ = self._detect([row])
        assert not any(r["flag"] == "SENTIMENT_EXTREME" for r in records)

    def test_anomaly_report_written_to_disk(self, tmp_path):
        rows = [_row("BA", insider_usd=1e12 * 0.009, market_cap=1e12, cap_tier="large")]
        records, _ = self._detect(rows, run_id="r123", tmp_path=tmp_path)
        assert (tmp_path / "anomaly_report_r123.json").exists()

    def test_anomaly_report_latest_written(self, tmp_path):
        rows = [_row("BA", insider_usd=1e12 * 0.009, market_cap=1e12, cap_tier="large")]
        self._detect(rows, run_id="r456", tmp_path=tmp_path)
        assert (tmp_path / "anomaly_report_latest.json").exists()

    def test_anomaly_report_schema_valid(self, tmp_path):
        rows = [_row("BA", insider_usd=1e12 * 0.009, market_cap=1e12, cap_tier="large")]
        records, _ = self._detect(rows, run_id="r789", tmp_path=tmp_path)
        for record in records:
            for key in ("run_id", "ticker", "timestamp", "flag", "value", "threshold", "action", "source"):
                assert key in record, f"missing key {key} in {record}"

    def test_no_file_written_when_no_anomalies(self, tmp_path):
        rows = [_row(f"T{i}") for i in range(3)]
        records, _ = self._detect(rows, run_id="clean", tmp_path=tmp_path)
        assert records == []
        assert not (tmp_path / "anomaly_report_clean.json").exists()
        assert not (tmp_path / "anomaly_report_latest.json").exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```
python -m pytest tests/test_validator.py::TestAnomalyDetection -v
```

Expected: `ERROR` — `ImportError: cannot import name 'detect_anomalies'`

- [ ] **Step 3: Add `AnomalyRecord` + `detect_anomalies` to `validator.py`**

```python
# ── AnomalyRecord + detect_anomalies ──────────────────────────────────────────

def _write_atomic(path: Path, data: Any) -> None:
    """Write JSON atomically using os.replace() — safe for concurrent readers."""
    payload = json.dumps(data, indent=2, ensure_ascii=False).encode("utf-8")
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(payload)
            fh.flush()
            try:
                os.fsync(fh.fileno())
            except OSError:
                pass
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def detect_anomalies(
    rows: List[Dict[str, Any]],
    run_id: str,
    log_dir: Path,
) -> List[Dict[str, Any]]:
    """Stage 2 circuit breakers — run after normalization, before final ranking.

    Checks:
      VOLUME_SPIKE       — volume_spike > 10× universe mean
      INSIDER_CAP_LIMIT  — insider_usd / market_cap > tier ceiling
      SENTIMENT_EXTREME  — news_score > 0.95 or < 0.05
      STALE_DATA         — _stale_data flag set by validate_dates (flag_only)

    Writes anomaly_report_{run_id}.json and anomaly_report_latest.json.
    Returns list of AnomalyRecord dicts (empty = no anomalies, no file written).
    """
    now_str = datetime.now(timezone.utc).isoformat()
    records: List[Dict[str, Any]] = []

    def _record(ticker: str, flag: str, value: float, threshold: float,
                action: str, source: str = "") -> Dict[str, Any]:
        return {
            "run_id":    run_id,
            "ticker":    ticker,
            "timestamp": now_str,
            "flag":      flag,
            "value":     round(value, 6),
            "threshold": round(threshold, 6),
            "action":    action,
            "source":    source,
        }

    # ── VOLUME_SPIKE ──────────────────────────────────────────────────────────
    vol_values = [float(r.get("volume_spike") or 0) for r in rows]
    universe_mean_vol = float(np.mean(vol_values)) if vol_values else 0.0
    spike_threshold = universe_mean_vol * 10.0
    for row in rows:
        v = float(row.get("volume_spike") or 0)
        if universe_mean_vol > 0 and v > spike_threshold:
            records.append(_record(
                row.get("ticker", "?"), "VOLUME_SPIKE",
                value=v, threshold=spike_threshold, action="flag_only",
                source=row.get("insider_source", ""),
            ))

    # ── INSIDER_CAP_LIMIT ─────────────────────────────────────────────────────
    for row in rows:
        usd = row.get("insider_usd")
        cap = row.get("market_cap")
        tier = row.get("cap_tier", "large")
        if not usd or not cap or math.isnan(float(usd)) or math.isnan(float(cap)):
            continue
        ceiling = _TIER_CEILING.get(tier, _TIER_CEILING["large"])
        ratio = float(usd) / float(cap)
        if ratio > ceiling:
            records.append(_record(
                row.get("ticker", "?"), "INSIDER_CAP_LIMIT",
                value=ratio, threshold=ceiling, action="flag_only",
                source=row.get("insider_source", ""),
            ))

    # ── SENTIMENT_EXTREME ─────────────────────────────────────────────────────
    for row in rows:
        ns = row.get("news_score")
        if ns is None:
            continue
        v = float(ns)
        if v > 0.95:
            records.append(_record(
                row.get("ticker", "?"), "SENTIMENT_EXTREME",
                value=v, threshold=0.95, action="flag_only",
            ))
        elif v < 0.05:
            records.append(_record(
                row.get("ticker", "?"), "SENTIMENT_EXTREME",
                value=v, threshold=0.05, action="flag_only",
            ))

    # ── STALE_DATA (propagate flag from validate_dates) ───────────────────────
    for row in rows:
        if row.get("_stale_data"):
            records.append(_record(
                row.get("ticker", "?"), "STALE_DATA",
                value=0.0, threshold=0.0, action="flag_only",
                source=row.get("insider_source", ""),
            ))

    if not records:
        return []

    # ── Write audit files ──────────────────────────────────────────────────────
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    _write_atomic(log_dir / f"anomaly_report_{run_id}.json", records)
    _write_atomic(log_dir / "anomaly_report_latest.json", records)

    log.info("detect_anomalies: %d anomaly record(s) written for run %s", len(records), run_id)
    return records
```

- [ ] **Step 4: Run all anomaly detection tests**

```
python -m pytest tests/test_validator.py::TestAnomalyDetection -v
```

Expected: `11 passed`

- [ ] **Step 5: Run full test_validator.py**

```
python -m pytest tests/test_validator.py -v
```

Expected: All tests pass (≥28 tests, 0 failures)

- [ ] **Step 6: Commit**

```
git add backend/market_intel/validator.py tests/test_validator.py
git commit -m "feat(validator): detect_anomalies — circuit breakers + atomic JSON audit report"
```

---

## Task 7: Integrate `validate_raw` into `generate_top_lists.py`

**Files:**
- Modify: `backend/market_intel/generate_top_lists.py`

`generate_top_lists.py` receives the list of scored rows. We add one filter at the point where `top_buys` is assembled: exclude any row with `_validation_failed=True`.

- [ ] **Step 1: Find the top_buys assembly line**

Open `backend/market_intel/generate_top_lists.py`. Search for where `top_buys` is built — it's in the `generate()` function, around line 335–360. It looks like:

```python
top_buys = sorted(large_caps, key=lambda r: r["final_score"], reverse=True)[:5]
```

- [ ] **Step 2: Add the validation filter**

Immediately before the `top_buys` assignment, add:

```python
# Exclude tickers that failed Stage 1 validation — prevents false-positive buy signals
large_caps  = [r for r in large_caps  if not r.get("_validation_failed")]
mid_caps    = [r for r in mid_caps    if not r.get("_validation_failed")]
small_caps  = [r for r in small_caps  if not r.get("_validation_failed")]
```

- [ ] **Step 3: Run existing tests to check no regression**

```
python -m pytest tests/test_pipeline_integrity.py -v
```

Expected: All 28 existing tests still pass.

- [ ] **Step 4: Commit**

```
git add backend/market_intel/generate_top_lists.py
git commit -m "feat(generate_top_lists): exclude _validation_failed tickers from top_buys"
```

---

## Task 8: Discord integration — ⚠️ on flagged tickers

**Files:**
- Modify: `scripts/send_toplists_discord.py`

- [ ] **Step 1: Add `_load_anomaly_report` helper**

In `scripts/send_toplists_discord.py`, after the existing `_load_satellite` function, add:

```python
def _load_anomaly_report(log_dir: Path) -> Dict[str, List[str]]:
    """Load anomaly_report_latest.json. Returns {ticker: [flags]} dict.

    Returns empty dict on any failure — anomaly display is best-effort.
    """
    path = log_dir / "anomaly_report_latest.json"
    if not path.exists():
        return {}
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            return {}
        result: Dict[str, List[str]] = {}
        for rec in records:
            if not isinstance(rec, dict):
                continue
            ticker = rec.get("ticker", "")
            flag   = rec.get("flag", "")
            if ticker and flag and ticker != "__SOURCE__":
                result.setdefault(ticker, []).append(flag)
        return result
    except Exception as exc:
        log.warning("anomaly_report_latest.json unreadable: %s", exc)
        return {}
```

- [ ] **Step 2: Pass anomaly flags into `_ticker_detail_field`**

In `_ticker_detail_field(rank, entry)`, add an optional `anomaly_flags` parameter:

```python
def _ticker_detail_field(
    rank: int,
    entry: Dict[str, Any],
    anomaly_flags: Optional[List[str]] = None,
) -> Dict[str, Any]:
```

In the first line of the field value, append ` ⚠️` if `anomaly_flags` is non-empty:

```python
    flag_tag = "  ⚠️" if anomaly_flags else ""
    lines = [
        f"{medal} **{ticker}**{ceo}{flag_tag}  —  *{badge}*  —  `{score:.2f}`  `{bar}`",
        ...
    ]
```

- [ ] **Step 3: Pass anomaly map to `_ticker_fields`**

Update `_ticker_fields`:

```python
def _ticker_fields(
    entries: List[Dict],
    anomaly_map: Optional[Dict[str, List[str]]] = None,
) -> List[Dict[str, Any]]:
    anomaly_map = anomaly_map or {}
    return [
        _ticker_detail_field(i, e, anomaly_flags=anomaly_map.get(e.get("ticker", "")))
        for i, e in enumerate(entries[:5], 1)
    ]
```

- [ ] **Step 4: Load anomaly report in `build_payload` and pass it through**

In `build_payload`, the anomaly report isn't available because it's log_dir-scoped. Instead, accept an optional `anomaly_map` parameter:

```python
def build_payload(
    top_lists: Dict[str, Any],
    satellite: Optional[Dict[str, Any]] = None,
    anomaly_map: Optional[Dict[str, List[str]]] = None,
) -> Dict[str, Any]:
```

In the fields section:
```python
    if top_buys:
        fields.append(_top_conviction_field(top_buys))
        fields.extend(_ticker_fields(top_buys, anomaly_map=anomaly_map))
```

- [ ] **Step 5: Load and pass anomaly_map in `main()`**

In the `main()` function, after loading satellite:

```python
    satellite    = _load_satellite(args.log_dir)
    anomaly_map  = _load_anomaly_report(args.log_dir)
    payload      = build_payload(top_lists, satellite=satellite, anomaly_map=anomaly_map)
```

- [ ] **Step 6: Add STALE_SOURCE alert to description**

In `build_payload`, after the existing alert block construction, add:

```python
    # Stale source alert from anomaly report
    if anomaly_map:
        stale_sources = [
            v for flags in anomaly_map.values() for v in flags if v == "STALE_SOURCE"
        ]
        if stale_sources:
            alerts.append("⚠️  STALE DATA SOURCE detected — scores may be unreliable. Check anomaly_report_latest.json.")
```

- [ ] **Step 7: Run Discord tests**

```
python -m pytest tests/test_send_toplists_discord.py tests/test_discord_formatter.py -v
```

Expected: All 25 tests pass.

- [ ] **Step 8: Commit**

```
git add scripts/send_toplists_discord.py
git commit -m "feat(discord): load anomaly_report_latest.json, append ⚠️ to flagged tickers"
```

---

## Task 9: Final — run full test suite + push

- [ ] **Step 1: Run all tests**

```
python -m pytest tests/ -v --timeout=60 2>&1 | tail -30
```

Expected: All tests pass. Zero failures.

- [ ] **Step 2: Run ruff lint**

```
python -m ruff check backend/market_intel/validator.py tests/test_validator.py --select E,F,W --ignore E501
```

Expected: No violations.

- [ ] **Step 3: Push**

```
git push
```

---

## Self-Review

**Spec coverage check:**

| Spec requirement | Task |
|-----------------|------|
| `validate_dates()` with staleness | Task 4 |
| `validate_amounts()` zero/negative → NaN | Task 3 |
| `validate_tickers()` regex | Task 2 |
| `validate_raw()` failure_threshold + structured error | Task 5 |
| `Normalizer.winsorize` | Task 1 |
| `Normalizer.log_scale_insider` with tiers | Task 1 |
| `Normalizer.cross_sectional_norm` | Task 1 |
| `detect_anomalies` 4 circuit breakers | Task 6 |
| `AnomalyRecord` schema | Task 6 |
| `anomaly_report_{run_id}.json` per-run | Task 6 |
| `anomaly_report_latest.json` atomic write | Task 6 |
| `os.replace()` for atomic writes | Task 6 |
| `Literal["small","mid","large"]` type | Task 1 |
| `generate_top_lists.py` excludes `_validation_failed` | Task 7 |
| Discord ⚠️ on flagged tickers | Task 8 |
| `STALE_SOURCE` quarantines all source rows | Task 4 |
| `PipelineIntegrityError` structured summary | Task 5 |
| No external deps beyond pandas/numpy | All tasks |
| Full type annotations | All tasks |
| No imports from `scripts/` in validator.py | Task 1 |
| CI isolation — no HTTP in tests | All tasks (in-memory fixtures) |

**Placeholder scan:** No TBDs, no "implement later", no "similar to task N". All code blocks are complete. ✓

**Type consistency:** `ValidationIssue` defined in Task 2, used in Tasks 3–5. `AnomalyRecord` is a plain `Dict[str, Any]` throughout — consistent. `Tuple[bool, List[ValidationIssue]]` return type consistent across all validators. ✓
