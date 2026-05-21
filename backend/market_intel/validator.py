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
import re as _re
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
        limits: Tuple[float, float] = (0.02, 0.98),
    ) -> np.ndarray:
        """Winsorize series at [lo, hi] fractional limits (0.02 = 2nd pct).

        The underlying winsorize() in normalize.py uses lo/hi as percentile
        values on the 0–100 scale, so fractional limits are multiplied by 100.
        Default of (0.02, 0.98) clips the top/bottom 2% — sufficient to cap
        the 1% tail outliers in typical ~100-row pipeline batches.
        """
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


# ── Validation data structures ────────────────────────────────────────────────

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
      - INVALID_DATE:  unparseable -> _validation_failed=True
      - FUTURE_DATE:   > now + 60s -> _validation_failed=True
      - STALE_DATA:    age > max_age_days -> _stale_data=True (flag_only, NOT failed)

    Source-level check (quarantines all rows from that source):
      - STALE_SOURCE:  source last_updated > source_stale_hours -> _validation_failed=True

    Never raises.
    """
    from datetime import timedelta
    issues: List[ValidationIssue] = []
    now = datetime.now(timezone.utc)

    # ── Source-level staleness check first ─────────────────────────────────────
    stale_sources: set = set()
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

        if dt > now + timedelta(seconds=_CLOCK_SKEW_SECONDS):
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
        counts: Dict[str, int] = defaultdict(int)
        for issue in all_issues:
            if issue.code != "STALE_DATA":
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
        if not usd or not cap:
            continue
        try:
            usd_f = float(usd)
            cap_f = float(cap)
        except (TypeError, ValueError):
            continue
        if math.isnan(usd_f) or math.isnan(cap_f) or cap_f <= 0:
            continue
        ceiling = _TIER_CEILING.get(tier, _TIER_CEILING["large"])
        ratio = usd_f / cap_f
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
