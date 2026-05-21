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
        assert result.max() <= 0.1 + 1e-6
        assert result.min() >= 0.1 - 1e-6

    def test_cross_sectional_norm_mean_near_neutral(self):
        import numpy as np
        from backend.market_intel.validator import Normalizer
        series = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        result = Normalizer.cross_sectional_norm(series)
        # min-max scaled to [0,1] — mean ≈ 0.5 for symmetric input
        assert abs(float(result.mean()) - 0.5) < 0.1


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
        rows = [_row("GOOD") for _ in range(4)] + [_row("BAD", market_cap=0.0)]
        clean, quarantined, issues = validate_raw(rows, _source_meta(), failure_threshold=0.30)
        assert len(clean) == 4
        assert len(quarantined) == 1
        assert quarantined[0]["ticker"] == "BAD"

    def test_all_clean_returns_all_rows(self):
        from backend.market_intel.validator import validate_raw
        tickers = ["AAPL", "MSFT", "TSLA", "GOOG", "AMZN"]
        rows = [_row(ticker) for ticker in tickers]
        clean, quarantined, issues = validate_raw(rows, _source_meta())
        assert len(clean) == 5
        assert quarantined == []


# ── TestAnomalyDetection ───────────────────────────────────────────────────────

class TestAnomalyDetection:
    def _detect(self, rows, run_id="test-run", tmp_path=None):
        from backend.market_intel.validator import detect_anomalies
        if tmp_path is None:
            import tempfile, pathlib
            tmp_path = pathlib.Path(tempfile.mkdtemp())
        return detect_anomalies(rows, run_id=run_id, log_dir=tmp_path), tmp_path

    def test_volume_spike_flagged(self):
        # 19 rows at 1.0, spike at 200 → mean=10.95, threshold=109.5 < 200
        rows = [_row(f"A{i}", volume_spike=1.0) for i in range(19)]
        rows.append(_row("SPIKE", volume_spike=200.0))
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
