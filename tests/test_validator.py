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
