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
