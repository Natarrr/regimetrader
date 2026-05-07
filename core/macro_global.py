"""core/macro_global.py
Global macro momentum engine — FRED via pandas_datareader.

Fetches GDP, PMI, Jobless Claims and Consumer Confidence for US / EU / Asia
and computes 3-month and 6-month rolling Z-scores (Momentum Macro).
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_LOOKBACK_DAYS = 3 * 365

# FRED series IDs per zone/indicator
_SERIES: Dict[str, Dict[str, str]] = {
    "US": {
        "GDP":      "GDPC1",            # Real GDP, Quarterly
        "PMI":      "BSCICP03USM665S",  # OECD Business Confidence US
        "Claims":   "ICSA",             # Initial Jobless Claims, Weekly
        "ConsConf": "UMCSENT",          # U of Michigan Consumer Sentiment
    },
    "EU": {
        "GDP":      "CLVMEURSCAB1GQEA19",  # EA19 Real GDP, Quarterly
        "PMI":      "BSCICP03EZM665S",     # OECD Business Confidence EA
        "Unemp":    "LRHUTTTTEZM156S",     # Harmonised Unemployment Rate EA
        "ConsConf": "CSCICP03EZM665S",     # OECD Consumer Confidence EA
    },
    "Asia": {
        "Production": "JPNPROINDMISMEI",   # Japan Industrial Production
        "Unemp":      "LRUN64TTJPM156S",   # Japan Unemployment Rate
        "ConsConf":   "CSCICP03JPM665S",   # OECD Consumer Confidence Japan
        "Leading":    "JPNLOLITONOSTSAM",  # Japan OECD Leading Indicator
    },
}


class GlobalMacroEngine:
    """
    Fetches FRED macro series and computes 3M/6M momentum Z-scores.

    Usage
    -----
        engine = GlobalMacroEngine()
        data = engine.fetch_all()
        # data["US"]["GDP"] → {"latest": float, "z3": float, "z6": float,
        #                       "z_composite": float, "trend": str, "series": str}
    """

    def __init__(self) -> None:
        self._cache: Dict[str, Optional[pd.Series]] = {}

    # ── FRED fetch ─────────────────────────────────────────────────────────────

    def _fetch_fred(self, series_id: str) -> Optional[pd.Series]:
        if series_id in self._cache:
            return self._cache[series_id]
        try:
            # Use FRED public CSV endpoint directly — avoids pandas_datareader
            # which is broken on Python 3.14 (deprecate_kwarg signature mismatch).
            import requests
            from io import StringIO
            start = (datetime.today() - timedelta(days=_LOOKBACK_DAYS)).strftime("%Y-%m-%d")
            resp = requests.get(
                "https://fred.stlouisfed.org/graph/fredgraph.csv",
                params={"id": series_id},
                timeout=20,
            )
            resp.raise_for_status()
            df = pd.read_csv(StringIO(resp.text), parse_dates=["DATE"], index_col="DATE")
            df.columns = [series_id]
            df = df.replace(".", float("nan")).astype(float).dropna()
            s = df.loc[df.index >= start, series_id].dropna()
            self._cache[series_id] = s if len(s) >= 4 else None
        except Exception as exc:
            logger.warning("[MACRO] FRED %s failed: %s", series_id, exc)
            self._cache[series_id] = None
        return self._cache[series_id]

    # ── Momentum Z-score ───────────────────────────────────────────────────────

    @staticmethod
    def _momentum(series: Optional[pd.Series]) -> Dict[str, Any]:
        """
        Compute latest value, 3M/6M Z-scores, and directional trend.

        Resamples to monthly (end-of-month), computes rolling % changes,
        then standardises to Z-scores relative to the full available history.
        """
        if series is None or len(series) < 4:
            return {"latest": None, "z3": 0.0, "z6": 0.0, "z_composite": 0.0, "trend": "n/a"}

        monthly = series.resample("ME").last().dropna()
        if len(monthly) < 4:
            return {
                "latest": float(monthly.iloc[-1]) if len(monthly) else None,
                "z3": 0.0, "z6": 0.0, "z_composite": 0.0, "trend": "neutral",
            }

        latest = float(monthly.iloc[-1])

        def _z(chg: pd.Series) -> float:
            if len(chg) < 3:
                return 0.0
            sigma = chg.std()
            return 0.0 if sigma < 1e-10 else float((chg.iloc[-1] - chg.mean()) / sigma)

        z3 = _z(monthly.pct_change(3).dropna())
        z6 = _z(monthly.pct_change(6).dropna())
        zc = round(0.6 * z3 + 0.4 * z6, 3)
        trend = "expanding" if zc > 0.5 else "contracting" if zc < -0.5 else "neutral"
        return {
            "latest":      latest,
            "z3":          round(z3, 3),
            "z6":          round(z6, 3),
            "z_composite": zc,
            "trend":       trend,
        }

    # ── Public API ─────────────────────────────────────────────────────────────

    def fetch_zone(self, zone: str) -> Dict[str, Dict[str, Any]]:
        """Fetch all indicators for one zone (US / EU / Asia)."""
        return {
            ind: {**self._momentum(self._fetch_fred(sid)), "series": sid}
            for ind, sid in _SERIES.get(zone, {}).items()
        }

    def fetch_all(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Fetch all zones. Returns zone → indicator → metrics dict."""
        return {zone: self.fetch_zone(zone) for zone in _SERIES}

    def zone_score(self, zone_data: Dict[str, Dict[str, Any]]) -> float:
        """
        Aggregate composite Z-scores for a zone into a [0, 1] gauge.
        0 = severe contraction · 0.5 = neutral · 1 = strong expansion.
        """
        zcs = [
            v["z_composite"]
            for v in zone_data.values()
            if v.get("latest") is not None and "z_composite" in v
        ]
        if not zcs:
            return 0.5
        avg = float(np.mean(zcs))
        return round(float(1.0 / (1.0 + np.exp(-avg * 0.7))), 4)
