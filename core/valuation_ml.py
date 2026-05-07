"""core/valuation_ml.py
ML-enhanced DCF valuation engine.

Fetches macro features from FRED and uses Ridge regression (scikit-learn) to
predict revenue growth. Returns Classic DCF fair value vs ML-Adjusted fair value.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_WACC_DEFAULT      = 0.09
_TERM_GROW_DEFAULT = 0.025
_YEARS             = 5

_MACRO_SERIES = {
    "us_leading": "USSLIND",        # US Leading Index
    "us_conf":    "UMCSENT",        # U of Michigan Consumer Sentiment
    "us_pmi":     "BSCICP03USM665S",# OECD Business Confidence US
}


@dataclass
class DCFResult:
    """Full ML-DCF valuation result for one ticker."""

    ticker: str
    current_price: float
    classic_fv: float       # DCF using reported revenueGrowth
    ml_fv: float            # DCF using ML-predicted growth
    classic_upside: float   # (classic_fv / price - 1) × 100
    ml_upside: float        # (ml_fv / price - 1) × 100
    base_growth: float      # reported revenueGrowth
    ml_growth: float        # Ridge-predicted growth
    model: str              # model name used
    macro_confidence: float # 0–1: fraction of FRED series fetched

    @property
    def classic_tag(self) -> str:
        s = "+" if self.classic_upside >= 0 else ""
        return f"{s}{self.classic_upside:.1f}%"

    @property
    def ml_tag(self) -> str:
        s = "+" if self.ml_upside >= 0 else ""
        return f"{s}{self.ml_upside:.1f}%"


class MLDCFEngine:
    """
    ML-enhanced 5-year Discounted Cash Flow engine.

    Steps
    -----
    1. Fetch FRED macro indicators (US Leading Index, Consumer Conf, PMI).
    2. Fetch ticker fundamentals from yfinance (EPS, revenueGrowth, price).
    3. Train Ridge on synthetic macro→growth dataset anchored on FRED data.
    4. Run Classic DCF (base growth) and ML-DCF (predicted growth).
    5. Return DCFResult with both fair values and upside/discount percentages.
    """

    def __init__(
        self,
        wacc: float = _WACC_DEFAULT,
        terminal_growth: float = _TERM_GROW_DEFAULT,
    ) -> None:
        self.wacc = wacc
        self.terminal_growth = terminal_growth

    # ── Macro features ─────────────────────────────────────────────────────────

    def _fetch_macro(self) -> Tuple[Dict[str, float], float]:
        """Returns (features_dict, confidence). confidence = fraction of series fetched."""
        import requests
        from io import StringIO
        from datetime import datetime, timedelta

        start = (datetime.today() - timedelta(days=400)).strftime("%Y-%m-%d")
        feats:   Dict[str, float] = {}
        fetched = 0

        for key, sid in _MACRO_SERIES.items():
            try:
                resp = requests.get(
                    "https://fred.stlouisfed.org/graph/fredgraph.csv",
                    params={"id": sid},
                    timeout=15,
                )
                resp.raise_for_status()
                df = pd.read_csv(StringIO(resp.text), parse_dates=["DATE"], index_col="DATE")
                df.columns = [sid]
                s = df.replace(".", float("nan")).astype(float).dropna().iloc[:, 0]
                s = s[s.index >= start]
                if len(s) >= 2:
                    feats[key] = float(s.iloc[-1])
                    prev = s.iloc[max(0, len(s) - 4)]
                    feats[f"{key}_mom"] = float((s.iloc[-1] - prev) / (abs(prev) + 1e-9))
                    fetched += 1
            except Exception as exc:
                logger.debug("[ML-DCF] FRED %s: %s", sid, exc)

        return feats, fetched / len(_MACRO_SERIES)

    # ── Ticker fundamentals ────────────────────────────────────────────────────

    @staticmethod
    def _fundamentals(ticker: str) -> Dict[str, float]:
        import yfinance as yf
        info = yf.Ticker(ticker).info or {}
        price = float(info.get("currentPrice") or info.get("regularMarketPrice") or 0)
        eps   = float(info.get("trailingEps") or 0)
        return {
            "price":      max(price, 0.01),
            "eps":        eps,
            "rev_growth": float(info.get("revenueGrowth") or 0.10),
        }

    # ── ML growth prediction ───────────────────────────────────────────────────

    @staticmethod
    def _predict_growth(macro: Dict[str, float], base: float) -> Tuple[float, str]:
        """
        Ridge regression on synthetic macro→growth training set.

        We bootstrap a training set where growth = base + macro sensitivity
        (US Leading Index deviation, consumer confidence momentum, PMI momentum).
        This encodes established macro-growth relationships while remaining
        anchored on the current macro readings from FRED.
        """
        from sklearn.linear_model import Ridge           # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore

        rng   = np.random.default_rng(0)
        n_syn = 300

        # Feature names and approximate normalization centres
        feat_names = [
            "us_leading", "us_conf", "us_pmi",
            "us_leading_mom", "us_conf_mom", "us_pmi_mom",
        ]
        centers = [100.0, 80.0, 100.0, 0.0, 0.0, 0.0]
        scales  = [5.0,   12.0,  5.0,  0.05, 0.05, 0.05]

        X_syn = rng.normal(centers, scales, size=(n_syn, len(feat_names)))

        # Target: growth driven by leading index deviation and momentum signals
        y_syn = (
            base
            + 0.005 * (X_syn[:, 0] - centers[0])   # leading index gap
            + 0.008 * X_syn[:, 3]                   # leading momentum
            + 0.004 * X_syn[:, 4]                   # confidence momentum
            + rng.normal(0, 0.02, n_syn)
        )

        X_pred = np.array([[macro.get(k, c) for k, c in zip(feat_names, centers)]])

        sc        = StandardScaler()
        X_syn_sc  = sc.fit_transform(X_syn)
        X_pred_sc = sc.transform(X_pred)

        pred = float(Ridge(alpha=1.0).fit(X_syn_sc, y_syn).predict(X_pred_sc)[0])
        return float(np.clip(pred, -0.30, 0.50)), "Ridge"

    # ── DCF calculation ────────────────────────────────────────────────────────

    def _dcf(self, eps: float, growth: float) -> float:
        """Simple 5-year EPS-based DCF with Gordon Growth terminal value."""
        if eps <= 0:
            return 0.0
        fcf = eps
        pv  = 0.0
        for yr in range(1, _YEARS + 1):
            fcf *= 1 + growth
            pv  += fcf / (1 + self.wacc) ** yr
        # Terminal value: FCF grows at terminal_growth in perpetuity
        tv = fcf * (1 + self.terminal_growth) / (self.wacc - self.terminal_growth)
        pv += tv / (1 + self.wacc) ** _YEARS
        return max(0.0, pv)

    # ── Public API ─────────────────────────────────────────────────────────────

    def value(self, ticker: str) -> DCFResult:
        """Run full ML-DCF valuation for a single ticker."""
        macro, conf = self._fetch_macro()
        fund        = self._fundamentals(ticker)

        price = fund["price"]
        eps   = fund["eps"]
        base  = fund["rev_growth"]

        classic_fv = self._dcf(eps, base)

        if macro and conf >= 0.33:
            ml_growth, model = self._predict_growth(macro, base)
        else:
            ml_growth, model = base, "Ridge (no macro)"

        ml_fv = self._dcf(eps, ml_growth)

        def _pct(fv: float) -> float:
            if price <= 0 or fv <= 0:
                return 0.0
            return round((fv / price - 1) * 100, 2)

        return DCFResult(
            ticker=ticker,
            current_price=round(price, 2),
            classic_fv=round(classic_fv, 2),
            ml_fv=round(ml_fv, 2),
            classic_upside=_pct(classic_fv),
            ml_upside=_pct(ml_fv),
            base_growth=round(base, 4),
            ml_growth=round(ml_growth, 4),
            model=model,
            macro_confidence=round(conf, 2),
        )
