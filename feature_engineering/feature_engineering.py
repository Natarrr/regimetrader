# Path: feature_engineering/feature_engineering.py
"""Feature engineering pipeline for HMM regime classification.

Constructs a feature matrix from daily OHLCV price data:
    - log_return:      daily log price change
    - rolling_vol_21:  21-day rolling realised standard deviation
    - momentum_12_1m:  Jegadeesh-Titman 12-1 month cross-sectional momentum
    - rsi_14:          14-day Wilder RSI (normalised to [0,1])

The feature matrix uses ONLY lagged/trailing information — no look-ahead bias.
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_MIN_BARS_MOMENTUM = 252   # 12 months of trading days
_RSI_PERIOD        = 14
_VOL_PERIOD        = 21


class FeatureEngineer:
    """Converts raw OHLCV DataFrame into feature arrays for RegimeClassifier."""

    def build(
        self, bars: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """Build feature matrix from OHLCV *bars*.

        Returns:
            features   — (T, 4) float32 array: [log_ret, vol21, mom12_1, rsi14]
            returns    — (T,) float64 array of raw log returns (same index)
            meta       — dict with index, feature_names, valid_start (row index
                         of first fully-populated row)

        All features are causal: row t uses only data from t and earlier.
        """
        if bars.empty or "Close" not in bars.columns:
            return np.empty((0, 4), dtype=np.float32), np.array([]), {}

        closes = bars["Close"].dropna().astype(float)
        log_ret = np.log(closes / closes.shift(1)).fillna(0.0)

        # Rolling 21-day realised volatility
        vol21 = log_ret.rolling(_VOL_PERIOD).std().fillna(0.0)

        # 12-1 month momentum (skip most recent month = 21 trading days)
        mom12_1 = pd.Series(np.nan, index=closes.index)
        if len(closes) >= _MIN_BARS_MOMENTUM:
            formation_end   = closes.shift(21)     # skip 1 month
            formation_start = closes.shift(252)    # go back 12 months
            mom12_1 = (formation_end / formation_start - 1.0).fillna(0.0)
        else:
            mom12_1 = pd.Series(0.0, index=closes.index)

        # RSI-14 (normalised to [0,1])
        rsi_raw = _compute_rsi(log_ret, _RSI_PERIOD)
        rsi_norm = (rsi_raw / 100.0).fillna(0.5)

        df = pd.DataFrame({
            "log_ret":  log_ret,
            "vol21":    vol21,
            "mom12_1":  mom12_1,
            "rsi14":    rsi_norm,
        }).dropna()

        features = df[["log_ret", "vol21", "mom12_1", "rsi14"]].values.astype(np.float32)
        returns  = df["log_ret"].values.astype(np.float64)

        # Warm-up: first _MIN_BARS_MOMENTUM rows have zeroed momentum
        valid_start = max(0, _MIN_BARS_MOMENTUM - len(df) + len(df))

        meta = {
            "index":         df.index,
            "feature_names": ["log_ret", "vol21", "mom12_1", "rsi14"],
            "valid_start":   valid_start,
            "n_obs":         len(df),
        }
        return features, returns, meta


def _compute_rsi(log_returns: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI from log returns, in [0, 100]."""
    delta = log_returns.copy()
    gain  = delta.clip(lower=0)
    loss  = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)
