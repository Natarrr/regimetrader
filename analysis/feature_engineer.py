"""analysis/feature_engineer.py
OHLCV feature builder for the HMM regime classifier.

Markowitz (1952 Nobel) — risk is multi-dimensional; representing it via a
standardised feature matrix allows the HMM to learn state-conditional
covariance structures rather than scalar thresholds.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureEngineer:
    """Build a scaled feature matrix for the HMM regime classifier.

    Feature layout (column order is fixed — RegimeClassifier uses col 0 as returns):
      0  log_ret       — daily log-return                  (returns_feature_index=0)
      1  vol_20        — 20-day realised volatility (std of log-returns)
      2  vol_ratio     — volume / 20-day avg volume  (clipped to [0, 5])
      3  range_ratio   — (high - low) / close        (intraday range proxy)
      4  rsi_norm      — RSI(14) normalised to [0, 1]
    """

    def build(
        self,
        ohlcv: pd.DataFrame,
        fit_scaler: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
        """Markowitz (1952 Nobel) — standardise heterogeneous risk features.

        Args:
            ohlcv:      DataFrame with columns Close, High, Low, Volume.
            fit_scaler: If True (default), fit a new StandardScaler on this data.

        Returns:
            (features, returns, scaler) where features is (N, 5) float64,
            returns is (N,) log-return array, and scaler is the fitted scaler.
        """
        close  = ohlcv["Close"].astype(float)
        if isinstance(close, (int, float, np.number)):
            close = pd.Series([close])
        high   = ohlcv["High"].astype(float)
        if isinstance(high, (int, float, np.number)):
            high = pd.Series([high])
        low    = ohlcv["Low"].astype(float)
        if isinstance(low, (int, float, np.number)):
            low = pd.Series([low])
        volume = ohlcv["Volume"].astype(float)
        if isinstance(volume, (int, float, np.number)):
            volume = pd.Series([volume])

        log_ret = np.log(close / close.shift(1))
        vol_20  = log_ret.rolling(20).std()
        _vol_mean = volume.rolling(20).mean().replace(0, np.nan)
        # Indices and volatility products (^VIX, ^CRSLDX) report zero volume;
        # substitute 1.0 (neutral: at-average-volume) so dropna() keeps all rows.
        vol_ratio   = (volume / _vol_mean).clip(0, 5).fillna(1.0)
        range_ratio = (high - low) / close.replace(0, np.nan)

        delta    = close.diff()
        gain     = delta.clip(lower=0).rolling(14).mean()
        loss     = (-delta.clip(upper=0)).rolling(14).mean()
        rs       = gain / loss.replace(0, 1e-10)
        rsi_norm = (100 - (100 / (1 + rs))) / 100.0

        feat_df = pd.DataFrame({
            "log_ret":     log_ret,
            "vol_20":      vol_20,
            "vol_ratio":   vol_ratio,
            "range_ratio": range_ratio,
            "rsi_norm":    rsi_norm,
        }).dropna()

        if feat_df.empty:
            raise ValueError(
                f"Feature engineering resulted in empty DataFrame after NaN removal. "
                f"Input had {len(ohlcv)} rows; after feature calculations and dropna(), "
                f"0 rows remain. Check that input data is sufficient (at least 20+ trading days)."
            )

        returns  = feat_df["log_ret"].values.astype(np.float64)
        scaler   = StandardScaler()
        features = scaler.fit_transform(feat_df.values)

        return features.astype(np.float64), returns, scaler
