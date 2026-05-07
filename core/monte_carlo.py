"""core/monte_carlo.py
GBM Monte Carlo risk simulator — VaR, CVaR, fan chart paths.

Runs N Geometric Brownian Motion paths for a weighted portfolio and returns
Value-at-Risk (95%/99%) and CVaR (Expected Shortfall) at 30/90/252-day horizons.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TRADING_DAYS = 252
_DEFAULT_MU    = 0.10 / _TRADING_DAYS   # SPY-like annual drift, daily
_DEFAULT_SIGMA = 0.18 / np.sqrt(_TRADING_DAYS)


@dataclass
class SimResult:
    """Results from a Monte Carlo simulation run."""

    horizons: Dict[int, Dict[str, float]]
    paths: np.ndarray       # shape (n_paths, max_horizon), P&L in USD
    portfolio_value: float
    symbols: List[str]

    def summary(self, horizon: int = 30) -> str:
        h = self.horizons.get(horizon, {})
        return (
            f"H={horizon}d | E[R]={h.get('expected_return', 0):+.2%} "
            f"VaR95={h.get('var95', 0):.2%} VaR99={h.get('var99', 0):.2%} "
            f"CVaR95={h.get('cvar95', 0):.2%}"
        )


class RiskSimulator:
    """
    Geometric Brownian Motion Monte Carlo for a weighted equity portfolio.

    Parameters
    ----------
    weights : {symbol: weight} — positive fractions summing ≤ 1.0
    portfolio_value : total portfolio value in USD
    n_paths : number of simulation paths (default 10 000)
    history_days : trading days used to estimate μ/σ from price history
    """

    def __init__(
        self,
        weights: Dict[str, float],
        portfolio_value: float,
        n_paths: int = 10_000,
        history_days: int = 252,
    ) -> None:
        self.weights = {k: v for k, v in weights.items() if v > 0}
        self.portfolio_value = max(portfolio_value, 1.0)
        self.n_paths = n_paths
        self.history_days = history_days

    # ── Parameter estimation ───────────────────────────────────────────────────

    def _estimate_params(self):
        """
        Estimate daily log-return μ and σ per asset from yfinance history.
        Falls back to SPY-like parameters if data is insufficient.

        Returns (mu_arr, sigma_arr, symbols, weights_arr).
        """
        import yfinance as yf

        symbols = list(self.weights.keys())
        w_arr   = np.array([self.weights[s] for s in symbols])
        n       = len(symbols)

        fallback = (
            np.full(n, _DEFAULT_MU),
            np.full(n, _DEFAULT_SIGMA),
            symbols,
            w_arr,
        )

        if not symbols:
            return fallback

        try:
            period = f"{self.history_days + 60}d"
            tickers_arg = symbols[0] if len(symbols) == 1 else symbols
            raw = yf.download(
                tickers_arg,
                period=period,
                auto_adjust=True,
                progress=False,
                group_by="column",
            )
            if isinstance(raw.columns, pd.MultiIndex):
                prices = raw["Close"][symbols].dropna()
            else:
                # Single-ticker download — column is just the ticker or "Close"
                close_col = symbols[0] if symbols[0] in raw.columns else "Close"
                prices = raw[[close_col]].rename(columns={close_col: symbols[0]})

            log_ret = np.log(prices / prices.shift(1)).dropna()
            if len(log_ret) < 20:
                raise ValueError("insufficient history")

            mu    = log_ret.mean().values
            sigma = log_ret.std().values
            return mu, sigma, symbols, w_arr

        except Exception as exc:
            logger.warning("[MC] param estimation failed (%s) — using defaults", exc)
            return fallback

    # ── Simulation ─────────────────────────────────────────────────────────────

    def run(self, horizons: Optional[List[int]] = None) -> SimResult:
        """
        Run GBM Monte Carlo and compute risk metrics.

        Parameters
        ----------
        horizons : list of day counts for risk reporting (default [30, 90, 252])

        Returns
        -------
        SimResult with per-horizon VaR/CVaR and full paths matrix.
        """
        if horizons is None:
            horizons = [30, 90, 252]

        max_h = max(horizons)
        mu, sigma, symbols, w_arr = self._estimate_params()

        rng = np.random.default_rng(seed=42)

        # GBM log-space drift and diffusion per day
        drift     = mu - 0.5 * sigma ** 2        # (n_assets,)
        diffusion = sigma                          # (n_assets,)

        # Random shocks: shape (n_paths, max_h, n_assets)
        Z = rng.standard_normal((self.n_paths, max_h, len(symbols)))

        # Cumulative log returns → portfolio return
        daily_log = drift[None, None, :] + diffusion[None, None, :] * Z
        cum_log   = np.cumsum(daily_log, axis=1)      # (n_paths, max_h, n)
        asset_ret = np.exp(cum_log) - 1.0             # (n_paths, max_h, n)
        port_ret  = asset_ret @ w_arr                 # (n_paths, max_h)

        # Compute risk metrics per horizon
        results: Dict[int, Dict[str, float]] = {}
        for h in horizons:
            r    = port_ret[:, h - 1]
            var95 = float(np.percentile(r, 5))
            var99 = float(np.percentile(r, 1))
            tail  = r[r <= var95]
            cvar95 = float(tail.mean()) if len(tail) else var95
            results[h] = {
                "expected_return": float(r.mean()),
                "median_return":   float(np.median(r)),
                "var95":  -var95,   # reported as positive loss magnitude
                "var99":  -var99,
                "cvar95": -cvar95,
                "p_loss": float((r < 0).mean()),
                "p10":    float(np.percentile(r, 10)),
                "p90":    float(np.percentile(r, 90)),
            }

        # P&L in USD (n_paths × max_horizon)
        paths_usd = port_ret * self.portfolio_value

        return SimResult(
            horizons=results,
            paths=paths_usd,
            portfolio_value=self.portfolio_value,
            symbols=symbols,
        )
