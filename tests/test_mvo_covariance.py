"""tests/test_mvo_covariance.py — Ledoit-Wolf-shrunk async covariance (P1.2 audit).

The previous `np.cov` on rolling 2-day returns produced a rank-deficient/singular
matrix whenever observations < assets, which makes SLSQP error-maximize. These
tests lock the two fixes: non-overlapping sampling + LW shrinkage → always PD.
"""
from __future__ import annotations

import numpy as np

from backend.market_intel.portfolio_optimizer import (
    _MIN_COV_OBS,
    build_async_covariance,
    run_optimizer,
)


def _gbm(n_assets: int, n_prices: int, seed: int = 0) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    series: dict[str, list[float]] = {}
    for a in range(n_assets):
        rets = rng.normal(0.0005, 0.02, n_prices - 1)
        prices = 100.0 * np.exp(np.cumsum(np.insert(rets, 0, 0.0)))
        series[f"T{a}"] = prices.tolist()
    return series


class TestAsyncCovariance:
    def test_none_when_too_few_prices(self):
        # One price short of the 2·obs+1 floor → caller must fall back.
        assert build_async_covariance(_gbm(5, 2 * _MIN_COV_OBS)) is None

    def test_positive_definite_when_assets_exceed_obs(self):
        # 20 assets, 21 prices → 10 non-overlapping obs < 20 assets: the raw
        # sample covariance is singular; Ledoit-Wolf must still be PD.
        n = 20
        cov = build_async_covariance(_gbm(n, 2 * _MIN_COV_OBS + 1))
        assert cov is not None and cov.shape == (n, n)
        assert float(np.linalg.eigvalsh(cov).min()) > 0.0   # positive-definite
        assert np.allclose(cov, cov.T)                       # symmetric
        assert np.isfinite(cov).all()

    def test_well_conditioned_vs_singular_raw(self):
        n = 30
        cov = build_async_covariance(_gbm(n, 2 * _MIN_COV_OBS + 1, seed=7))
        assert np.linalg.cond(cov) < 1e6                     # well-conditioned

    def test_run_optimizer_with_shrunk_cov(self):
        n = 12
        series = _gbm(n, 2 * _MIN_COV_OBS + 5, seed=3)
        tickers = list(series.keys())
        scores = [0.5 + 0.01 * i for i in range(n)]
        sectors = ["Tech" if i % 2 else "Health" for i in range(n)]
        w, _ = run_optimizer(
            tickers, scores, sectors, vix=18.0, price_series=series)
        assert set(w) == set(tickers)
        assert all(v >= -1e-9 for v in w.values())
        assert sum(w.values()) <= 1.0 + 1e-6                 # cash-buffer model
