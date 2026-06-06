# Path: research/tests/test_train_lgbm.py
"""Tests for train_lgbm helpers — no LightGBM training (fast)."""
import numpy as np
import pytest

from research.scripts.train_lgbm import (
    _blend_and_constrain,
    _shap_to_stable_weights,
    _build_folds,
    WEIGHT_FLOOR,
    WEIGHT_CAP_MULTIPLIER,
    BLEND_ALPHA,
)
from research.scripts.ic_engine import FACTORS, ACADEMIC_WEIGHTS_US
import pandas as pd
from datetime import date, timedelta


def _make_df(n_dates: int = 20, n_tickers: int = 10) -> pd.DataFrame:
    records = []
    base = date(2025, 1, 1)
    fridays = [base + timedelta(weeks=i) for i in range(n_dates)]
    for snap in fridays:
        for i in range(n_tickers):
            rec = {"snapshot_date": str(snap), "ticker": f"T{i:03d}",
                   "forward_return_21d": np.random.randn() * 0.02}
            for f in FACTORS:
                rec[f] = np.random.uniform(0, 1)
            records.append(rec)
    df = pd.DataFrame(records)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    return df


def test_blend_and_constrain_sum_to_one():
    shap_stable = {f: 1.0 / len(FACTORS) for f in FACTORS}
    result = _blend_and_constrain(shap_stable, investigate_factors=set())
    assert abs(sum(result.values()) - 1.0) < 1e-5


def test_blend_and_constrain_floor_applied():
    shap_stable = {f: 0.001 if f == "news_buzz" else 0.12 for f in FACTORS}
    result = _blend_and_constrain(shap_stable, investigate_factors=set())
    assert result["news_buzz"] >= WEIGHT_FLOOR - 1e-8


def test_blend_and_constrain_cap_applied():
    shap_stable = {f: 0.001 for f in FACTORS}
    shap_stable["insider_conviction"] = 0.99
    result = _blend_and_constrain(shap_stable, investigate_factors=set())
    max_allowed = WEIGHT_CAP_MULTIPLIER * max(ACADEMIC_WEIGHTS_US["insider_conviction"], WEIGHT_FLOOR)
    # After normalization the cap may be lower, but before normalization it was capped
    assert result["insider_conviction"] <= max_allowed + 1e-5


def test_blend_and_constrain_investigate_factor_keeps_academic():
    shap_stable = {f: 0.1 for f in FACTORS}
    investigate = {"congress"}
    result = _blend_and_constrain(shap_stable, investigate_factors=investigate)
    assert result["congress"] > 0


def test_shap_stability_check_penalizes_high_cv():
    fold1 = np.array([0.5, 0.5, 0.0, 0.0])
    fold2 = np.array([0.0, 0.5, 0.5, 0.0])
    active = ["insider_conviction", "insider_breadth", "congress", "news_sentiment"]
    mean_d, cv_d, stable_d = _shap_to_stable_weights([fold1, fold2], active)
    assert stable_d["congress"] <= mean_d["congress"] + 1e-8


def test_shap_stability_check_single_fold():
    fold1 = np.array([0.3, 0.2, 0.1, 0.4])
    active = ["insider_conviction", "insider_breadth", "congress", "news_sentiment"]
    mean_d, cv_d, stable_d = _shap_to_stable_weights([fold1], active)
    for f in active:
        assert abs(stable_d[f] - mean_d[f]) < 1e-6


def test_build_folds_returns_two_folds():
    np.random.seed(42)
    df = _make_df(n_dates=20)
    folds = _build_folds(df, n_splits=2)
    assert len(folds) == 2


def test_build_folds_no_overlap():
    np.random.seed(42)
    df = _make_df(n_dates=20)
    folds = _build_folds(df, n_splits=2)
    train1_dates = set(folds[0][0]["snapshot_date"])
    val1_dates = set(folds[0][1]["snapshot_date"])
    assert train1_dates.isdisjoint(val1_dates)


def test_optimal_weights_constraints():
    """Integration: blend + constrain must always satisfy all hard constraints."""
    import random
    random.seed(99)
    for _ in range(20):
        shap_stable = {f: random.uniform(0, 1) for f in FACTORS}
        total = sum(shap_stable.values())
        shap_stable = {k: v / total for k, v in shap_stable.items()}
        investigate = set(random.sample(FACTORS, k=2))
        result = _blend_and_constrain(shap_stable, investigate)
        for f in FACTORS:
            if f not in investigate:
                assert result[f] >= WEIGHT_FLOOR - 1e-7, f"{f}: {result[f]} < {WEIGHT_FLOOR}"
        assert abs(sum(result.values()) - 1.0) < 1e-5
