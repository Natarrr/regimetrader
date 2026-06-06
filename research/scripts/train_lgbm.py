# Path: research/scripts/train_lgbm.py
"""Walk-forward LightGBM training + SHAP stability → optimal_weights.json.

Run from repo root after run_ic_analysis.py completes:
    python research/scripts/train_lgbm.py

Output: research/optimal_weights.json
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import shap

from research.scripts.build_qlib_dataset import load_ndjson
from research.scripts.ic_engine import FACTORS, ACADEMIC_WEIGHTS_US, weight_recommendation
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("train_lgbm")

_IN = Path("research/data/backfill/factor_scores.ndjson")
_IC_REPORT = Path("research/ic_report.json")
_OUT = Path("research/optimal_weights.json")

BLEND_ALPHA = 0.6        # trust data 60%, academic prior 40%
WEIGHT_FLOOR = 0.05
WEIGHT_CAP_MULTIPLIER = 2.0

# Monotone constraints: +1 = increasing, -1 = decreasing, 0 = unconstrained
MONOTONE_CONSTRAINTS: dict[str, int] = {
    "insider_conviction": +1,
    "insider_breadth":    +1,
    "congress":           +1,
    "news_sentiment":     +1,
    "news_buzz":           0,
    "momentum_long":      +1,
    "volume_attention":    0,
    "analyst_consensus":  +1,
    "quality_piotroski":  +1,
}


def _get_investigate_factors(ic_report: dict) -> set[str]:
    return {f for f, m in ic_report.items() if m.get("weight_recommendation") == "investigate"}


def _build_folds(df, n_splits: int = 2) -> list[tuple]:
    """Walk-forward expanding-window folds on snapshot_date."""
    dates = sorted(df["snapshot_date"].unique())
    fold_size = len(dates) // (n_splits + 1)
    folds = []
    for i in range(n_splits):
        train_cutoff = dates[(i + 1) * fold_size - 1]
        val_cutoff = dates[min((i + 2) * fold_size - 1, len(dates) - 1)]
        train = df[df["snapshot_date"] <= train_cutoff]
        val = df[(df["snapshot_date"] > train_cutoff) & (df["snapshot_date"] <= val_cutoff)]
        if len(train) > 0 and len(val) > 0:
            folds.append((train, val))
    return folds


def _train_fold(
    train_df,
    val_df,
    active_factors: list[str],
) -> tuple[lgb.LGBMRegressor, np.ndarray, float]:
    """Train one LightGBM fold. Returns (model, shap_values, val_ic)."""
    mono_list = [MONOTONE_CONSTRAINTS[f] for f in active_factors]

    X_train = train_df[active_factors].values
    y_train = train_df["forward_return_21d"].values
    X_val = val_df[active_factors].values
    y_val = val_df["forward_return_21d"].values

    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 15,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "monotone_constraints": mono_list,
        "verbose": -1,
        "random_state": 42,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)],
    )

    # SHAP values on validation set
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_val)
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)

    # Val IC (rank correlation of predictions vs actual returns)
    preds = model.predict(X_val)
    val_ic = float(spearmanr(preds, y_val).correlation)

    return model, mean_abs_shap, val_ic


def _shap_to_stable_weights(
    shap_per_fold: list[np.ndarray],
    active_factors: list[str],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    """Apply SHAP stability check, return (shap_mean, shap_cv, shap_stable)."""
    arr = np.array(shap_per_fold)  # (n_folds, n_factors)
    mean_arr = arr.mean(axis=0)
    std_arr = arr.std(axis=0) if len(arr) > 1 else np.zeros(len(active_factors))
    cv_arr = np.where(mean_arr > 1e-8, std_arr / mean_arr, 1.0)

    stability_multiplier = np.clip(1 - cv_arr, 0.3, 1.0)
    stable = mean_arr * stability_multiplier
    if stable.sum() > 0:
        stable = stable / stable.sum()

    shap_mean = {f: round(float(mean_arr[i]), 6) for i, f in enumerate(active_factors)}
    shap_cv = {f: round(float(cv_arr[i]), 6) for i, f in enumerate(active_factors)}
    shap_stable_w = {f: round(float(stable[i]), 6) for i, f in enumerate(active_factors)}
    return shap_mean, shap_cv, shap_stable_w


def _blend_and_constrain(
    shap_stable: dict[str, float],
    investigate_factors: set[str],
) -> dict[str, float]:
    """Blend SHAP weights with academic prior, apply floor + cap, re-normalize."""
    final: dict[str, float] = {}
    for factor in FACTORS:
        academic_w = ACADEMIC_WEIGHTS_US[factor]
        if factor in investigate_factors:
            final[factor] = academic_w  # unchanged
            continue
        data_w = shap_stable.get(factor, 0.0)
        blended = BLEND_ALPHA * data_w + (1 - BLEND_ALPHA) * academic_w
        blended = max(blended, WEIGHT_FLOOR)
        blended = min(blended, WEIGHT_CAP_MULTIPLIER * max(academic_w, WEIGHT_FLOOR))
        final[factor] = blended

    # Iterative floor projection: normalize, then clamp below-floor weights and
    # redistribute the deficit among unclamped weights until stable.  This
    # guarantees WEIGHT_FLOOR holds after normalization.
    total = sum(final.values())
    weights = {k: v / total for k, v in final.items()}

    for _ in range(len(FACTORS) + 1):
        floored = {k: v for k, v in weights.items() if v < WEIGHT_FLOOR}
        if not floored:
            break
        clamped = {k: WEIGHT_FLOOR for k in floored}
        remaining_budget = 1.0 - sum(clamped.values())
        free = {k: v for k, v in weights.items() if k not in floored}
        free_total = sum(free.values())
        if free_total <= 0:
            # Degenerate: all factors are at floor, spread evenly
            n = len(weights)
            weights = {k: 1.0 / n for k in weights}
            break
        scale = remaining_budget / free_total
        weights = {k: WEIGHT_FLOOR if k in floored else v * scale
                   for k, v in weights.items()}

    final = {k: round(v, 8) for k, v in weights.items()}
    assert abs(sum(final.values()) - 1.0) < 1e-5, f"Weights sum to {sum(final.values())}"
    return final


def main() -> None:
    if not _IN.exists():
        raise FileNotFoundError(f"{_IN} not found — run backfill_factors.py first")
    if not _IC_REPORT.exists():
        raise FileNotFoundError(f"{_IC_REPORT} not found — run run_ic_analysis.py first")

    ic_report = json.loads(_IC_REPORT.read_text())
    investigate_factors = _get_investigate_factors(ic_report)
    active_factors = [f for f in FACTORS if f not in investigate_factors]
    log.info("Active factors: %s", active_factors)
    log.info("Investigate factors (excluded): %s", investigate_factors)

    df = load_ndjson(_IN)
    df = df.dropna(subset=active_factors + ["forward_return_21d"])
    log.info("Training on %d records", len(df))

    folds = _build_folds(df, n_splits=2)
    log.info("Walk-forward folds: %d", len(folds))

    shap_per_fold: list[np.ndarray] = []
    val_ics: list[float] = []

    for fold_idx, (train_df, val_df) in enumerate(folds):
        log.info("Fold %d: train=%d val=%d", fold_idx + 1, len(train_df), len(val_df))
        _, fold_shap, fold_ic = _train_fold(train_df, val_df, active_factors)
        shap_per_fold.append(fold_shap)
        val_ics.append(fold_ic)
        log.info("  Val IC: %.4f", fold_ic)

    shap_mean_d, shap_cv_d, shap_stable_d = _shap_to_stable_weights(
        shap_per_fold, active_factors
    )
    final_weights = _blend_and_constrain(shap_stable_d, investigate_factors)

    # Build output JSON
    weights_detail: dict[str, dict] = {}
    for factor in FACTORS:
        academic_w = ACADEMIC_WEIGHTS_US[factor]
        is_investigate = factor in investigate_factors
        weights_detail[factor] = {
            "academic": academic_w,
            "shap_per_fold": [round(float(shap_per_fold[i][active_factors.index(factor)]), 6)
                              for i in range(len(shap_per_fold))]
                             if factor in active_factors else None,
            "shap_cv":             shap_cv_d.get(factor),
            "stability_multiplier": round(
                float(np.clip(1 - shap_cv_d.get(factor, 1.0), 0.3, 1.0)), 6
            ) if factor in active_factors else None,
            "shap_stable":         shap_stable_d.get(factor),
            "final":               final_weights[factor],
            "weight_recommendation": ic_report.get(factor, {}).get("weight_recommendation", "hold"),
        }

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "blend_alpha": BLEND_ALPHA,
        "weight_floor": WEIGHT_FLOOR,
        "lgbm_val_ic_per_fold": [round(ic, 6) for ic in val_ics],
        "lgbm_val_ic_mean": round(float(np.mean(val_ics)), 6),
        "investigate_factors": sorted(investigate_factors),
        "weights": weights_detail,
    }

    _OUT.write_text(json.dumps(output, indent=2))
    log.info("Optimal weights written to %s", _OUT)

    # Print summary
    print("\n── Calibrated Weights ───────────────────────────────────")
    print(f"{'Factor':<22} {'Academic':>9} {'Final':>9} {'Delta':>8}")
    print("-" * 52)
    for factor in FACTORS:
        d = weights_detail[factor]
        delta = d["final"] - d["academic"]
        marker = " *" if factor in investigate_factors else ""
        print(f"{factor:<22} {d['academic']:>9.4f} {d['final']:>9.4f} {delta:>+8.4f}{marker}")
    print(f"\nVal IC per fold: {val_ics}")


if __name__ == "__main__":
    main()
