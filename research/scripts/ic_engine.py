# Path: research/scripts/ic_engine.py
"""Pure IC computation functions — no I/O, no side effects.

All functions operate on plain Python lists or numpy arrays.
Imported by run_ic_analysis.py and Notebook 01.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.stats import spearmanr

FACTORS = [
    "insider_conviction", "insider_breadth", "congress",
    "news_sentiment", "news_buzz", "momentum_long",
    "volume_attention", "analyst_consensus", "quality_piotroski",
]

WeightRecommendation = Literal["increase", "hold", "decrease", "investigate"]

# Academic weights from config/weights.py WEIGHTS_US (v2.2-global)
ACADEMIC_WEIGHTS_US: dict[str, float] = {
    "insider_conviction": 0.30,
    "insider_breadth":    0.15,
    "congress":           0.22,
    "news_sentiment":     0.10,
    "news_buzz":          0.05,
    "momentum_long":      0.15,
    "volume_attention":   0.03,
    "analyst_consensus":  0.00,
    "quality_piotroski":  0.00,
}


def rank_ic_per_snapshot(
    factor_scores: np.ndarray,  # shape (n_tickers,)
    forward_returns: np.ndarray,  # shape (n_tickers,)
) -> float:
    """Spearman rank IC for a single cross-section.

    Returns NaN if insufficient variation (all identical scores).
    """
    if len(factor_scores) < 3:
        return float("nan")
    if np.std(factor_scores) < 1e-8:
        return 0.0
    corr, _ = spearmanr(factor_scores, forward_returns, nan_policy="omit")
    return float(corr) if not np.isnan(corr) else 0.0


def compute_factor_ic(
    factor_name: str,
    df_records: list[dict],
) -> dict:
    """Compute all IC metrics for one factor across all (ticker, date) records.

    df_records: list of dicts with keys: snapshot_date, <factor_name>, forward_return_21d.
    Returns dict matching ic_report.json schema (minus weight_recommendation).
    """
    from collections import defaultdict
    by_date: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for rec in df_records:
        score = rec.get(factor_name)
        fwd = rec.get("forward_return_21d")
        if score is None or fwd is None:
            continue
        by_date[rec["snapshot_date"]].append((float(score), float(fwd)))

    ics_by_month: dict[str, list[float]] = defaultdict(list)
    all_ics: list[float] = []
    for snap_date, pairs in sorted(by_date.items()):
        scores = np.array([p[0] for p in pairs])
        returns = np.array([p[1] for p in pairs])
        ic = rank_ic_per_snapshot(scores, returns)
        if not np.isnan(ic):
            all_ics.append(ic)
            month = snap_date[:7]  # "YYYY-MM"
            ics_by_month[month].append(ic)

    if not all_ics:
        return {
            "mean_ic": 0.0,
            "ic_ir": 0.0,
            "ic_positive_rate": 0.0,
            "monthly_ic": {},
        }

    arr = np.array(all_ics)
    mean_ic = float(arr.mean())
    std_ic = float(arr.std()) if len(arr) > 1 else 1e-8
    ic_ir = mean_ic / std_ic if std_ic > 1e-8 else 0.0
    ic_positive_rate = float((arr > 0).mean())
    monthly_ic = {m: round(float(np.mean(v)), 6) for m, v in sorted(ics_by_month.items())}

    return {
        "mean_ic":          round(mean_ic, 6),
        "ic_ir":            round(ic_ir, 6),
        "ic_positive_rate": round(ic_positive_rate, 6),
        "monthly_ic":       monthly_ic,
    }


def weight_recommendation(
    mean_ic: float,
    ic_ir: float,
    ic_positive_rate: float,
) -> WeightRecommendation:
    """Derive mechanical weight recommendation from IC metrics.

    Rules (spec § Phase 2):
        mean_ic < 0                         → "investigate"
        ic_ir > 0.5 AND ic_pos_rate >= 0.60 → "increase"
        ic_ir >= 0.3                        → "hold"
        else                                → "decrease"
    """
    if mean_ic < 0:
        return "investigate"
    if ic_ir > 0.5 and ic_positive_rate >= 0.60:
        return "increase"
    if ic_ir >= 0.3:
        return "hold"
    return "decrease"


def build_ic_report(df_records: list[dict]) -> dict:
    """Build the full ic_report.json dict for all 9 factors."""
    report = {}
    for factor in FACTORS:
        metrics = compute_factor_ic(factor, df_records)
        rec = weight_recommendation(
            metrics["mean_ic"],
            metrics["ic_ir"],
            metrics["ic_positive_rate"],
        )
        report[factor] = {**metrics, "weight_recommendation": rec}
    return report
