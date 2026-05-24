"""regime_trader.monitoring.factor_orthogonality — permanent factor orthogonality monitoring.

Fix #8: López de Prado (AFML ch. 8) — "feature engineering is not done after one
validation, it requires permanent monitoring of the assumed structure."

After each pipeline run, compute the cross-sectional Spearman correlation matrix
of the 7 scored factors on the US universe. If any pair exceeds CORRELATION_WARN_THRESHOLD,
emit WARNING. If any pair exceeds CORRELATION_ERROR_THRESHOLD, emit ERROR (factors are
essentially redundant and should be re-engineered).

Design choices:
- Spearman (not Pearson): factor scores are bounded [0,1] with skewed distributions
  post-shrinkage; Spearman is rank-based and robust to this.
- US-only by default: EU/Asia have structurally absent factors (None values for 5/7
  factors), which would produce misleading pairwise NaN handling and shrink the
  observation count. The orthogonality structure that matters most is on the rich
  US scoring where all 7 factors fire.
- use_neutralized=True: orthogonality must hold *after* cross-sectional neutralization —
  that is the residual information fed to portfolio construction. Raw scores may
  legitimately co-move during market regimes; neutralized scores should not.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np

log = logging.getLogger(__name__)

FACTORS_TO_MONITOR = [
    "insider_conviction_score",
    "insider_breadth_score",
    "congress_score",
    "news_sentiment_score",
    "news_buzz_score",
    "momentum_long_score",
    "volume_attention_score",
]

CORRELATION_WARN_THRESHOLD: float = 0.50   # Spearman |ρ| — warn
CORRELATION_ERROR_THRESHOLD: float = 0.75  # Spearman |ρ| — factors essentially identical


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation between two 1-D arrays (no scipy dependency)."""
    n = len(x)
    if n < 2:
        return float("nan")
    rx = x.argsort().argsort().astype(float)
    ry = y.argsort().argsort().astype(float)
    dx = rx - rx.mean()
    dy = ry - ry.mean()
    denom = np.sqrt((dx ** 2).sum() * (dy ** 2).sum())
    if denom == 0.0:
        return float("nan")
    return float(np.dot(dx, dy) / denom)


def compute_factor_correlation_matrix(
    results: list[dict[str, Any]],
    factors: list[str] | None = None,
    use_neutralized: bool = True,
    market_filter: str | None = "US",
    min_observations: int = 30,
) -> dict[str, Any]:
    """Compute cross-sectional Spearman correlation matrix of factors.

    Args:
        results:          List of scored ticker dicts from run_pipeline.run().
        factors:          Factor names to evaluate. Defaults to FACTORS_TO_MONITOR.
        use_neutralized:  If True, use *_score_neutral columns (post–Fix #1).
                          If False, use raw *_score columns.
        market_filter:    Only include rows with r["market"] == market_filter
                          (or "USA" for legacy US rows). Pass None to use all markets.
        min_observations: Minimum rows required; returns error dict if below threshold.

    Returns:
        {
            "computed_at": iso_timestamp,
            "market": market_filter or "ALL",
            "n_observations": int,
            "factors_evaluated": list[str],
            "factors_skipped": list[str],
            "correlation_matrix": {factor_i: {factor_j: rho_ij}},
            "max_abs_correlation": float,
            "max_pair": [str, str] | [],
            "warnings": list[str],
            "errors": list[str],
        }
        On failure: {"error": str, "computed_at": iso_timestamp}
    """
    if factors is None:
        factors = FACTORS_TO_MONITOR

    ts = datetime.now(timezone.utc).isoformat()

    try:
        # ── Filter by market ──────────────────────────────────────────────────
        if market_filter is not None:
            # US rows may carry market="USA" (legacy) or market="US"
            _us_values = {"US", "USA"} if market_filter in ("US", "USA") else {market_filter}
            rows = [r for r in results if r.get("market", "USA") in _us_values]
        else:
            rows = list(results)

        if len(rows) < min_observations:
            msg = (
                f"Insufficient observations for orthogonality diagnostic: "
                f"{len(rows)} rows (market={market_filter}, need ≥{min_observations}). "
                f"Factor orthogonality not computed."
            )
            log.warning(msg)
            return {"error": msg, "computed_at": ts}

        # ── Build factor arrays, handle neutralized vs raw ────────────────────
        suffix = "_neutral" if use_neutralized else ""
        evaluated: list[str] = []
        skipped: list[str] = []
        arrays: dict[str, np.ndarray] = {}

        for factor in factors:
            col = f"{factor}{suffix}" if use_neutralized else factor
            # Collect non-None values aligned across all rows
            vals = []
            for r in rows:
                v = r.get(col)
                if v is None and use_neutralized:
                    # Fall back to raw score if neutral column absent (e.g. bucket too small)
                    v = r.get(factor)
                vals.append(v)

            none_count = sum(1 for v in vals if v is None)
            if none_count > len(vals) * 0.5:
                skipped.append(factor)
                log.debug(
                    "Orthogonality: skipping %s — %.0f%% None values",
                    factor, none_count / len(vals) * 100,
                )
                continue

            # Replace remaining None with median of available values (minimal imputation)
            numeric = [float(v) for v in vals if v is not None]
            median_fill = float(np.median(numeric)) if numeric else 0.0
            arr = np.array([float(v) if v is not None else median_fill for v in vals])
            arrays[factor] = arr
            evaluated.append(factor)

        if len(evaluated) < 2:
            msg = f"Too few factors with sufficient data: {evaluated}. Skipped: {skipped}."
            log.warning("Orthogonality diagnostic: %s", msg)
            return {"error": msg, "computed_at": ts}

        # ── Pairwise Spearman correlation matrix ──────────────────────────────
        matrix: dict[str, dict[str, float]] = {}
        max_abs_rho = 0.0
        max_pair: list[str] = []
        warnings: list[str] = []
        errors: list[str] = []

        for fi in evaluated:
            matrix[fi] = {}
            for fj in evaluated:
                if fi == fj:
                    matrix[fi][fj] = 1.0
                    continue
                rho = _spearman_rho(arrays[fi], arrays[fj])
                matrix[fi][fj] = round(rho, 4) if not np.isnan(rho) else None

                abs_rho = abs(rho) if not np.isnan(rho) else 0.0
                if abs_rho > max_abs_rho:
                    max_abs_rho = abs_rho
                    max_pair = [fi, fj]

                # Emit warnings/errors only once per unordered pair
                if fi < fj and not np.isnan(rho):
                    if abs_rho >= CORRELATION_ERROR_THRESHOLD:
                        msg = (
                            f"({fi}, {fj}): |ρ|={abs_rho:.3f} ≥ {CORRELATION_ERROR_THRESHOLD} "
                            f"— factors are not effectively independent"
                        )
                        errors.append(msg)
                    elif abs_rho >= CORRELATION_WARN_THRESHOLD:
                        msg = (
                            f"({fi}, {fj}): |ρ|={abs_rho:.3f} ≥ {CORRELATION_WARN_THRESHOLD} "
                            f"— correlation approaching redundancy threshold"
                        )
                        warnings.append(msg)

        return {
            "computed_at":         ts,
            "market":              market_filter or "ALL",
            "n_observations":      len(rows),
            "factors_evaluated":   evaluated,
            "factors_skipped":     skipped,
            "correlation_matrix":  matrix,
            "max_abs_correlation": round(max_abs_rho, 4),
            "max_pair":            max_pair,
            "warnings":            warnings,
            "errors":              errors,
        }

    except Exception as exc:
        msg = f"compute_factor_correlation_matrix failed: {exc}"
        log.error(msg)
        return {"error": msg, "computed_at": ts}
