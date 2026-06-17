"""src.monitoring.factor_orthogonality — permanent factor orthogonality monitoring.

Fix #8: López de Prado (AFML ch. 8) — "feature engineering is not done after one
validation, it requires permanent monitoring of the assumed structure."

After each pipeline run, compute the cross-sectional Spearman correlation matrix
of the live US-scored factors. If any pair exceeds CORRELATION_WARN_THRESHOLD,
emit WARNING. If any pair exceeds CORRELATION_ERROR_THRESHOLD, emit ERROR (factors are
essentially redundant and should be re-engineered).

Fix #8.1 — Sparsity-aware correlation:
On a large-cap S&P 500-style universe, several factors (congress, news_buzz) have
< 5% non-zero density. When 95%+ of tickers share a value of 0.0, Spearman rank
correlation becomes dominated by tied ranks and produces artifactually high ρ even
between conceptually independent factors. These are not design redundancies — they
are a property of the data distribution on this universe.

Pairs where either factor has density < SPARSITY_THRESHOLD are excluded from
max_abs_correlation, warnings, and errors, but reported in low_density_pairs for
full transparency. The full correlation_matrix is always exposed for manual inspection.

Design choices:
- Spearman (not Pearson): factor scores are bounded [0,1] with skewed distributions
  post-shrinkage; Spearman is rank-based and robust to this.
- US-only by default: EU/Asia have structurally absent factors (None values for
  most US-thesis factors), which would produce misleading pairwise NaN handling and
  shrink the observation count. The orthogonality structure that matters most is on
  the rich US scoring where the full live factor set fires.
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

# Full live US factor set (WEIGHTS_US, v2.2-global). The original list covered
# only 7 of the 11 factors that actually fire for US tickers, so the diagnostic
# was blind to redundancy among the consensus/quality signals added in v2.4
# (analyst_consensus, quality_piotroski, transcript_tone, revenue_revision).
# INTL-only fundamentals (fcf_yield, amihud_shock, pb_value_up, roic_quality)
# and analyst_revision/price_target_upside score 0.0 for US by design and would
# only be sparsity-excluded here, so they are intentionally omitted.
FACTORS_TO_MONITOR = [
    "insider_conviction_score",
    "insider_breadth_score",
    "congress_score",
    "news_sentiment_score",
    "news_buzz_score",
    "momentum_long_score",
    "volume_attention_score",
    "analyst_consensus_score",
    "quality_piotroski_score",
    "transcript_tone_score",
    "revenue_revision_score",
    "inst_flow_13f_score",   # v2.5 — 13F whale flow; watch vs the insider factors
]

CORRELATION_WARN_THRESHOLD: float = 0.50   # Spearman |ρ| — warn
CORRELATION_ERROR_THRESHOLD: float = 0.75  # Spearman |ρ| — factors essentially identical

# Fix #8.1: minimum fraction of non-zero observations for a factor to participate
# in reliable correlation measurement. Below this threshold, tied-rank artefacts
# dominate and Spearman ρ is statistically uninformative.
SPARSITY_THRESHOLD: float = 0.20


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
        use_neutralized:  If True, use *_score_neutral columns (post-Fix #1).
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
            "factor_densities": {factor: float},        # fraction of non-zero values
            "correlation_matrix": {fi: {fj: rho}},     # full matrix, all pairs
            "max_abs_correlation": float,               # dense pairs only (Fix #8.1)
            "max_pair": [str, str] | [],                # dense pairs only
            "warnings": list[str],                      # dense pairs only
            "errors": list[str],                        # dense pairs only
            "low_density_pairs": list[dict],            # sparsity-excluded pairs
            "reliable_correlations_only": bool,         # True when sparsity filter active
        }
        On failure: {"error": str, "computed_at": iso_timestamp}
    """
    if factors is None:
        factors = FACTORS_TO_MONITOR

    ts = datetime.now(timezone.utc).isoformat()

    try:
        # ── Filter by market ──────────────────────────────────────────────────
        if market_filter is not None:
            _us_values = {"US", "USA"} if market_filter in ("US", "USA") else {market_filter}
            rows = [r for r in results if r.get("market", "USA") in _us_values]
        else:
            rows = list(results)

        if len(rows) < min_observations:
            msg = (
                f"Insufficient observations for orthogonality diagnostic: "
                f"{len(rows)} rows (market={market_filter}, need >={min_observations}). "
                f"Factor orthogonality not computed."
            )
            log.warning(msg)
            return {"error": msg, "computed_at": ts}

        # ── Build factor arrays, handle neutralized vs raw ────────────────────
        suffix = "_neutral" if use_neutralized else ""
        evaluated: list[str] = []
        skipped: list[str] = []
        arrays: dict[str, np.ndarray] = {}
        densities: dict[str, float] = {}

        n_total = len(rows)
        for factor in factors:
            col = f"{factor}{suffix}" if use_neutralized else factor
            vals = []
            for r in rows:
                v = r.get(col)
                if v is None and use_neutralized:
                    v = r.get(factor)
                vals.append(v)

            none_count = sum(1 for v in vals if v is None)
            if none_count > n_total * 0.5:
                skipped.append(factor)
                log.debug(
                    "Orthogonality: skipping %s — %.0f%% None values",
                    factor, none_count / n_total * 100,
                )
                continue

            numeric = [float(v) for v in vals if v is not None]
            median_fill = float(np.median(numeric)) if numeric else 0.0
            arr = np.array([float(v) if v is not None else median_fill for v in vals])
            arrays[factor] = arr
            evaluated.append(factor)

            # Fix #8.1: compute density = fraction of strictly non-zero values
            nonzero = int(np.count_nonzero(arr))
            densities[factor] = round(nonzero / n_total, 4)

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
        low_density_pairs: list[dict[str, Any]] = []

        for fi in evaluated:
            matrix[fi] = {}
            for fj in evaluated:
                if fi == fj:
                    matrix[fi][fj] = 1.0
                    continue
                rho = _spearman_rho(arrays[fi], arrays[fj])
                matrix[fi][fj] = round(rho, 4) if not np.isnan(rho) else None

                if fi >= fj:
                    continue  # process unordered pairs once

                abs_rho = abs(rho) if not np.isnan(rho) else 0.0
                d_i = densities.get(fi, 1.0)
                d_j = densities.get(fj, 1.0)
                is_sparse = d_i < SPARSITY_THRESHOLD or d_j < SPARSITY_THRESHOLD

                if is_sparse:
                    # Fix #8.1: sparsity artefact — record but exclude from scoring
                    low_density_pairs.append({
                        "factor_a":    fi,
                        "factor_b":    fj,
                        "correlation": round(rho, 4) if not np.isnan(rho) else None,
                        "density_a":   d_i,
                        "density_b":   d_j,
                        "reason":      (
                            f"{'both factors' if d_i < SPARSITY_THRESHOLD and d_j < SPARSITY_THRESHOLD else 'one factor'} "
                            f"below SPARSITY_THRESHOLD={SPARSITY_THRESHOLD}"
                        ),
                    })
                    continue  # does not contribute to max_abs_rho / warnings / errors

                if abs_rho > max_abs_rho:
                    max_abs_rho = abs_rho
                    max_pair = [fi, fj]

                if not np.isnan(rho):
                    if abs_rho >= CORRELATION_ERROR_THRESHOLD:
                        errors.append(
                            f"({fi}, {fj}): |rho|={abs_rho:.3f} >= {CORRELATION_ERROR_THRESHOLD}"
                            f" -- factors are not effectively independent"
                        )
                    elif abs_rho >= CORRELATION_WARN_THRESHOLD:
                        warnings.append(
                            f"({fi}, {fj}): |rho|={abs_rho:.3f} >= {CORRELATION_WARN_THRESHOLD}"
                            f" -- correlation approaching redundancy threshold"
                        )

        any_sparse = len(low_density_pairs) > 0

        log.info(
            "Factor orthogonality (US, neutralized, dense factors only): "
            "max |rho| = %.3f between (%s, %s) across %d observations. "
            "Low-density pairs (sparsity-excluded): %d.",
            max_abs_rho,
            max_pair[0] if max_pair else "n/a",
            max_pair[1] if max_pair else "n/a",
            len(rows),
            len(low_density_pairs),
        )

        if low_density_pairs:
            log.info(
                "Sparsity-excluded pairs (tied-rank artefacts, not design redundancies): %s",
                ", ".join(
                    f"{p['factor_a'].replace('_score','')}<->{p['factor_b'].replace('_score','')}"
                    for p in low_density_pairs[:6]
                ),
            )

        return {
            "computed_at":              ts,
            "market":                   market_filter or "ALL",
            "n_observations":           len(rows),
            "factors_evaluated":        evaluated,
            "factors_skipped":          skipped,
            "factor_densities":         densities,
            "correlation_matrix":       matrix,
            "max_abs_correlation":      round(max_abs_rho, 4),
            "max_pair":                 max_pair,
            "warnings":                 warnings,
            "errors":                   errors,
            "low_density_pairs":        low_density_pairs,
            "reliable_correlations_only": any_sparse,
        }

    except Exception as exc:
        msg = f"compute_factor_correlation_matrix failed: {exc}"
        log.error(msg)
        return {"error": msg, "computed_at": ts}
