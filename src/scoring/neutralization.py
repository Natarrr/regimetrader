"""src/scoring/neutralization.py
Cross-sectional factor neutralization by sector × cap_tier bucket.

Theory — Grinold & Kahn (2000), "Active Portfolio Management" ch. 7:
    IC is only a valid predictor of active return AFTER removing common-factor
    exposures.  Raw scores contaminated by sector/size biases produce spurious
    IC estimates and degrade IR = IC × √breadth.

    Procedure:
      1. For each (market, sector, cap_tier) bucket with >= min_bucket_size
         non-zero observations, compute the bucket mean and std over the factor,
         EXCLUDING zero observations (dead-feed tickers should not distort the
         distribution).
      2. z-score each non-zero observation within the bucket.
      3. Map z-scores to [0, 1] via the logistic function (avoids hard clipping):
             score_neutral = sigmoid(z) = 1 / (1 + exp(-z))
         then clip to [0.01, 0.99] to preserve tail ordering.
      4. Zero-score tickers (dead feed) receive 0.0 in the neutralized column
         and a _neutralization_fallback of "zero" — they are excluded from
         bucket stats but their output is preserved as a meaningful floor.
      5. For buckets smaller than min_bucket_size, fall back to cap_tier only.
         If that is also < min_bucket_size, leave the score unchanged ("raw").

Bucket key includes "market" so EU/Asia tickers are never mixed with US.
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

# Audit P3.2 — no module-level default factor list. The old `_FACTORS_DEFAULT`
# carried legacy 5-factor names (edgar_score/insider_score/…) that no live key
# matches; any caller relying on it would silently neutralize NOTHING (the
# silent-zero trap). All callers pass an explicit `factors=` tuple, so the
# argument is now REQUIRED — a future omission fails loudly, never silently.
_SIGMOID_CLIP_LO = 0.01
_SIGMOID_CLIP_HI = 0.99

# Scores live on a [0, 1] scale; bucket std below this is floating-point
# noise from non-representable values (e.g., 6 × 0.7 sums inexactly leaving
# std ≈ 2e-16), NOT genuine cross-sectional spread. An exact `std == 0.0`
# check amplifies noise/noise into full z-scores.
_STD_EPS = 1e-9


def _sigmoid(z: float) -> float:
    # Numerically stable: clamp large |z| to avoid overflow in exp
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def _bucket_stats(values: list[float]) -> tuple[float, float]:
    """Return (mean, std) for a non-empty list; std=0 when all identical."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return mean, math.sqrt(variance)


def neutralize_factors(
    results: list[dict[str, Any]],
    factors: tuple[str, ...],
    group_by: tuple[str, ...] = ("sector", "cap_tier"),
    min_bucket_size: int = 5,
    fallback_group_by: tuple[str, ...] = ("cap_tier",),
    output_suffix: str = "_neutral",
    none_passthrough: bool = False,
    zero_is_dead: Optional[Dict[str, bool]] = None,
) -> list[dict[str, Any]]:
    """Apply cross-sectional z-score neutralization within sector × cap_tier buckets.

    For each (market, sector, cap_tier) bucket with >= min_bucket_size non-zero
    tickers, z-score each factor within the bucket then map via sigmoid to [0,1].
    Zero-score tickers (dead feed) are excluded from bucket stats and receive 0.0
    in their neutralized output.

    Buckets smaller than min_bucket_size fall back to cap_tier-only grouping,
    then to raw (unchanged) if still too small.

    Args:
        results:          List of ticker dicts from run_pipeline.run().
        factors:          Factor key names to neutralize.
        group_by:         Primary grouping keys (default: sector + cap_tier).
        min_bucket_size:  Minimum non-zero observations required to apply
                          bucket-level stats (default: 5).
        fallback_group_by: Keys for the fallback bucket (default: cap_tier only).
        output_suffix:    Suffix appended to each neutralized factor key.
        none_passthrough: v3.0 missing-data semantics (OPT-IN — default False
                          keeps v2.2 byte-identical): a raw value of None is
                          "data unavailable", excluded from bucket stats and
                          emitted as ``{factor}{output_suffix} = None`` so the
                          pillar aggregator reweights, instead of being
                          coerced to a bearish dead 0.0.
        zero_is_dead:     Per-factor override (default: all True = v2.2).
                          False marks a SIGNED factor (centered at 0.5) whose
                          true 0.0 is a real worst-case observation: it enters
                          bucket μ/σ and receives sigmoid(z), never the dead
                          floor (mass-point defense applies to unsigned only).

    Returns:
        New list of dicts with added ``{factor}{output_suffix}`` keys.
        Original score keys are preserved unchanged.
        Each dict also gains ``_neutralization_fallback`` ∈
        {"sector_cap_tier", "cap_tier", "raw", "zero", "none"}.
    """
    if not results:
        return results

    # ------------------------------------------------------------------
    # Build bucket index: bucket_key → list of (ticker_index, value)
    # Exclude zeros from stat computation but track their indices separately.
    # bucket_key = (market, *group_by_values) to keep markets isolated.
    # ------------------------------------------------------------------
    def _bucket_key(row: dict, keys: tuple[str, ...]) -> tuple:
        market = row.get("market", "USA")
        return (market,) + tuple(row.get(k, "Unknown") for k in keys)

    output = [dict(r) for r in results]  # shallow copy per row

    for factor in factors:
        dead_on_zero = True if zero_is_dead is None else zero_is_dead.get(factor, True)

        # Collect active values per primary bucket
        primary_nonzero: dict[tuple, list[tuple[int, float]]] = defaultdict(list)
        fallback_nonzero: dict[tuple, list[tuple[int, float]]] = defaultdict(list)
        zero_indices: list[int] = []
        none_indices: list[int] = []

        for idx, row in enumerate(output):
            raw = row.get(factor)
            if raw is None and none_passthrough:
                # v3: data unavailable — never enters stats, emits None.
                none_indices.append(idx)
                continue
            val = float(raw or 0.0)
            if val == 0.0 and dead_on_zero:
                zero_indices.append(idx)
                continue
            pk = _bucket_key(row, group_by)
            fk = _bucket_key(row, fallback_group_by)
            primary_nonzero[pk].append((idx, val))
            fallback_nonzero[fk].append((idx, val))

        # Track which tickers got primary treatment (avoid double-processing)
        processed: set[int] = set()

        # ── Primary bucket pass ────────────────────────────────────────
        for pk, entries in primary_nonzero.items():
            if len(entries) < min_bucket_size:
                continue  # will be handled in fallback pass
            values = [v for _, v in entries]
            mean, std = _bucket_stats(values)
            for idx, val in entries:
                if std <= _STD_EPS:
                    neutral = 0.5
                else:
                    z = (val - mean) / std
                    neutral = _sigmoid(z)
                neutral = max(_SIGMOID_CLIP_LO, min(_SIGMOID_CLIP_HI, neutral))
                output[idx][f"{factor}{output_suffix}"] = round(neutral, 6)
                output[idx]["_neutralization_fallback"] = "sector_cap_tier"
                processed.add(idx)

        # ── Fallback bucket pass ──────────────────────────────────────
        for fk, entries in fallback_nonzero.items():
            unprocessed = [(i, v) for i, v in entries if i not in processed]
            if not unprocessed:
                continue
            if len(unprocessed) < min_bucket_size:
                # raw — leave original score, just copy it
                for idx, val in unprocessed:
                    output[idx][f"{factor}{output_suffix}"] = round(val, 6)
                    output[idx]["_neutralization_fallback"] = "raw"
                    processed.add(idx)
                continue
            values = [v for _, v in unprocessed]
            mean, std = _bucket_stats(values)
            for idx, val in unprocessed:
                if std <= _STD_EPS:
                    neutral = 0.5
                else:
                    z = (val - mean) / std
                    neutral = _sigmoid(z)
                neutral = max(_SIGMOID_CLIP_LO, min(_SIGMOID_CLIP_HI, neutral))
                output[idx][f"{factor}{output_suffix}"] = round(neutral, 6)
                output[idx]["_neutralization_fallback"] = "cap_tier"
                processed.add(idx)

        # ── Zero-score tickers (dead feed) ────────────────────────────
        for idx in zero_indices:
            output[idx][f"{factor}{output_suffix}"] = 0.0
            if "_neutralization_fallback" not in output[idx]:
                output[idx]["_neutralization_fallback"] = "zero"

        # ── None tickers (data unavailable — v3 passthrough) ──────────
        for idx in none_indices:
            output[idx][f"{factor}{output_suffix}"] = None
            if "_neutralization_fallback" not in output[idx]:
                output[idx]["_neutralization_fallback"] = "none"

    # ------------------------------------------------------------------
    # Diagnostic logging
    # ------------------------------------------------------------------
    _log_diagnostics(output, factors, output_suffix, min_bucket_size, group_by)

    return output


def _log_diagnostics(
    results: list[dict],
    factors: tuple[str, ...],
    suffix: str,
    min_bucket_size: int,
    group_by: tuple[str, ...],
) -> None:
    fallback_counts: dict[str, int] = defaultdict(int)
    for row in results:
        fb = row.get("_neutralization_fallback", "unknown")
        fallback_counts[fb] += 1

    bucket_sizes: dict[tuple, int] = defaultdict(int)
    for row in results:
        market = row.get("market", "USA")
        key = (market,) + tuple(row.get(k, "Unknown") for k in group_by)
        bucket_sizes[key] += 1

    log.info(
        "Neutralization: %d buckets formed (min_size=%d). "
        "Coverage — sector×cap_tier: %d | cap_tier: %d | raw: %d | zero: %d",
        len(bucket_sizes),
        min_bucket_size,
        fallback_counts.get("sector_cap_tier", 0),
        fallback_counts.get("cap_tier", 0),
        fallback_counts.get("raw", 0),
        fallback_counts.get("zero", 0),
    )

    import statistics as _stats

    for factor in factors:
        raw_vals = [
            float(r.get(factor, 0.0) or 0.0)
            for r in results
            if float(r.get(factor, 0.0) or 0.0) > 0.0
        ]
        neu_vals = [
            float(r.get(f"{factor}{suffix}", 0.0) or 0.0)
            for r in results
            if float(r.get(f"{factor}{suffix}", 0.0) or 0.0) > 0.0
        ]
        if len(raw_vals) >= 2 and len(neu_vals) >= 2:
            log.info(
                "  %s: raw μ=%.3f σ=%.3f → neutral μ=%.3f σ=%.3f",
                factor,
                _stats.mean(raw_vals),
                _stats.pstdev(raw_vals),
                _stats.mean(neu_vals),
                _stats.pstdev(neu_vals),
            )

    # Congress sector-bias sanity check: correlation raw vs neutral cross-sector
    _log_congress_correlation(results, suffix)


def _log_congress_correlation(results: list[dict], suffix: str) -> None:
    """Log Pearson r between congress_score raw and neutralized (should be 0 < r < 0.9)."""
    pairs = [
        (float(r.get("congress_score", 0.0) or 0.0),
         float(r.get(f"congress_score{suffix}", 0.0) or 0.0))
        for r in results
        if float(r.get("congress_score", 0.0) or 0.0) > 0.0
           and float(r.get(f"congress_score{suffix}", 0.0) or 0.0) > 0.0
    ]
    if len(pairs) < 5:
        log.info("  congress_score correlation: insufficient non-zero pairs (%d)", len(pairs))
        return
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    n = len(pairs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in pairs)
    denom = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    if denom == 0:
        log.info("  congress_score correlation: undefined (zero variance)")
        return
    r = num / denom
    flag = "" if 0.0 < r < 0.9 else " ⚠ OUTSIDE expected range"
    log.info("  congress_score raw↔neutral Pearson r = %.3f%s", r, flag)
