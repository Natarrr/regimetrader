# Path: src/scoring/engine_v3.py
"""src/scoring/engine_v3.py
v3.0 pillar scoring engine — pure math, no I/O.

Loop (plan "regime_trader Scoring Engine v3.0", Deliverable 2):
    1. assert_region_isolation  — market check (always raises on mismatch) +
       out-of-mask factor guard (strict: raise; prod: force None + evidence).
    2. NO raw-space winsorization — scorers emit bounded [0,1] via fixed
       economic clip ranges; sample-dependent winsorization before sector
       z-scoring would truncate structurally-extreme sectors.
    3. neutralize_factors — sector-bucket z → sigmoid [0.01, 0.99], with v3
       semantics: none_passthrough=True, per-factor zero_is_dead from
       FactorSpec (signed factors' true 0.0 enters bucket stats).
    4. Pillar aggregation (first-class columns) with None-reweighting at
       both the factor level (within pillar) and the pillar level (base).
    5. US-only one-sided surge interaction
       [Cohen, Malloy & Pomorski 2012], then the Piotroski gate.
    6. weight_coverage_v3 / _low_coverage_v3 diagnostics; region-wide
       factor-blackout telemetry (None-rate ≥ 90%).

Grinold & Kahn (2000) ch.7: IC is only meaningful after removing
common-factor exposures — bucket keys are market-prefixed and each region
is a SEPARATE score_universe_v3() call, so cross-region statistics are
impossible by construction (contamination layer 3).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, FrozenSet, List, Optional

from src.config.factor_matrix import (
    FACTOR_MATRIX_V3,
    PILLAR_WEIGHTS,
    PILLARS,
    REGION_FACTOR_MASK,
    REGIONS,
    SURGE_LAMBDA,
    SURGE_TAU,
    US_STRUCTURAL_ONLY,
)
from src.config.weights import _piotroski_gate_multiplier
from src.scoring.market_config import LOW_COVERAGE_THRESHOLD, MIN_BUCKET_SIZE
from src.scoring.neutralization import neutralize_factors

log = logging.getLogger(__name__)

# Markets accepted per scoring pool (rows carry pipeline market strings).
_REGION_MARKETS: Dict[str, FrozenSet[str]] = {
    "US": frozenset({"US", "USA"}),
    "EU": frozenset({"EU", "EUROPE"}),
    "ASIA": frozenset({"ASIA"}),
}

# Region-wide None-rate at/above which a factor is flagged as a feed blackout.
_BLACKOUT_NONE_RATE = 0.90

# Raw Piotroski F-score row key (matches run_pipeline / fmp_fetcher output).
_PIOTROSKI_RAW_KEY = "quality_piotroski_raw"

# Every factor any region knows about — the complement of a region's mask is
# what must never carry a value in that region's rows.
_ALL_FACTORS: FrozenSet[str] = frozenset(
    set().union(*REGION_FACTOR_MASK.values()) | US_STRUCTURAL_ONLY
)


def _strict_mode(strict: Optional[bool]) -> bool:
    if strict is not None:
        return strict
    return os.getenv("STRICT_REGION_GUARD", "") == "1"


def assert_region_isolation(
    rows: List[Dict[str, Any]],
    region: str,
    strict: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Contamination layer 2 — defensive runtime guard.

    Market mismatch ALWAYS raises (a row from another pool is a pipeline
    bug, never maskable). Out-of-mask factors with non-None values raise in
    strict mode (STRICT_REGION_GUARD=1 — CI / staging drill); in prod they
    are forced to None on a copy, recorded in ``_contamination_masked``,
    and logged at ERROR. Input rows are never mutated.
    """
    if region not in REGIONS:
        raise ValueError(f"unknown region {region!r} — expected one of {REGIONS}")
    is_strict = _strict_mode(strict)
    allowed_markets = _REGION_MARKETS[region]
    foreign = sorted(_ALL_FACTORS - REGION_FACTOR_MASK[region])

    out: List[Dict[str, Any]] = []
    for row in rows:
        market = str(row.get("market", "USA")).upper()
        if market not in allowed_markets:
            raise ValueError(
                f"region isolation violated: {row.get('ticker', '?')} has "
                f"market={market!r}, cannot be scored in the {region} pool"
            )
        leaked = [n for n in foreign if row.get(f"{n}_score") is not None]
        row = dict(row)
        if leaked:
            if is_strict:
                raise ValueError(
                    f"cross-contamination (strict): {row.get('ticker', '?')} "
                    f"carries out-of-mask factors {leaked} in the {region} pool"
                )
            log.error(
                "cross-contamination masked: %s carries %s in the %s pool — "
                "forced to None (upstream guard breach, investigate)",
                row.get("ticker", "?"), leaked, region,
            )
            for name in leaked:
                row[f"{name}_score"] = None
            row["_contamination_masked"] = leaked
        out.append(row)
    return out


def _detect_blackouts(rows: List[Dict[str, Any]], region: str) -> List[str]:
    """Region-wide feed-blackout telemetry on RAW factor columns."""
    n = len(rows)
    if n == 0:
        return []
    blackout = [
        name for name in FACTOR_MATRIX_V3[region]
        if sum(1 for r in rows if r.get(f"{name}_score") is None) / n
        >= _BLACKOUT_NONE_RATE
    ]
    if blackout:
        log.error(
            "factor blackout in %s pool: %s is None for >=%d%% of the "
            "universe — pillar reweighting absorbs it (NO neutral 0.5 "
            "injection), but the feed needs investigation",
            region, blackout, int(_BLACKOUT_NONE_RATE * 100),
        )
    return blackout


def score_universe_v3(
    rows: List[Dict[str, Any]],
    region: str,
    strict: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    """Score one regional universe under the v3.0 pillar matrix.

    Adds per row: ``pillar_{fundamental,consensus,alternative}_score``,
    ``final_score_v3``, ``weight_coverage_v3``, ``_low_coverage_v3``,
    plus ``_factor_blackout`` when a region-wide feed outage is detected
    and ``_contamination_masked`` when layer-2 masking fired.
    """
    if not rows:
        return []
    rows = assert_region_isolation(rows, region, strict=strict)

    matrix = FACTOR_MATRIX_V3[region]
    factor_keys = tuple(f"{name}_score" for name in matrix)
    zero_is_dead = {
        f"{name}_score": spec.zero_is_dead for name, spec in matrix.items()
    }
    blackout = _detect_blackouts(rows, region)

    scored = neutralize_factors(
        rows,
        factors=factor_keys,
        min_bucket_size=MIN_BUCKET_SIZE,
        none_passthrough=True,
        zero_is_dead=zero_is_dead,
    )

    region_pillar_w = PILLAR_WEIGHTS[region]
    for row in scored:
        # ── Pillar aggregation with factor-level None-reweighting ────────
        pillar_vals: Dict[str, Optional[float]] = {}
        coverage = 0.0
        for pillar in PILLARS:
            num = den = 0.0
            for name, spec in matrix.items():
                if spec.pillar != pillar:
                    continue
                neutral = row.get(f"{name}_score_neutral")
                if neutral is None:
                    continue
                num += spec.weight * neutral
                den += spec.weight
                coverage += spec.weight
            value = (num / den) if den > 0 else None
            pillar_vals[pillar] = value
            row[f"pillar_{pillar}_score"] = (
                round(value, 6) if value is not None else None
            )

        if blackout:
            row["_factor_blackout"] = list(blackout)
        row["weight_coverage_v3"] = round(coverage, 6)
        row["_low_coverage_v3"] = coverage < LOW_COVERAGE_THRESHOLD

        # ── Base with pillar-level None-reweighting ───────────────────────
        available = {p: v for p, v in pillar_vals.items() if v is not None}
        if not available:
            row["final_score_v3"] = None
            row["_low_coverage_v3"] = True
            continue
        w_sum = sum(region_pillar_w[p] for p in available)
        base = sum(region_pillar_w[p] * v for p, v in available.items()) / w_sum

        # ── One-sided surge interaction (US only; λ=0 ex-US) ──────────────
        final = base
        if region == "US":
            alt = pillar_vals["alternative"]
            fund = pillar_vals["fundamental"]
            if alt is not None and fund is not None:
                bonus = (
                    SURGE_LAMBDA
                    * max(0.0, alt - SURGE_TAU)
                    * max(0.0, fund - 0.5)
                )
                final = max(0.0, min(1.0, base + bonus))

        # ── Piotroski distress gate (after surge, per design) ─────────────
        final *= _piotroski_gate_multiplier(row.get(_PIOTROSKI_RAW_KEY))
        row["final_score_v3"] = round(final, 6)

    return scored
