# Path: src/config/factor_matrix.py
#
# FACTOR_MATRIX_V3 — canonical source for the v3.0 pillar scoring matrices.
# Version: v3.0-pillars (2026-06)
#
# Design (docs: plan "regime_trader Scoring Engine v3.0"):
#   Each region gets EXACTLY 9 factors in 3 pillars × 3 factors:
#     fundamental  — Fundamental Quality & Health
#     consensus    — Market Sentiment & Consensus Momentum
#     alternative  — Alternative / Micro-Structural Flow
#
#   Regional theses:
#     US   "Alternative Alpha"   — P3-heavy (0.45): Form 4 insider composite
#          [Cohen, Malloy & Pomorski 2012], congressional flow, 13F deltas.
#     EU   "Quality & Value"     — P1-heavy (0.45): Piotroski 2000, Damodaran
#          FCF yield, Fama & French 1992 P/B; consensus momentum compensates
#          for structurally absent alt-data.
#     ASIA "Growth & Reversion"  — P2-heavy (0.40): estimate revisions
#          [Chan, Jegadeesh & Lakonishok 1996], margin trajectory,
#          price-to-consensus gaps; Amihud 2002 illiquidity premium
#          [Kim & Lee 2014] justifies amihud_shock at 0.12.
#
# Missing-data semantics (enforced by engine_v3 + neutralization):
#   signed=True   → factor centers at 0.5; a true 0.0 is a real observation
#                   (zero_is_dead=False); unavailability is ALWAYS None.
#   signed=False  → 0.0 is a genuine dead signal (zero_is_dead=True),
#                   excluded from bucket μ/σ and re-attached after
#                   neutralization (mass-point defense).
#
# Cross-contamination layer 1 (structural): EU/ASIA matrices simply do not
# contain US-structural factors — weights cannot exist. Layers 2-3 live in
# src/scoring/engine_v3.py (runtime mask) and neutralization bucket keys.
#
# Weights sum check enforced at module load time via assert (CLAUDE.md §3).
# Any modification must maintain sum == 1.0 per region, 3×3 pillar shape.
#
# Pre-registered contingencies (see plan §Deliverable 1):
#   C1 momentum re-entry at 0.05 (per-region, exact IC trigger).
#   C2 inst_concentration → shareholder_yield [Boudoukh, Michaely,
#      Richardson & Roberts 2007] if intl 13F coverage < 30%.
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Tuple

WEIGHTS_VERSION_V3 = "v3.0-pillars"

REGIONS: Tuple[str, ...] = ("US", "EU", "ASIA")
PILLARS: Tuple[str, ...] = ("fundamental", "consensus", "alternative")

# ── US surge-vs-fundamentals interaction (engine_v3, US only) ─────────────────
# final = clip(base + λ·max(0, pillar_alt − τ)·max(0, pillar_fund − 0.5), 0, 1)
# One-sided: corroboration bonus only, no penalty — turnaround insider buying
# leads trailing fundamentals [Cohen, Malloy & Pomorski 2012]; distress is
# already suppressed by the Piotroski gate (F<3 → ×0.0).
SURGE_TAU: float = 0.80
SURGE_LAMBDA: float = 0.5
# Pillars are convex combinations of sigmoid outputs clipped at [0.01, 0.99],
# so surge ≤ 0.19 and conf ≤ 0.49 → bound is 0.04655 (NOT the naive 0.0475).
SURGE_MAX_BONUS: float = SURGE_LAMBDA * (0.99 - SURGE_TAU) * (0.99 - 0.50)

# ── Missing-data protocol: signed factors center at 0.5 ───────────────────────
SIGNED_FACTORS: FrozenSet[str] = frozenset({
    "analyst_revision",
    "pead_surprise",
    "price_target_upside",
    "revision_velocity",
    "margin_expansion",
    "inst_flow_13f",
    "inst_concentration",
})

# ── Cross-contamination: factors that must NEVER score outside the US ─────────
# congress         — US STOCK Act / Quiver S3 only (no EU/Asia equivalent)
# transcript_tone  — FMP earning-call-transcript-latest US-only (not in any
#                    v3 matrix; listed for runtime-mask completeness)
# insider_alpha / inst_flow_13f / pead_surprise / quality_dupont — by matrix
#                    design (US thesis factors), guarded identically.
US_STRUCTURAL_ONLY: FrozenSet[str] = frozenset({
    "congress",
    "inst_flow_13f",
    "pead_surprise",
    "insider_alpha",
    "quality_dupont",
    "transcript_tone",
})


@dataclass(frozen=True)
class FactorSpec:
    """One factor's slot in a regional matrix.

    weight        — final-score weight (region weights sum to 1.0).
    pillar        — one of PILLARS.
    sources       — FMP stable/ endpoints (or named feeds) the factor reads.
    signed        — True: centered at 0.5, 0.0 is a real observation.
    zero_is_dead  — True: raw 0.0 is a dead signal, excluded from bucket stats.
    us_only       — True: structurally barred outside the US (runtime mask).
    """

    weight: float
    pillar: str
    sources: Tuple[str, ...]
    signed: bool
    zero_is_dead: bool
    us_only: bool


def _spec(
    name: str, weight: float, pillar: str, *sources: str
) -> Tuple[str, FactorSpec]:
    """Build (name, FactorSpec) deriving protocol flags from the master sets."""
    return name, FactorSpec(
        weight=weight,
        pillar=pillar,
        sources=sources,
        signed=name in SIGNED_FACTORS,
        zero_is_dead=name not in SIGNED_FACTORS,
        us_only=name in US_STRUCTURAL_ONLY,
    )


FACTOR_MATRIX_V3: Dict[str, Dict[str, FactorSpec]] = {
    # ── US — "Alternative Alpha Vector" (P1=0.30 / P2=0.25 / P3=0.45) ────────
    "US": dict([
        # P1 — quality_dupont: negative-range-preserving DuPont composite
        _spec("quality_dupont", 0.12, "fundamental",
              "ratios-ttm-bulk", "ratios-ttm"),
        # P1 — Damodaran (2006): TTM FCF / EV
        _spec("fcf_yield", 0.10, "fundamental",
              "cash-flow-statement", "enterprise-values"),
        # P1 — Piotroski (2000); also retains the multiplicative final gate
        _spec("quality_piotroski", 0.08, "fundamental",
              "ratios-ttm-bulk", "ratios-ttm"),
        # P2 — Chan, Jegadeesh & Lakonishok (1996); damped toward 0.5
        _spec("analyst_revision", 0.08, "consensus", "analyst-estimates"),
        # P2 — Bernard & Thomas (1989) post-earnings announcement drift
        _spec("pead_surprise", 0.09, "consensus", "earnings"),
        # P2 — Brav & Lehavy (2003); same-symbol + GBX/ratio guards
        _spec("price_target_upside", 0.08, "consensus",
              "price-target-consensus", "quote"),
        # P3 — Cohen, Malloy & Pomorski (2012) + Lakonishok & Lee (2001):
        #      conviction + breadth residual + velocity/C-suite micro term
        _spec("insider_alpha", 0.30, "alternative",
              "edgar-form4", "insider-trading/search"),
        # P3 — STOCK Act flow with 30d/180d surge multiplier
        _spec("congress", 0.05, "alternative", "quiver-s3-stock-watcher"),
        # P3 — 13F position deltas (investorsHolding / inc-red / ownership Δ)
        _spec("inst_flow_13f", 0.10, "alternative",
              "institutional-ownership/symbol-positions-summary"),
    ]),
    # ── EU — "Quality & Value Vector" (P1=0.45 / P2=0.35 / P3=0.20) ──────────
    "EU": dict([
        _spec("quality_piotroski", 0.12, "fundamental",
              "ratios-ttm-bulk", "ratios-ttm"),
        _spec("fcf_yield", 0.18, "fundamental",
              "cash-flow-statement", "enterprise-values"),
        # Fama & French (1992) P/B inversion with <1.0 bonus
        _spec("pb_value_up", 0.15, "fundamental", "ratios-ttm", "quote"),
        # Givoly & Lakonishok (1979) — the bulk consensus momentum filter
        _spec("analyst_consensus", 0.10, "consensus",
              "upgrades-downgrades-consensus-bulk"),
        _spec("analyst_revision", 0.13, "consensus", "analyst-estimates"),
        _spec("price_target_upside", 0.12, "consensus",
              "price-target-consensus", "quote"),
        # Synthetic alt: institutional ownership concentration (13F endpoint)
        _spec("inst_concentration", 0.07, "alternative",
              "institutional-ownership/symbol-positions-summary"),
        # Synthetic alt: payout sustainability (income-quality value tilt)
        _spec("dividend_sustain", 0.08, "alternative",
              "ratios-ttm", "cash-flow-statement"),
        # Amihud (2002) illiquidity shock
        _spec("amihud_shock", 0.05, "alternative", "historical-price-eod/full"),
    ]),
    # ── ASIA — "Growth & Reversion Vector" (P1=0.35 / P2=0.40 / P3=0.25) ─────
    "ASIA": dict([
        # TTM OPM trajectory; quarterly with discrete-window validation,
        # annual fallback (filingDate-anchored)
        _spec("margin_expansion", 0.13, "fundamental", "income-statement"),
        # Greenblatt (2005): (ROE + ROCE) / 2
        _spec("roic_quality", 0.10, "fundamental", "ratios-ttm"),
        _spec("quality_piotroski", 0.12, "fundamental",
              "ratios-ttm-bulk", "ratios-ttm"),
        _spec("analyst_revision", 0.15, "consensus", "analyst-estimates"),
        # CJL (1996) second derivative of the revision path
        _spec("revision_velocity", 0.10, "consensus", "analyst-estimates"),
        _spec("price_target_upside", 0.15, "consensus",
              "price-target-consensus", "quote"),
        _spec("inst_concentration", 0.07, "alternative",
              "institutional-ownership/symbol-positions-summary"),
        _spec("dividend_sustain", 0.06, "alternative",
              "ratios-ttm", "cash-flow-statement"),
        # Kim & Lee (2014): APAC illiquidity premium → 2.4× the EU weight
        _spec("amihud_shock", 0.12, "alternative", "historical-price-eod/full"),
    ]),
}

# Declared independently of the matrix so load-time asserts cross-check the
# two sources against each other (a typo in either fails the import).
PILLAR_WEIGHTS: Dict[str, Dict[str, float]] = {
    "US":   {"fundamental": 0.30, "consensus": 0.25, "alternative": 0.45},
    "EU":   {"fundamental": 0.45, "consensus": 0.35, "alternative": 0.20},
    "ASIA": {"fundamental": 0.35, "consensus": 0.40, "alternative": 0.25},
}

REGION_FACTOR_MASK: Dict[str, FrozenSet[str]] = {
    region: frozenset(matrix) for region, matrix in FACTOR_MATRIX_V3.items()
}

# Flat weight dicts for callers that don't need FactorSpec metadata.
WEIGHTS_US_V3: Dict[str, float] = {
    name: spec.weight for name, spec in FACTOR_MATRIX_V3["US"].items()
}
WEIGHTS_EU_V3: Dict[str, float] = {
    name: spec.weight for name, spec in FACTOR_MATRIX_V3["EU"].items()
}
WEIGHTS_APAC_V3: Dict[str, float] = {
    name: spec.weight for name, spec in FACTOR_MATRIX_V3["ASIA"].items()
}

# ── Load-time integrity asserts (import fails loudly on any drift) ────────────
assert set(FACTOR_MATRIX_V3) == set(REGIONS)
assert set(PILLAR_WEIGHTS) == set(REGIONS)

for _region in REGIONS:
    _matrix = FACTOR_MATRIX_V3[_region]
    _total = sum(_s.weight for _s in _matrix.values())
    assert abs(_total - 1.0) < 1e-6, (
        f"FACTOR_MATRIX_V3[{_region}] sums to {_total:.8f}, not 1.0"
    )
    assert len(_matrix) == 9, (
        f"FACTOR_MATRIX_V3[{_region}] has {len(_matrix)} factors, expected 9"
    )
    _pillar_total = sum(PILLAR_WEIGHTS[_region].values())
    assert abs(_pillar_total - 1.0) < 1e-6, (
        f"PILLAR_WEIGHTS[{_region}] sums to {_pillar_total:.8f}, not 1.0"
    )
    for _pillar in PILLARS:
        _members = [_s for _s in _matrix.values() if _s.pillar == _pillar]
        assert len(_members) == 3, (
            f"{_region}/{_pillar}: {len(_members)} factors, expected 3"
        )
        _psum = sum(_s.weight for _s in _members)
        assert abs(_psum - PILLAR_WEIGHTS[_region][_pillar]) < 1e-6, (
            f"{_region}/{_pillar}: factor sum {_psum:.8f} != declared "
            f"{PILLAR_WEIGHTS[_region][_pillar]:.8f}"
        )

for _region in ("EU", "ASIA"):
    _leaked = set(FACTOR_MATRIX_V3[_region]) & US_STRUCTURAL_ONLY
    assert not _leaked, (
        f"US-structural factors leaked into {_region} matrix: {sorted(_leaked)}"
    )
