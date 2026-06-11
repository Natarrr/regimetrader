"""tests/scoring/test_factor_matrix.py — v3.0 pillar factor matrix invariants.

Tests:
  1. Shape: exactly 9 factors per region, exactly 3 per pillar.
  2. Weight integrity: per-region sum == 1.0; per-pillar sums match PILLAR_WEIGHTS.
  3. Cross-contamination: US-structural factors absent from EU/ASIA matrices/masks.
  4. Signed/zero_is_dead semantics per the missing-data protocol.
  5. Surge constants incl. the one-sided bound 0.04655 (0.99 sigmoid clip).

No live network calls. Pure config validation.
"""
from __future__ import annotations

import pytest

from src.config.factor_matrix import (
    FACTOR_MATRIX_V3,
    FactorSpec,
    PILLAR_WEIGHTS,
    PILLARS,
    REGION_FACTOR_MASK,
    REGIONS,
    SIGNED_FACTORS,
    SURGE_LAMBDA,
    SURGE_MAX_BONUS,
    SURGE_TAU,
    US_STRUCTURAL_ONLY,
    WEIGHTS_APAC_V3,
    WEIGHTS_EU_V3,
    WEIGHTS_US_V3,
    WEIGHTS_VERSION_V3,
)

_EXPECTED_FACTORS = {
    "US": {
        "quality_dupont", "fcf_yield", "quality_piotroski",
        "analyst_revision", "pead_surprise", "price_target_upside",
        "insider_alpha", "congress", "inst_flow_13f",
    },
    "EU": {
        "quality_piotroski", "fcf_yield", "pb_value_up",
        "analyst_consensus", "analyst_revision", "price_target_upside",
        "inst_concentration", "dividend_sustain", "amihud_shock",
    },
    "ASIA": {
        "margin_expansion", "roic_quality", "quality_piotroski",
        "analyst_revision", "revision_velocity", "price_target_upside",
        "inst_concentration", "dividend_sustain", "amihud_shock",
    },
}

_EXPECTED_PILLAR_WEIGHTS = {
    "US":   {"fundamental": 0.30, "consensus": 0.25, "alternative": 0.45},
    "EU":   {"fundamental": 0.45, "consensus": 0.35, "alternative": 0.20},
    "ASIA": {"fundamental": 0.35, "consensus": 0.40, "alternative": 0.25},
}

_EXPECTED_WEIGHTS = {
    "US": {
        "quality_dupont": 0.12, "fcf_yield": 0.10, "quality_piotroski": 0.08,
        "analyst_revision": 0.08, "pead_surprise": 0.09, "price_target_upside": 0.08,
        "insider_alpha": 0.30, "congress": 0.05, "inst_flow_13f": 0.10,
    },
    "EU": {
        "quality_piotroski": 0.12, "fcf_yield": 0.18, "pb_value_up": 0.15,
        "analyst_consensus": 0.10, "analyst_revision": 0.13, "price_target_upside": 0.12,
        "inst_concentration": 0.07, "dividend_sustain": 0.08, "amihud_shock": 0.05,
    },
    "ASIA": {
        "margin_expansion": 0.13, "roic_quality": 0.10, "quality_piotroski": 0.12,
        "analyst_revision": 0.15, "revision_velocity": 0.10, "price_target_upside": 0.15,
        "inst_concentration": 0.07, "dividend_sustain": 0.06, "amihud_shock": 0.12,
    },
}

_EXPECTED_SIGNED = {
    "analyst_revision", "pead_surprise", "price_target_upside",
    "revision_velocity", "margin_expansion", "inst_flow_13f", "inst_concentration",
}


# ── Shape invariants ──────────────────────────────────────────────────────────

class TestMatrixShape:
    def test_regions(self):
        assert REGIONS == ("US", "EU", "ASIA")
        assert set(FACTOR_MATRIX_V3.keys()) == set(REGIONS)

    def test_pillars(self):
        assert PILLARS == ("fundamental", "consensus", "alternative")

    @pytest.mark.parametrize("region", ["US", "EU", "ASIA"])
    def test_exactly_nine_factors(self, region):
        assert len(FACTOR_MATRIX_V3[region]) == 9

    @pytest.mark.parametrize("region", ["US", "EU", "ASIA"])
    def test_exactly_three_per_pillar(self, region):
        for pillar in PILLARS:
            n = sum(
                1 for spec in FACTOR_MATRIX_V3[region].values()
                if spec.pillar == pillar
            )
            assert n == 3, f"{region}/{pillar} has {n} factors, expected 3"

    @pytest.mark.parametrize("region", ["US", "EU", "ASIA"])
    def test_expected_factor_membership(self, region):
        assert set(FACTOR_MATRIX_V3[region].keys()) == _EXPECTED_FACTORS[region]

    @pytest.mark.parametrize("region", ["US", "EU", "ASIA"])
    def test_specs_well_formed(self, region):
        for name, spec in FACTOR_MATRIX_V3[region].items():
            assert isinstance(spec, FactorSpec), name
            assert spec.weight > 0, name
            assert spec.pillar in PILLARS, name
            assert isinstance(spec.sources, tuple) and len(spec.sources) > 0, name
            assert all(isinstance(s, str) and s for s in spec.sources), name


# ── Weight integrity (CLAUDE.md: sum-to-1 assertion convention) ───────────────

class TestWeightIntegrity:
    @pytest.mark.parametrize("region", ["US", "EU", "ASIA"])
    def test_region_weights_sum_to_one(self, region):
        total = sum(spec.weight for spec in FACTOR_MATRIX_V3[region].values())
        assert abs(total - 1.0) < 1e-6, f"{region} sums to {total:.8f}"

    @pytest.mark.parametrize("region", ["US", "EU", "ASIA"])
    def test_pillar_weights_match_factor_sums(self, region):
        for pillar in PILLARS:
            pillar_sum = sum(
                spec.weight for spec in FACTOR_MATRIX_V3[region].values()
                if spec.pillar == pillar
            )
            assert pillar_sum == pytest.approx(
                PILLAR_WEIGHTS[region][pillar], abs=1e-6
            ), f"{region}/{pillar}"

    @pytest.mark.parametrize("region", ["US", "EU", "ASIA"])
    def test_exact_pillar_weights(self, region):
        assert PILLAR_WEIGHTS[region] == pytest.approx(
            _EXPECTED_PILLAR_WEIGHTS[region], abs=1e-6
        )

    @pytest.mark.parametrize("region", ["US", "EU", "ASIA"])
    def test_exact_factor_weights(self, region):
        actual = {n: s.weight for n, s in FACTOR_MATRIX_V3[region].items()}
        assert actual == pytest.approx(_EXPECTED_WEIGHTS[region], abs=1e-6)

    def test_derived_weight_dicts_match_matrix(self):
        for weights, region in (
            (WEIGHTS_US_V3, "US"),
            (WEIGHTS_EU_V3, "EU"),
            (WEIGHTS_APAC_V3, "ASIA"),
        ):
            expected = {n: s.weight for n, s in FACTOR_MATRIX_V3[region].items()}
            assert weights == pytest.approx(expected, abs=1e-9), region

    def test_version_string(self):
        assert WEIGHTS_VERSION_V3 == "v3.0-pillars"


# ── Cross-contamination structural layer ─────────────────────────────────────

class TestContaminationStructure:
    def test_us_structural_only_contents(self):
        assert US_STRUCTURAL_ONLY >= {
            "congress", "inst_flow_13f", "pead_surprise",
            "insider_alpha", "quality_dupont", "transcript_tone",
        }

    @pytest.mark.parametrize("region", ["EU", "ASIA"])
    def test_no_us_structural_factors_in_intl_matrix(self, region):
        leaked = set(FACTOR_MATRIX_V3[region].keys()) & US_STRUCTURAL_ONLY
        assert leaked == set(), f"{region} matrix leaks US factors: {leaked}"

    @pytest.mark.parametrize("region", ["EU", "ASIA"])
    def test_no_us_structural_factors_in_intl_mask(self, region):
        leaked = REGION_FACTOR_MASK[region] & US_STRUCTURAL_ONLY
        assert leaked == set(), f"{region} mask leaks US factors: {leaked}"

    @pytest.mark.parametrize("region", ["US", "EU", "ASIA"])
    def test_masks_match_matrices(self, region):
        assert REGION_FACTOR_MASK[region] == frozenset(FACTOR_MATRIX_V3[region])

    def test_us_only_flag_consistency(self):
        for region in REGIONS:
            for name, spec in FACTOR_MATRIX_V3[region].items():
                assert spec.us_only == (name in US_STRUCTURAL_ONLY), (
                    f"{region}/{name}: us_only={spec.us_only}"
                )


# ── Missing-data protocol semantics ───────────────────────────────────────────

class TestSignedSemantics:
    def test_signed_factor_set(self):
        assert SIGNED_FACTORS == frozenset(_EXPECTED_SIGNED)

    def test_zero_is_dead_iff_unsigned(self):
        for region in REGIONS:
            for name, spec in FACTOR_MATRIX_V3[region].items():
                expected_signed = name in SIGNED_FACTORS
                assert spec.signed == expected_signed, f"{region}/{name}"
                assert spec.zero_is_dead == (not expected_signed), (
                    f"{region}/{name}: signed factors must never treat 0.0 as dead"
                )


# ── US surge interaction constants ────────────────────────────────────────────

class TestSurgeConstants:
    def test_tau_and_lambda(self):
        assert SURGE_TAU == pytest.approx(0.80)
        assert SURGE_LAMBDA == pytest.approx(0.5)

    def test_max_bonus_consistent_with_sigmoid_clip(self):
        # Pillars are convex combinations of sigmoid outputs clipped at 0.99,
        # so surge ≤ 0.19 and conf ≤ 0.49 — the bound is 0.04655, not 0.0475.
        assert SURGE_MAX_BONUS == pytest.approx(
            SURGE_LAMBDA * (0.99 - SURGE_TAU) * (0.99 - 0.50), abs=1e-9
        )
        assert SURGE_MAX_BONUS == pytest.approx(0.04655, abs=1e-6)
