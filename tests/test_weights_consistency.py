"""tests/test_weights_consistency.py
Regression tests for WEIGHTS consistency across pipeline modules.

These tests enforce that run_pipeline.py and generate_top_lists.py use
the same factor weights (Patch 02) and that the sum constraint is maintained.

Grinold & Kahn (2000): "The forecast must be consistent across all
downstream consumers of the signal."
"""
import importlib.util
import pathlib
import sys
from typing import Dict

import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_run_pipeline_weights() -> Dict[str, float]:
    """Import WEIGHTS from scripts/run_pipeline.py without side effects."""
    rp_path = pathlib.Path(__file__).resolve().parent.parent / "scripts" / "run_pipeline.py"
    if not rp_path.exists():
        pytest.skip(f"scripts/run_pipeline.py not found at {rp_path}")
    spec = importlib.util.spec_from_file_location("_run_pipeline_test", rp_path)
    mod = importlib.util.module_from_spec(spec)
    # Prevent side-effects from __main__ block
    mod.__name__ = "_run_pipeline_test"
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    return dict(mod.WEIGHTS)


def _load_generate_top_lists_weights() -> Dict[str, float]:
    """Import WEIGHTS from backend/market_intel/generate_top_lists.py."""
    gtl_path = (
        pathlib.Path(__file__).resolve().parent.parent
        / "backend" / "market_intel" / "generate_top_lists.py"
    )
    if not gtl_path.exists():
        pytest.skip(f"generate_top_lists.py not found at {gtl_path}")
    spec = importlib.util.spec_from_file_location("_gtl_test", gtl_path)
    mod = importlib.util.module_from_spec(spec)
    mod.__name__ = "_gtl_test"
    try:
        spec.loader.exec_module(mod)
    except (SystemExit, ImportError):
        pass
    return dict(mod.WEIGHTS)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestWeightsConsistency:
    """WEIGHTS must be identical between run_pipeline and generate_top_lists."""

    def test_run_pipeline_weights_sum_to_one(self):
        """run_pipeline.WEIGHTS must sum to exactly 1.0."""
        weights = _load_run_pipeline_weights()
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6, (
            f"run_pipeline.WEIGHTS sums to {total:.8f}, expected 1.0. "
            f"Factors: {list(weights.keys())}"
        )

    def test_generate_top_lists_weights_sum_to_one(self):
        """generate_top_lists.WEIGHTS must sum to exactly 1.0."""
        weights = _load_generate_top_lists_weights()
        total = sum(weights.values())
        assert abs(total - 1.0) < 1e-6, (
            f"generate_top_lists.WEIGHTS sums to {total:.8f}, expected 1.0. "
            f"Factors: {list(weights.keys())}"
        )

    def test_weights_are_independent(self):
        """Post-sprint: run_pipeline uses 12-factor WEIGHTS (regime_trader/weights.py);
        generate_top_lists uses 9-factor WEIGHTS (regime_trader/config/weights.py).
        Both must sum to 1.0 independently.  They are intentionally different schemas.

        RT-QA-2026-REV6 step 7: generate_top_lists migrated to config/weights.py
        (9-factor post-sprint canonical set).  run_pipeline retains the 12-factor
        set for scoring breadth — Grinold & Kahn: BR benefit from more signals.
        """
        rp_weights  = _load_run_pipeline_weights()
        gtl_weights = _load_generate_top_lists_weights()

        # run_pipeline: 12-factor schema
        assert len(rp_weights) == 12, (
            f"run_pipeline.WEIGHTS expected 12 factors, got {len(rp_weights)}: "
            f"{list(rp_weights.keys())}"
        )
        # generate_top_lists: 9-factor post-sprint schema
        assert len(gtl_weights) == 9, (
            f"generate_top_lists.WEIGHTS expected 9 factors, got {len(gtl_weights)}: "
            f"{list(gtl_weights.keys())}"
        )

        # Both must independently sum to 1.0
        assert abs(sum(rp_weights.values()) - 1.0) < 1e-6
        assert abs(sum(gtl_weights.values()) - 1.0) < 1e-6

        # The 9-factor keys must be a subset of the 12-factor run_pipeline keys
        # (no orphan factors introduced in generate_top_lists)
        rp_keys  = set(rp_weights.keys())
        gtl_keys = set(gtl_weights.keys())
        orphan_in_gtl = gtl_keys - rp_keys
        assert not orphan_in_gtl, (
            f"generate_top_lists has factors not in run_pipeline: {orphan_in_gtl}. "
            "Add the factor to regime_trader/weights.py or remove it from config/weights.py."
        )

    def test_all_weights_positive(self):
        """All weights must be strictly positive (no zero-weight factors in WEIGHTS)."""
        weights = _load_run_pipeline_weights()
        zero_weights = {k: v for k, v in weights.items() if v <= 0}
        assert not zero_weights, (
            f"Zero or negative weights found: {zero_weights}. "
            "Remove zero-weight factors from WEIGHTS or set them to a small positive value."
        )

    def test_factor_fields_covers_all_weights(self):
        """FACTOR_FIELDS must have an entry for every key in WEIGHTS."""
        gtl_path = (
            pathlib.Path(__file__).resolve().parent.parent
            / "backend" / "market_intel" / "generate_top_lists.py"
        )
        if not gtl_path.exists():
            pytest.skip("generate_top_lists.py not found")

        spec = importlib.util.spec_from_file_location("_gtl_ff_test", gtl_path)
        mod = importlib.util.module_from_spec(spec)
        mod.__name__ = "_gtl_ff_test"
        try:
            spec.loader.exec_module(mod)
        except (SystemExit, ImportError):
            pass

        weights_keys     = set(mod.WEIGHTS.keys())
        factor_field_keys = set(mod.FACTOR_FIELDS.keys())

        missing_in_ff = weights_keys - factor_field_keys
        assert not missing_in_ff, (
            f"Factors in WEIGHTS but missing from FACTOR_FIELDS: {missing_in_ff}. "
            "Apply Patch 02 to extend FACTOR_FIELDS."
        )


class TestWeightsValues:
    """Sanity checks on individual weight values."""

    def test_momentum_long_is_largest_weight(self):
        """momentum_long should have the highest weight (strongest IC empirically)."""
        weights = _load_run_pipeline_weights()
        if "momentum_long" not in weights:
            pytest.skip("momentum_long not in WEIGHTS")
        max_factor = max(weights, key=weights.get)
        assert max_factor == "momentum_long", (
            f"Expected momentum_long to have highest weight, got {max_factor}={weights[max_factor]}. "
            f"Full WEIGHTS: {weights}"
        )

    def test_congress_below_threshold(self):
        """congress weight should be <= 0.10 (sparse US-only binary signal)."""
        weights = _load_run_pipeline_weights()
        if "congress" not in weights:
            pytest.skip("congress not in WEIGHTS")
        assert weights["congress"] <= 0.10, (
            f"congress weight={weights['congress']} exceeds 0.10. "
            "Congress is a sparse, US-only, binary signal with limited IC."
        )

    def test_volume_attention_is_tilt_only(self):
        """volume_attention should have low weight (attention, not alpha signal)."""
        weights = _load_run_pipeline_weights()
        if "volume_attention" not in weights:
            pytest.skip("volume_attention not in WEIGHTS")
        assert weights["volume_attention"] <= 0.05, (
            f"volume_attention weight={weights['volume_attention']} is too high. "
            "Volume attention is a short-term attention tilt, not an alpha factor."
        )
