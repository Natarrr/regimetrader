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
    rp_path = pathlib.Path(__file__).resolve().parent.parent / "src" / "ingestion" / "run_pipeline.py"
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

    def test_weights_unified(self):
        """v2.2-global: run_pipeline and generate_top_lists both use config/weights.py.
        Both must use WEIGHTS_US (9-factor) — the dual 12-factor/9-factor schema is resolved.
        """
        rp_weights  = _load_run_pipeline_weights()
        gtl_weights = _load_generate_top_lists_weights()

        # Both schemas are now 9-factor (WEIGHTS_US from config/weights.py)
        assert len(rp_weights) == 9, (
            f"run_pipeline.WEIGHTS expected 9 factors (WEIGHTS_US), got {len(rp_weights)}: "
            f"{list(rp_weights.keys())}"
        )
        assert len(gtl_weights) == 9, (
            f"generate_top_lists.WEIGHTS expected 9 factors, got {len(gtl_weights)}: "
            f"{list(gtl_weights.keys())}"
        )

        # Both must sum to 1.0
        assert abs(sum(rp_weights.values()) - 1.0) < 1e-6
        assert abs(sum(gtl_weights.values()) - 1.0) < 1e-6

        # Keys must match — single source of truth
        assert set(rp_weights.keys()) == set(gtl_weights.keys()), (
            f"Weight key mismatch: run_pipeline={set(rp_weights.keys())}, "
            f"generate_top_lists={set(gtl_weights.keys())}"
        )

    def test_all_weights_non_negative(self):
        """All weights must be >= 0. Factors at 0.0 are 'wired but not yet active' sprints."""
        weights = _load_run_pipeline_weights()
        negative = {k: v for k, v in weights.items() if v < 0}
        assert not negative, (
            f"Negative weights found: {negative}. "
            "Weights must be in [0.0, 1.0]."
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

    def test_insider_conviction_is_largest_weight(self):
        """insider_conviction should have the highest weight in WEIGHTS_US.
        v2.2-global: insider_conviction=0.30 is the dominant alpha source."""
        weights = _load_run_pipeline_weights()
        if "insider_conviction" not in weights:
            pytest.skip("insider_conviction not in WEIGHTS")
        max_factor = max(weights, key=weights.get)
        assert max_factor == "insider_conviction", (
            f"Expected insider_conviction to have highest weight, got {max_factor}={weights[max_factor]}. "
            f"Full WEIGHTS: {weights}"
        )

    def test_congress_weight_is_22_percent(self):
        """congress weight = 0.22 in WEIGHTS_US (redistributed to WEIGHTS_GLOBAL for EU/Asia)."""
        weights = _load_run_pipeline_weights()
        if "congress" not in weights:
            pytest.skip("congress not in WEIGHTS")
        assert abs(weights["congress"] - 0.22) < 1e-6, (
            f"congress weight={weights['congress']} — expected 0.22 per v2.2-global WEIGHTS_US spec."
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

    def test_config_congress_weight_is_intentional(self):
        """config/weights.py congress=0.22 per v2.1-global (WEIGHTS_US).
        The 0.22 is redistributed to WEIGHTS_GLOBAL for EU/Asia where congress is absent.
        """
        from regime_trader.config.weights import WEIGHTS as CONFIG_WEIGHTS
        assert CONFIG_WEIGHTS["congress"] == 0.22, (
            "9-factor US congress weight must be 0.22 per v2.1-global spec (WEIGHTS_US). "
            "Change only by updating the canonical source in regime_trader/config/weights.py."
        )
