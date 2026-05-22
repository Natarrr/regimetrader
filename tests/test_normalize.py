"""tests/test_normalize.py
Unit tests for regime_trader.scoring.normalize.

Markowitz (1990 Nobel) — signals must be comparable and bounded before
portfolio construction. Tests validate math against known inputs including
a 2008/2020 synthetic crash outlier that would dominate without winsorizing.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from regime_trader.scoring.normalize import (
    winsorize,
    normalize_score,
    fallback_reweight,
    build_explain,
    persist_explain,
    load_explain,
)


# ── winsorize ─────────────────────────────────────────────────────────────────

class TestWinsorize:
    def test_no_outliers_unchanged(self):
        arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        w   = winsorize(arr, lo=1, hi=99)
        # With only 5 elements, 1st pct ≈ min and 99th ≈ max
        assert float(np.min(w)) >= float(np.min(arr)) - 1e-6
        assert float(np.max(w)) <= float(np.max(arr)) + 1e-6

    def test_crash_outlier_capped(self):
        """2008-style: 999 normal values + 1 extreme outlier → outlier capped at P99.
        Uses 999 normals so np.percentile interpolation stays near the normal tail."""
        rng  = np.random.default_rng(42)
        arr  = np.concatenate([rng.normal(50, 5, 999), [1_000_000.0]])
        w    = winsorize(arr, lo=1, hi=99)
        assert float(np.max(w)) < 200.0   # outlier removed

    def test_empty_array_returns_empty(self):
        w = winsorize(np.array([]), lo=1, hi=99)
        assert w.size == 0

    def test_uniform_array_unchanged(self):
        arr = np.ones(10)
        w   = winsorize(arr)
        np.testing.assert_array_almost_equal(w, arr)

    def test_output_shape_preserved(self):
        arr = np.arange(100.0)
        w   = winsorize(arr)
        assert w.shape == (100,)


# ── normalize_score ───────────────────────────────────────────────────────────

class TestNormalizeScore:
    def test_output_in_0_100(self):
        arr  = np.array([1.0, 2.0, 3.0, 10.0, 100.0])
        norm = normalize_score(arr)
        assert float(np.min(norm)) >= 0.0 - 1e-9
        assert float(np.max(norm)) <= 100.0 + 1e-9

    def test_min_maps_to_0(self):
        arr  = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        norm = normalize_score(arr, lo_pct=0, hi_pct=100)  # no winsorizing
        assert float(np.min(norm)) == pytest.approx(0.0, abs=1e-6)

    def test_max_maps_to_100(self):
        arr  = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        norm = normalize_score(arr, lo_pct=0, hi_pct=100)
        assert float(np.max(norm)) == pytest.approx(100.0, abs=1e-6)

    def test_constant_array_returns_zeros(self):
        arr  = np.ones(10)
        norm = normalize_score(arr)
        np.testing.assert_array_almost_equal(norm, np.zeros(10))

    def test_empty_returns_empty(self):
        assert normalize_score(np.array([])).size == 0

    def test_custom_output_range(self):
        arr  = np.array([0.0, 0.5, 1.0])
        norm = normalize_score(arr, lo_pct=0, hi_pct=100, out_min=50.0, out_max=150.0)
        assert float(np.min(norm)) == pytest.approx(50.0, abs=1e-6)
        assert float(np.max(norm)) == pytest.approx(150.0, abs=1e-6)

    def test_2008_crash_outlier_does_not_dominate(self):
        """2020 COVID analog: 1000 normal scores + 1 extreme → still scale 0–100."""
        rng  = np.random.default_rng(0)
        arr  = np.concatenate([rng.uniform(0, 1, 999), [99999.0]])
        norm = normalize_score(arr)
        # Almost all values should map to most of the [0, 100] range
        median_norm = float(np.median(norm))
        assert 30 < median_norm < 70   # not collapsed near 0


# ── fallback_reweight ─────────────────────────────────────────────────────────

class TestFallbackReweight:
    def test_all_available_preserves_relative_weights(self):
        w = np.array([0.4, 0.4, 0.2])
        m = [True, True, True]
        r = fallback_reweight(w, m)
        np.testing.assert_array_almost_equal(r, w)
        assert float(r.sum()) == pytest.approx(1.0, abs=1e-9)

    def test_missing_component_redistributed(self):
        """If insider (weight 0.4) is missing, its weight goes to the others."""
        w = np.array([0.4, 0.4, 0.2])
        m = [True, False, True]
        r = fallback_reweight(w, m)
        assert r[1] == pytest.approx(0.0)
        assert float(r.sum()) == pytest.approx(1.0, abs=1e-9)
        # Remaining weights proportional to [0.4, 0.2] → [0.667, 0.333]
        assert r[0] == pytest.approx(0.4 / 0.6, abs=1e-6)
        assert r[2] == pytest.approx(0.2 / 0.6, abs=1e-6)

    def test_all_missing_returns_uniform(self):
        w = np.array([0.5, 0.3, 0.2])
        m = [False, False, False]
        r = fallback_reweight(w, m)
        np.testing.assert_array_almost_equal(r, [1/3, 1/3, 1/3])

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            fallback_reweight([0.5, 0.5], [True])

    def test_output_sums_to_one(self):
        for seed in range(10):
            rng = np.random.default_rng(seed)
            w = rng.dirichlet([1, 1, 1, 1])
            m = rng.random(4) > 0.3
            r = fallback_reweight(w, m.tolist())
            assert float(r.sum()) == pytest.approx(1.0, abs=1e-9)


# ── build_explain + persist/load ──────────────────────────────────────────────

class TestBuildExplain:
    def test_output_keys_present(self):
        explain = build_explain(
            ticker       = "AAPL",
            scores       = {"insider": 0.8, "momentum": 0.6, "fundamental": 0.5},
            weights      = {"insider": 0.4, "momentum": 0.4, "fundamental": 0.2},
            evidence_ids = ["abc123", "def456"],
        )
        assert explain["ticker"] == "AAPL"
        assert "composite" in explain
        assert "breakdown" in explain
        assert "evidence" in explain
        assert "computed_at" in explain

    def test_composite_bounded_0_1(self):
        explain = build_explain(
            "TST",
            {"a": 0.9, "b": 0.1},
            {"a": 0.5, "b": 0.5},
        )
        assert 0.0 <= explain["composite"] <= 1.0

    def test_missing_component_handled(self):
        """If a component score is None, it should not crash."""
        explain = build_explain(
            "TST",
            {"a": 0.9, "b": None},
            {"a": 0.6, "b": 0.4},
        )
        assert "composite" in explain

    def test_breakdown_per_component(self):
        explain = build_explain(
            "TST",
            {"x": 0.8, "y": 0.6},
            {"x": 0.5, "y": 0.5},
        )
        assert "x" in explain["breakdown"]
        assert "y" in explain["breakdown"]
        for v in explain["breakdown"].values():
            assert "raw" in v
            assert "weight" in v
            assert "contribution" in v

    def test_evidence_ids_in_output(self):
        explain = build_explain(
            "TST", {}, {}, evidence_ids=["ev001", "ev002"]
        )
        assert "ev001" in explain["evidence"]
        assert "ev002" in explain["evidence"]


class TestPersistLoadExplain:
    def test_roundtrip(self, tmp_path: Path):
        explain = build_explain("AAPL", {"x": 0.7}, {"x": 1.0})
        persist_explain("AAPL", explain, cache_root=tmp_path)
        loaded = load_explain("AAPL", cache_root=tmp_path)
        assert loaded is not None
        assert loaded["ticker"] == "AAPL"
        assert loaded["composite"] == explain["composite"]

    def test_missing_returns_none(self, tmp_path: Path):
        assert load_explain("MISSING_ZZZ", cache_root=tmp_path) is None

    def test_file_is_atomic(self, tmp_path: Path):
        """No .tmp files should remain after persist."""
        persist_explain("MSFT", build_explain("MSFT", {}, {}), cache_root=tmp_path)
        leftovers = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
        assert leftovers == []
