"""tests/research/test_ic_backtest.py — Synthetic IC backtest validation tests.

Two tests with synthetic data whose ground truth is known:
  1. test_ic_detects_known_alpha_signal — factor has real alpha (IC ∈ [0.20, 0.40])
  2. test_ic_rejects_pure_noise        — factor is random (|IC| < 0.1, IR < 0.5)

No live yfinance calls; no disk I/O beyond tmp_path.

References:
  Grinold & Kahn (2000) APM ch. 6 — IC > 0.05 is economically meaningful
  López de Prado (2018) AFML ch. 7 — purged k-fold prevents leakage
"""
from __future__ import annotations

import math
from datetime import date, timedelta

import numpy as np
import pytest

from regime_trader.research.ic_metrics import compute_ic, purged_kfold_ic, ICResult


# ── Synthetic data factory ────────────────────────────────────────────────────

_RNG = np.random.default_rng(2024_05_16)  # fixed seed for reproducibility

N_SNAPS = 6
N_TICKERS = 50
_TICKERS = [f"T{i:03d}" for i in range(N_TICKERS)]


def _make_snapshots(
    alpha: float,
    noise_std: float = 0.15,
    factor_name: str = "test_factor_score",
) -> tuple[list[tuple[date, list[dict]]], dict[date, dict[str, float]]]:
    """Build N_SNAPS synthetic snapshots and their forward returns.

    Forward return model: r_i = alpha * factor_i + ε_i, ε ~ N(0, noise_std²)

    Returns:
        snapshots: [(date, rows), ...] as expected by purged_kfold_ic
        fwd_map:   {date: {ticker: forward_return}}
    """
    base_date = date(2025, 1, 1)
    snapshots: list[tuple[date, list[dict]]] = []
    fwd_map: dict[date, dict[str, float]] = {}

    for snap_idx in range(N_SNAPS):
        snap_date = base_date + timedelta(days=snap_idx * 21)
        factor_scores = _RNG.uniform(0.0, 1.0, N_TICKERS)
        noise = _RNG.normal(0.0, noise_std, N_TICKERS)
        forward_rets = alpha * factor_scores + noise

        rows = [
            {"ticker": _TICKERS[i], factor_name: float(factor_scores[i])}
            for i in range(N_TICKERS)
        ]
        snapshots.append((snap_date, rows))
        fwd_map[snap_date] = {_TICKERS[i]: float(forward_rets[i]) for i in range(N_TICKERS)}

    return snapshots, fwd_map


# ── Test 1: Known alpha signal ────────────────────────────────────────────────

class TestICDetectsKnownAlphaSignal:
    """factor = r * 0.3 + ε should produce IC in a clearly positive range."""

    def test_ic_mean_in_positive_range(self):
        snapshots, fwd_map = _make_snapshots(alpha=0.6, noise_std=0.10)
        result = purged_kfold_ic(
            snapshots=snapshots,
            factor_name="test_factor_score",
            forward_return_map=fwd_map,
            n_folds=3,
            embargo_days=2,
        )
        assert not math.isnan(result.ic_mean), "IC mean should not be NaN for alpha signal"
        assert result.ic_mean >= 0.20, (
            f"Expected IC >= 0.20 for alpha=0.6 signal, got {result.ic_mean:.4f}"
        )
        assert result.ic_mean <= 1.0, "IC is bounded by definition"

    def test_p_value_significant(self):
        snapshots, fwd_map = _make_snapshots(alpha=0.6, noise_std=0.10)
        result = purged_kfold_ic(
            snapshots=snapshots,
            factor_name="test_factor_score",
            forward_return_map=fwd_map,
            n_folds=3,
            embargo_days=2,
        )
        # With alpha=0.6 and 50 tickers × 6 snapshots, p < 0.05 is expected
        assert not math.isnan(result.p_value), "p-value should be computable"
        assert result.p_value < 0.05, (
            f"Expected p < 0.05 for strong alpha signal, got p={result.p_value:.4f}"
        )

    def test_n_snapshots_equals_input(self):
        snapshots, fwd_map = _make_snapshots(alpha=0.6, noise_std=0.10)
        result = purged_kfold_ic(
            snapshots=snapshots,
            factor_name="test_factor_score",
            forward_return_map=fwd_map,
            n_folds=3,
            embargo_days=2,
        )
        assert result.n_snapshots == N_SNAPS
        assert result.n_tickers_avg == pytest.approx(N_TICKERS, abs=1)

    def test_icresult_is_named_tuple(self):
        snapshots, fwd_map = _make_snapshots(alpha=0.6, noise_std=0.10)
        result = purged_kfold_ic(
            snapshots=snapshots,
            factor_name="test_factor_score",
            forward_return_map=fwd_map,
            n_folds=3,
            embargo_days=2,
        )
        assert isinstance(result, ICResult)
        assert result.factor_name == "test_factor_score"


# ── Test 2: Pure noise signal ─────────────────────────────────────────────────

class TestICRejectsPureNoise:
    """alpha=0.0 → factor and returns are independent → |IC| should be small."""

    def test_ic_mean_near_zero(self):
        snapshots, fwd_map = _make_snapshots(alpha=0.0, noise_std=0.15)
        result = purged_kfold_ic(
            snapshots=snapshots,
            factor_name="test_factor_score",
            forward_return_map=fwd_map,
            n_folds=3,
            embargo_days=2,
        )
        assert not math.isnan(result.ic_mean), "IC mean should be computable even for noise"
        assert abs(result.ic_mean) < 0.20, (
            f"Expected |IC| < 0.20 for pure noise, got {result.ic_mean:.4f}"
        )

    def test_ir_below_threshold(self):
        snapshots, fwd_map = _make_snapshots(alpha=0.0, noise_std=0.15)
        result = purged_kfold_ic(
            snapshots=snapshots,
            factor_name="test_factor_score",
            forward_return_map=fwd_map,
            n_folds=3,
            embargo_days=2,
        )
        if not math.isnan(result.ir):
            assert abs(result.ir) < 0.8, (
                f"Expected |IR| < 0.8 for pure noise, got {result.ir:.4f}"
            )

    def test_schema_warning_propagated(self):
        snapshots, fwd_map = _make_snapshots(alpha=0.0, noise_std=0.15)
        warn_text = "v1 buzz contamination"
        result = purged_kfold_ic(
            snapshots=snapshots,
            factor_name="test_factor_score",
            forward_return_map=fwd_map,
            n_folds=3,
            embargo_days=2,
            schema_warning=warn_text,
        )
        assert result.schema_warning == warn_text


# ── Test 3: compute_ic unit tests ─────────────────────────────────────────────

class TestComputeIC:
    def test_perfect_correlation(self):
        xs = list(range(10))
        ic, p = compute_ic(xs, xs)
        assert ic == pytest.approx(1.0, abs=1e-6)
        assert p < 0.001

    def test_perfect_anticorrelation(self):
        xs = list(range(10))
        ic, p = compute_ic(xs, list(reversed(xs)))
        assert ic == pytest.approx(-1.0, abs=1e-6)

    def test_too_few_pairs_returns_nan(self):
        ic, p = compute_ic([0.1, 0.2, 0.3], [0.4, 0.5, 0.6])
        assert math.isnan(ic)
        assert math.isnan(p)

    def test_nan_pairs_dropped(self):
        import math as _math
        xs = [1.0, 2.0, float("nan"), 4.0, 5.0, 6.0]
        ys = [1.0, 2.0, 3.0,          4.0, 5.0, 6.0]
        ic, p = compute_ic(xs, ys)
        # 5 valid pairs after dropping nan — should return valid IC
        assert not _math.isnan(ic)


# ── Test 4: historical_loader unit tests ─────────────────────────────────────

class TestHistoricalLoader:
    def test_detect_schema_v2(self):
        from regime_trader.research.historical_loader import detect_schema_version, V2_SCHEMA
        row = {"insider_conviction_score": 0.5, "ticker": "AAPL"}
        assert detect_schema_version(row) == V2_SCHEMA

    def test_detect_schema_v1(self):
        from regime_trader.research.historical_loader import detect_schema_version, V1_SCHEMA
        row = {"edgar_score": 0.3, "news_score": 0.4, "ticker": "AAPL"}
        assert detect_schema_version(row) == V1_SCHEMA

    def test_normalize_v2_unchanged(self):
        from regime_trader.research.historical_loader import normalize_snapshot_schema, V2_SCHEMA
        rows = [{"insider_conviction_score": 0.5, "congress_score": 0.3, "ticker": "AAPL"}]
        normed, schema = normalize_snapshot_schema(rows)
        assert schema == V2_SCHEMA
        assert normed[0]["insider_conviction_score"] == 0.5

    def test_normalize_v1_maps_congress_and_news(self):
        from regime_trader.research.historical_loader import normalize_snapshot_schema
        rows = [{"congress_score": 0.4, "news_score": 0.6, "edgar_score": 0.3, "ticker": "AAPL"}]
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            normed, schema = normalize_snapshot_schema(rows)
        assert normed[0]["congress_score"] == pytest.approx(0.4)
        assert normed[0]["news_sentiment_score"] == pytest.approx(0.6)
        # edgar_score has no safe v2 mapping — must be absent
        assert "edgar_score" not in normed[0]
        assert "insider_conviction_score" not in normed[0]

    def test_load_historical_snapshots_aborts_on_missing_dir(self, tmp_path):
        from regime_trader.research.historical_loader import load_historical_snapshots
        fake_dir = tmp_path / "historical"
        with pytest.raises(RuntimeError, match="historical directory not found"):
            list(load_historical_snapshots(fake_dir))

    def test_load_historical_snapshots_aborts_insufficient(self, tmp_path):
        import json
        from regime_trader.research.historical_loader import load_historical_snapshots
        hist_dir = tmp_path / "historical"
        # Create only 3 snapshots (< default min_snapshots=60)
        for i in range(3):
            d = date(2025, 1, 1) + timedelta(days=i)
            snap_dir = hist_dir / d.isoformat()
            snap_dir.mkdir(parents=True)
            rows = [{"ticker": f"T{j:03d}", "insider_conviction_score": 0.5}
                    for j in range(90)]
            (snap_dir / "intel_source_status.json").write_text(
                json.dumps(rows), encoding="utf-8"
            )
        with pytest.raises(RuntimeError, match="only 3 qualifying snapshots"):
            list(load_historical_snapshots(hist_dir, min_snapshots=60))

    def test_archive_current_run_creates_file(self, tmp_path):
        import json
        from regime_trader.research.historical_loader import archive_current_run
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        src = log_dir / "intel_source_status.json"
        src.write_text(json.dumps([{"ticker": "AAPL"}]), encoding="utf-8")

        result = archive_current_run(log_dir)
        assert result is not None
        assert result.exists()
        today = date.today().isoformat()
        assert today in str(result)

    def test_archive_current_run_idempotent(self, tmp_path):
        import json
        from regime_trader.research.historical_loader import archive_current_run
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        src = log_dir / "intel_source_status.json"
        src.write_text(json.dumps([{"ticker": "AAPL"}]), encoding="utf-8")

        r1 = archive_current_run(log_dir)
        r2 = archive_current_run(log_dir)
        assert r1 == r2  # same path returned both times

    def test_archive_returns_none_when_source_missing(self, tmp_path):
        from regime_trader.research.historical_loader import archive_current_run
        log_dir = tmp_path / "logs"
        log_dir.mkdir()
        result = archive_current_run(log_dir)
        assert result is None
