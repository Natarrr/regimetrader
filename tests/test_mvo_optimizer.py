"""tests/test_mvo_optimizer.py
Portfolio optimizer hard constraint tests.
"""
from __future__ import annotations

import pytest
from backend.market_intel.portfolio_optimizer import run_optimizer


def _make_tickers(n: int) -> list[str]:
    return [f"T{i:02d}" for i in range(n)]


def _make_scores(n: int) -> list[float]:
    # Spread scores to give MVO something to work with
    return [0.5 + 0.01 * i for i in range(n)]


class TestMVOConstraints:
    def test_weights_sum_at_most_one(self):
        tickers = _make_tickers(10)
        scores = _make_scores(10)
        sectors = ["Tech"] * 5 + ["Healthcare"] * 5
        weights, _ = run_optimizer(tickers, scores, sectors, vix=20.0)
        assert sum(weights.values()) <= 1.0 + 1e-6

    def test_all_weights_nonnegative(self):
        tickers = _make_tickers(10)
        scores = _make_scores(10)
        sectors = ["Tech"] * 10
        weights, _ = run_optimizer(tickers, scores, sectors, vix=20.0)
        for t, w in weights.items():
            assert w >= -1e-8, f"{t} has negative weight {w}"

    def test_no_position_exceeds_max(self):
        tickers = _make_tickers(15)
        scores = _make_scores(15)
        sectors = ["Sector"] * 15
        weights, _ = run_optimizer(tickers, scores, sectors, vix=20.0)
        for t, w in weights.items():
            assert w <= 0.10 + 1e-6, f"{t} weight {w:.4f} exceeds 10% cap"

    def test_sector_weight_does_not_exceed_cap(self):
        # 5 tickers all in "Tech": unconstrained, 5 × 10% = 50% > 30% sector cap.
        # The sector constraint must be the binding limit here.
        tickers = [f"T{i:02d}" for i in range(5)]
        scores = [0.9 - 0.01 * i for i in range(5)]
        sectors = ["Tech"] * 5
        weights, _ = run_optimizer(tickers, scores, sectors, vix=20.0)
        tech_total = sum(weights[t] for t, s in zip(tickers, sectors) if s == "Tech")
        assert tech_total <= 0.30 + 1e-5, f"Tech sector weight {tech_total:.4f} exceeds 30% cap"

    def test_all_tickers_have_weight_key(self):
        tickers = _make_tickers(5)
        scores = _make_scores(5)
        sectors = ["A", "B", "C", "D", "E"]
        weights, _ = run_optimizer(tickers, scores, sectors, vix=20.0)
        for t in tickers:
            assert t in weights

    def test_method_is_valid_string(self):
        tickers = _make_tickers(5)
        scores = _make_scores(5)
        sectors = ["A"] * 5
        _, method = run_optimizer(tickers, scores, sectors, vix=20.0)
        assert method in {"sharpe", "min_variance", "risk_parity", "score_proportional"}


class TestScoreCalibration:
    def test_output_within_return_bounds(self):
        import numpy as np
        from backend.market_intel.portfolio_optimizer import _calibrate_scores_to_returns
        scores = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
        returns = _calibrate_scores_to_returns(scores)
        assert all(0.04 - 1e-6 <= r <= 0.18 + 1e-6 for r in returns)

    def test_monotone_order_preserved(self):
        import numpy as np
        from backend.market_intel.portfolio_optimizer import _calibrate_scores_to_returns
        scores = np.array([0.3, 0.6, 0.9])
        returns = _calibrate_scores_to_returns(scores)
        assert returns[0] < returns[1] < returns[2]


class TestAsyncCovariance:
    def test_2day_cov_returns_matrix(self):
        import numpy as np
        from backend.market_intel.portfolio_optimizer import build_async_covariance
        prices = {"A": list(range(100, 200)), "B": list(range(50, 150))}
        cov = build_async_covariance(prices)
        assert cov is not None
        assert cov.shape == (2, 2)

    def test_cov_is_symmetric(self):
        import numpy as np
        from backend.market_intel.portfolio_optimizer import build_async_covariance
        prices = {
            "A": [100 + i + (i % 3) for i in range(60)],
            "B": [50 + i * 0.5 for i in range(60)],
        }
        cov = build_async_covariance(prices)
        assert np.allclose(cov, cov.T, atol=1e-10)

    def test_returns_none_for_short_series(self):
        from backend.market_intel.portfolio_optimizer import build_async_covariance
        prices = {"A": [100, 101, 102]}   # < 5 observations
        cov = build_async_covariance(prices)
        assert cov is None

    def test_returns_none_for_empty(self):
        from backend.market_intel.portfolio_optimizer import build_async_covariance
        assert build_async_covariance({}) is None


class TestMinVarianceMode:
    def test_weights_sum_to_one(self):
        tickers = _make_tickers(6)
        scores  = _make_scores(6)
        sectors = ["A", "B", "C", "D", "E", "F"]
        weights, _ = run_optimizer(tickers, scores, sectors, vix=20.0,
            mode="min_variance", position_floor=0.10, position_ceiling=0.25)
        assert abs(sum(weights.values()) - 1.0) <= 1e-4

    def test_floor_respected(self):
        tickers = _make_tickers(5)
        scores  = _make_scores(5)
        sectors = list("ABCDE")
        weights, _ = run_optimizer(tickers, scores, sectors, vix=20.0,
            mode="min_variance", position_floor=0.10, position_ceiling=0.25)
        for t, w in weights.items():
            assert w >= 0.10 - 1e-6

    def test_ceiling_respected(self):
        tickers = _make_tickers(5)
        scores  = _make_scores(5)
        sectors = list("ABCDE")
        weights, _ = run_optimizer(tickers, scores, sectors, vix=20.0,
            mode="min_variance", position_floor=0.10, position_ceiling=0.25)
        for t, w in weights.items():
            assert w <= 0.25 + 1e-6

    def test_method_label(self):
        tickers = _make_tickers(4)
        scores  = _make_scores(4)
        sectors = list("ABCD")
        _, method = run_optimizer(tickers, scores, sectors, vix=20.0,
            mode="min_variance", position_floor=0.10, position_ceiling=0.25)
        assert method == "min_variance"


class TestLargeCapAnchor:
    def test_anchor_equal_weight(self):
        from backend.market_intel.portfolio_optimizer import build_large_cap_anchors
        entries = [
            {"ticker": "MSFT", "market_cap": 3_000_000_000_000, "final_score": 0.94},
            {"ticker": "AAPL", "market_cap": 2_800_000_000_000, "final_score": 0.91},
        ]
        anchors = build_large_cap_anchors(entries)
        assert len(anchors) == 2
        assert abs(anchors[0]["allocation"] - anchors[1]["allocation"]) < 1e-6

    def test_sub_10b_excluded(self):
        from backend.market_intel.portfolio_optimizer import build_large_cap_anchors
        entries = [
            {"ticker": "LRG", "market_cap": 15_000_000_000},
            {"ticker": "MID", "market_cap": 5_000_000_000},
        ]
        anchors = build_large_cap_anchors(entries)
        assert len(anchors) == 1 and anchors[0]["ticker"] == "LRG"

    def test_empty_when_none_qualify(self):
        from backend.market_intel.portfolio_optimizer import build_large_cap_anchors
        entries = [{"ticker": "SMALL", "market_cap": 500_000_000}]
        assert build_large_cap_anchors(entries) == []

    def test_tier_label_set(self):
        from backend.market_intel.portfolio_optimizer import build_large_cap_anchors
        entries = [{"ticker": "BIG", "market_cap": 50_000_000_000}]
        anchor = build_large_cap_anchors(entries)[0]
        assert anchor["tier"] == "LARGE_CAP_ANCHOR"


class TestAdVLiquidityGate:
    def test_adv_ceiling_reduces_allocation(self):
        from backend.market_intel.portfolio_optimizer import adv_capacity_ceiling
        # ADV=$1M, portfolio=$50M, max_adv_pct=3% → cap = 3%×$1M/$50M = 0.0006
        result = adv_capacity_ceiling(
            current_ceiling=0.25,
            adv_20d_usd=1_000_000,
            portfolio_value_usd=50_000_000,
            max_adv_pct=0.03,
        )
        assert result < 0.25
        assert result == pytest.approx(0.0006, abs=1e-6)

    def test_adv_gate_no_op_when_liquid(self):
        from backend.market_intel.portfolio_optimizer import adv_capacity_ceiling
        # ADV=$100M, portfolio=$10M → constraint = 30% > 25% ceiling → unchanged
        result = adv_capacity_ceiling(
            current_ceiling=0.25,
            adv_20d_usd=100_000_000,
            portfolio_value_usd=10_000_000,
            max_adv_pct=0.03,
        )
        assert result == pytest.approx(0.25, abs=1e-6)

    def test_zero_portfolio_returns_ceiling(self):
        from backend.market_intel.portfolio_optimizer import adv_capacity_ceiling
        result = adv_capacity_ceiling(0.25, 1_000_000, 0)
        assert result == pytest.approx(0.25)
