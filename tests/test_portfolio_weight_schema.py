"""tests/test_portfolio_weight_schema.py
Schema validation: run_optimizer() returns a dict with all tickers keyed,
all weights non-negative, and the sum at most 1.0 (cash-buffer model).
"""
from __future__ import annotations

import pytest
from backend.market_intel.portfolio_optimizer import run_optimizer


def _run(n: int = 20, vix: float = 20.0) -> tuple[dict, str]:
    tickers = [f"T{i:02d}" for i in range(n)]
    scores = [0.9 - 0.01 * i for i in range(n)]
    sectors = ["Tech" if i % 2 == 0 else "Health" for i in range(n)]
    return run_optimizer(tickers, scores, sectors, vix=vix)


class TestPortfolioWeightSchema:
    def test_all_tickers_present_in_output(self):
        n = 20
        tickers = [f"T{i:02d}" for i in range(n)]
        weights, _ = _run(n)
        for t in tickers:
            assert t in weights, f"{t} missing from weights dict"

    def test_all_weights_nonnegative(self):
        weights, _ = _run(20)
        for t, w in weights.items():
            assert w >= 0.0, f"{t} has negative weight {w}"

    def test_weight_sum_at_most_one(self):
        weights, _ = _run(20)
        assert sum(weights.values()) <= 1.0 + 1e-6

    def test_weight_sum_at_most_one_panic_vix(self):
        weights, _ = _run(20, vix=35.0)
        assert sum(weights.values()) <= 1.0 + 1e-6

    def test_no_weight_exceeds_position_cap(self):
        weights, _ = _run(20)
        for t, w in weights.items():
            assert w <= 0.10 + 1e-6, f"{t} weight {w:.4f} exceeds position cap"

    def test_method_returned_as_string(self):
        _, method = _run(20)
        assert isinstance(method, str)
        assert method in {"MVO", "risk_parity", "score_proportional"}

    def test_weight_dict_keys_are_strings(self):
        weights, _ = _run(5)
        for k in weights:
            assert isinstance(k, str)

    def test_weight_dict_values_are_floats(self):
        weights, _ = _run(5)
        for w in weights.values():
            assert isinstance(w, float)
