"""tests/test_piotroski_gate_flag.py — P1.3 missing-Piotroski convergence flag.

The live gate keeps its conservative 0.375 haircut on a missing F-score by
default (live unchanged). PIOTROSKI_GATE_NEUTRAL_MISSING=1 converges it onto the
neutral 1.0 contract (absence not bearish) — to be enabled only after backtest.
"""
from __future__ import annotations

import pytest

from src.config.weights import _piotroski_gate_multiplier


class TestPiotroskiGateMissingFlag:
    def test_default_missing_is_conservative_haircut(self, monkeypatch):
        monkeypatch.delenv("PIOTROSKI_GATE_NEUTRAL_MISSING", raising=False)
        assert _piotroski_gate_multiplier(None) == pytest.approx(0.375)

    def test_flag_makes_missing_neutral(self, monkeypatch):
        monkeypatch.setenv("PIOTROSKI_GATE_NEUTRAL_MISSING", "1")
        assert _piotroski_gate_multiplier(None) == 1.0

    @pytest.mark.parametrize("raw,expected", [(0, 0.0), (2, 0.0), (3, 0.6),
                                              (5, 0.6), (6, 1.0), (9, 1.0)])
    def test_non_missing_unchanged_by_flag(self, raw, expected, monkeypatch):
        # The flag only affects the None branch; real F-scores gate identically.
        monkeypatch.setenv("PIOTROSKI_GATE_NEUTRAL_MISSING", "1")
        assert _piotroski_gate_multiplier(raw) == expected
        monkeypatch.delenv("PIOTROSKI_GATE_NEUTRAL_MISSING", raising=False)
        assert _piotroski_gate_multiplier(raw) == expected
