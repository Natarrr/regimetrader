"""tests/test_fmt_insider_pct.py
Unit tests for _fmt_insider_pct() in regime_trader/ui/streamlit_app.py.

Thaler (2017 Nobel) — nudge theory: a well-formatted percentage nudges the
analyst toward correct interpretation of insider conviction magnitude.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Stub streamlit before importing the app module ────────────────────────────
def _cache_data_stub(*a, **kw):
    def dec(f):
        f.clear = lambda: None
        return f
    return dec

_st_mock = MagicMock()
_st_mock.cache_data = _cache_data_stub
_st_mock.set_page_config = MagicMock()

sys.modules.setdefault("streamlit", _st_mock)

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def fmt():
    """Return the _fmt_insider_pct function from the app module."""
    with patch.dict(sys.modules, {"streamlit": _st_mock}):
        import importlib
        import regime_trader.ui.streamlit_app as app
        importlib.reload(app)  # ensure fresh load with mocked st
        return app._fmt_insider_pct


# ── None / invalid ────────────────────────────────────────────────────────────

class TestFmtInsiderPctInvalid:
    def test_none_returns_dash(self, fmt):
        assert fmt(None) == "—"

    def test_string_non_numeric_returns_dash(self, fmt):
        assert fmt("n/a") == "—"

    def test_empty_string_returns_dash(self, fmt):
        assert fmt("") == "—"

    def test_list_returns_dash(self, fmt):
        assert fmt([0.01]) == "—"


# ── Fraction path (|val| ≤ 1.0 and > 0) — multiply by 100 ───────────────────

class TestFmtInsiderPctFractionPath:
    def test_small_fraction(self, fmt):
        # 0.0005 → 0.05 % → "0.0500%"
        assert fmt(0.0005) == "0.0500%"

    def test_typical_fraction(self, fmt):
        # 0.05 → 5.0 % → "5.0000%"
        assert fmt(0.05) == "5.0000%"

    def test_one_exactly_is_fraction(self, fmt):
        # 1.0 is at the boundary — treated as fraction → 100.0000%
        assert fmt(1.0) == "100.0000%"

    def test_negative_fraction(self, fmt):
        # -0.02 → abs(-0.02) = 0.02 ≤ 1.0, so fraction path → -2.0000%
        assert fmt(-0.02) == "-2.0000%"


# ── Percentage path (|val| > 1.0) — already a percent, pass through ──────────

class TestFmtInsiderPctPercentPath:
    def test_large_value_passthrough(self, fmt):
        # 5.0 > 1.0 → treat as already-percent → "5.0000%"
        assert fmt(5.0) == "5.0000%"

    def test_value_gt_100(self, fmt):
        assert fmt(150.0) == "150.0000%"

    def test_negative_large(self, fmt):
        assert fmt(-3.5) == "-3.5000%"


# ── String-numeric passthrough ────────────────────────────────────────────────

class TestFmtInsiderPctStringNumeric:
    def test_string_float(self, fmt):
        # "0.001" is a valid float → fraction path → 0.1000%
        assert fmt("0.001") == "0.1000%"

    def test_string_large(self, fmt):
        assert fmt("10.0") == "10.0000%"


# ── Zero edge case ────────────────────────────────────────────────────────────

class TestFmtInsiderPctZero:
    def test_zero_is_fraction_boundary(self, fmt):
        # 0 < abs(0) is False → falls to passthrough → 0.0000%
        assert fmt(0) == "0.0000%"

    def test_zero_float(self, fmt):
        assert fmt(0.0) == "0.0000%"
