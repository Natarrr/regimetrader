# Path: tests/test_exit_rules.py
"""pytest — validates Batch Floor, PT signal, and breakout extension logic."""
from __future__ import annotations
import pytest
from regime_trader.risk.exit_rules import (
    compute_batch_floor,
    compute_pt_signal,
    compute_breakout_extension,
    enrich_with_exit_anchors,
    format_card_line,
)


class TestComputeBatchFloor:
    def test_basic(self):
        # 100.0 - 2.5 * 2.0 = 95.0
        assert compute_batch_floor(100.0, 2.0) == pytest.approx(95.0, abs=1e-4)

    def test_custom_multiplier(self):
        assert compute_batch_floor(100.0, 4.0, multiplier=2.0) == pytest.approx(92.0, abs=1e-4)

    def test_floor_below_price(self):
        assert compute_batch_floor(200.0, 5.0) < 200.0

    def test_precision(self):
        result = compute_batch_floor(150.0, 3.5)
        assert result == pytest.approx(150.0 - 2.5 * 3.5, abs=1e-4)


class TestComputePtSignal:
    def test_no_pt(self):
        result = compute_pt_signal(100.0, None)
        assert result["upside_pct"] is None
        assert result["take_profit_alert"] is False

    def test_upside_calculation(self):
        result = compute_pt_signal(100.0, 120.0)
        assert result["upside_pct"] == pytest.approx(20.0, abs=0.01)

    def test_no_alert_above_5pct(self):
        # (120-114)/114 ≈ 5.26% > threshold
        assert compute_pt_signal(114.0, 120.0)["take_profit_alert"] is False

    def test_alert_within_5pct(self):
        # (120-116)/116 ≈ 3.45% < threshold
        assert compute_pt_signal(116.0, 120.0)["take_profit_alert"] is True

    def test_alert_at_pt(self):
        # price == PT → upside = 0% ≤ threshold
        assert compute_pt_signal(120.0, 120.0)["take_profit_alert"] is True

    def test_zero_price_returns_safe_defaults(self):
        result = compute_pt_signal(0.0, 120.0)
        assert result["upside_pct"] is None
        assert result["take_profit_alert"] is False


class TestBreakoutExtension:
    def test_no_extension_below_pt(self):
        result = compute_breakout_extension(
            current_price=100.0, price_target=120.0, atr_14=3.0,
            rsi_14=60.0, vwap_ratio=1.01,
        )
        assert result["breakout_extension"] is False
        assert result["extended_target"] is None

    def test_extension_above_pt_with_accumulation(self):
        result = compute_breakout_extension(
            current_price=125.0, price_target=120.0, atr_14=3.0,
            rsi_14=62.0, vwap_ratio=1.02,
        )
        assert result["breakout_extension"] is True
        # Extended target = 120 + 1.5 * 3.0 = 124.5
        assert result["extended_target"] == pytest.approx(124.5, abs=1e-4)

    def test_no_extension_above_pt_without_rsi(self):
        result = compute_breakout_extension(
            current_price=125.0, price_target=120.0, atr_14=3.0,
            rsi_14=45.0, vwap_ratio=1.02,
        )
        assert result["breakout_extension"] is False

    def test_no_extension_above_pt_without_vwap(self):
        result = compute_breakout_extension(
            current_price=125.0, price_target=120.0, atr_14=3.0,
            rsi_14=62.0, vwap_ratio=0.98,
        )
        assert result["breakout_extension"] is False

    def test_no_extension_none_rsi_none_vwap(self):
        result = compute_breakout_extension(
            current_price=125.0, price_target=120.0, atr_14=3.0,
            rsi_14=None, vwap_ratio=None,
        )
        assert result["breakout_extension"] is False


class TestEnrichWithExitAnchors:
    def test_attaches_exit_anchors(self):
        entry = {"ticker": "AAPL", "current_price": 170.0, "price_target": 190.0}
        result = enrich_with_exit_anchors(entry, atr_14=4.0)
        assert "exit_anchors" in result
        anchors = result["exit_anchors"]
        assert anchors["atr_14"] == pytest.approx(4.0, abs=1e-4)
        assert anchors["batch_floor"] == pytest.approx(170.0 - 2.5 * 4.0, abs=1e-4)

    def test_no_atr_gives_none_floor(self):
        entry = {"ticker": "AAPL", "current_price": 170.0}
        result = enrich_with_exit_anchors(entry, atr_14=None)
        assert result["exit_anchors"]["batch_floor"] is None


class TestFormatCardLine:
    def _make_entry(self, *, breakout=False, ext_tgt=None):
        return {
            "ticker": "AAPL",
            "current_price": 170.0,
            "price_target": 190.0,
            "exit_anchors": {
                "batch_floor": 162.5,
                "upside_pct": 11.76,
                "take_profit_alert": False,
                "breakout_extension": breakout,
                "extended_target": ext_tgt,
            },
        }

    def test_contains_batch_floor_label(self):
        line = format_card_line(self._make_entry())
        assert "Batch Floor" in line

    def test_no_hard_stop_label(self):
        line = format_card_line(self._make_entry())
        assert "Hard Stop" not in line

    def test_ticker_in_line(self):
        assert "AAPL" in format_card_line(self._make_entry())

    def test_spot_price_in_line(self):
        assert "170.00" in format_card_line(self._make_entry())

    def test_floor_value_in_line(self):
        line = format_card_line(self._make_entry())
        assert "162.50" in line or "162.5" in line

    def test_breakout_extension_appended(self):
        line = format_card_line(self._make_entry(breakout=True, ext_tgt=196.5))
        assert "BREAKOUT EXTENSION" in line
        assert "196.50" in line

    def test_no_breakout_extension_when_false(self):
        line = format_card_line(self._make_entry(breakout=False))
        assert "BREAKOUT EXTENSION" not in line
