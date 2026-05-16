"""tests/test_revolut_parser.py — unit tests for Revolut XLSX parser"""
from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import openpyxl
import pytest

from regime_trader.services.revolut_parser import parse_xlsx, net_positions_from_rows


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_xlsx(rows: list[tuple]) -> Path:
    """Write an in-memory XLSX with Revolut header + given rows, return tmp path."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(("Date", "Ticker", "Type", "Quantity", "Price per share", "Total Amount", "Currency", "FX Rate"))
    for row in rows:
        ws.append(row)
    tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
    wb.save(tmp.name)
    return Path(tmp.name)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestNetPositionsFromRows:
    def test_single_buy_creates_position(self):
        rows = [
            {"ticker": "MSFT", "type": "BUY - MARKET", "qty": 2.0, "price": 393.22, "currency": "USD"},
        ]
        result = net_positions_from_rows(rows)
        assert len(result) == 1
        assert result[0]["ticker"] == "MSFT"
        assert result[0]["net_qty"] == pytest.approx(2.0)
        assert result[0]["avg_cost"] == pytest.approx(393.22)

    def test_full_sell_removes_position(self):
        rows = [
            {"ticker": "COIN", "type": "BUY - MARKET",  "qty": 5.0,  "price": 200.0, "currency": "USD"},
            {"ticker": "COIN", "type": "SELL - MARKET", "qty": 5.0,  "price": 180.0, "currency": "USD"},
        ]
        result = net_positions_from_rows(rows)
        assert result == []

    def test_partial_sell_keeps_remainder(self):
        rows = [
            {"ticker": "DDOG", "type": "BUY - MARKET",  "qty": 4.0, "price": 100.0, "currency": "USD"},
            {"ticker": "DDOG", "type": "SELL - MARKET", "qty": 1.5, "price": 110.0, "currency": "USD"},
        ]
        result = net_positions_from_rows(rows)
        assert len(result) == 1
        assert result[0]["net_qty"] == pytest.approx(2.5)

    def test_weighted_avg_cost_basis(self):
        rows = [
            {"ticker": "OXY", "type": "BUY - MARKET", "qty": 5.0, "price": 50.0, "currency": "USD"},
            {"ticker": "OXY", "type": "BUY - MARKET", "qty": 5.0, "price": 60.0, "currency": "USD"},
        ]
        result = net_positions_from_rows(rows)
        assert result[0]["avg_cost"] == pytest.approx(55.0)  # (5*50 + 5*60) / 10

    def test_dividend_rows_ignored(self):
        rows = [
            {"ticker": "ORCL", "type": "BUY - MARKET", "qty": 2.0, "price": 176.0, "currency": "USD"},
            {"ticker": "ORCL", "type": "DIVIDEND",      "qty": None, "price": None, "currency": "USD"},
        ]
        result = net_positions_from_rows(rows)
        assert len(result) == 1
        assert result[0]["net_qty"] == pytest.approx(2.0)

    def test_source_field_is_revolut(self):
        rows = [
            {"ticker": "PANW", "type": "BUY - MARKET", "qty": 3.0, "price": 169.0, "currency": "USD"},
        ]
        result = net_positions_from_rows(rows)
        assert result[0]["source"] == "revolut"


class TestParseXlsx:
    def test_parses_real_format(self):
        xlsx_path = _make_xlsx([
            ("2026-04-14T13:40:22.074Z", "NUVL", "BUY - MARKET", 2.38, "USD 104.84", "USD 250", "USD", 1.18),
            ("2026-04-14T13:40:52.225Z", "MSTR", "BUY - MARKET", 1.42, "USD 139.71", "USD 199", "USD", 1.18),
            ("2026-04-10T16:40:37.280Z", "GZF",  "SELL - MARKET", 15,  "EUR 29.23",  "EUR 438", "EUR", 1.0),
        ])
        result = parse_xlsx(xlsx_path, ticker_map={})
        tickers = {p["ticker"] for p in result}
        assert "NUVL" in tickers
        assert "MSTR" in tickers
        assert "GZF" not in tickers  # net_qty = -15 (sell with no buy)

    def test_price_string_with_currency_prefix_parsed(self):
        xlsx_path = _make_xlsx([
            ("2026-04-14T13:43:01.610Z", "MSFT", "BUY - MARKET", 1.27, "USD 393.22", "USD 500", "USD", 1.18),
        ])
        result = parse_xlsx(xlsx_path, ticker_map={})
        assert result[0]["avg_cost"] == pytest.approx(393.22, abs=0.01)

    def test_ticker_map_applied(self):
        xlsx_path = _make_xlsx([
            ("2026-04-14T13:00:00Z", "AIR1", "BUY - MARKET", 10.0, "EUR 28.0", "EUR 280", "EUR", 1.0),
        ])
        result = parse_xlsx(xlsx_path, ticker_map={"AIR1": "EADSY"})
        assert result[0]["ticker"] == "EADSY"
        assert result[0]["revolut_ticker"] == "AIR1"

    def test_cash_events_ignored(self):
        xlsx_path = _make_xlsx([
            ("2026-04-14T13:43:02Z", None, "CASH TOP-UP",    None, None, "USD 501", "USD", 1.18),
            ("2026-04-14T13:42:31Z", None, "CASH WITHDRAWAL", None, None, "EUR -458", "EUR", 1.0),
            ("2026-04-14T13:40:22Z", "OXY", "BUY - MARKET",  5.0, "USD 54.50", "USD 272", "USD", 1.18),
        ])
        result = parse_xlsx(xlsx_path, ticker_map={})
        assert len(result) == 1
        assert result[0]["ticker"] == "OXY"
