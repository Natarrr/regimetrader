# Alpha Engine Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Stock Picker and Portfolio Advisor pages, make Revolut the primary portfolio source, and fix two consistency bugs — all without touching the existing backend scoring pipeline.

**Architecture:** Incremental extension following existing patterns. New pages live in `pages/` with a `render()` function loaded by `_load_page_module()`. New services live in `regime_trader/services/` and `regime_trader/ui/`. The sidebar is refactored from `st.radio()` to `st.sidebar.button()` with session-state tracking. All data reads from existing JSON files (`intel_source_status.json`, `top_lists.json`) or `data/revolut_portfolio.json` — zero new pipeline calls from the UI.

**Tech Stack:** Python 3.11, Streamlit, openpyxl, yfinance, plotly, anthropic SDK (optional, gated), pytest, unittest.mock

---

## File Map

| Action | Path | Responsibility |
|--------|------|---------------|
| Modify | `regime_trader/ui/streamlit_app.py` | Sidebar refactor, Live Monitor Revolut-first, Portfolio Sync tab, explainability fix |
| Modify | `backend/market_intel/generate_top_lists.py` | Add `_sector_picks()`, write `sector_picks` to `top_lists.json` |
| Create | `regime_trader/services/revolut_parser.py` | Parse Revolut XLSX → net positions + avg cost |
| Create | `data/revolut_ticker_map.json` | Revolut symbol → scoring universe symbol mapping |
| Create | `pages/6_Stock_Picker.py` | Monthly pick leaderboard — sector + cap-tier, read-only |
| Create | `pages/7_Portfolio_Advisor.py` | Daily position advice — table + expand drawer |
| Create | `regime_trader/ui/portfolio_advisor_engine.py` | Hybrid signal: quant score lookup + Claude narrative |
| Create | `tests/test_revolut_parser.py` | Unit tests for revolut_parser |
| Create | `tests/test_sector_picks.py` | Unit tests for _sector_picks |
| Create | `tests/test_portfolio_advisor_engine.py` | Unit tests for portfolio_advisor_engine |

---

## Task 1: Fix explainability label mismatch

**Files:**
- Modify: `regime_trader/ui/streamlit_app.py` (find `_render_explainability`)

The `_render_explainability()` function references `insider_score / institutional_score / momentum_score` (old schema). The pipeline produces `edgar_score / insider_score / congress_score / news_score / macro_score`. This causes wrong labels and wrong weights in the UI.

- [ ] **Step 1: Locate and replace the `_WEIGHTS` dict inside `_render_explainability()`**

Find this block (around line 886):
```python
_WEIGHTS = {
    "insider_score":       ("Insider",       0.25),
    "institutional_score": ("Institutional", 0.20),
    "momentum_score":      ("Momentum",      0.20),
    "smart_money_score":   ("Smart Money",   0.35),
}
```

Replace with:
```python
_WEIGHTS = {
    "edgar_score":    ("Edgar",    0.30),
    "insider_score":  ("Insider",  0.25),
    "congress_score": ("Congress", 0.20),
    "news_score":     ("News",     0.15),
    "macro_score":    ("Macro",    0.10),
}
```

- [ ] **Step 2: Commit**

```bash
git add regime_trader/ui/streamlit_app.py
git commit -m "fix(ui): align explainability labels with 5-factor schema"
```

---

## Task 2: Add sector picks to generate_top_lists.py

**Files:**
- Modify: `backend/market_intel/generate_top_lists.py`
- Create: `tests/test_sector_picks.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_sector_picks.py`:
```python
"""tests/test_sector_picks.py — unit tests for _sector_picks()"""
from __future__ import annotations
import pytest
from backend.market_intel.generate_top_lists import _sector_picks

_ENTRIES = [
    {"ticker": "XOM",  "sector": "Energy",                   "final_score": 0.80, "cap_tier": "large", "market_cap": 4e11},
    {"ticker": "CVX",  "sector": "Energy",                   "final_score": 0.70, "cap_tier": "large", "market_cap": 3e11},
    {"ticker": "OXY",  "sector": "Energy",                   "final_score": 0.60, "cap_tier": "mid",   "market_cap": 5e10},
    {"ticker": "ENPH", "sector": "Energy",                   "final_score": 0.55, "cap_tier": "small", "market_cap": 1e10},
    {"ticker": "NVDA", "sector": "Information Technology",   "final_score": 0.90, "cap_tier": "large", "market_cap": 2e12},
    {"ticker": "MSFT", "sector": "Information Technology",   "final_score": 0.85, "cap_tier": "large", "market_cap": 3e12},
    {"ticker": "AAPL", "sector": "Information Technology",   "final_score": 0.82, "cap_tier": "large", "market_cap": 3e12},
    {"ticker": "DELL", "sector": "Information Technology",   "final_score": 0.50, "cap_tier": "mid",   "market_cap": 8e10},
    {"ticker": "PFE",  "sector": "Healthcare",               "final_score": 0.65, "cap_tier": "large", "market_cap": 1.5e11},
    {"ticker": "TMO",  "sector": "Healthcare",               "final_score": 0.72, "cap_tier": "large", "market_cap": 2e11},
    {"ticker": "PANW", "sector": "Communication Services",   "final_score": 0.88, "cap_tier": "mid",   "market_cap": 9e10},
    {"ticker": "META", "sector": "Communication Services",   "final_score": 0.78, "cap_tier": "large", "market_cap": 1e12},
    {"ticker": "FCX",  "sector": "Materials",                "final_score": 0.66, "cap_tier": "mid",   "market_cap": 6e10},
    {"ticker": "NEM",  "sector": "Financials",               "final_score": 0.91, "cap_tier": "large", "market_cap": 3e10},
]


def test_returns_dict_with_target_sectors():
    result = _sector_picks(_ENTRIES)
    for sector in ["Energy", "Materials", "Communication Services", "Healthcare", "Information Technology"]:
        assert sector in result


def test_top_n_per_sector_sorted_descending():
    result = _sector_picks(_ENTRIES, n=3)
    energy = result["Energy"]
    assert len(energy) == 3
    scores = [e["final_score"] for e in energy]
    assert scores == sorted(scores, reverse=True)


def test_sector_with_fewer_than_n_tickers():
    result = _sector_picks(_ENTRIES, n=3)
    materials = result["Materials"]
    assert len(materials) == 1  # only FCX in Materials


def test_non_target_sector_excluded():
    result = _sector_picks(_ENTRIES, n=3)
    all_tickers = [e["ticker"] for picks in result.values() for e in picks]
    assert "NEM" not in all_tickers  # Financials is not a target sector


def test_empty_entries_returns_empty_lists():
    result = _sector_picks([], n=3)
    for picks in result.values():
        assert picks == []
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd "c:/Users/ntard/Projects/Trading dashboard/regime_trader"
.venv/Scripts/pytest tests/test_sector_picks.py -v
```
Expected: `ImportError: cannot import name '_sector_picks'`

- [ ] **Step 3: Add `_sector_picks` and wire into `generate()` in `backend/market_intel/generate_top_lists.py`**

Add the constant and function after the existing `_MID_CUTOFF` line:
```python
_TARGET_SECTORS = [
    "Energy",
    "Materials",
    "Communication Services",
    "Healthcare",
    "Information Technology",
]


def _sector_picks(entries: List[Dict[str, Any]], n: int = 3) -> Dict[str, List[Dict[str, Any]]]:
    result: Dict[str, List[Dict[str, Any]]] = {}
    for sector in _TARGET_SECTORS:
        candidates = [e for e in entries if e.get("sector") == sector]
        result[sector] = sorted(candidates, key=lambda e: e["final_score"], reverse=True)[:n]
    return result
```

In the `generate()` function, add sector picks to `top_lists` dict (after the existing keys):
```python
    top_lists: Dict[str, Any] = {
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "source_run_id": run_id,
        "ticker_count":  len(entries),
        "weights":       WEIGHTS,
        "top_buys":      top_buys,
        "mid_caps":      mid_caps,
        "small_caps":    small_caps,
        "sector_picks":  _sector_picks(entries),
    }
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
.venv/Scripts/pytest tests/test_sector_picks.py -v
```
Expected: 5 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add backend/market_intel/generate_top_lists.py tests/test_sector_picks.py
git commit -m "feat(pipeline): add sector_picks to top_lists.json for 5 target sectors"
```

---

## Task 3: Revolut parser service + ticker map

**Files:**
- Create: `regime_trader/services/revolut_parser.py`
- Create: `data/revolut_ticker_map.json`
- Create: `tests/test_revolut_parser.py`

- [ ] **Step 1: Create the ticker map file**

Create `data/revolut_ticker_map.json`:
```json
{
  "AIR1": "EADSY",
  "URNU": "URA",
  "MCHA": "MCHI",
  "ENXB": "ENXB",
  "UDIV1": "UDIV1",
  "4MMR": "4MMR",
  "CEBT": "CEBT",
  "GZF": "GZF"
}
```

Note: tickers with no clean US equivalent keep their Revolut symbol. The Portfolio Advisor engine resolves them as `not_scored` at lookup time.

- [ ] **Step 2: Write the failing tests**

Create `tests/test_revolut_parser.py`:
```python
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
```

- [ ] **Step 3: Run tests to confirm they fail**

```bash
.venv/Scripts/pytest tests/test_revolut_parser.py -v
```
Expected: `ModuleNotFoundError: No module named 'regime_trader.services.revolut_parser'`

- [ ] **Step 4: Implement `regime_trader/services/revolut_parser.py`**

```python
"""regime_trader/services/revolut_parser.py
Parse a Revolut trading account statement (.xlsx) into net positions.

Revolut XLSX columns:
  Date | Ticker | Type | Quantity | Price per share | Total Amount | Currency | FX Rate

Transaction types handled:
  BUY - MARKET / BUY - LIMIT / BUY - STOP  → add to position
  SELL - MARKET / SELL - LIMIT / SELL - STOP → reduce position
  DIVIDEND, CASH TOP-UP, CASH WITHDRAWAL    → ignored
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import openpyxl

_BUY_TYPES  = {"BUY - MARKET", "BUY - LIMIT", "BUY - STOP"}
_SELL_TYPES = {"SELL - MARKET", "SELL - LIMIT", "SELL - STOP"}

_DEFAULT_MAP = Path(__file__).parent.parent.parent / "data" / "revolut_ticker_map.json"


def _load_default_ticker_map() -> Dict[str, str]:
    if _DEFAULT_MAP.exists():
        return json.loads(_DEFAULT_MAP.read_text(encoding="utf-8"))
    return {}


def _parse_price(raw: Any) -> float:
    """Parse price field which may be 'USD 104.84', 'EUR 29.23', or a bare float."""
    if raw is None:
        return 0.0
    parts = str(raw).strip().split()
    try:
        return float(parts[-1])
    except (ValueError, IndexError):
        return 0.0


def net_positions_from_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compute net positions from a list of normalised transaction dicts.

    Each dict must have: ticker, type, qty (float|None), price (float|None), currency.
    Returns only positions with net_qty > 1e-6.
    """
    buys:    Dict[str, List[tuple[float, float]]] = {}
    net_qty: Dict[str, float] = {}
    currency_map: Dict[str, str] = {}

    for row in rows:
        tx_type = str(row.get("type", "")).strip()
        ticker  = row.get("ticker")
        qty     = row.get("qty")
        price   = row.get("price") or 0.0
        currency = str(row.get("currency", "USD")).strip()

        if not ticker or qty is None:
            continue

        qty = float(qty)

        if tx_type in _BUY_TYPES:
            net_qty[ticker] = net_qty.get(ticker, 0.0) + qty
            buys.setdefault(ticker, []).append((qty, float(price)))
            currency_map[ticker] = currency
        elif tx_type in _SELL_TYPES:
            net_qty[ticker] = net_qty.get(ticker, 0.0) - qty

    positions = []
    for ticker, remaining in net_qty.items():
        if remaining <= 1e-6:
            continue
        buy_list = buys.get(ticker, [])
        total_qty_bought = sum(q for q, _ in buy_list)
        total_cost = sum(q * p for q, p in buy_list)
        avg_cost = total_cost / total_qty_bought if total_qty_bought > 0 else 0.0
        positions.append({
            "ticker":          ticker,
            "revolut_ticker":  ticker,
            "net_qty":         round(remaining, 8),
            "avg_cost":        round(avg_cost, 4),
            "currency":        currency_map.get(ticker, "USD"),
            "source":          "revolut",
        })

    return sorted(positions, key=lambda p: p["ticker"])


def parse_xlsx(
    filepath: str | Path,
    ticker_map: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Parse a Revolut XLSX statement into a list of net positions.

    Args:
        filepath:   Path to the .xlsx file.
        ticker_map: Optional {revolut_symbol: universe_symbol} dict.
                    Defaults to data/revolut_ticker_map.json.

    Returns:
        List of position dicts sorted by ticker.
    """
    if ticker_map is None:
        ticker_map = _load_default_ticker_map()

    wb = openpyxl.load_workbook(filepath, data_only=True)
    ws = wb.active

    # Locate the header row (first row where col 0 == "Date")
    headers: Optional[List[str]] = None
    header_row_idx = 0
    for i, row in enumerate(ws.iter_rows(values_only=True), 1):
        if row and str(row[0]).strip() == "Date":
            headers = [str(c).strip() if c is not None else "" for c in row]
            header_row_idx = i
            break

    if headers is None:
        raise ValueError(f"Could not find header row in {filepath}")

    col = {h: i for i, h in enumerate(headers)}

    rows = []
    for raw in ws.iter_rows(min_row=header_row_idx + 1, values_only=True):
        rows.append({
            "ticker":   str(raw[col["Ticker"]]).strip() if raw[col["Ticker"]] else None,
            "type":     str(raw[col["Type"]]).strip()   if raw[col["Type"]]   else "",
            "qty":      raw[col["Quantity"]],
            "price":    _parse_price(raw[col["Price per share"]]),
            "currency": str(raw[col["Currency"]]).strip() if col.get("Currency") is not None and raw[col.get("Currency", 0)] else "USD",
        })

    positions = net_positions_from_rows(rows)

    # Apply ticker mapping
    for pos in positions:
        original = pos["ticker"]
        mapped   = ticker_map.get(original, original)
        pos["ticker"]          = mapped
        pos["revolut_ticker"]  = original

    return positions
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
.venv/Scripts/pytest tests/test_revolut_parser.py -v
```
Expected: 9 tests PASSED

- [ ] **Step 6: Commit**

```bash
git add regime_trader/services/revolut_parser.py data/revolut_ticker_map.json tests/test_revolut_parser.py
git commit -m "feat(services): add Revolut XLSX parser — nets BUY/SELL into positions with avg cost"
```

---

## Task 4: Sidebar refactor — section headers + button nav

**Files:**
- Modify: `regime_trader/ui/streamlit_app.py`

The current `_render_sidebar()` uses `st.radio()`, which cannot render section headers. Replace with `st.sidebar.button()` per page and `st.session_state["_nav_page"]` for selection tracking.

- [ ] **Step 1: Update `_NAV_PAGES` to include the two new pages**

Find the existing `_NAV_PAGES` list and replace it:
```python
_NAV_PAGES = [
    ("📊 Dashboard",              None,                    None),
    ("📅 Stock Picker",           "6_Stock_Picker.py",     "stock_picker"),
    ("💼 Portfolio Advisor",      "7_Portfolio_Advisor.py","portfolio_advisor"),
    ("💰 Monetary Pulse",         "1_Monetary_Pulse.py",   "monetary_pulse"),
    ("📈 Volatility Brain",       "2_Volatility_Brain.py", "volatility_brain"),
    ("🔭 Valuation Radar",        "3_Valuation_Radar.py",  "valuation_radar"),
    ("🕸️ Contagion Web",          "4_Contagion_Web.py",    "contagion_web"),
    ("🎯 Regime Prediction",      "5_Regime_Prediction.py","regime_prediction"),
]
```

- [ ] **Step 2: Replace `_render_sidebar()` with button-based nav**

Replace the entire `_render_sidebar()` function:
```python
_NAV_SECTION_CSS = """
<style>
.rt-nav-section {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    color: rgba(255,255,255,0.30);
    padding: 10px 4px 3px;
    text-transform: uppercase;
}
</style>
"""

_ALPHA_ENGINE_PAGES = {"📅 Stock Picker", "💼 Portfolio Advisor"}
_QUANT_MODEL_PAGES  = {"💰 Monetary Pulse", "📈 Volatility Brain",
                        "🔭 Valuation Radar", "🕸️ Contagion Web", "🎯 Regime Prediction"}


def _render_sidebar() -> str:
    """Render sidebar navigation with section headers. Returns selected page label."""
    with st.sidebar:
        st.markdown(_NAV_SECTION_CSS, unsafe_allow_html=True)
        st.markdown("## 🧭 Navigate")

        if "_nav_page" not in st.session_state:
            st.session_state["_nav_page"] = "📊 Dashboard"

        def _nav_btn(label: str) -> None:
            is_active = st.session_state["_nav_page"] == label
            if st.button(
                label,
                key=f"_nav_{label}",
                use_container_width=True,
                type="primary" if is_active else "secondary",
            ):
                st.session_state["_nav_page"] = label
                st.rerun()

        st.markdown('<div class="rt-nav-section">── Alpha Engine ──</div>', unsafe_allow_html=True)
        _nav_btn("📅 Stock Picker")
        _nav_btn("💼 Portfolio Advisor")

        st.markdown('<div class="rt-nav-section">── Quant Models ──</div>', unsafe_allow_html=True)
        _nav_btn("💰 Monetary Pulse")
        _nav_btn("📈 Volatility Brain")
        _nav_btn("🔭 Valuation Radar")
        _nav_btn("🕸️ Contagion Web")
        _nav_btn("🎯 Regime Prediction")

        st.markdown('<div class="rt-nav-section">── Dashboard ──</div>', unsafe_allow_html=True)
        _nav_btn("📊 Dashboard")

        st.divider()
        st.markdown("## ⚙️ Settings")

        with st.expander("Cache controls", expanded=False):
            if st.button("Clear engine state cache", key="clear_disc"):
                _load_market_state.clear()
                _load_discovery.clear()
                st.success("Engine state + discovery cache cleared.")
            if st.button("Clear commodity cache", key="clear_comm"):
                _load_commodity_prices.clear()
                _load_macro_indicators.clear()
                st.success("Commodity / macro cache cleared.")
            if st.button("Clear account cache", key="clear_acct"):
                _load_alpaca_account.clear()
                _load_regime.clear()
                _load_vix_history.clear()
                _load_portfolio_history.clear()
                st.success("Account / regime cache cleared.")

        with st.expander("Environment", expanded=False):
            st.caption(f"FMP key: **{'set' if _HAS_FMP else 'missing'}**")
            st.caption(f"Alpaca key: **{'set' if _HAS_ALPACA else 'missing'}**")
            st.caption(f"Paper trading: **{_ALPACA_PAPER}**")
            st.caption(f"EDGAR User-Agent: `{_EDGAR_USER_AGENT}`")

    return st.session_state.get("_nav_page", "📊 Dashboard")
```

- [ ] **Step 3: Update `main()` dispatch to use the new page labels**

In `main()`, the `_page_map` lookup already uses the labels from `_NAV_PAGES`. The only change is the Dashboard guard:
```python
def main() -> None:
    selected = _render_sidebar()
    _page_map = {label: (fn, mn) for label, fn, mn in _NAV_PAGES}

    if selected == "📊 Dashboard" or selected not in _page_map:
        _render_dashboard()
        return

    filename, mod_name = _page_map[selected]
    if filename is None:
        _render_dashboard()
        return

    mod = _load_page_module(mod_name, filename)
    if mod is None:
        st.error(
            f"**{selected}** could not be loaded. "
            "Check that all backend dependencies are installed and try again."
        )
        return

    if not hasattr(mod, "render"):
        st.error(f"Page module `{filename}` has no `render()` function.")
        return

    try:
        mod.render()
    except Exception as exc:
        log.exception("Page render failed: %s — %s", selected, exc)
        st.error(f"**{selected}** encountered an error: {exc}")
```

- [ ] **Step 4: Remove stub tabs from `_render_dashboard()`**

Replace `_render_dashboard()`:
```python
def _render_dashboard() -> None:
    """Render the main dashboard with four tabs."""
    st.title("Regime Trader Dashboard")
    tabs = st.tabs([
        "📊 Live Monitor",
        "🧠 Market Intel",
        "🌍 Macro Intel",
        "🔄 Portfolio Sync",
    ])
    with tabs[0]:
        _render_live_monitor()
    with tabs[1]:
        _render_market_intel()
    with tabs[2]:
        _render_macro_intel()
    with tabs[3]:
        _render_portfolio_sync()
```

- [ ] **Step 5: Commit**

```bash
git add regime_trader/ui/streamlit_app.py
git commit -m "feat(ui): sidebar section headers — Alpha Engine / Quant Models / Dashboard"
```

---

## Task 5: Portfolio Sync tab — Revolut importer

**Files:**
- Modify: `regime_trader/ui/streamlit_app.py` (replace `_render_portfolio_sync()`)

- [ ] **Step 1: Add `_REVOLUT_PORTFOLIO_PATH` constant** near the other path constants at the top of `streamlit_app.py`:

```python
_REVOLUT_PORTFOLIO_PATH = _ROOT / "data" / "revolut_portfolio.json"
```

- [ ] **Step 2: Replace `_render_portfolio_sync()`**

```python
def _render_portfolio_sync() -> None:
    """Render the Portfolio Sync tab — Revolut XLSX importer."""
    from regime_trader.services.revolut_parser import parse_xlsx
    from regime_trader.utils.io import save_json_atomic
    import pandas as pd

    st.header("Portfolio Sync — Revolut Import")

    # ── Last import badge ──────────────────────────────────────────────────────
    if _REVOLUT_PORTFOLIO_PATH.exists():
        try:
            existing = json.loads(_REVOLUT_PORTFOLIO_PATH.read_text(encoding="utf-8"))
            imported_at = existing.get("imported_at", "unknown")
            n_pos = len(existing.get("positions", []))
            st.info(f"📋 Last import: **{imported_at}** · {n_pos} positions loaded")
        except Exception:
            pass

    st.markdown(
        "Upload your Revolut trading account statement (`.xlsx`). "
        "The parser nets all BUY/SELL transactions to derive your current holdings."
    )

    uploaded = st.file_uploader(
        "Revolut account statement",
        type=["xlsx"],
        key="revolut_upload",
    )

    if not uploaded:
        return

    # ── Parse ──────────────────────────────────────────────────────────────────
    try:
        import tempfile, shutil
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
            shutil.copyfileobj(uploaded, tmp)
            tmp_path = tmp.name
        positions = parse_xlsx(tmp_path)
    except Exception as exc:
        st.error(f"Failed to parse file: {exc}")
        return

    if not positions:
        st.warning("No open positions found in this statement.")
        return

    # ── Preview ────────────────────────────────────────────────────────────────
    st.subheader(f"Preview — {len(positions)} positions")
    df = pd.DataFrame(positions)
    df_display = df[["ticker", "revolut_ticker", "net_qty", "avg_cost", "currency"]].copy()
    df_display.columns = ["Ticker", "Revolut Symbol", "Net Qty", "Avg Cost", "Currency"]
    df_display["Avg Cost"] = df_display["Avg Cost"].map("{:.4f}".format)

    st.dataframe(df_display, use_container_width=True, hide_index=True)

    # ── Confirm ────────────────────────────────────────────────────────────────
    if st.button("✅ Confirm & Save Portfolio", type="primary"):
        from datetime import datetime, timezone
        payload = {
            "imported_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "positions":   positions,
        }
        _REVOLUT_PORTFOLIO_PATH.parent.mkdir(parents=True, exist_ok=True)
        save_json_atomic(_REVOLUT_PORTFOLIO_PATH, payload)
        st.success(
            f"Portfolio saved — {len(positions)} positions. "
            "Live Monitor and Portfolio Advisor will now use your Revolut data."
        )
        st.rerun()
```

- [ ] **Step 3: Commit**

```bash
git add regime_trader/ui/streamlit_app.py
git commit -m "feat(ui): Portfolio Sync tab — Revolut XLSX importer with preview and disk persistence"
```

---

## Task 6: Live Monitor tab — Revolut-first

**Files:**
- Modify: `regime_trader/ui/streamlit_app.py` (update `_render_live_monitor()`)

- [ ] **Step 1: Add `_load_revolut_positions()` cached loader** after the other cached loaders:

```python
@st.cache_data(ttl=300, show_spinner=False)
def _load_revolut_positions() -> Optional[Dict[str, Any]]:
    """Load persisted Revolut portfolio. Returns None if not yet imported."""
    try:
        return json.loads(_REVOLUT_PORTFOLIO_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as exc:
        log.warning("revolut_portfolio.json read failed: %s", exc)
        return None
```

- [ ] **Step 2: Add `_fetch_live_prices()` helper** (yfinance, no API key needed):

```python
def _fetch_live_prices(tickers: list[str]) -> Dict[str, float]:
    """Fetch latest close prices for a list of tickers via yfinance. Returns {} on failure."""
    if not tickers:
        return {}
    try:
        import yfinance as yf
        raw = yf.download(tickers, period="2d", interval="1d",
                          progress=False, auto_adjust=True, group_by="ticker")
        prices: Dict[str, float] = {}
        for t in tickers:
            try:
                if len(tickers) == 1:
                    col_data = raw["Close"].dropna()
                else:
                    col_data = raw[t]["Close"].dropna()
                if not col_data.empty:
                    prices[t] = float(col_data.iloc[-1])
            except Exception:
                pass
        return prices
    except Exception as exc:
        log.warning("live price fetch failed: %s", exc)
        return {}
```

- [ ] **Step 3: Add `_render_revolut_portfolio()` helper**:

```python
def _render_revolut_portfolio() -> None:
    """Render Revolut portfolio block — live prices + P&L vs avg cost."""
    import pandas as pd

    data = _load_revolut_positions()

    if data is None:
        st.info(
            "No Revolut portfolio imported yet. "
            "Go to **Portfolio Sync** tab to upload your statement."
        )
        return

    positions = data.get("positions", [])
    imported_at = data.get("imported_at", "—")

    st.caption(f"Revolut portfolio · Imported: **{imported_at}** · {len(positions)} positions")

    us_tickers = [p["ticker"] for p in positions]
    with st.spinner("Fetching live prices…"):
        prices = _fetch_live_prices(us_tickers)

    rows = []
    total_cost   = 0.0
    total_value  = 0.0

    for pos in positions:
        ticker   = pos["ticker"]
        qty      = pos["net_qty"]
        avg_cost = pos["avg_cost"]
        price    = prices.get(ticker)

        cost_basis = qty * avg_cost
        mkt_value  = qty * price if price else None
        unreal_pl  = (mkt_value - cost_basis) if mkt_value is not None else None
        unreal_pct = (unreal_pl / cost_basis * 100) if (unreal_pl is not None and cost_basis > 0) else None

        total_cost  += cost_basis
        if mkt_value:
            total_value += mkt_value

        rows.append({
            "Ticker":      ticker,
            "Revolut":     pos.get("revolut_ticker", ticker),
            "Qty":         f"{qty:.4f}",
            "Avg Cost":    f"{avg_cost:.2f} {pos.get('currency','USD')}",
            "Price":       f"{price:.2f}" if price else "—",
            "Mkt Value":   f"{mkt_value:,.2f}" if mkt_value else "—",
            "Unreal. P&L": f"{unreal_pl:+,.2f}" if unreal_pl is not None else "—",
            "Unreal. %":   f"{unreal_pct:+.2f}%" if unreal_pct is not None else "—",
        })

    total_pl  = total_value - total_cost if total_value else 0.0
    total_pct = (total_pl / total_cost * 100) if total_cost > 0 else 0.0

    c1, c2, c3 = st.columns(3)
    c1.metric("Portfolio Value",  f"{total_value:,.2f}" if total_value else "—")
    c2.metric("Total Cost Basis", f"{total_cost:,.2f}")
    c3.metric("Total P&L",        f"{total_pl:+,.2f}", f"{total_pct:+.2f}%")

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
```

- [ ] **Step 4: Update `_render_live_monitor()` to put Revolut first, Alpaca in expander**

At the top of `_render_live_monitor()`, after the regime block and before the existing Alpaca code, insert the Revolut block. Then wrap the entire existing Alpaca section in an expander. The regime row and VIX sparkline stay at the top:

Replace the body of `_render_live_monitor()` after the regime data load:
```python
    # ── Row 1: Regime ─────────────────────────────────────────────────────────
    st.metric("Regime", f"{regime_icon} {regime}", f"VIX {vix:.1f}" if vix else None)

    st.divider()

    # ── Revolut Portfolio (primary) ────────────────────────────────────────────
    st.subheader("📈 Revolut Portfolio")
    _render_revolut_portfolio()

    st.divider()

    # ── VIX sparkline ──────────────────────────────────────────────────────────
    _render_vix_sparkline()

    # ── Alpaca paper trading (secondary) ──────────────────────────────────────
    with st.expander("📋 Paper Trading (Alpaca)", expanded=False):
        if not _HAS_ALPACA:
            st.caption("ALPACA_API_KEY / ALPACA_SECRET_KEY not set.")
        else:
            with st.spinner("Loading Alpaca account…"):
                acct = _load_alpaca_account()
            if "error" in acct:
                st.error(f"Alpaca connection failed: {acct['error']}")
            else:
                paper_tag = " · Paper" if acct["paper"] else ""
                pnl       = acct["daily_pnl"]
                pnl_pct   = acct["daily_pnl_pct"]
                col1, col2, col3 = st.columns(3)
                col1.metric("Portfolio Value",  f"${acct['portfolio_value']:,.2f}", f"{pnl_pct:+.2f}% today")
                col2.metric("Daily P&L",        f"${pnl:+,.2f}", f"{pnl_pct:+.2f}%")
                col3.metric("Cash",             f"${acct['cash']:,.2f}")
                st.caption(f"Alpaca · Status: **{acct['status']}**{paper_tag}")
                period = st.radio("period", ["1W","1M","3M","1Y"], index=1, horizontal=True,
                                  label_visibility="collapsed", key="alpaca_period")
                port_history = _load_portfolio_history(period)
                if port_history:
                    _render_portfolio_chart(port_history)
                import pandas as pd
                positions = acct.get("positions", [])
                if positions:
                    df = pd.DataFrame(positions)
                    st.dataframe(df, use_container_width=True, hide_index=True)
```

- [ ] **Step 5: Commit**

```bash
git add regime_trader/ui/streamlit_app.py
git commit -m "feat(ui): Live Monitor — Revolut portfolio primary, Alpaca paper trading in expander"
```

---

## Task 7: Stock Picker page

**Files:**
- Create: `pages/6_Stock_Picker.py`

- [ ] **Step 1: Create `pages/6_Stock_Picker.py`**

```python
"""pages/6_Stock_Picker.py
Monthly stock pick leaderboard — sector picks + cap-tier picks.
Reads logs/top_lists.json (produced by edgar_3x pipeline). Zero API calls.

Run via sidebar: "📅 Stock Picker"
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

log = logging.getLogger(__name__)

_ROOT       = Path(__file__).parent.parent
_TOP_LISTS  = _ROOT / "logs" / "top_lists.json"

_BADGE_COLOR = {
    "HIGH BUY":     "#00d26a",
    "TACTICAL BUY": "#f5a623",
    "WATCHLIST":    "#888888",
}

_SECTOR_EMOJI = {
    "Energy":                   "⚡",
    "Materials":                "🪨",
    "Communication Services":   "📡",
    "Healthcare":               "🏥",
    "Information Technology":   "💻",
}


@st.cache_data(ttl=3600, show_spinner=False)
def _load_top_lists() -> Optional[Dict[str, Any]]:
    try:
        return json.loads(_TOP_LISTS.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as exc:
        log.warning("top_lists.json load failed: %s", exc)
        return None


def _score_bar(score: float, width: int = 10) -> str:
    filled = round(score * width)
    return "█" * filled + "░" * (width - filled)


def _render_ticker_table(entries: List[Dict[str, Any]], show_watchlist: bool = False) -> None:
    if not entries:
        st.caption("No tickers in this category.")
        return

    import pandas as pd

    rows = []
    for i, e in enumerate(entries, 1):
        badge  = e.get("badge", "WATCHLIST")
        if badge == "WATCHLIST" and not show_watchlist:
            continue
        score  = e.get("final_score", 0.0)
        color  = _BADGE_COLOR.get(badge, "#888")
        f      = e.get("factors", {})
        rows.append({
            "#":        i,
            "Ticker":   e.get("ticker", "?"),
            "Cap":      e.get("cap_tier", "?").capitalize(),
            "Score":    f"{score:.3f}",
            "Bar":      _score_bar(score),
            "Badge":    badge,
            "CEO Buy":  "✅" if e.get("ceo_buy") else "",
            "Edgar":    f"{f.get('edgar',0):.2f}",
            "Insider":  f"{f.get('insider',0):.2f}",
            "Congress": f"{f.get('congress',0):.2f}",
            "News":     f"{f.get('news',0):.2f}",
            "Macro":    f"{f.get('macro',0):.2f}",
        })

    if not rows:
        st.caption("No HIGH BUY or TACTICAL BUY tickers. Toggle 'Show Watchlist' to see all.")
        return

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render() -> None:
    st.title("📅 Stock Picker")
    st.caption("Monthly pick leaderboard powered by the 5-factor scoring engine. Informational only.")

    # ── Load data ──────────────────────────────────────────────────────────────
    col_ref, col_ts = st.columns([1, 6])
    if col_ref.button("↻ Refresh", key="sp_refresh"):
        _load_top_lists.clear()
        st.rerun()

    data = _load_top_lists()

    if data is None:
        st.error(
            "**⚠️ No data** — `logs/top_lists.json` not found.\n\n"
            "Run the edgar_3x pipeline to generate picks:\n"
            "```\npython -m backend.market_intel.generate_top_lists --force\n```"
        )
        return

    generated_at = data.get("generated_at", "—")
    ticker_count = data.get("ticker_count", 0)
    col_ts.caption(f"Pipeline ran: **{generated_at}** · {ticker_count} tickers scored")

    show_watchlist = st.toggle("Show WATCHLIST tickers", value=False, key="sp_show_watchlist")

    st.divider()

    # ── Section 1: Sector Picks ────────────────────────────────────────────────
    st.subheader("Sector Picks — Top 3 per Sector")

    sector_picks: Dict[str, List] = data.get("sector_picks", {})

    if not sector_picks:
        st.warning(
            "Sector picks not in this snapshot. Re-run the pipeline with the updated "
            "`generate_top_lists.py` to populate sector data."
        )
    else:
        for sector, emoji in _SECTOR_EMOJI.items():
            picks = sector_picks.get(sector, [])
            label = f"{emoji} {sector} ({len(picks)} picks)"
            with st.expander(label, expanded=True):
                _render_ticker_table(picks, show_watchlist=show_watchlist)

    st.divider()

    # ── Section 2: Cap-Tier Overview ──────────────────────────────────────────
    st.subheader("Cap-Tier Overview")

    col_tb, col_mc, col_sc = st.columns(3)

    with col_tb:
        st.markdown("**🏆 Top Buys**")
        _render_ticker_table(data.get("top_buys", []), show_watchlist=show_watchlist)

    with col_mc:
        st.markdown("**⬡ Mid Caps**")
        _render_ticker_table(data.get("mid_caps", []), show_watchlist=show_watchlist)

    with col_sc:
        st.markdown("**◆ Small Caps**")
        _render_ticker_table(data.get("small_caps", []), show_watchlist=show_watchlist)
```

- [ ] **Step 2: Verify the page loads (smoke test)**

```bash
.venv/Scripts/python -c "
import sys; sys.path.insert(0, '.')
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location('stock_picker', 'pages/6_Stock_Picker.py')
mod = importlib.util.module_from_spec(spec)
print('render callable:', callable(getattr(mod, 'render', None)) or 'MISSING — will fail after exec')
spec.loader.exec_module(mod)
assert callable(mod.render), 'render() not defined'
print('OK')
"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add pages/6_Stock_Picker.py
git commit -m "feat(pages): Stock Picker — sector picks + cap-tier leaderboard"
```

---

## Task 8: Portfolio Advisor engine

**Files:**
- Create: `regime_trader/ui/portfolio_advisor_engine.py`
- Create: `tests/test_portfolio_advisor_engine.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_portfolio_advisor_engine.py`:
```python
"""tests/test_portfolio_advisor_engine.py"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from regime_trader.ui.portfolio_advisor_engine import (
    compute_signal,
    compute_health_score,
    find_swap_candidate,
    PositionAdvice,
    _signal_age_days,
)


# ── Signal thresholds ─────────────────────────────────────────────────────────

class TestComputeSignal:
    def test_high_score_is_add(self):
        assert compute_signal(0.70, regime="Bull") == "ADD"

    def test_mid_score_is_hold(self):
        assert compute_signal(0.55, regime="Bull") == "HOLD"

    def test_low_score_is_reduce(self):
        assert compute_signal(0.38, regime="Bull") == "REDUCE"

    def test_very_low_score_is_exit(self):
        assert compute_signal(0.20, regime="Bull") == "EXIT"

    def test_kill_switch_regime_forces_exit(self):
        assert compute_signal(0.80, regime="Crash") == "EXIT"

    def test_boundary_065_is_add(self):
        assert compute_signal(0.65, regime="Neutral") == "ADD"

    def test_boundary_045_is_hold(self):
        assert compute_signal(0.45, regime="Neutral") == "HOLD"

    def test_boundary_030_is_reduce(self):
        assert compute_signal(0.30, regime="Neutral") == "REDUCE"


# ── Portfolio health score ────────────────────────────────────────────────────

class TestComputeHealthScore:
    def test_weighted_average_by_value(self):
        positions = [
            {"ticker": "AAPL", "final_score": 0.80, "market_value": 800.0},
            {"ticker": "COIN", "final_score": 0.20, "market_value": 200.0},
        ]
        score = compute_health_score(positions)
        # 0.80 * 0.8 + 0.20 * 0.2 = 0.64 + 0.04 = 0.68
        assert score == pytest.approx(0.68, abs=1e-4)

    def test_empty_returns_zero(self):
        assert compute_health_score([]) == 0.0

    def test_single_position(self):
        positions = [{"ticker": "X", "final_score": 0.75, "market_value": 1000.0}]
        assert compute_health_score(positions) == pytest.approx(0.75)


# ── Signal age ────────────────────────────────────────────────────────────────

class TestSignalAge:
    def test_recent_run_shows_small_age(self):
        from datetime import datetime, timezone, timedelta
        recent = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
        status = {"computed_at": recent}
        assert _signal_age_days(status) == pytest.approx(3, abs=1)

    def test_missing_computed_at_returns_none(self):
        assert _signal_age_days({}) is None


# ── Swap candidates ───────────────────────────────────────────────────────────

class TestFindSwapCandidate:
    _TOP_LISTS = {
        "top_buys": [
            {"ticker": "NVDA", "sector": "Information Technology", "final_score": 0.90, "badge": "HIGH BUY"},
            {"ticker": "AAPL", "sector": "Information Technology", "final_score": 0.85, "badge": "HIGH BUY"},
        ],
        "mid_caps": [
            {"ticker": "PANW", "sector": "Communication Services", "final_score": 0.80, "badge": "HIGH BUY"},
        ],
        "small_caps": [],
        "sector_picks": {},
    }

    def test_returns_top_unowned_in_same_sector(self):
        held = {"AAPL"}
        result = find_swap_candidate("MSFT", "Information Technology", held, self._TOP_LISTS)
        assert result is not None
        assert result["ticker"] == "NVDA"

    def test_no_swap_when_all_owned(self):
        held = {"NVDA", "AAPL"}
        result = find_swap_candidate("MSFT", "Information Technology", held, self._TOP_LISTS)
        assert result is None

    def test_no_swap_for_unknown_sector(self):
        result = find_swap_candidate("XYZ", "Utilities", set(), self._TOP_LISTS)
        assert result is None

    def test_does_not_suggest_the_reduce_ticker_itself(self):
        held = set()
        result = find_swap_candidate("NVDA", "Information Technology", held, self._TOP_LISTS)
        assert result is not None
        assert result["ticker"] != "NVDA"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
.venv/Scripts/pytest tests/test_portfolio_advisor_engine.py -v
```
Expected: `ModuleNotFoundError: No module named 'regime_trader.ui.portfolio_advisor_engine'`

- [ ] **Step 3: Implement `regime_trader/ui/portfolio_advisor_engine.py`**

```python
"""regime_trader/ui/portfolio_advisor_engine.py
Hybrid scoring engine for the Portfolio Advisor page.

Reads scores from logs/intel_source_status.json (no new API calls).
Optionally generates a 2-sentence Claude narrative per position.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

log = logging.getLogger(__name__)

_ROOT               = Path(__file__).parent.parent.parent
_STATUS_PATH        = _ROOT / "logs" / "intel_source_status.json"
_TOP_LISTS_PATH     = _ROOT / "logs" / "top_lists.json"

_KILL_SWITCH_REGIMES = {"Crash", "Panic"}


@dataclass
class PositionAdvice:
    ticker:           str
    revolut_ticker:   str
    net_qty:          float
    avg_cost:         float
    currency:         str
    source:           str
    signal:           str          # ADD | HOLD | REDUCE | EXIT
    final_score:      Optional[float]
    factors:          Dict[str, float]
    signal_age_days:  Optional[int]
    swap_candidate:   Optional[Dict[str, Any]]
    narrative:        Optional[str]   # Claude 2-sentence text, populated later
    not_in_universe:  bool
    market_value:     float = 0.0    # filled by UI from live price


# ── Core logic (pure, testable) ───────────────────────────────────────────────

def compute_signal(score: float, regime: str) -> str:
    if regime in _KILL_SWITCH_REGIMES:
        return "EXIT"
    if score >= 0.65:
        return "ADD"
    if score >= 0.45:
        return "HOLD"
    if score >= 0.30:
        return "REDUCE"
    return "EXIT"


def compute_health_score(positions: List[Dict[str, Any]]) -> float:
    """Weighted average final_score by market_value across all positions."""
    total_value = sum(p.get("market_value", 0.0) for p in positions)
    if total_value <= 0:
        return 0.0
    return sum(
        p.get("final_score", 0.0) * p.get("market_value", 0.0)
        for p in positions
    ) / total_value


def _signal_age_days(status: Dict[str, Any]) -> Optional[int]:
    raw = status.get("computed_at")
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - ts).days
    except Exception:
        return None


def find_swap_candidate(
    ticker: str,
    sector: str,
    held_tickers: Set[str],
    top_lists: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Return the top-scored unowned ticker in the same sector, or None."""
    all_entries = (
        top_lists.get("top_buys", []) +
        top_lists.get("mid_caps", []) +
        top_lists.get("small_caps", [])
    )
    candidates = [
        e for e in all_entries
        if e.get("sector") == sector
        and e.get("ticker") != ticker
        and e.get("ticker") not in held_tickers
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda e: e.get("final_score", 0.0))


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_status() -> Dict[str, Any]:
    try:
        return json.loads(_STATUS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception as exc:
        log.warning("intel_source_status.json load failed: %s", exc)
        return {}


def _load_top_lists() -> Dict[str, Any]:
    try:
        return json.loads(_TOP_LISTS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception as exc:
        log.warning("top_lists.json load failed: %s", exc)
        return {}


def _build_score_index(status: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build ticker → scored_row lookup from intel_source_status.json."""
    return {r["ticker"]: r for r in status.get("results", []) if r.get("ticker")}


# ── Public API ────────────────────────────────────────────────────────────────

def build_advice(
    positions: List[Dict[str, Any]],
    regime: str,
) -> List[PositionAdvice]:
    """Score and signal all positions. Returns PositionAdvice list."""
    status     = _load_status()
    top_lists  = _load_top_lists()
    score_idx  = _build_score_index(status)
    age_days   = _signal_age_days(status)
    held       = {p["ticker"] for p in positions}

    advice_list = []
    for pos in positions:
        ticker   = pos["ticker"]
        row      = score_idx.get(ticker)

        if row is None:
            advice_list.append(PositionAdvice(
                ticker          = ticker,
                revolut_ticker  = pos.get("revolut_ticker", ticker),
                net_qty         = pos["net_qty"],
                avg_cost        = pos["avg_cost"],
                currency        = pos.get("currency", "USD"),
                source          = pos.get("source", "revolut"),
                signal          = "—",
                final_score     = None,
                factors         = {},
                signal_age_days = age_days,
                swap_candidate  = None,
                narrative       = None,
                not_in_universe = True,
            ))
            continue

        final_score = float(row.get("edgar_score",0)*0.30 + row.get("insider_score",0)*0.25 +
                            row.get("congress_score",0)*0.20 + row.get("news_score",0)*0.15 +
                            row.get("momentum_score",0)*0.10)
        signal = compute_signal(final_score, regime)
        swap   = find_swap_candidate(ticker, row.get("sector",""), held, top_lists) \
                 if signal in ("REDUCE", "EXIT") else None

        advice_list.append(PositionAdvice(
            ticker          = ticker,
            revolut_ticker  = pos.get("revolut_ticker", ticker),
            net_qty         = pos["net_qty"],
            avg_cost        = pos["avg_cost"],
            currency        = pos.get("currency", "USD"),
            source          = pos.get("source", "revolut"),
            signal          = signal,
            final_score     = round(final_score, 4),
            factors         = {
                "edgar":    round(float(row.get("edgar_score",   0)), 4),
                "insider":  round(float(row.get("insider_score", 0)), 4),
                "congress": round(float(row.get("congress_score",0)), 4),
                "news":     round(float(row.get("news_score",    0)), 4),
                "macro":    round(float(row.get("momentum_score",0)), 4),
            },
            signal_age_days = age_days,
            swap_candidate  = swap,
            narrative       = None,
            not_in_universe = False,
        ))

    return advice_list
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
.venv/Scripts/pytest tests/test_portfolio_advisor_engine.py -v
```
Expected: 12 tests PASSED

- [ ] **Step 5: Commit**

```bash
git add regime_trader/ui/portfolio_advisor_engine.py tests/test_portfolio_advisor_engine.py
git commit -m "feat(engine): Portfolio Advisor engine — signal thresholds, health score, swap candidates"
```

---

## Task 9: Portfolio Advisor page

**Files:**
- Create: `pages/7_Portfolio_Advisor.py`

- [ ] **Step 1: Create `pages/7_Portfolio_Advisor.py`**

```python
"""pages/7_Portfolio_Advisor.py
Daily portfolio advice — Revolut-first, table + expand drawer.

Signals: ADD / HOLD / REDUCE / EXIT derived from 5-factor quant scores.
Claude narrative: 2-sentence explanation per position (gated by ANTHROPIC_API_KEY,
cached in st.session_state to avoid repeated API calls).

Run via sidebar: "💼 Portfolio Advisor"
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from regime_trader.ui.portfolio_advisor_engine import (
    PositionAdvice,
    build_advice,
    compute_health_score,
)

log = logging.getLogger(__name__)

_ROOT                   = Path(__file__).parent.parent
_REVOLUT_PORTFOLIO_PATH = _ROOT / "data" / "revolut_portfolio.json"
_ANTHROPIC_KEY          = os.getenv("ANTHROPIC_API_KEY", "")
_HAS_CLAUDE             = bool(_ANTHROPIC_KEY)

_SIGNAL_COLOR = {
    "ADD":    "#00d26a",
    "HOLD":   "#60a5fa",
    "REDUCE": "#f5a623",
    "EXIT":   "#ff4d6d",
    "—":      "#888888",
}

_SIGNAL_ICON = {
    "ADD":    "➕",
    "HOLD":   "=",
    "REDUCE": "⬇️",
    "EXIT":   "🚫",
    "—":      "—",
}


def _load_revolut() -> Optional[Dict[str, Any]]:
    try:
        return json.loads(_REVOLUT_PORTFOLIO_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as exc:
        log.warning("revolut_portfolio.json: %s", exc)
        return None


def _load_regime() -> str:
    try:
        import yfinance as yf
        from regime_trader.models.regime_detector import vix_rule
        df = yf.download("^VIX", period="2d", interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            return "Unknown"
        vix = float(df["Close"].squeeze().dropna().iloc[-1])
        return vix_rule(vix)
    except Exception:
        return "Unknown"


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_live_prices(tickers: tuple[str, ...]) -> Dict[str, float]:
    if not tickers:
        return {}
    try:
        import yfinance as yf
        ticker_list = list(tickers)
        raw = yf.download(ticker_list, period="2d", interval="1d",
                          progress=False, auto_adjust=True,
                          group_by="ticker" if len(ticker_list) > 1 else None)
        prices: Dict[str, float] = {}
        for t in ticker_list:
            try:
                col = raw[t]["Close"] if len(ticker_list) > 1 else raw["Close"]
                prices[t] = float(col.dropna().iloc[-1])
            except Exception:
                pass
        return prices
    except Exception:
        return {}


def _get_claude_narrative(ticker: str, advice: PositionAdvice, regime: str) -> str:
    """Generate or retrieve cached Claude 2-sentence narrative."""
    cache_key = f"_advisor_narrative_{ticker}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    if not _HAS_CLAUDE:
        return ""

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=_ANTHROPIC_KEY)
        f = advice.factors
        prompt = (
            f"You are a quantitative analyst. In exactly 2 sentences, explain why "
            f"{ticker} signals {advice.signal} in the current {regime} regime. "
            f"Factor scores: Edgar={f.get('edgar',0):.2f}, Insider={f.get('insider',0):.2f}, "
            f"Congress={f.get('congress',0):.2f}, News={f.get('news',0):.2f}, "
            f"Macro={f.get('macro',0):.2f}. Overall score: {advice.final_score:.3f}. "
            f"Be specific about which factor drives the signal."
        )
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=120,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip()
        st.session_state[cache_key] = text
        return text
    except Exception as exc:
        log.warning("Claude narrative failed for %s: %s", ticker, exc)
        return ""


def _render_regime_banner(regime: str) -> None:
    _COLOR = {"Bull": "#00FFA3", "Neutral": "#60A5FA",
               "Bear": "#FFB347", "Panic": "#FF6B6B", "Crash": "#FF2222"}
    color = _COLOR.get(regime, "#9E9E9E")
    st.markdown(
        f'<div style="background:{color}18;border:1px solid {color};border-radius:8px;'
        f'padding:10px 20px;margin:8px 0;">'
        f'<span style="color:{color};font-size:1.1em;">Regime: <strong>{regime}</strong></span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render() -> None:
    import pandas as pd

    st.title("💼 Portfolio Advisor")
    st.caption("Daily buy/sell/hold signals on your Revolut positions. Scores from last pipeline run.")

    # ── Load data ──────────────────────────────────────────────────────────────
    revolut_data = _load_revolut()
    if revolut_data is None:
        st.info(
            "No Revolut portfolio found. Upload your statement in the **Portfolio Sync** tab "
            "(Dashboard → Portfolio Sync)."
        )
        return

    positions   = revolut_data.get("positions", [])
    imported_at = revolut_data.get("imported_at", "—")

    if not positions:
        st.warning("Imported portfolio has no positions.")
        return

    with st.spinner("Detecting regime…"):
        regime = _load_regime()

    _render_regime_banner(regime)
    st.caption(f"Revolut portfolio · Imported: **{imported_at}** · {len(positions)} positions")

    # ── Build advice ───────────────────────────────────────────────────────────
    with st.spinner("Scoring positions…"):
        advice_list = build_advice(positions, regime)

    # ── Fetch live prices ──────────────────────────────────────────────────────
    scored_tickers = tuple(a.ticker for a in advice_list if not a.not_in_universe)
    with st.spinner("Fetching live prices…"):
        prices = _fetch_live_prices(scored_tickers)

    # Attach market_value + build health score input
    for adv in advice_list:
        price = prices.get(adv.ticker)
        if price:
            adv.market_value = adv.net_qty * price

    health_positions = [
        {"ticker": a.ticker, "final_score": a.final_score or 0.0, "market_value": a.market_value}
        for a in advice_list if a.final_score is not None
    ]
    health_score = compute_health_score(health_positions)

    # ── Portfolio health summary ───────────────────────────────────────────────
    signal_counts = {s: sum(1 for a in advice_list if a.signal == s)
                     for s in ("ADD", "HOLD", "REDUCE", "EXIT")}

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Health Score", f"{health_score:.2f}",
              help="Weighted avg factor score by position value. ≥0.65 = healthy.")
    c2.metric("➕ ADD",    signal_counts.get("ADD",    0))
    c3.metric("= HOLD",   signal_counts.get("HOLD",   0))
    c4.metric("⬇️ REDUCE", signal_counts.get("REDUCE", 0))
    c5.metric("🚫 EXIT",  signal_counts.get("EXIT",   0))

    if not _HAS_CLAUDE:
        st.caption("💡 Set ANTHROPIC_API_KEY in .env to enable Claude narratives per position.")

    st.divider()

    # ── Position table ─────────────────────────────────────────────────────────
    sort_order = {"EXIT": 0, "REDUCE": 1, "ADD": 2, "HOLD": 3, "—": 4}
    advice_list.sort(key=lambda a: sort_order.get(a.signal, 9))

    for adv in advice_list:
        signal_color = _SIGNAL_COLOR.get(adv.signal, "#888")
        signal_icon  = _SIGNAL_ICON.get(adv.signal, "—")
        score_str    = f"{adv.final_score:.3f}" if adv.final_score is not None else "N/A"
        price        = prices.get(adv.ticker)
        unreal_pl    = adv.market_value - adv.net_qty * adv.avg_cost if adv.market_value else None
        unreal_pct   = (unreal_pl / (adv.net_qty * adv.avg_cost) * 100) \
                       if (unreal_pl is not None and adv.avg_cost > 0) else None

        age_str = f"Signal: {adv.signal_age_days}d old" if adv.signal_age_days is not None else ""
        age_warn = adv.signal_age_days is not None and adv.signal_age_days > 30

        header = (
            f"**{adv.ticker}**"
            + (f" *(Revolut: {adv.revolut_ticker})*" if adv.revolut_ticker != adv.ticker else "")
            + f"  ·  "
            + f"<span style='color:{signal_color};font-weight:700;'>{signal_icon} {adv.signal}</span>"
            + f"  ·  Score: **{score_str}**"
            + (f"  ·  {unreal_pl:+,.2f} ({unreal_pct:+.1f}%)" if unreal_pl is not None else "")
            + (f"  ·  ⚠️ {age_str}" if age_warn else (f"  ·  {age_str}" if age_str else ""))
            + ("  ·  *not in universe*" if adv.not_in_universe else "")
        )

        with st.expander(header, expanded=False):
            if adv.not_in_universe:
                st.caption("This ticker is not in the scoring universe — no factor data available.")
            else:
                # Factor bars
                f = adv.factors
                factor_data = [
                    {"Factor": "📋 Edgar",    "Weight": "30%", "Score": f"{f.get('edgar',0):.3f}",    "Bar": "█" * round(f.get('edgar',0)*10)    + "░" * (10-round(f.get('edgar',0)*10))},
                    {"Factor": "🏦 Insider",  "Weight": "25%", "Score": f"{f.get('insider',0):.3f}",  "Bar": "█" * round(f.get('insider',0)*10)  + "░" * (10-round(f.get('insider',0)*10))},
                    {"Factor": "🏛️ Congress", "Weight": "20%", "Score": f"{f.get('congress',0):.3f}", "Bar": "█" * round(f.get('congress',0)*10) + "░" * (10-round(f.get('congress',0)*10))},
                    {"Factor": "📰 News",     "Weight": "15%", "Score": f"{f.get('news',0):.3f}",     "Bar": "█" * round(f.get('news',0)*10)     + "░" * (10-round(f.get('news',0)*10))},
                    {"Factor": "📈 Macro",    "Weight": "10%", "Score": f"{f.get('macro',0):.3f}",    "Bar": "█" * round(f.get('macro',0)*10)    + "░" * (10-round(f.get('macro',0)*10))},
                ]
                st.dataframe(pd.DataFrame(factor_data), use_container_width=True, hide_index=True)

                # Claude narrative
                if _HAS_CLAUDE and adv.signal != "—":
                    with st.spinner(f"Generating narrative for {adv.ticker}…"):
                        narrative = _get_claude_narrative(adv.ticker, adv, regime)
                    if narrative:
                        st.markdown(f"> {narrative}")

                # Swap candidate
                if adv.swap_candidate:
                    swap = adv.swap_candidate
                    st.info(
                        f"🔄 **Consider rotating into {swap['ticker']}** "
                        f"(score {swap.get('final_score',0):.2f}, {swap.get('badge','')}, "
                        f"same sector)"
                    )

            # Raw position data
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Qty",       f"{adv.net_qty:.4f}")
            col_b.metric("Avg Cost",  f"{adv.avg_cost:.2f} {adv.currency}")
            col_c.metric("Live Price",f"{price:.2f}" if price else "—")
```

- [ ] **Step 2: Smoke test**

```bash
.venv/Scripts/python -c "
import sys; sys.path.insert(0, '.')
import importlib.util
spec = importlib.util.spec_from_file_location('portfolio_advisor', 'pages/7_Portfolio_Advisor.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
assert callable(mod.render)
print('OK')
"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add pages/7_Portfolio_Advisor.py
git commit -m "feat(pages): Portfolio Advisor — daily signals, health score, swap candidates, Claude narrative"
```

---

## Task 10: Run full test suite and verify

- [ ] **Step 1: Run all tests**

```bash
.venv/Scripts/pytest tests/ -v --tb=short -x
```
Expected: All tests pass (0 failures). New tests: `test_sector_picks`, `test_revolut_parser`, `test_portfolio_advisor_engine`.

- [ ] **Step 2: Start the dashboard and verify navigation**

```bash
.venv/Scripts/streamlit run streamlit_app.py
```

Check:
- Sidebar shows "── Alpha Engine ──" header with Stock Picker + Portfolio Advisor buttons
- Sidebar shows "── Quant Models ──" with all 5 quant pages
- Dashboard tab bar shows 4 tabs (Live Monitor, Market Intel, Macro Intel, Portfolio Sync) — no Trade Log or Regime History stubs
- Portfolio Sync tab accepts an `.xlsx` upload and shows preview
- Live Monitor shows Revolut block first (with "No portfolio imported yet" if not uploaded), Alpaca in collapsed expander
- Stock Picker page loads and shows sector expanders (may show "re-run pipeline" if `top_lists.json` has no `sector_picks` key yet)
- Portfolio Advisor page shows "No Revolut portfolio found" until Portfolio Sync is used

- [ ] **Step 3: Final commit**

```bash
git add .
git commit -m "chore: verify Alpha Engine dashboard — all pages, sidebar, tests passing"
```
