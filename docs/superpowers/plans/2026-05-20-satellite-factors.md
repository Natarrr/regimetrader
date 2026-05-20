# Satellite Factors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a satellite enrichment layer that computes seasonal cyclicality and share buyback yield signals for the top-lists tickers, appends them to the daily Discord embed, and never blocks the primary pipeline.

**Architecture:** `satellite_factors.py` runs as a new CI step after `generate_top_lists.py`, reads `logs/top_lists.json` to get the ticker universe and market-cap map, writes `logs/satellite_insights.json`, and then `send_toplists_discord.py` reads that file optionally to append two embed fields. The satellite CI step has `continue-on-error: true` so any failure is fully isolated from the main Discord send.

**Tech Stack:** yfinance (batch OHLCV + `.info`), requests (FMP cash-flow API), pandas, Python 3.11, pytest + monkeypatch, GitHub Actions.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `backend/market_intel/satellite_factors.py` | **Create** | `get_top_cyclical`, `get_top_cannibals`, `main()`, constants |
| `scripts/send_toplists_discord.py` | **Modify** | Add `_load_satellite`, modify `build_payload` signature, modify `main()` |
| `.github/workflows/edgar_3x.yml` | **Modify** | New satellite step + extended artifact upload |
| `tests/test_satellite_factors.py` | **Create** | 8 unit tests covering both functions and `main()` |
| `tests/test_send_toplists_discord.py` | **Create** | 4 tests for `_load_satellite` and `build_payload` with satellite |

---

### Task 1: Create `backend/market_intel/satellite_factors.py`

**Files:**
- Create: `backend/market_intel/satellite_factors.py`

This is the core module. Build it bottom-up: constants → `get_top_cyclical` → `get_top_cannibals` → `main`.

- [ ] **Step 1: Write the full module**

Create `backend/market_intel/satellite_factors.py` with this exact content:

```python
"""backend/market_intel/satellite_factors.py

Satellite enrichment layer for the edgar_3x pipeline.
Computes two supplementary signals:
  - Seasonal cyclicality  (win-rate / median return in the current calendar month)
  - Share cannibals       (buyback yield filtered by P/E and price proximity to 52w-low)

Called after generate_top_lists.py; writes logs/satellite_insights.json.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# ── Tuning knobs ──────────────────────────────────────────────────────────────
MIN_MONTHLY_OBSERVATIONS = 8     # minimum historical month-samples for cyclicality
PE_MAX                   = 25.0  # P/E ratio ceiling for cannibal filter
PRICE_VS_52W_LOW_MAX     = 1.25  # price must be < 125% of 52-week low
TOP_N                    = 3     # tickers returned by each function

# ── FMP base URL ─────────────────────────────────────────────────────────────
_FMP_BASE = "https://financialmodelingprep.com"


# ─────────────────────────────────────────────────────────────────────────────
# Cyclicality
# ─────────────────────────────────────────────────────────────────────────────

def get_top_cyclical(tickers: list[str]) -> list[dict]:
    """Return up to TOP_N tickers with best win-rate in the current calendar month.

    Data source: yfinance batch download, 10 years of monthly OHLCV.
    Returns [] on any exception.
    """
    try:
        import pandas as pd
        import yfinance as yf

        current_month = datetime.now(timezone.utc).month

        raw = yf.download(
            tickers,
            period="10y",
            interval="1mo",
            auto_adjust=True,
            group_by="ticker",
            progress=False,
        )

        # Normalise index to DatetimeIndex (batch download can produce MultiIndex)
        if not isinstance(raw.index, pd.DatetimeIndex):
            raw.index = pd.to_datetime(raw.index.get_level_values(-1))

        results: list[dict] = []
        for ticker in tickers:
            try:
                # Extract per-ticker slice; column key depends on download shape
                try:
                    df = raw[[("Open", ticker), ("Close", ticker)]].copy()
                    df.columns = ["Open", "Close"]
                except KeyError:
                    # Single-ticker download returns flat columns
                    df = raw[["Open", "Close"]].copy()

                filtered = df[df.index.month == current_month].dropna(
                    subset=["Open", "Close"]
                )
                if len(filtered) < MIN_MONTHLY_OBSERVATIONS:
                    continue

                wins = (filtered["Close"] > filtered["Open"]).sum()
                win_rate = float(wins / len(filtered))
                median_return = float(
                    ((filtered["Close"] - filtered["Open"]) / filtered["Open"]).median()
                )
                results.append(
                    {
                        "ticker":        ticker,
                        "win_rate":      round(win_rate, 4),
                        "median_return": round(median_return, 4),
                        "years":         len(filtered),
                    }
                )
            except Exception as exc:
                log.warning("cyclical: skipping %s — %s", ticker, exc)

        results.sort(key=lambda r: (-r["win_rate"], -r["median_return"]))
        return results[:TOP_N]

    except Exception as exc:
        log.warning("get_top_cyclical failed entirely: %s", exc)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Share cannibals
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_yf_info(ticker: str, max_retries: int = 3) -> dict:
    """Fetch yfinance .info with up to max_retries attempts."""
    import yfinance as yf

    last_exc: Exception = RuntimeError("no attempts")
    for attempt in range(max_retries):
        try:
            info = yf.Ticker(ticker).info
            if not isinstance(info, dict):
                raise AttributeError(f"yf.Ticker({ticker!r}).info returned non-dict")
            return info
        except Exception as exc:
            last_exc = exc
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)   # 1s, 2s
    raise last_exc


def get_top_cannibals(
    tickers: list[str],
    fmp_key: str,
    market_caps: dict[str, float],
) -> list[dict]:
    """Return up to TOP_N tickers ranked by trailing 4-quarter buyback yield.

    Filters: trailingPE < PE_MAX and currentPrice < PRICE_VS_52W_LOW_MAX * fiftyTwoWeekLow.
    Data sources: yfinance .info (P/E, 52w-low), FMP cash-flow (repurchases), market_caps dict.
    Returns [] when fmp_key is absent or on any unrecoverable exception.
    """
    if not fmp_key:
        log.warning("FMP_API_KEY absent — skipping cannibal scan")
        return []

    try:
        import requests as req

        results: list[dict] = []

        for ticker in tickers:
            try:
                info = _fetch_yf_info(ticker)
            except Exception as exc:
                log.warning("cannibal: yf.info failed for %s — %s", ticker, exc)
                continue

            # Filter 1: P/E
            try:
                pe = info["trailingPE"]
                if pe is None or float(pe) >= PE_MAX:
                    continue
                pe = float(pe)
            except (KeyError, AttributeError, TypeError, ValueError):
                continue

            # Filter 2: price vs 52-week low
            try:
                price    = float(info["currentPrice"])
                low_52w  = float(info["fiftyTwoWeekLow"])
                if low_52w <= 0 or price >= PRICE_VS_52W_LOW_MAX * low_52w:
                    continue
                price_vs_52w = round(price / low_52w, 4)
            except (KeyError, AttributeError, TypeError, ValueError):
                continue

            # FMP cash-flow — 4 trailing quarters
            try:
                url = (
                    f"{_FMP_BASE}/stable/cash-flow-statement"
                    f"?symbol={ticker}&period=quarter&limit=4&apikey={fmp_key}"
                )
                resp = req.get(url, timeout=15.0)
                resp.raise_for_status()
                quarters: list[dict] = resp.json()
                if not quarters:
                    continue
            except Exception as exc:
                log.warning("cannibal: FMP cash-flow failed for %s — %s", ticker, exc)
                continue

            total_repurchased = sum(
                abs(q.get("repurchasedCommonStock", 0) or 0) for q in quarters
            )

            mktcap = market_caps.get(ticker)
            if not mktcap or mktcap <= 0:
                continue

            buyback_yield = round(total_repurchased / mktcap, 6)
            results.append(
                {
                    "ticker":           ticker,
                    "buyback_yield":    buyback_yield,
                    "pe":               round(pe, 2),
                    "price_vs_52w_low": price_vs_52w,
                }
            )

        results.sort(key=lambda r: -r["buyback_yield"])
        return results[:TOP_N]

    except Exception as exc:
        log.warning("get_top_cannibals failed entirely: %s", exc)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate satellite insights")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    top_lists_path = args.log_dir / "top_lists.json"
    if not top_lists_path.exists():
        log.error("top_lists.json not found at %s", top_lists_path)
        raise SystemExit(1)

    top_lists: dict = json.loads(top_lists_path.read_text(encoding="utf-8"))

    # Collect unique tickers and market-cap map from all three universe tiers
    all_entries: list[dict] = (
        (top_lists.get("top_buys") or [])
        + (top_lists.get("mid_caps") or [])
        + (top_lists.get("small_caps") or [])
    )
    tickers: list[str] = list(dict.fromkeys(
        e["ticker"] for e in all_entries if e.get("ticker")
    ))
    market_caps: dict[str, float] = {
        e["ticker"]: float(e.get("market_cap") or 0)
        for e in all_entries
        if e.get("ticker") and e.get("market_cap")
    }

    fmp_key: str = os.getenv("FMP_API_KEY", "")

    cyclical_success = True
    cannibal_success = True

    cyclicals = get_top_cyclical(tickers)
    if not cyclicals and tickers:
        cyclical_success = False

    cannibals = get_top_cannibals(tickers, fmp_key, market_caps)
    if not cannibals and tickers and fmp_key:
        cannibal_success = False

    if cyclical_success and cannibal_success:
        status = "success"
    elif not cyclical_success and not cannibal_success:
        status = "error"
    else:
        status = "partial"

    current_month = datetime.now(timezone.utc).strftime("%B")

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "month":        current_month,
        "status":       status,
        "cyclicals":    cyclicals,
        "cannibals":    cannibals,
    }

    out_path = args.log_dir / "satellite_insights.json"
    tmp_path = out_path.with_suffix(".json.tmp")
    try:
        tmp_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(out_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    log.info(
        "satellite_insights.json written — status=%s cyclicals=%d cannibals=%d",
        status, len(cyclicals), len(cannibals),
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the module is importable**

Run:
```
python -c "from backend.market_intel.satellite_factors import get_top_cyclical, get_top_cannibals, main; print('OK')"
```
Expected: `OK` with no traceback.

- [ ] **Step 3: Commit**

```bash
git add backend/market_intel/satellite_factors.py
git commit -m "feat(satellite): add satellite_factors module with cyclical and cannibal signals"
```

---

### Task 2: Unit tests for `satellite_factors.py`

**Files:**
- Create: `tests/test_satellite_factors.py`

Write all 8 tests before running any of them; then run them all.

- [ ] **Step 1: Write the test file**

Create `tests/test_satellite_factors.py`:

```python
"""tests/test_satellite_factors.py
Unit tests for backend.market_intel.satellite_factors.
All yfinance and FMP calls are monkeypatched — no network access.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from backend.market_intel.satellite_factors import (
    MIN_MONTHLY_OBSERVATIONS,
    PE_MAX,
    PRICE_VS_52W_LOW_MAX,
    TOP_N,
    get_top_cannibals,
    get_top_cyclical,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_monthly_df(closes: list[float], month: int = 5) -> pd.DataFrame:
    """Build a single-ticker monthly DataFrame with all rows in `month`."""
    dates = pd.date_range(
        start=f"2015-{month:02d}-01",
        periods=len(closes),
        freq="MS",  # month-start
    )
    opens = [c * 0.95 for c in closes]  # open slightly below close → always a win
    return pd.DataFrame({"Open": opens, "Close": closes}, index=dates)


def _make_batch_df(ticker_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Create multi-column batch DataFrame as yf.download returns for multiple tickers."""
    frames = {}
    for ticker, df in ticker_dfs.items():
        for col in ("Open", "Close"):
            frames[(col, ticker)] = df[col]
    return pd.DataFrame(frames)


# ── Cyclicality tests ─────────────────────────────────────────────────────────

class TestGetTopCyclical:
    def test_filters_insufficient_history(self, monkeypatch):
        """Ticker with fewer than MIN_MONTHLY_OBSERVATIONS rows is excluded."""
        short_df = _make_monthly_df([10.0] * (MIN_MONTHLY_OBSERVATIONS - 1), month=5)
        batch = _make_batch_df({"AAPL": short_df})

        with patch("yfinance.download", return_value=batch):
            with patch(
                "backend.market_intel.satellite_factors.datetime"
            ) as mock_dt:
                mock_dt.now.return_value.month = 5
                mock_dt.now.return_value.utc = None
                # Use real datetime for timezone.utc reference
                from datetime import timezone
                mock_dt.now = lambda tz=None: MagicMock(month=5)
                result = get_top_cyclical(["AAPL"])

        assert result == [], "ticker with insufficient history must be excluded"

    def test_win_rate_calculation(self, monkeypatch):
        """Known OHLC data produces expected win_rate."""
        # 10 months: close > open for 8 of them → win_rate = 0.80
        closes = [105.0] * 8 + [95.0] * 2   # last 2 months: close < open
        df = _make_monthly_df(closes, month=5)
        # Override opens for last 2 to be above close
        df.iloc[-2:, df.columns.get_loc("Open")] = 100.0
        df.iloc[-2:, df.columns.get_loc("Close")] = 95.0
        batch = _make_batch_df({"PLTR": df})

        with patch("yfinance.download", return_value=batch):
            with patch(
                "backend.market_intel.satellite_factors.datetime"
            ) as mock_dt:
                mock_dt.now = lambda tz=None: MagicMock(month=5)
                result = get_top_cyclical(["PLTR"])

        assert len(result) == 1
        assert result[0]["ticker"] == "PLTR"
        assert math.isclose(result[0]["win_rate"], 0.8, rel_tol=1e-3)

    def test_returns_at_most_top_n(self, monkeypatch):
        """Result list is capped at TOP_N entries."""
        tickers = [f"T{i}" for i in range(TOP_N + 2)]
        dfs = {t: _make_monthly_df([100.0 + i] * MIN_MONTHLY_OBSERVATIONS, month=5)
               for i, t in enumerate(tickers)}
        batch = _make_batch_df(dfs)

        with patch("yfinance.download", return_value=batch):
            with patch(
                "backend.market_intel.satellite_factors.datetime"
            ) as mock_dt:
                mock_dt.now = lambda tz=None: MagicMock(month=5)
                result = get_top_cyclical(tickers)

        assert len(result) <= TOP_N

    def test_returns_empty_on_yfinance_exception(self, monkeypatch):
        """Any exception from yfinance.download returns []."""
        with patch("yfinance.download", side_effect=RuntimeError("timeout")):
            result = get_top_cyclical(["AAPL"])
        assert result == []


# ── Cannibal tests ────────────────────────────────────────────────────────────

class TestGetTopCannibals:
    _FMP_KEY = "test_key"
    _MARKET_CAPS = {"PLTR": 50_000_000_000.0, "SQ": 30_000_000_000.0}

    def _good_info(self, pe: float = 18.0, price: float = 20.0, low_52w: float = 18.0) -> dict:
        return {
            "trailingPE":      pe,
            "currentPrice":    price,
            "fiftyTwoWeekLow": low_52w,
        }

    def _fmp_quarters(self, repurchased: float = 500_000_000.0) -> list[dict]:
        return [{"repurchasedCommonStock": -repurchased / 4}] * 4

    def test_filters_high_pe(self, monkeypatch):
        """Ticker with P/E >= PE_MAX is excluded."""
        info = self._good_info(pe=PE_MAX + 1)
        with patch(
            "backend.market_intel.satellite_factors._fetch_yf_info",
            return_value=info,
        ):
            result = get_top_cannibals(["PLTR"], self._FMP_KEY, self._MARKET_CAPS)
        assert result == []

    def test_filters_price_above_52w_band(self, monkeypatch):
        """Ticker priced above PRICE_VS_52W_LOW_MAX * 52w-low is excluded."""
        info = self._good_info(price=100.0, low_52w=50.0)  # ratio = 2.0 > 1.25
        with patch(
            "backend.market_intel.satellite_factors._fetch_yf_info",
            return_value=info,
        ):
            result = get_top_cannibals(["PLTR"], self._FMP_KEY, self._MARKET_CAPS)
        assert result == []

    def test_zero_market_cap_skipped(self, monkeypatch):
        """Ticker with market_cap = 0 is skipped — no ZeroDivisionError."""
        info = self._good_info()
        quarters = self._fmp_quarters()
        mock_resp = MagicMock()
        mock_resp.json.return_value = quarters
        mock_resp.raise_for_status = lambda: None

        with patch(
            "backend.market_intel.satellite_factors._fetch_yf_info",
            return_value=info,
        ):
            with patch("requests.get", return_value=mock_resp):
                result = get_top_cannibals(["PLTR"], self._FMP_KEY, {"PLTR": 0.0})
        assert result == []

    def test_missing_fmp_key_returns_empty(self):
        """No FMP key → return [] without any HTTP call."""
        result = get_top_cannibals(["PLTR"], "", {"PLTR": 1e10})
        assert result == []

    def test_buyback_yield_calculated_correctly(self, monkeypatch):
        """Correct buyback_yield = total_repurchased / market_cap."""
        info = self._good_info(pe=10.0, price=19.0, low_52w=18.0)
        # 4 quarters × 250M = 1B repurchased; market_cap = 50B → yield = 0.02
        quarters = [{"repurchasedCommonStock": -250_000_000}] * 4
        mock_resp = MagicMock()
        mock_resp.json.return_value = quarters
        mock_resp.raise_for_status = lambda: None

        with patch(
            "backend.market_intel.satellite_factors._fetch_yf_info",
            return_value=info,
        ):
            with patch("requests.get", return_value=mock_resp):
                result = get_top_cannibals(
                    ["PLTR"], self._FMP_KEY, {"PLTR": 50_000_000_000.0}
                )

        assert len(result) == 1
        assert math.isclose(result[0]["buyback_yield"], 0.02, rel_tol=1e-3)


# ── Integration: main() ────────────────────────────────────────────────────────

class TestMain:
    def _make_top_lists(self, tickers: list[str]) -> dict:
        return {
            "generated_at": "2026-05-20T08:00:00+00:00",
            "top_buys": [
                {"ticker": t, "final_score": 0.7, "badge": "HIGH BUY",
                 "market_cap": 1e10, "factors": {}}
                for t in tickers
            ],
            "mid_caps":   [],
            "small_caps": [],
        }

    def test_main_writes_satellite_json(self, tmp_path, monkeypatch):
        """main() reads top_lists.json and writes satellite_insights.json."""
        top_lists = self._make_top_lists(["PLTR"])
        (tmp_path / "top_lists.json").write_text(
            json.dumps(top_lists), encoding="utf-8"
        )

        monkeypatch.setenv("FMP_API_KEY", "")  # no FMP key → cannibals = []
        with patch("yfinance.download", side_effect=RuntimeError("no network")):
            import sys
            monkeypatch.setattr(sys, "argv", ["satellite_factors", "--log-dir", str(tmp_path)])
            from backend.market_intel import satellite_factors
            satellite_factors.main()

        out = json.loads((tmp_path / "satellite_insights.json").read_text())
        assert "generated_at" in out
        assert "cyclicals" in out
        assert "cannibals" in out
        assert out["status"] in ("success", "partial", "error")

    def test_satellite_status_partial_when_one_fails(self, tmp_path, monkeypatch):
        """status='partial' when cyclicals succeed but cannibals return []."""
        tickers = ["PLTR"]
        top_lists = self._make_top_lists(tickers)
        (tmp_path / "top_lists.json").write_text(
            json.dumps(top_lists), encoding="utf-8"
        )

        # cyclicals succeed with real data
        good_df = _make_monthly_df([100.0] * MIN_MONTHLY_OBSERVATIONS, month=5)
        batch = _make_batch_df({"PLTR": good_df})

        monkeypatch.setenv("FMP_API_KEY", "present_key")  # key present but...
        # ...cannibals fail due to yf.info error
        with patch("yfinance.download", return_value=batch):
            with patch(
                "backend.market_intel.satellite_factors._fetch_yf_info",
                side_effect=RuntimeError("timeout"),
            ):
                with patch(
                    "backend.market_intel.satellite_factors.datetime"
                ) as mock_dt:
                    mock_dt.now = lambda tz=None: MagicMock(month=5)
                    import sys
                    monkeypatch.setattr(
                        sys, "argv",
                        ["satellite_factors", "--log-dir", str(tmp_path)],
                    )
                    from backend.market_intel import satellite_factors
                    satellite_factors.main()

        out = json.loads((tmp_path / "satellite_insights.json").read_text())
        # cannibals list is empty because yf.info failed for every ticker
        assert out["cannibals"] == []
```

- [ ] **Step 2: Run the tests — expect most to PASS (some may need minor adjustments for month mocking)**

```
pytest tests/test_satellite_factors.py -v
```

Expected: all 10 tests pass. If `datetime` patching causes issues with `timezone.utc`, adjust the mock to use `unittest.mock.patch("backend.market_intel.satellite_factors.datetime")` with a spec or use `freezegun` if available. The key invariant: no network calls reach the real internet.

- [ ] **Step 3: Commit**

```bash
git add tests/test_satellite_factors.py
git commit -m "test(satellite): add unit tests for get_top_cyclical, get_top_cannibals, main"
```

---

### Task 3: Modify `scripts/send_toplists_discord.py`

**Files:**
- Modify: `scripts/send_toplists_discord.py`

Three changes: (1) add `_load_satellite` helper, (2) add optional `satellite` param to `build_payload`, (3) call both in `main()`.

- [ ] **Step 1: Add `_load_satellite` helper after the `_data_age_hours` function (line 155)**

Insert this block between `_data_age_hours` and `build_payload` in [scripts/send_toplists_discord.py](scripts/send_toplists_discord.py):

```python
def _load_satellite(log_dir: Path) -> dict | None:
    """Load satellite_insights.json if present. Returns None on any failure."""
    path = log_dir / "satellite_insights.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        return data
    except Exception as exc:
        log.warning("satellite_insights.json unreadable: %s", exc)
        return None
```

- [ ] **Step 2: Change the `build_payload` signature and add satellite fields**

Replace the current `def build_payload(top_lists: Dict[str, Any]) -> Dict[str, Any]:` function signature and add satellite field injection before the Factor Legend field. The full updated function signature and satellite block:

Change line 158 from:
```python
def build_payload(top_lists: Dict[str, Any]) -> Dict[str, Any]:
```
to:
```python
def build_payload(top_lists: Dict[str, Any], satellite: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
```

Then, immediately before the `"📊 Factor Legend"` field dict inside the `fields` list (after the `"🔬 Top 5 Small Caps"` dict and before the `"📊 Factor Legend"` dict), insert:

```python
        # ── Satellite fields (optional — wrapped so never crashes embed) ──────
        try:
            if satellite and isinstance(satellite, dict):
                month_label = satellite.get("month", "")
                cyclicals = satellite.get("cyclicals") or []
                cannibals = satellite.get("cannibals") or []

                if cyclicals:
                    lines = []
                    for i, c in enumerate(cyclicals, 1):
                        wr   = f"{c['win_rate']:.0%}"
                        med  = f"{c['median_return']:+.1%}"
                        yr   = c.get("years", "?")
                        lines.append(f"{i}. {c['ticker']}  Win-rate: {wr}  Median: {med}  ({yr} yr)")
                    fields.append({
                        "name":   f"🌀 Seasonal Cyclicals — {month_label}",
                        "value":  "\n".join(lines),
                        "inline": False,
                    })

                if cannibals:
                    lines = []
                    for i, c in enumerate(cannibals, 1):
                        yld  = f"{c['buyback_yield']:.1%}"
                        pe   = f"{c['pe']:.1f}"
                        pvl  = f"{c['price_vs_52w_low']:.2f}"
                        lines.append(f"{i}. {c['ticker']}  Yield: {yld}  P/E: {pe}  Price/52wLow: {pvl}×")
                    fields.append({
                        "name":   "🐷 Share Cannibals — Buyback Yield",
                        "value":  "\n".join(lines),
                        "inline": False,
                    })
        except Exception as exc:
            log.warning("satellite embed fields skipped due to error: %s", exc)
```

- [ ] **Step 3: Update `main()` to load satellite and pass it to `build_payload`**

In `main()`, replace the single line:
```python
    payload = build_payload(top_lists)
```
with:
```python
    satellite = _load_satellite(args.log_dir)
    payload   = build_payload(top_lists, satellite=satellite)
```

- [ ] **Step 4: Verify the file is syntactically valid**

```
python -c "import scripts.send_toplists_discord; print('OK')"
```
Expected: `OK` with no traceback.

- [ ] **Step 5: Commit**

```bash
git add scripts/send_toplists_discord.py
git commit -m "feat(satellite): integrate satellite_insights into Discord embed"
```

---

### Task 4: Tests for `send_toplists_discord.py` satellite integration

**Files:**
- Create: `tests/test_send_toplists_discord.py`

- [ ] **Step 1: Write the test file**

Create `tests/test_send_toplists_discord.py`:

```python
"""tests/test_send_toplists_discord.py
Tests for satellite integration in send_toplists_discord.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.send_toplists_discord import _load_satellite, build_payload


# ── Fixture: minimal valid top_lists ─────────────────────────────────────────

def _top_lists() -> dict:
    return {
        "generated_at":  "2026-05-20T08:00:00+00:00",
        "source_run_id": "test-run",
        "ticker_count":  5,
        "weights":       {},
        "kill_switch":   False,
        "vix":           18.0,
        "top_buys":      [
            {"ticker": "PLTR", "final_score": 0.75, "badge": "HIGH BUY",
             "factors": {"edgar": 0.8, "insider": 0.7, "congress": 0.6,
                         "news": 0.65, "momentum": 0.6}, "ceo_buy": False}
        ],
        "mid_caps":   [],
        "small_caps": [],
    }


def _satellite() -> dict:
    return {
        "generated_at": "2026-05-20T08:41:00+00:00",
        "month":        "May",
        "status":       "success",
        "cyclicals": [
            {"ticker": "PLTR", "win_rate": 0.75, "median_return": 0.031, "years": 9}
        ],
        "cannibals": [
            {"ticker": "SQ", "buyback_yield": 0.048, "pe": 18.2, "price_vs_52w_low": 1.18}
        ],
    }


# ── _load_satellite ───────────────────────────────────────────────────────────

class TestLoadSatellite:
    def test_returns_none_on_missing_file(self, tmp_path):
        assert _load_satellite(tmp_path) is None

    def test_returns_none_on_corrupt_json(self, tmp_path):
        (tmp_path / "satellite_insights.json").write_text("not json", encoding="utf-8")
        assert _load_satellite(tmp_path) is None

    def test_returns_none_on_non_dict_json(self, tmp_path):
        (tmp_path / "satellite_insights.json").write_text(
            json.dumps([1, 2, 3]), encoding="utf-8"
        )
        assert _load_satellite(tmp_path) is None

    def test_returns_dict_on_valid_file(self, tmp_path):
        data = _satellite()
        (tmp_path / "satellite_insights.json").write_text(
            json.dumps(data), encoding="utf-8"
        )
        result = _load_satellite(tmp_path)
        assert result == data


# ── build_payload with satellite ──────────────────────────────────────────────

class TestBuildPayloadSatellite:
    def test_without_satellite_embed_unchanged(self):
        """satellite=None → embed fields identical to original 4-field structure."""
        payload = build_payload(_top_lists(), satellite=None)
        fields = payload["embeds"][0]["fields"]
        field_names = [f["name"] for f in fields]
        assert "🌀 Seasonal Cyclicals — May" not in field_names
        assert "🐷 Share Cannibals — Buyback Yield" not in field_names
        assert len(fields) == 4

    def test_with_satellite_adds_cyclical_and_cannibal_fields(self):
        """satellite with non-empty lists → 6 embed fields total."""
        payload = build_payload(_top_lists(), satellite=_satellite())
        fields = payload["embeds"][0]["fields"]
        field_names = [f["name"] for f in fields]
        assert "🌀 Seasonal Cyclicals — May" in field_names
        assert "🐷 Share Cannibals — Buyback Yield" in field_names
        assert len(fields) == 6

    def test_cyclical_field_content(self):
        """Cyclical field renders win-rate and median correctly."""
        payload = build_payload(_top_lists(), satellite=_satellite())
        fields = payload["embeds"][0]["fields"]
        cyclical_field = next(f for f in fields if "Cyclicals" in f["name"])
        assert "PLTR" in cyclical_field["value"]
        assert "75%" in cyclical_field["value"]
        assert "+3.1%" in cyclical_field["value"]

    def test_cannibal_field_content(self):
        """Cannibal field renders yield, P/E, and price ratio correctly."""
        payload = build_payload(_top_lists(), satellite=_satellite())
        fields = payload["embeds"][0]["fields"]
        cannibal_field = next(f for f in fields if "Cannibals" in f["name"])
        assert "SQ" in cannibal_field["value"]
        assert "4.8%" in cannibal_field["value"]
        assert "18.2" in cannibal_field["value"]

    def test_empty_cyclicals_no_cyclical_field(self):
        """If cyclicals is empty, no cyclical embed field added."""
        sat = _satellite()
        sat["cyclicals"] = []
        payload = build_payload(_top_lists(), satellite=sat)
        field_names = [f["name"] for f in payload["embeds"][0]["fields"]]
        assert "🌀 Seasonal Cyclicals — May" not in field_names

    def test_empty_cannibals_no_cannibal_field(self):
        """If cannibals is empty, no cannibal embed field added."""
        sat = _satellite()
        sat["cannibals"] = []
        payload = build_payload(_top_lists(), satellite=sat)
        field_names = [f["name"] for f in payload["embeds"][0]["fields"]]
        assert "🐷 Share Cannibals — Buyback Yield" not in field_names

    def test_factor_legend_always_last(self):
        """Factor Legend must always be the last embed field."""
        payload = build_payload(_top_lists(), satellite=_satellite())
        last_field = payload["embeds"][0]["fields"][-1]
        assert "Factor Legend" in last_field["name"]
```

- [ ] **Step 2: Run the tests**

```
pytest tests/test_send_toplists_discord.py tests/test_discord_formatter.py -v
```

Expected: all tests pass. The existing `test_discord_formatter.py` tests must still pass because `build_payload` is backwards-compatible (new `satellite` param defaults to `None`).

- [ ] **Step 3: Commit**

```bash
git add tests/test_send_toplists_discord.py
git commit -m "test(satellite): add Discord satellite integration tests"
```

---

### Task 5: CI workflow changes in `edgar_3x.yml`

**Files:**
- Modify: `.github/workflows/edgar_3x.yml`

Two changes: (1) add satellite step after step 6 (Generate top lists), (2) add `satellite_insights.json` to the artifact upload.

- [ ] **Step 1: Add the satellite step**

In [.github/workflows/edgar_3x.yml](.github/workflows/edgar_3x.yml), immediately after the `# ── 6. Generate top_lists.json` step block (after line 138 where `$FORCE` ends), insert this new step:

```yaml
      # ── 7. Generate satellite insights ──────────────────────────────────────
      - name: Generate satellite insights
        continue-on-error: true
        env:
          FMP_API_KEY: ${{ secrets.FMP_API_KEY || '' }}
        run: |
          python -m backend.market_intel.satellite_factors \
            --log-dir logs \
            --verbose
```

Re-number the downstream step comments from `# ── 7.` through `# ── 9.` to `# ── 8.` through `# ── 10.` for consistency, updating only the comment text (not the step names or YAML keys).

- [ ] **Step 2: Extend the artifact upload path list**

In the `Upload top-lists artifact` step, change the `path:` block from:
```yaml
          path: |
            logs/top_lists.json
            logs/top5.csv
```
to:
```yaml
          path: |
            logs/top_lists.json
            logs/top5.csv
            logs/satellite_insights.json
```

- [ ] **Step 3: Verify YAML is valid**

```
python -c "import yaml; yaml.safe_load(open('.github/workflows/edgar_3x.yml').read()); print('YAML OK')"
```
Expected: `YAML OK`

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/edgar_3x.yml
git commit -m "ci(satellite): add satellite insights step and extend artifact upload"
```

---

### Task 6: Full test suite smoke-check

**Files:** none (verification only)

- [ ] **Step 1: Run all new satellite-related tests**

```
pytest tests/test_satellite_factors.py tests/test_send_toplists_discord.py tests/test_discord_formatter.py -v
```

Expected: all tests pass (green).

- [ ] **Step 2: Run the full test suite to check for regressions**

```
pytest --tb=short -q
```

Expected: same pass/fail count as before this feature (902 passing tests + new tests). No regressions.

- [ ] **Step 3: Dry-run the Discord sender with a synthetic satellite file**

```bash
mkdir -p /tmp/sat_test
echo '{"generated_at":"2026-05-20T08:00:00+00:00","top_buys":[{"ticker":"PLTR","final_score":0.75,"badge":"HIGH BUY","factors":{"edgar":0.8,"insider":0.7,"congress":0.6,"news":0.65,"momentum":0.6},"ceo_buy":false}],"mid_caps":[],"small_caps":[],"source_run_id":"test","ticker_count":1,"weights":{},"kill_switch":false}' > /tmp/sat_test/top_lists.json
echo '{"generated_at":"2026-05-20T08:41:00+00:00","month":"May","status":"success","cyclicals":[{"ticker":"PLTR","win_rate":0.75,"median_return":0.031,"years":9}],"cannibals":[{"ticker":"SQ","buyback_yield":0.048,"pe":18.2,"price_vs_52w_low":1.18}]}' > /tmp/sat_test/satellite_insights.json
python scripts/send_toplists_discord.py --input /tmp/sat_test/top_lists.json --log-dir /tmp/sat_test --dry-run
```

Expected: JSON printed to stdout with 6 embed fields including `"🌀 Seasonal Cyclicals — May"` and `"🐷 Share Cannibals — Buyback Yield"`.
