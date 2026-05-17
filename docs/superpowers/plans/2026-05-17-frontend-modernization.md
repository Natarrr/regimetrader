# Frontend Modernization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all stale `macro`→`momentum` field references across the frontend and Discord alert, add 4 pass-through evidence fields to the backend pipeline, and surface per-ticker evidence cards in Stock Picker and Portfolio Advisor.

**Architecture:** Six targeted file modifications — two backend pass-throughs (no scoring changes), four frontend fixes plus UI evidence sections. All changes are additive or direct substitutions; no existing test suite logic changes.

**Tech Stack:** Python 3.11, Streamlit, pandas, pytest, anthropic SDK (already installed)

---

## File Map

| File | Action | What changes |
|------|--------|--------------|
| `scripts/run_pipeline.py` | Modify | Add `news_source`, `insider_usd`, `momentum_spy_relative`, `volume_spike` to `_score_ticker()` return dict |
| `backend/market_intel/generate_top_lists.py` | Modify | Pass the 4 new evidence fields through `_to_entry()` |
| `regime_trader/ui/portfolio_advisor_engine.py` | Modify | Fix weights (28/23/22/15/12); rename `"macro"`→`"momentum"` key |
| `pages/7_Portfolio_Advisor.py` | Modify | Fix labels + weights in factor table; fix Claude prompt; add evidence section |
| `pages/6_Stock_Picker.py` | Modify | Rename `"Macro"` column → `"Momentum"`; add per-ticker evidence expanders |
| `scripts/send_toplists_discord.py` | Modify | Fix `_FACTOR_EMOJI` key + `_format_factor_line` loop: `"macro"`→`"momentum"` |
| `tests/test_portfolio_advisor_engine.py` | Modify | Add assertions for correct weights and `"momentum"` key |
| `tests/test_discord_formatter.py` | Create | Test `_format_factor_line` reads `"momentum"` key correctly |

---

### Task 1: Add evidence pass-through fields to `_score_ticker()`

**Context:** `scripts/run_pipeline.py` contains a nested function `_score_ticker()` at line 752. It already computes `price_data = fetch_price_data(ticker)` (which returns `return_20d`, `spy_return_20d`, `volume_spike`) and `total_purchases_usd`. We need to expose 4 values in the return dict so they flow through to `intel_source_status.json`. The `news_source` field is determined by whether `FINNHUB_API_KEY` was set.

**Files:**
- Modify: `scripts/run_pipeline.py:801–816` (the `return {...}` dict inside `_score_ticker`)

- [ ] **Step 1: Write the failing test**

Add this test class to `tests/test_scoring_signals.py` (class `TestQuiverEvidenceInResults` already exists — add the new class after it):

```python
class TestEvidencePassthroughFields:
    """_score_ticker() must include the 4 evidence pass-through fields."""

    def test_score_ticker_result_contains_evidence_fields(self):
        from scripts.run_pipeline import run
        import tempfile, csv, json
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            tickers_file = tdp / "tickers.csv"
            log_dir = tdp / "logs"
            log_dir.mkdir()
            with tickers_file.open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["ticker", "sector", "cap_tier"])
                w.writeheader()
                w.writerow({"ticker": "AAPL", "sector": "Tech", "cap_tier": "large"})

            import pandas as pd, numpy as np
            dates = pd.date_range("2026-04-01", periods=30, freq="B")
            fake_df = pd.DataFrame({
                "Close":  pd.Series(np.linspace(100, 110, 30), index=dates),
                "Volume": pd.Series([1_000_000] * 25 + [3_000_000] * 5, index=dates),
            })
            fake_spy = pd.DataFrame({
                "Close": pd.Series(np.linspace(500, 510, 30), index=dates)
            })

            with patch("yfinance.download", side_effect=lambda sym, **kw: fake_spy if sym == "SPY" else fake_df), \
                 patch("yfinance.Ticker") as mock_ticker, \
                 patch("scripts.run_pipeline._sec_get", side_effect=Exception("no SEC")), \
                 patch("scripts.run_pipeline.fetch_fmp_profiles", return_value={"AAPL": 3e12}), \
                 patch("scripts.run_pipeline.fetch_congress_buys", return_value={}), \
                 patch("scripts.run_pipeline.score_news_finnhub", return_value=0.55):
                mock_ticker.return_value.news = []
                status = run(tickers_file, log_dir, max_workers=1)

            r = status["results"][0]
            assert "news_source" in r,           "news_source missing"
            assert "insider_usd" in r,           "insider_usd missing"
            assert "momentum_spy_relative" in r, "momentum_spy_relative missing"
            assert "volume_spike" in r,          "volume_spike missing"
            assert r["news_source"] in ("finnhub", "yfinance", "none")
            assert isinstance(r["insider_usd"], float)
            assert isinstance(r["momentum_spy_relative"], float)
            assert isinstance(r["volume_spike"], float)
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_scoring_signals.py::TestEvidencePassthroughFields -v
```

Expected: FAIL — `KeyError: 'news_source'` or `AssertionError`

- [ ] **Step 3: Add the 4 fields to the `_score_ticker()` return dict**

In `scripts/run_pipeline.py`, find the success `return` dict inside `_score_ticker()` (around line 801). The current return ends with:

```python
            return {
                "ticker":          ticker,
                "sector":          sector.get(ticker, "Unknown"),
                "cap_tier":        cap_tier.get(ticker, "large"),
                "market_cap":      mktcap,
                "edgar_score":     e_score,
                "insider_score":   i_score,
                "congress_score":  c_score,
                "news_score":      n_score,
                "momentum_score":  m_score,
                "ceo_buy":         ceo_buy,
                "form4_count":     form4_count,
                "quiver_evidence": quiver_evidence,
                "_edgar_ok":       edgar_ok,
                "_scoring_error":  False,
            }
```

Replace with:

```python
            news_source = "none"
            if finnhub_key:
                news_source = "finnhub" if n_score > 0.0 else "none"
            else:
                news_source = "yfinance" if n_score > 0.0 else "none"

            return {
                "ticker":                  ticker,
                "sector":                  sector.get(ticker, "Unknown"),
                "cap_tier":                cap_tier.get(ticker, "large"),
                "market_cap":              mktcap,
                "edgar_score":             e_score,
                "insider_score":           i_score,
                "congress_score":          c_score,
                "news_score":              n_score,
                "momentum_score":          m_score,
                "ceo_buy":                 ceo_buy,
                "form4_count":             form4_count,
                "quiver_evidence":         quiver_evidence,
                "news_source":             news_source,
                "insider_usd":             float(total_purchases_usd),
                "momentum_spy_relative":   float(price_data["return_20d"] - price_data["spy_return_20d"]),
                "volume_spike":            float(price_data["volume_spike"]),
                "_edgar_ok":               edgar_ok,
                "_scoring_error":          False,
            }
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/test_scoring_signals.py::TestEvidencePassthroughFields -v
```

Expected: PASS

- [ ] **Step 5: Run full test suite**

```
pytest tests/ -q --tb=short
```

Expected: all green

- [ ] **Step 6: Commit**

```
git add scripts/run_pipeline.py tests/test_scoring_signals.py
git commit -m "feat(pipeline): add evidence pass-through fields to _score_ticker()"
```

---

### Task 2: Pass evidence fields through `_to_entry()` in `generate_top_lists.py`

**Context:** `backend/market_intel/generate_top_lists.py` `_to_entry()` at line 202 builds the entry dict that ends up in `top_lists.json`. The 4 new fields arrive in `row` (the raw result dict from `intel_source_status.json`). We add them as pass-through `.get()` reads so Stock Picker and Discord can use them without recomputing anything.

**Files:**
- Modify: `backend/market_intel/generate_top_lists.py:216–228` (`_to_entry` return dict)

- [ ] **Step 1: Write the failing test**

Add this class to `tests/test_cross_sectional.py` (after the existing classes):

```python
class TestToEntryEvidencePassthrough:
    def test_evidence_fields_present_in_entry(self):
        from backend.market_intel.generate_top_lists import _to_entry

        row = {
            "ticker": "AAPL", "sector": "Tech", "cap_tier": "large",
            "market_cap": 3e12, "ceo_buy": True, "form4_count": 3,
            "news_source": "finnhub",
            "insider_usd": 2_500_000.0,
            "momentum_spy_relative": 0.042,
            "volume_spike": 2.3,
        }
        norm = {"edgar": 0.8, "insider": 0.7, "congress": 0.6, "news": 0.5, "momentum": 0.4}
        entry = _to_entry(row, norm)

        assert entry["news_source"] == "finnhub"
        assert entry["insider_usd"] == pytest.approx(2_500_000.0)
        assert entry["momentum_spy_relative"] == pytest.approx(0.042)
        assert entry["volume_spike"] == pytest.approx(2.3)

    def test_evidence_fields_default_when_absent(self):
        from backend.market_intel.generate_top_lists import _to_entry

        row = {"ticker": "X", "sector": "?", "cap_tier": "large", "market_cap": 0}
        norm = {"edgar": 0.5, "insider": 0.5, "congress": 0.5, "news": 0.5, "momentum": 0.5}
        entry = _to_entry(row, norm)

        assert entry["news_source"] == "none"
        assert entry["insider_usd"] == pytest.approx(0.0)
        assert entry["momentum_spy_relative"] == pytest.approx(0.0)
        assert entry["volume_spike"] == pytest.approx(1.0)
```

Note: `tests/test_cross_sectional.py` already imports `pytest`.

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_cross_sectional.py::TestToEntryEvidencePassthrough -v
```

Expected: FAIL — `KeyError` or `AssertionError`

- [ ] **Step 3: Add the 4 pass-through fields to `_to_entry()` return dict**

In `backend/market_intel/generate_top_lists.py`, find the `_to_entry` return dict (around line 216):

```python
    return {
        "ticker":          row.get("ticker", "?"),
        "sector":          row.get("sector", "Unknown"),
        "cap_tier":        row.get("cap_tier", "large"),
        "market_cap":      float(row.get("market_cap", 0.0)),
        "raw_score":       raw_score,
        "final_score":     score,
        "badge":           _badge(score),
        "ceo_buy":         bool(row.get("ceo_buy", False)),
        "form4_count":     int(row.get("form4_count", 0)),
        "factors":         norm_factors,
        "quiver_evidence": quiver_evidence or {},
    }
```

Replace with:

```python
    return {
        "ticker":                  row.get("ticker", "?"),
        "sector":                  row.get("sector", "Unknown"),
        "cap_tier":                row.get("cap_tier", "large"),
        "market_cap":              float(row.get("market_cap", 0.0)),
        "raw_score":               raw_score,
        "final_score":             score,
        "badge":                   _badge(score),
        "ceo_buy":                 bool(row.get("ceo_buy", False)),
        "form4_count":             int(row.get("form4_count", 0)),
        "factors":                 norm_factors,
        "quiver_evidence":         quiver_evidence or {},
        "news_source":             row.get("news_source", "none"),
        "insider_usd":             float(row.get("insider_usd", 0.0)),
        "momentum_spy_relative":   float(row.get("momentum_spy_relative", 0.0)),
        "volume_spike":            float(row.get("volume_spike", 1.0)),
    }
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/test_cross_sectional.py::TestToEntryEvidencePassthrough -v
```

Expected: PASS

- [ ] **Step 5: Run full test suite**

```
pytest tests/ -q --tb=short
```

Expected: all green

- [ ] **Step 6: Commit**

```
git add backend/market_intel/generate_top_lists.py tests/test_cross_sectional.py
git commit -m "feat(generate_top_lists): pass evidence fields through _to_entry()"
```

---

### Task 3: Fix `portfolio_advisor_engine.py` — weights and `"macro"` key

**Context:** `regime_trader/ui/portfolio_advisor_engine.py` `build_advice()` has stale weights (0.30/0.25/0.20/0.15/0.10) and outputs `"macro"` as the momentum factor key. Both must be fixed: weights → 0.28/0.23/0.22/0.15/0.12, key → `"momentum"`.

**Files:**
- Modify: `regime_trader/ui/portfolio_advisor_engine.py:165–191`
- Modify: `tests/test_portfolio_advisor_engine.py` (add weight + key assertions)

- [ ] **Step 1: Write the failing tests**

Add this class to `tests/test_portfolio_advisor_engine.py`:

```python
class TestBuildAdviceWeightsAndKeys:
    """build_advice() must use 28/23/22/15/12 weights and 'momentum' key."""

    def _make_status(self) -> dict:
        return {
            "computed_at": "2026-05-17T10:00:00Z",
            "results": [{
                "ticker":         "AAPL",
                "sector":         "Information Technology",
                "cap_tier":       "large",
                "edgar_score":    1.0,
                "insider_score":  0.0,
                "congress_score": 0.0,
                "news_score":     0.0,
                "momentum_score": 0.0,
            }],
        }

    def test_final_score_uses_correct_weights(self):
        import json, tempfile
        from pathlib import Path
        from unittest.mock import patch

        status = self._make_status()
        with tempfile.TemporaryDirectory() as td:
            status_path = Path(td) / "intel_source_status.json"
            status_path.write_text(json.dumps(status))
            top_lists_path = Path(td) / "top_lists.json"
            top_lists_path.write_text(json.dumps({}))

            with patch("regime_trader.ui.portfolio_advisor_engine._STATUS_PATH", status_path), \
                 patch("regime_trader.ui.portfolio_advisor_engine._TOP_LISTS_PATH", top_lists_path):
                from regime_trader.ui.portfolio_advisor_engine import build_advice
                result = build_advice(
                    [{"ticker": "AAPL", "net_qty": 10, "avg_cost": 150.0}],
                    regime="Bull",
                )

        assert len(result) == 1
        adv = result[0]
        # edgar_score=1.0, all others=0 → final_score should be 0.28 (not 0.30)
        assert adv.final_score == pytest.approx(0.28, abs=1e-4)

    def test_factors_dict_has_momentum_not_macro(self):
        import json, tempfile
        from pathlib import Path
        from unittest.mock import patch

        status = self._make_status()
        with tempfile.TemporaryDirectory() as td:
            status_path = Path(td) / "intel_source_status.json"
            status_path.write_text(json.dumps(status))
            top_lists_path = Path(td) / "top_lists.json"
            top_lists_path.write_text(json.dumps({}))

            with patch("regime_trader.ui.portfolio_advisor_engine._STATUS_PATH", status_path), \
                 patch("regime_trader.ui.portfolio_advisor_engine._TOP_LISTS_PATH", top_lists_path):
                from regime_trader.ui.portfolio_advisor_engine import build_advice
                result = build_advice(
                    [{"ticker": "AAPL", "net_qty": 10, "avg_cost": 150.0}],
                    regime="Bull",
                )

        adv = result[0]
        assert "momentum" in adv.factors,     "'momentum' key missing from factors"
        assert "macro"    not in adv.factors, "'macro' key must not be in factors"
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_portfolio_advisor_engine.py::TestBuildAdviceWeightsAndKeys -v
```

Expected: FAIL — `final_score` is 0.30 not 0.28; `"macro"` present, `"momentum"` absent

- [ ] **Step 3: Fix weights and key in `build_advice()`**

In `regime_trader/ui/portfolio_advisor_engine.py`, replace lines 165–191:

```python
        final_score = float(
            row.get("edgar_score", 0) * 0.30 +
            row.get("insider_score", 0) * 0.25 +
            row.get("congress_score", 0) * 0.20 +
            row.get("news_score", 0) * 0.15 +
            row.get("momentum_score", 0) * 0.10
        )
        signal = compute_signal(final_score, regime)
        swap   = find_swap_candidate(ticker, row.get("sector", ""), held, top_lists) \
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
```

With:

```python
        final_score = float(
            row.get("edgar_score", 0) * 0.28 +
            row.get("insider_score", 0) * 0.23 +
            row.get("congress_score", 0) * 0.22 +
            row.get("news_score", 0) * 0.15 +
            row.get("momentum_score", 0) * 0.12
        )
        signal = compute_signal(final_score, regime)
        swap   = find_swap_candidate(ticker, row.get("sector", ""), held, top_lists) \
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
                "momentum": round(float(row.get("momentum_score",0)), 4),
            },
            signal_age_days = age_days,
            swap_candidate  = swap,
            narrative       = None,
            not_in_universe = False,
        ))
```

- [ ] **Step 4: Run test to verify it passes**

```
pytest tests/test_portfolio_advisor_engine.py::TestBuildAdviceWeightsAndKeys -v
```

Expected: PASS

- [ ] **Step 5: Run full test suite**

```
pytest tests/ -q --tb=short
```

Expected: all green

- [ ] **Step 6: Commit**

```
git add regime_trader/ui/portfolio_advisor_engine.py tests/test_portfolio_advisor_engine.py
git commit -m "fix(advisor-engine): correct weights to 28/23/22/15/12, rename macro->momentum key"
```

---

### Task 4: Fix `pages/7_Portfolio_Advisor.py` — labels, Claude prompt, evidence section

**Context:** `pages/7_Portfolio_Advisor.py` has three stale references to fix and one new section to add:
1. Factor table (line 269–273): weights show "30%/25%/20%/15%/10%" → must be "28%/23%/22%/15%/12%"; label "📈 Macro" → "📈 Momentum"; key `f.get('macro',0)` → `f.get('momentum',0)`
2. Claude prompt (line 117): `factors.get('macro', 0)` → `factors.get('momentum', 0)` and label `Macro=` → `Momentum=`
3. After the factor bar table, add a `📊 Signal Evidence` collapsible section that displays the 4 evidence fields sourced from `intel_source_status.json` via `PositionAdvice`.

For the evidence section, `PositionAdvice` does not currently carry the evidence fields. We extend `PositionAdvice` with optional evidence fields and populate them in `build_advice()` from the same `row` dict.

**Files:**
- Modify: `regime_trader/ui/portfolio_advisor_engine.py` (add 4 evidence fields to `PositionAdvice` dataclass; populate in `build_advice`)
- Modify: `pages/7_Portfolio_Advisor.py` (fix labels/weights, fix Claude prompt, add evidence section)

- [ ] **Step 1: Add evidence fields to `PositionAdvice` and `build_advice()`**

In `regime_trader/ui/portfolio_advisor_engine.py`, replace the `PositionAdvice` dataclass definition:

```python
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
```

With:

```python
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
    # Evidence pass-through fields (populated from intel_source_status.json)
    news_source:             str   = "none"
    insider_usd:             float = 0.0
    momentum_spy_relative:   float = 0.0
    volume_spike:            float = 1.0
    quiver_evidence:         Dict[str, Any] = field(default_factory=dict)
```

Then in `build_advice()`, extend the `PositionAdvice(...)` constructor call (the one for scored positions, inside the loop) to include:

```python
            advice_list.append(PositionAdvice(
                ticker                = ticker,
                revolut_ticker        = pos.get("revolut_ticker", ticker),
                net_qty               = pos["net_qty"],
                avg_cost              = pos["avg_cost"],
                currency              = pos.get("currency", "USD"),
                source                = pos.get("source", "revolut"),
                signal                = signal,
                final_score           = round(final_score, 4),
                factors               = {
                    "edgar":    round(float(row.get("edgar_score",   0)), 4),
                    "insider":  round(float(row.get("insider_score", 0)), 4),
                    "congress": round(float(row.get("congress_score",0)), 4),
                    "news":     round(float(row.get("news_score",    0)), 4),
                    "momentum": round(float(row.get("momentum_score",0)), 4),
                },
                signal_age_days       = age_days,
                swap_candidate        = swap,
                narrative             = None,
                not_in_universe       = False,
                news_source           = row.get("news_source", "none"),
                insider_usd           = float(row.get("insider_usd", 0.0)),
                momentum_spy_relative = float(row.get("momentum_spy_relative", 0.0)),
                volume_spike          = float(row.get("volume_spike", 1.0)),
                quiver_evidence       = row.get("quiver_evidence", {}),
            ))
```

- [ ] **Step 2: Fix factor table labels, weights, and key in `7_Portfolio_Advisor.py`**

In `pages/7_Portfolio_Advisor.py`, replace the `factor_data` list (lines 268–274):

```python
                factor_data = [
                    {"Factor": "📋 Edgar",    "Weight": "30%", "Score": f"{f.get('edgar',    0):.3f}", "Bar": _factor_bar(f.get('edgar',    0))},
                    {"Factor": "🏦 Insider",  "Weight": "25%", "Score": f"{f.get('insider',  0):.3f}", "Bar": _factor_bar(f.get('insider',  0))},
                    {"Factor": "🏛️ Congress", "Weight": "20%", "Score": f"{f.get('congress', 0):.3f}", "Bar": _factor_bar(f.get('congress', 0))},
                    {"Factor": "📰 News",     "Weight": "15%", "Score": f"{f.get('news',     0):.3f}", "Bar": _factor_bar(f.get('news',     0))},
                    {"Factor": "📈 Macro",    "Weight": "10%", "Score": f"{f.get('macro',    0):.3f}", "Bar": _factor_bar(f.get('macro',    0))},
                ]
```

With:

```python
                factor_data = [
                    {"Factor": "📋 Edgar",    "Weight": "28%", "Score": f"{f.get('edgar',    0):.3f}", "Bar": _factor_bar(f.get('edgar',    0))},
                    {"Factor": "🏦 Insider",  "Weight": "23%", "Score": f"{f.get('insider',  0):.3f}", "Bar": _factor_bar(f.get('insider',  0))},
                    {"Factor": "🏛️ Congress", "Weight": "22%", "Score": f"{f.get('congress', 0):.3f}", "Bar": _factor_bar(f.get('congress', 0))},
                    {"Factor": "📰 News",     "Weight": "15%", "Score": f"{f.get('news',     0):.3f}", "Bar": _factor_bar(f.get('news',     0))},
                    {"Factor": "📈 Momentum", "Weight": "12%", "Score": f"{f.get('momentum', 0):.3f}", "Bar": _factor_bar(f.get('momentum', 0))},
                ]
```

- [ ] **Step 3: Fix the Claude prompt in `_get_claude_narrative()`**

In `pages/7_Portfolio_Advisor.py`, find this line in `_get_claude_narrative()` (around line 115–118):

```python
            f"Macro={factors.get('macro', 0):.2f}. Overall score: {score_display}. "
```

Replace with:

```python
            f"Momentum={factors.get('momentum', 0):.2f}. Overall score: {score_display}. "
```

- [ ] **Step 4: Add evidence section after the factor bars table**

In `pages/7_Portfolio_Advisor.py`, after the block:

```python
                st.dataframe(pd.DataFrame(factor_data), use_container_width=True, hide_index=True)
```

Add:

```python
                # Evidence section — only shown when at least one evidence field is non-zero
                cong = adv.quiver_evidence.get("congress", {})
                cong_net = cong.get("net", 0)
                cong_buys = cong.get("purchases", 0)
                cong_sales = cong.get("sales", 0)
                cong_reps = cong.get("representatives", [])
                cong_days = cong.get("recency_days")

                has_insider = adv.insider_usd > 0 or adv.factors.get("insider", 0) > 0
                has_congress = (cong_buys > 0 or cong_sales > 0) or adv.factors.get("congress", 0) > 0
                has_news = adv.news_source != "none" or adv.factors.get("news", 0) > 0
                has_momentum = (adv.momentum_spy_relative != 0.0 or adv.volume_spike != 1.0) or adv.factors.get("momentum", 0) > 0
                has_edgar = adv.factors.get("edgar", 0) > 0

                if any([has_insider, has_congress, has_news, has_momentum, has_edgar]):
                    with st.expander("📊 Signal Evidence", expanded=False):
                        if has_insider:
                            pct_cap = (adv.insider_usd / adv.market_value * 100) if adv.market_value > 0 else 0.0
                            st.markdown(f"🏦 **Insider:** ${adv.insider_usd:,.0f} open-market purchase ({pct_cap:.2f}% of position value)")
                        if has_congress:
                            reps_str = ", ".join(cong_reps[:3]) if cong_reps else "—"
                            days_str = f" · {cong_days}d ago" if cong_days is not None else ""
                            net_str  = f"Net {cong_net:+d} ({cong_buys} buys, {cong_sales} sells){days_str}"
                            st.markdown(f"🏛️ **Congress:** {net_str} · [{reps_str}]")
                        if has_news:
                            source_label = {"finnhub": "Finnhub", "yfinance": "yfinance", "none": "—"}.get(adv.news_source, adv.news_source)
                            st.markdown(f"📰 **News:** Source: {source_label} · Score: {adv.factors.get('news', 0):.2f}")
                        if has_momentum:
                            rel_pct = adv.momentum_spy_relative * 100
                            st.markdown(f"📈 **Momentum:** {rel_pct:+.1f}% vs SPY · Volume: {adv.volume_spike:.1f}× avg")
                        if has_edgar:
                            ceo_str = "✅" if adv.factors.get("edgar", 0) > 0.5 else ""
                            st.markdown(f"📋 **EDGAR:** Score {adv.factors.get('edgar', 0):.2f} {ceo_str}")
```

- [ ] **Step 5: Run the full test suite (no Streamlit UI tests — manual validation required)**

```
pytest tests/ -q --tb=short
```

Expected: all green (no Streamlit unit tests affected)

- [ ] **Step 6: Commit**

```
git add regime_trader/ui/portfolio_advisor_engine.py pages/7_Portfolio_Advisor.py
git commit -m "fix(portfolio-advisor): fix labels/weights/prompt, add evidence section, extend PositionAdvice"
```

---

### Task 5: Fix `pages/6_Stock_Picker.py` — `"Macro"`→`"Momentum"` + evidence expanders

**Context:** `pages/6_Stock_Picker.py` renders a flat table in `_render_ticker_table()`. Line 77 reads `f.get('macro',0)` — this always returns 0.00 because `top_lists.json` uses `"momentum"`. We rename the column and fix the key. Then we restructure the rendering to add a per-ticker evidence expander below each table. Since evidence expanders require per-row render logic, we add a new helper `_render_ticker_list_with_evidence()` that replaces `_render_ticker_table` when evidence fields are present, and keep `_render_ticker_table` for the cap-tier overview columns (which don't have expand space in a 3-column layout).

**Files:**
- Modify: `pages/6_Stock_Picker.py`

- [ ] **Step 1: Fix the `"Macro"` column key to `"Momentum"` in `_render_ticker_table()`**

In `pages/6_Stock_Picker.py`, find line 77:

```python
            "Macro":    f"{f.get('macro',0):.2f}",
```

Replace with:

```python
            "Momentum": f"{f.get('momentum',0):.2f}",
```

- [ ] **Step 2: Add `_render_ticker_list_with_evidence()` function**

After the existing `_render_ticker_table()` function definition (around line 84), add:

```python
def _render_ticker_list_with_evidence(entries: List[Dict[str, Any]], show_watchlist: bool = False) -> None:
    """Render entries as expandable rows with evidence sub-sections."""
    if not entries:
        st.caption("No tickers in this category.")
        return

    shown = 0
    for i, e in enumerate(entries, 1):
        badge = e.get("badge", "WATCHLIST")
        if badge == "WATCHLIST" and not show_watchlist:
            continue
        shown += 1
        score  = e.get("final_score", 0.0)
        f      = e.get("factors", {})
        ticker = e.get("ticker", "?")
        ceo    = "✅ CEO Buy" if e.get("ceo_buy") else ""
        label  = f"**{i}. {ticker}** — {badge}  Score: {score:.3f}  {ceo}"

        with st.expander(label, expanded=False):
            # Factor mini-table
            factor_rows = [
                {"Factor": "📋 Edgar",    "W": "28%", "Score": f"{f.get('edgar',    0):.3f}"},
                {"Factor": "🏦 Insider",  "W": "23%", "Score": f"{f.get('insider',  0):.3f}"},
                {"Factor": "🏛️ Congress", "W": "22%", "Score": f"{f.get('congress', 0):.3f}"},
                {"Factor": "📰 News",     "W": "15%", "Score": f"{f.get('news',     0):.3f}"},
                {"Factor": "📈 Momentum", "W": "12%", "Score": f"{f.get('momentum', 0):.3f}"},
            ]
            st.dataframe(pd.DataFrame(factor_rows), use_container_width=True, hide_index=True)

            # Evidence sub-section
            insider_usd          = float(e.get("insider_usd", 0.0))
            news_source          = e.get("news_source", "none")
            momentum_spy_rel     = float(e.get("momentum_spy_relative", 0.0))
            volume_spike         = float(e.get("volume_spike", 1.0))
            qe                   = e.get("quiver_evidence", {})
            cong                 = qe.get("congress", {})
            cong_net             = cong.get("net", 0)
            cong_buys            = cong.get("purchases", 0)
            cong_sales           = cong.get("sales", 0)
            cong_days            = cong.get("recency_days")
            cong_reps            = cong.get("representatives", [])

            has_insider  = insider_usd > 0 or f.get("insider", 0) > 0
            has_congress = (cong_buys > 0 or cong_sales > 0) or f.get("congress", 0) > 0
            has_news     = news_source != "none" or f.get("news", 0) > 0
            has_momentum = (momentum_spy_rel != 0.0 or volume_spike != 1.0) or f.get("momentum", 0) > 0
            has_edgar    = f.get("edgar", 0) > 0

            evidence_lines = []
            if has_insider:
                evidence_lines.append(f"🏦 **Insider** · ${insider_usd:,.0f}")
            if has_congress:
                days_str = f" · {cong_days}d ago" if cong_days is not None else ""
                reps_str = ", ".join(cong_reps[:2]) if cong_reps else "—"
                evidence_lines.append(
                    f"🏛️ **Congress** · Net {cong_net:+d} ({cong_buys} buys, {cong_sales} sells){days_str} · [{reps_str}]"
                )
            if has_news:
                src = {"finnhub": "Finnhub", "yfinance": "yfinance"}.get(news_source, news_source)
                evidence_lines.append(f"📰 **News** · Source: {src} · Score: {f.get('news', 0):.2f}")
            if has_momentum:
                evidence_lines.append(
                    f"📈 **Momentum** · {momentum_spy_rel*100:+.1f}% vs SPY · {volume_spike:.1f}× avg vol"
                )
            if has_edgar:
                ceo_str = " · CEO Buy ✅" if e.get("ceo_buy") else ""
                evidence_lines.append(f"📋 **EDGAR** · Score {f.get('edgar', 0):.2f}{ceo_str}")

            if evidence_lines:
                st.markdown("\n\n".join(evidence_lines))
            else:
                st.caption("No evidence data available for this ticker.")

    if shown == 0:
        st.caption("No HIGH BUY or TACTICAL BUY tickers. Toggle 'Show Watchlist' to see all.")
```

- [ ] **Step 3: Use `_render_ticker_list_with_evidence()` in the Sector Picks section**

In `render()`, find the sector picks rendering block:

```python
        for sector, emoji in _SECTOR_EMOJI.items():
            picks = sector_picks.get(sector, [])
            label = f"{emoji} {sector} ({len(picks)} picks)"
            with st.expander(label, expanded=True):
                _render_ticker_table(picks, show_watchlist=show_watchlist)
```

Replace with:

```python
        for sector, emoji in _SECTOR_EMOJI.items():
            picks = sector_picks.get(sector, [])
            label = f"{emoji} {sector} ({len(picks)} picks)"
            with st.expander(label, expanded=True):
                _render_ticker_list_with_evidence(picks, show_watchlist=show_watchlist)
```

- [ ] **Step 4: Run the full test suite**

```
pytest tests/ -q --tb=short
```

Expected: all green

- [ ] **Step 5: Commit**

```
git add pages/6_Stock_Picker.py
git commit -m "fix(stock-picker): rename Macro->Momentum column, add per-ticker evidence expanders"
```

---

### Task 6: Fix `scripts/send_toplists_discord.py` — emoji map and format loop

**Context:** `send_toplists_discord.py` `_FACTOR_EMOJI` at line 60 has `"macro": "📈"` and `_format_factor_line()` at line 81 iterates the tuple `("edgar", "insider", "congress", "news", "macro")`. Since `top_lists.json` factors dict now uses `"momentum"`, the `"macro"` lookups always return the default 0.50. Two one-line changes fix this.

**Files:**
- Create: `tests/test_discord_formatter.py`
- Modify: `scripts/send_toplists_discord.py:60, 81`

- [ ] **Step 1: Write the failing test**

Create `tests/test_discord_formatter.py`:

```python
"""tests/test_discord_formatter.py
Unit tests for Discord formatter — factor line uses "momentum" key.
"""
from __future__ import annotations
import pytest


class TestFormatFactorLine:
    def test_reads_momentum_key(self):
        from scripts.send_toplists_discord import _format_factor_line
        factors = {
            "edgar": 0.80, "insider": 0.70, "congress": 0.60,
            "news": 0.55, "momentum": 0.65,
        }
        line = _format_factor_line(factors)
        assert "0.65" in line, "momentum value 0.65 not in output"

    def test_momentum_key_present_gives_non_default(self):
        from scripts.send_toplists_discord import _format_factor_line
        factors_with    = {"edgar": 0.5, "insider": 0.5, "congress": 0.5, "news": 0.5, "momentum": 0.99}
        factors_without = {"edgar": 0.5, "insider": 0.5, "congress": 0.5, "news": 0.5}
        line_with    = _format_factor_line(factors_with)
        line_without = _format_factor_line(factors_without)
        assert "0.99" in line_with,    "momentum 0.99 not rendered"
        assert "0.99" not in line_without

    def test_macro_key_ignored(self):
        from scripts.send_toplists_discord import _format_factor_line
        factors = {
            "edgar": 0.5, "insider": 0.5, "congress": 0.5,
            "news": 0.5, "macro": 0.99,
        }
        line = _format_factor_line(factors)
        assert "0.99" not in line, "'macro' key must not affect output"

    def test_output_contains_all_five_emojis(self):
        from scripts.send_toplists_discord import _format_factor_line
        factors = {"edgar": 0.5, "insider": 0.5, "congress": 0.5, "news": 0.5, "momentum": 0.5}
        line = _format_factor_line(factors)
        for emoji in ("📋", "🏦", "🏛️", "📰", "📈"):
            assert emoji in line, f"{emoji} missing from factor line"
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_discord_formatter.py -v
```

Expected: FAIL — `"0.65"` not in output (reads `"macro"` key → default 0.50 → outputs `"0.50"`)

- [ ] **Step 3: Fix `_FACTOR_EMOJI` dict key**

In `scripts/send_toplists_discord.py`, find lines 55–61:

```python
_FACTOR_EMOJI = {
    "edgar":    "📋",
    "insider":  "🏦",
    "congress": "🏛️",
    "news":     "📰",
    "macro": "📈",
}
```

Replace with:

```python
_FACTOR_EMOJI = {
    "edgar":    "📋",
    "insider":  "🏦",
    "congress": "🏛️",
    "news":     "📰",
    "momentum": "📈",
}
```

- [ ] **Step 4: Fix `_format_factor_line()` iteration tuple**

In `scripts/send_toplists_discord.py`, find line 81:

```python
    for key in ("edgar", "insider", "congress", "news", "macro"):
```

Replace with:

```python
    for key in ("edgar", "insider", "congress", "news", "momentum"):
```

- [ ] **Step 5: Run test to verify it passes**

```
pytest tests/test_discord_formatter.py -v
```

Expected: all 4 tests PASS

- [ ] **Step 6: Run full test suite**

```
pytest tests/ -q --tb=short
```

Expected: all green

- [ ] **Step 7: Commit**

```
git add scripts/send_toplists_discord.py tests/test_discord_formatter.py
git commit -m "fix(discord): rename macro->momentum in _FACTOR_EMOJI and _format_factor_line loop"
```

---

## Self-Review

### Spec coverage check

| Spec requirement | Task covering it |
|-----------------|-----------------|
| Fix `portfolio_advisor_engine.py` weights (28/23/22/15/12) | Task 3 |
| Fix `portfolio_advisor_engine.py` `"macro"`→`"momentum"` key | Task 3 |
| Fix `7_Portfolio_Advisor.py` labels + weights | Task 4 |
| Fix `7_Portfolio_Advisor.py` Claude prompt `Macro=`→`Momentum=` | Task 4 |
| Fix `6_Stock_Picker.py` `"Macro"` column → `"Momentum"` | Task 5 |
| Fix `send_toplists_discord.py` emoji map key | Task 6 |
| Fix `send_toplists_discord.py` format loop | Task 6 |
| Add `news_source`, `insider_usd`, `momentum_spy_relative`, `volume_spike` to `_score_ticker()` | Task 1 |
| Pass-through 4 evidence fields in `_to_entry()` | Task 2 |
| Evidence cards in Stock Picker (per-ticker expanders) | Task 5 |
| Evidence section in Portfolio Advisor | Task 4 |
| No scoring logic changes (pass-through only) | Tasks 1–2 (no score functions touched) |
| Factor order contract `["edgar","insider","congress","news","momentum"]` | Tasks 3,4,5,6 — all iterate in this order |
| Evidence hiding rule (omit line when both value and score zero) | Task 4 & 5 (`has_X` guards) |
| Tests: portfolio advisor weights + momentum key | Task 3 |
| Tests: Discord `_format_factor_line` | Task 6 |

All spec requirements covered. No placeholders.

### Type consistency

- `PositionAdvice.quiver_evidence: Dict[str, Any]` — populated in Task 4, read in Task 4 ✅
- `PositionAdvice.news_source: str` — default `"none"`, populated in Task 4, read in Task 4 ✅
- `_to_entry()` returns `"volume_spike": float` — default `1.0`, read in Task 5 with `float(e.get("volume_spike", 1.0))` ✅
- `_FACTOR_EMOJI["momentum"]` set in Task 6 before `_format_factor_line` iterates `"momentum"` ✅
