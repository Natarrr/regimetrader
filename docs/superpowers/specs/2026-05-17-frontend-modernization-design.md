# Frontend Modernization — 5-Factor Smart Money Alignment

**Date:** 2026-05-17  
**Author:** Nathan T  
**Status:** Approved for implementation

---

## Goal

Fix every stale field reference in the frontend (Stock Picker, Portfolio Advisor, Discord alerts) so it correctly reads the new 5-factor model (`momentum` not `macro`, weights 28/23/22/15/12). Then add evidence cards to Stock Picker and Portfolio Advisor that surface the underlying signal quality for each factor.

---

## Issues Found

### Bug 1 — `portfolio_advisor_engine.py`: stale weights + wrong key
**File:** `regime_trader/ui/portfolio_advisor_engine.py:165–190`

```python
# CURRENT (wrong)
final_score = (
    row.get("edgar_score", 0) * 0.30 +
    row.get("insider_score", 0) * 0.25 +
    row.get("congress_score", 0) * 0.20 +
    row.get("news_score", 0) * 0.15 +
    row.get("momentum_score", 0) * 0.10     # wrong weight
)
factors = {
    ...
    "macro": round(float(row.get("momentum_score", 0)), 4),  # wrong key
}

# TARGET (correct)
final_score = (
    row.get("edgar_score", 0) * 0.28 +
    row.get("insider_score", 0) * 0.23 +
    row.get("congress_score", 0) * 0.22 +
    row.get("news_score", 0) * 0.15 +
    row.get("momentum_score", 0) * 0.12     # correct weight
)
factors = {
    ...
    "momentum": round(float(row.get("momentum_score", 0)), 4),  # correct key
}
```

### Bug 2 — `pages/7_Portfolio_Advisor.py`: stale labels + Claude prompt
**File:** `pages/7_Portfolio_Advisor.py:269–273, 117`

```python
# CURRENT factor_data row (wrong label + wrong weight)
{"Factor": "📈 Macro",    "Weight": "10%", "Score": f"{f.get('macro',    0):.3f}", ...}

# TARGET
{"Factor": "📈 Momentum", "Weight": "12%", "Score": f"{f.get('momentum', 0):.3f}", ...}

# All other weights also stale:
# Edgar 30% → 28%, Insider 25% → 23%, Congress 20% → 22%

# CURRENT Claude prompt (wrong label, silently 0 after key fix)
f"Macro={factors.get('macro', 0):.2f}"

# TARGET
f"Momentum={factors.get('momentum', 0):.2f}"
```

### Bug 3 — `pages/6_Stock_Picker.py`: reads `"macro"` from `top_lists.json` which writes `"momentum"`
**File:** `pages/6_Stock_Picker.py:77`

```python
# CURRENT (silently returns 0.00 — key doesn't exist in factors dict)
"Macro":    f"{f.get('macro',0):.2f}",

# TARGET
"Momentum": f"{f.get('momentum',0):.2f}",
```

### Bug 4 — `scripts/send_toplists_discord.py`: factor emoji map + format loop
**File:** `scripts/send_toplists_discord.py:55–84`

```python
# CURRENT (emoji map has "macro" key; format loop iterates "macro" → always 0)
_FACTOR_EMOJI = {"edgar":..., "insider":..., "congress":..., "news":..., "macro": "📈"}
for key in ("edgar", "insider", "congress", "news", "macro"):
    v = factors.get(key, 0.50)   # "macro" never in factors → always 0.50

# TARGET
_FACTOR_EMOJI = {"edgar":..., "insider":..., "congress":..., "news":..., "momentum": "📈"}
for key in ("edgar", "insider", "congress", "news", "momentum"):
    v = factors.get(key, 0.50)
```

Also: the factor legend field text says "📈 Momentum" already — only the code is wrong.

---

## Backend Evidence Fields (light addition)

**Constraint:** These additions do not alter any scores, only expose already-computed values for UI and Discord. No scoring logic changes.

**File:** `scripts/run_pipeline.py` — `_score_ticker()` return dict

Add 3 pass-through fields to every result row so the frontend can display evidence without re-computing anything:

```python
return {
    # ... existing fields unchanged ...
    "news_source":          news_source,        # "finnhub" | "yfinance" | "none"
    "insider_usd":          total_purchases_usd, # raw $ value (already computed)
    "momentum_spy_relative": price_data["return_20d"] - price_data["spy_return_20d"],
    "volume_spike":         price_data["volume_spike"],
    # ... rest unchanged ...
}
```

`news_source` is determined by whether `FINNHUB_API_KEY` was set:
```python
news_source = "finnhub" if finnhub_key else "yfinance"
# Set to "none" if score_news_finnhub returns 0.0 AND yfinance also fails
```

Default value is `"none"` (not `"unknown"`) — clearer signal that no source succeeded.

These fields flow through `intel_source_status.json` → `generate_top_lists.py` → `top_lists.json` automatically (generate_top_lists passes them through via `row.get(...)` in `_to_entry()`).

**`generate_top_lists.py` `_to_entry()`** must pass these through:
```python
return {
    # ... existing fields ...
    "news_source":           row.get("news_source", "none"),
    "insider_usd":           float(row.get("insider_usd", 0.0)),
    "momentum_spy_relative": float(row.get("momentum_spy_relative", 0.0)),
    "volume_spike":          float(row.get("volume_spike", 1.0)),
}
```

---

## Evidence Cards Design

### Stock Picker — evidence expander per ticker
Under each ticker row (currently a flat table row), add an expander showing:

```
▶ AAPL evidence

  🏦 INSIDER   $2,450,000 · 0.49% of mktcap · 12d ago
  🏛️ CONGRESS  +3 net (5 buys, 2 sales) · 8d ago · [Rep. Smith, Rep. Jones]
  📰 NEWS      Source: Finnhub · Score: 0.72
  📈 MOMENTUM  +4.2% vs SPY · Volume: 2.3× avg
  📋 EDGAR     Form 4 filings: 3 · CEO Buy: ✅
```

Data sourced from `quiver_evidence` (congress), `insider_usd` (insider), `news_source` (news), `momentum_spy_relative` + `volume_spike` (momentum), `form4_count` + `ceo_buy` (edgar).

**Rendering rules:**

- If a field is 0/None/empty, show "—" not a number. Never show evidence that misleads (e.g., "0 buys, 0 sales" → show "No congressional activity").
- **Hide the entire line** when both the raw evidence value and the factor score are zero/null — e.g., if `insider_usd == 0` and `insider_score == 0`, omit the insider line entirely. Same rule applies to all factors. This avoids fake precision like "$0 purchase · 0.00% of mktcap".

### Portfolio Advisor — evidence section inside position expander
After the factor bar table, add a collapsible "📊 Signal Evidence" section:

```
📊 Signal Evidence
  🏦 Insider:   $1,200,000 open-market purchase (0.24% of mktcap, 21d ago)
  🏛️ Congress:  Net +2 (3 buys, 1 sell) · 15d ago · [Rep. A, Rep. B]
  📰 News:      Finnhub sentiment 0.68 (bullish)
  📈 Momentum:  +2.1% vs SPY · 1.8× volume
  📋 EDGAR:     3 Form 4 filings in 180d
```

Same data sources as Stock Picker.

---

## File Map

| File | Action | Change |
|------|--------|--------|
| `scripts/run_pipeline.py` | Modify | Add `news_source`, `insider_usd`, `momentum_spy_relative`, `volume_spike` to `_score_ticker()` return |
| `backend/market_intel/generate_top_lists.py` | Modify | Pass through 4 new evidence fields in `_to_entry()` |
| `regime_trader/ui/portfolio_advisor_engine.py` | Modify | Fix weights (28/23/22/15/12), fix `"macro"`→`"momentum"` key |
| `pages/7_Portfolio_Advisor.py` | Modify | Fix labels (Macro→Momentum, all weights), fix Claude prompt, add evidence section |
| `pages/6_Stock_Picker.py` | Modify | Fix `"Macro"`→`"Momentum"` column, add evidence expanders |
| `scripts/send_toplists_discord.py` | Modify | Fix `_FACTOR_EMOJI` key, fix `_format_factor_line()` loop |

---

## Data Flow

```
run_pipeline.py (_score_ticker)
  → intel_source_status.json
      {edgar_score, insider_score, congress_score, news_score, momentum_score,
       quiver_evidence, news_source, insider_usd, momentum_spy_relative, volume_spike}
  
  → generate_top_lists.py (_to_entry)
      → top_lists.json entries
          {factors: {edgar, insider, congress, news, momentum},
           quiver_evidence, news_source, insider_usd, momentum_spy_relative, volume_spike}

top_lists.json → pages/6_Stock_Picker.py  (evidence expanders)
top_lists.json → scripts/send_toplists_discord.py (factor line uses "momentum" key)
intel_source_status.json → portfolio_advisor_engine.py → pages/7_Portfolio_Advisor.py
```

**Factor order contract:** The canonical factor order is `["edgar", "insider", "congress", "news", "momentum"]`. All UIs (Stock Picker, Portfolio Advisor) and Discord must respect this order when iterating or displaying factors.

---

## Consistency Checklist

| Check | Before | After |
|-------|--------|-------|
| Portfolio Advisor weights | 30/25/20/15/10 | 28/23/22/15/12 |
| Portfolio Advisor factor key | `"macro"` | `"momentum"` |
| Portfolio Advisor label | "📈 Macro 10%" | "📈 Momentum 12%" |
| Portfolio Advisor Claude prompt | `factors.get('macro',0)` | `factors.get('momentum',0)` |
| Stock Picker column | `"Macro"` → `f.get('macro',0)` = 0 | `"Momentum"` → `f.get('momentum',0)` |
| Discord emoji map | `"macro": "📈"` | `"momentum": "📈"` |
| Discord format loop | iterates `"macro"` → 0 | iterates `"momentum"` → correct value |
| Evidence in Stock Picker | none | quiver_evidence + 4 new fields |
| Evidence in Portfolio Advisor | none | quiver_evidence + 4 new fields |

---

## Test Strategy

**Existing tests to run after each change:**
```bash
pytest tests/ -q --tb=short
```
No existing tests cover the UI pages (Streamlit) or Discord formatter. Write new tests for:

**`tests/test_portfolio_advisor_engine.py`** (already exists — check for stale assertions):
- `compute_signal()` still works
- `build_advice()` uses correct weights 0.28/0.23/0.22/0.15/0.12
- `factors` dict keys include `"momentum"` not `"macro"`

**`tests/test_discord_formatter.py`** (new):
- `_format_factor_line()` reads `"momentum"` key from factors dict
- `_format_factor_line()` produces non-zero value when `"momentum"` is present

**Manual validation:**
```bash
# 1. Run pipeline to get fresh data
python scripts/run_pipeline.py --tickers-file config/canary_top10.csv --verbose

# 2. Generate top_lists
python -m backend.market_intel.generate_top_lists --force --verbose

# 3. Verify new fields in intel_source_status.json
python -c "
import json
s = json.load(open('logs/intel_source_status.json'))
r = s['results'][0]
print('news_source:', r.get('news_source'))
print('insider_usd:', r.get('insider_usd'))
print('momentum_spy_relative:', r.get('momentum_spy_relative'))
print('volume_spike:', r.get('volume_spike'))
print('quiver_evidence:', r.get('quiver_evidence'))
"

# 4. Verify top_lists.json factors use "momentum" not "macro"
python -c "
import json
t = json.load(open('logs/top_lists.json'))
entry = t['top_buys'][0] if t.get('top_buys') else None
if entry:
    print('factors keys:', list(entry.get('factors',{}).keys()))
    print('Has momentum:', 'momentum' in entry.get('factors',{}))
    print('Has macro (should be False):', 'macro' in entry.get('factors',{}))
"

# 5. Discord dry-run
python scripts/send_toplists_discord.py --dry-run

# 6. Launch Streamlit
streamlit run regime_trader/ui/streamlit_app.py
```
