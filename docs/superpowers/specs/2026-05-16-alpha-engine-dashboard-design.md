# Alpha Engine Dashboard — Design Spec
**Date:** 2026-05-16  
**Status:** Approved by user  
**Scope:** Stock Picker page, Portfolio Advisor page, Revolut portfolio integration, sidebar refactor, workflow consistency fixes

---

## 1. Context & Goals

The user invests £1,000/month in small/mid-cap stocks selected by the existing 5-factor scoring engine (edgar 30%, insider 25%, congress 20%, news 15%, macro 10%). Their **real portfolio lives in Revolut** — uploaded weekly as an XLSX transaction log. Alpaca is a paper trading account (disconnected from real money) and is demoted to secondary status.

**Goals:**
1. Surface the monthly stock picks (by sector and cap tier) in a dedicated page — informational only, no allocation tool
2. Give daily buy/sell/hold advice on the actual Revolut portfolio via a Portfolio Advisor page
3. Make Revolut the primary portfolio source throughout the dashboard
4. Fix two consistency bugs in the existing UI
5. Add sector picks (top 3 per sector) to the scoring pipeline output

---

## 2. Architecture

### New files
| File | Purpose |
|------|---------|
| `pages/6_Stock_Picker.py` | Monthly pick leaderboard — sector picks + cap-tier picks, read-only |
| `pages/7_Portfolio_Advisor.py` | Daily position advice — Revolut-first, table + expand drawer |
| `regime_trader/services/revolut_parser.py` | Parses Revolut XLSX transaction log → net positions + avg cost basis |
| `regime_trader/ui/portfolio_advisor_engine.py` | Hybrid scoring: quant re-score + Claude 2-sentence narrative |

### Modified files (surgical changes only)
| File | Change |
|------|--------|
| `regime_trader/ui/streamlit_app.py` | Sidebar refactor (section headers + button nav), Live Monitor Revolut-first, Portfolio Sync tab wired to Revolut importer, explainability labels fixed |
| `backend/market_intel/generate_top_lists.py` | Add `_sector_picks()` → writes `sector_picks` key to `top_lists.json` |

### No changes to
- All backend scoring, pipeline, quant models
- GitHub Actions workflows
- `intel_source_status.json` schema
- Any quant model page (Monetary Pulse, Volatility Brain, Valuation Radar, Contagion Web, Regime Prediction)

---

## 3. Data Flow

```
Pipeline (edgar_3x.yml, daily 08:30 ET)
  └─→ logs/intel_source_status.json   (all 160 tickers, all factor scores)
        └─→ generate_top_lists.py
              └─→ logs/top_lists.json  (top_buys, mid_caps, small_caps, sector_picks)
                    └─→ Stock Picker page (reads JSON, zero API calls)

Revolut XLSX (uploaded weekly by user)
  └─→ revolut_parser.py
        └─→ data/revolut_portfolio.json  (persisted to disk)
              └─→ Live Monitor tab  (live prices via yfinance, real P&L)
              └─→ Portfolio Advisor page
                    └─→ portfolio_advisor_engine.py
                          ├─→ intel_source_status.json  (look up factor scores, zero new scoring)
                          └─→ Claude API  (2-sentence narrative, gated by ANTHROPIC_API_KEY, cached in session_state)

Alpaca API (paper trading — secondary)
  └─→ Collapsed expander in Live Monitor: "📋 Paper Trading (Alpaca)"
```

---

## 4. Sidebar Refactor

Replace `st.radio()` with `st.sidebar.button()` per page + `st.session_state["_nav_page"]` for selection tracking. Section headers via `st.sidebar.markdown()` with custom CSS styling.

**New structure:**
```
🧭 Navigate
── ALPHA ENGINE ──
  📅 Stock Picker
  💼 Portfolio Advisor
── QUANT MODELS ──
  💰 Monetary Pulse
  📈 Volatility Brain
  🔭 Valuation Radar
  🕸️ Contagion Web
  🎯 Regime Prediction
── DASHBOARD ──
  📊 Dashboard
```

Active page button gets visual highlight. Backward-compatible: existing pages load via same `_load_page_module()` mechanism.

---

## 5. Stock Picker Page (`pages/6_Stock_Picker.py`)

**Data source:** `logs/top_lists.json` — read-only, no API calls.

**Layout:**
1. **Freshness banner** — "Pipeline last ran: {generated_at} · Next: tomorrow 08:30 ET" + Refresh button (clears `@st.cache_data`)
2. **Sector Picks** — five `st.expander` panels, one per target sector:
   - ⚡ Energy | 🪨 Materials | 📡 Communication Services | 🏥 Healthcare | 💻 Information Technology
   - Each: ranked mini-table (Rank · Ticker · Cap Tier · Score · Badge · CEO Buy)
   - Tickers with `final_score < 0.40` hidden by default; toggle to show
3. **Cap-Tier Overview** — three columns (Top Buys · Mid Caps · Small Caps), top 5 each, compact read-only cards

**Backend change — `generate_top_lists.py`:**

Add `_sector_picks(entries, target_sectors, n=3) -> Dict[str, List]`:
- Filters full `entries` list to 5 target sectors
- Returns top `n` by `final_score` per sector
- Adds `sector_picks` key to `top_lists.json` output
- Target sectors: `["Energy", "Materials", "Communication Services", "Healthcare", "Information Technology"]`
- No change to existing `top_buys`, `mid_caps`, `small_caps` keys

---

## 6. Revolut Parser (`regime_trader/services/revolut_parser.py`)

**Input:** `.xlsx` file in Revolut trading account statement format.

**Revolut XLSX schema (confirmed from real file):**
```
Date | Ticker | Type | Quantity | Price per share | Total Amount | Currency | FX Rate
```
Transaction types: `BUY - MARKET`, `SELL - MARKET`, `DIVIDEND`, `CASH TOP-UP`, `CASH WITHDRAWAL`

**Logic:**
1. Load with `openpyxl`, find header row, filter to BUY/SELL rows only
2. Net quantity per ticker: BUY adds, SELL subtracts; drop tickers where `net_qty ≤ 0`
3. Weighted average cost basis from BUY transactions: `Σ(qty × price) / Σ(qty)`
4. Apply ticker mapping table (`data/revolut_ticker_map.json`) for known symbol differences (e.g., `AIR1 → AIR.PA`)
5. Flag tickers not resolvable to a US/universe symbol as `not_scored: true`

**Output schema (per position):**
```json
{
  "ticker": "DDOG",
  "revolut_ticker": "DDOG",
  "net_qty": 2.25,
  "avg_cost": 111.09,
  "currency": "USD",
  "fx_rate": 1.1828,
  "not_scored": false,
  "source": "revolut"
}
```

**Persistence:** saved to `data/revolut_portfolio.json` (atomic write via existing `save_json_atomic`). Each upload **overwrites** the previous file — latest statement wins, no append logic. Survives browser refreshes. Dashboard shows "Last imported: {date}" badge.

---

## 7. Live Monitor Tab (Revolut-First)

**Revised layout:**
1. **Regime banner** (unchanged — VIX rule, existing code)
2. **Revolut Portfolio** (primary block):
   - If `data/revolut_portfolio.json` exists: fetch latest **close price** via `yfinance` (15-min delayed, free), compute real P&L vs avg cost basis, display in GBP/EUR
   - Metrics row: Total Value · Total P&L · Day P&L · # Positions
   - Positions table with live price, unrealized P&L, unrealized %
   - "Upload new statement →" button linking to Portfolio Sync tab
   - If no file: prompt to upload via Portfolio Sync
3. **Paper Trading (Alpaca)** — `st.expander("📋 Paper Trading (Alpaca)", expanded=False)` containing existing Alpaca display code unchanged

---

## 8. Portfolio Advisor Page (`pages/7_Portfolio_Advisor.py`)

**Engine:** `regime_trader/ui/portfolio_advisor_engine.py`

### 8a. Signal thresholds
| Score | Signal |
|-------|--------|
| ≥ 0.65 | ➕ ADD |
| 0.45–0.64 | = HOLD |
| 0.30–0.44 | ⬇️ REDUCE |
| < 0.30 or regime kill-switch | 🚫 EXIT |

### 8b. Trader enhancements
1. **Portfolio Health Score** — `Σ(score × position_weight)` across all positions where `position_weight = position_value / total_portfolio_value`. Displayed as a gauge metric at the top. Sector breakdown bar below it — flags any sector > 40% of portfolio.
2. **Signal Age** — derived from the `computed_at` field in `intel_source_status.json` (run-level timestamp, same for all tickers in a given pipeline run). Age = today minus `computed_at`. Signals > 30 days old get ⚠️ badge. Shown as "Signal: 4 days old" in each row.
3. **Swap Candidates** — for REDUCE/EXIT positions: look up top-scored unowned ticker in the same sector from `top_lists.json`. Display as "🔄 Consider: NVDA (0.81, same sector)" in the expand drawer.

### 8c. Layout
**① Regime banner** (reused existing component)

**② Source toggle** — "Revolut" | "Alpaca (paper)" | "All" — defaults to Revolut

**③ Position table** (compact, click row to expand):
```
Ticker | Source | Signal | Score | Signal Age | Unreal. P&L | Day % | ⚠️
```

**④ Expand drawer per row:**
- Factor bars: edgar / insider / congress / news / macro (5 bars)
- Claude narrative: 2 sentences (generated once, cached in `st.session_state["advisor_narratives"][ticker]`)
- Swap candidate (if REDUCE/EXIT)
- Raw score numbers

### 8d. Engine logic (`portfolio_advisor_engine.py`)
```
load_revolut_positions(data/revolut_portfolio.json)
  + optional: load_alpaca_positions()
for each position:
  scores = lookup_scores(intel_source_status.json, ticker)  # no new API calls
  signal = apply_thresholds(scores, regime)
  age = compute_signal_age(intel_source_status.json, ticker)
  swap = find_swap_candidate(top_lists.json, ticker, sector) if signal in [REDUCE, EXIT]
  if ANTHROPIC_API_KEY and not cached:
    narrative = claude_client.analyze(ticker, scores, signal, pnl_context)  # 2 sentences
    cache in st.session_state
return List[PositionAdvice]
```

**Graceful degradation:** ticker not in `intel_source_status.json` → score `N/A`, signal `— (not in universe)`, no narrative, no swap candidate.

---

## 9. Portfolio Sync Tab (Revolut Importer)

Replaces `_render_portfolio_sync()` in the Dashboard page.

**Flow:**
1. Upload `.xlsx` file
2. Parse via `revolut_parser.py`
3. Preview table of derived positions (with deselect checkboxes)
4. Confirm → save to `data/revolut_portfolio.json`
5. Success banner: "Portfolio updated · {n} positions · {date}"

**Stub tabs removed:** Trade Log and Regime History tabs are removed from the Dashboard tab bar. Portfolio Sync remains as the 4th tab.

---

## 10. Consistency Fixes

**Fix 1 — Explainability label mismatch:**
`_render_explainability()` currently references `insider_score / institutional_score / momentum_score` (old schema). Replace with `edgar_score / insider_score / congress_score / news_score / macro_score` and update weights to match pipeline: edgar 30%, insider 25%, congress 20%, news 15%, macro 10%.

**Fix 2 — Sector picks in `top_lists.json`:**
Add `_sector_picks()` to `generate_top_lists.py` so Stock Picker has sector data. Without this fix, the sector panels would always be empty.

---

## 11. Out of Scope

- Automatic trade execution via Alpaca
- Persistent Claude narrative storage (disk) — session cache only
- Full Streamlit refactor / splitting `streamlit_app.py` into sub-modules
- Non-Revolut broker CSV formats
- Real-time price streaming (yfinance polling on refresh is sufficient)
