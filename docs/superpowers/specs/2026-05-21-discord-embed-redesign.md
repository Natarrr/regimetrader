# Discord Embed Redesign — Mobile-First v2

**Date:** 2026-05-21
**File:** `scripts/send_toplists_discord.py`
**Approach:** In-place refactor (Approach A) — modify `_ticker_detail_field` and `build_payload` only; all helpers unchanged.

---

## Goal

Restore at-a-glance readability on mobile by imposing strict visual hierarchy, whitespace anchors, and unified card anatomy across all ticker tiers. Integrate Buyback and VIX boosts into the existing signal flow.

---

## 1. Header Block

Two lines inside the embed description, followed by a blank line:

```
[REGIME TRADER] Daily Market Report — {DATE}
VIX {VAL} {ICON} {REGIME}
```

- `{DATE}` — formatted as `May 21, 2026`
- `{ICON}` — 🟢 / 🟡 / 🔴 from `get_market_regime()`
- `{REGIME}` — BULLISH / STABLE / BEARISH (bold)
- No ticker count, no pipeline name on this line (kept minimal per user decision)

---

## 2. Section Titles

Two bold bandeau titles separate the two ticker blocks. Rendered as embed field names (not inline):

- `🏆 Top 5 — Large Caps` — used as field name for a blank separator field
- `📈 Top 3 — Mid Caps ($2B–$10B)` — same pattern

These are Discord embed fields with `name = title` and `value = "​"` (zero-width space) to act as visual section headers.

---

## 3. Ticker Card Anatomy — Unified Format (Large Caps + Mid Caps)

Every ticker — whether Large Cap or Mid Cap — uses the identical 4-element structure:

```
Line 1:  {RANK} {TICKER} — {BADGE} — {SCORE} {BAR10}  [{BOOSTS}]
Line 2:  {SECTOR_EMOJI}{SECTOR} · {MARKET_CAP}
Line 3:  ────────────────────
Line 4:  👤{v}  🏛{v}  🔄{v}  📰{v}  🌐{v}
```

### Line 1 details

| Element | Rule |
|---|---|
| `{RANK}` | 🥇🥈🥉 for ranks 1–3 (Large Cap only); `` `1` `` `` `2` `` `` `3` `` for Mid Cap |
| `{TICKER}` | Bold, uppercase |
| `{BADGE}` | HIGH BUY / TACTICAL BUY / WATCHLIST |
| `{SCORE}` | `final_score` formatted `0.00` |
| `{BAR10}` | `_score_bar(score, width=10)` — fixed 10-char string `▓▓▓▓▓▓░░░░` |
| Congress boost | `🏛 +{boost:.2f}` if `congress_boost > 0` |
| Buyback boost | `🔄 +{conv:.2f}` if satellite cannibal match |
| Rank delta | `🟢+{n}` if promoted, `🔴{n}` if demoted |
| CEO buy | `⚡CEO` if `entry.ceo_buy` |
| Anomaly | `⚠️` if anomaly flags present |

All boost/signal tokens appear on Line 1, space-separated, after the bar.

### Line 3 (separator)

`────────────────────` — exactly 20 em-dashes. Mandatory visual anchor between metadata and factor matrix.

### Line 4 (factor matrix)

Emoji + raw value, space-separated. Only include factors present in `entry["factors"]`. Order: `👤 🏛 🔄 📰 🌐`. If buyback conviction present, append `🔄{conv:.2f}` (already shown on Line 1 as `+conv` — Line 4 shows raw factor value).

---

## 4. Ticker Budget & Truncation

- Budget: **1900 characters** total across all ticker field values
- On overflow after ticker N: append field `name="…"` `value="... [{N}/5] shown — full report in logs"`
- Large Caps: max 5 tickers from `top_buys`
- Mid Caps: max 3 tickers from `mid_caps`
- Mid Cap budget is separate — does not share the 1900-char Large Cap budget (each section tracked independently)

---

## 5. Sector Exposure

Field at the end of the embed. Format: one row per sector, two columns:

```
{EMOJI}{SECTOR} ({COUNT})    {CHIP1}{SCORE1}  {CHIP2}{SCORE2}
```

- Shows top-2 tickers by score per sector (across both Large + Mid caps combined)
- Sectors with only 1 ticker show 1 chip
- Sorted by descending count then descending top score
- Built by extending `_sector_heatmap()` to return structured data instead of a flat string

---

## 6. Footer

```
Run: {run_id}  |  Pipeline: {pipeline_name}
```

Pipeline name derived from `top_lists.get("pipeline", "EDGAR-first")`.

---

## 7. Implementation Scope

### Functions to modify

| Function | Change |
|---|---|
| `_ticker_detail_field()` | Adopt 4-line anatomy; add `mid_cap=False` param to switch rank display (medals vs backtick numbers) |
| `build_payload()` | Add section-title separator fields; add Mid Cap ticker loop (max 3, separate budget); restructure description to new 2-line header; update Sector Exposure to structured format |
| `_sector_heatmap()` | Extend to return `Dict[str, List[Tuple[str, float]]]` (sector → [(ticker, score)]) for chip rendering; keep old flat-string path for backward compat or replace entirely |

### Functions unchanged

`_score_bar`, `_fmt_cap`, `_factor_group`, `_truncate`, `get_market_regime`, `_buyback_conviction`, `_embed_color`, `_data_age_hours`, `_load_satellite`, `_load_anomaly_report`, `send_to_discord`, `main`, `run_tests` (tests updated to cover new format).

### Test updates required

- `run_tests()` assertion for section-title fields present when `top_buys` non-empty
- Assertion for Mid Cap fields when `mid_caps` non-empty
- Existing truncation test updated to expect `[N/5] shown — full report in logs` format
- Existing heatmap test updated for structured output

---

## 8. Discord Limits Compliance

| Limit | Value | Status |
|---|---|---|
| Embed title | 256 chars | ✅ unchanged |
| Field value | 1024 chars | ✅ `_truncate()` applied per field |
| Fields total | 25 max | ✅ 1 header + 1 sep + 5 large + 1 sep + 3 mid + 1 sector = 12 fields max |
| Total embed | 6000 chars | ✅ within budget |
