# Discord Embed Redesign — Mobile-First v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `scripts/send_toplists_discord.py` to render a mobile-first Discord embed with unified 4-line ticker cards for both Large Cap (Top 5) and Mid Cap (Top 3) tiers, structured section headers, and a per-sector ticker chip grid in the Sector Exposure block.

**Architecture:** In-place refactor of `_ticker_detail_field()`, `_sector_heatmap()`, and `build_payload()` only. All I/O helpers, retry logic, CLI, and domain functions (`_score_bar`, `_fmt_cap`, `get_market_regime`, etc.) are untouched. The built-in `run_tests()` suite is updated to cover the new format assertions.

**Tech Stack:** Python 3.9+, Discord Webhook API (JSON embed payload), no new dependencies.

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `scripts/send_toplists_discord.py` | Modify | All changes live here — 3 functions touched |

---

### Task 1: Update `_ticker_detail_field()` to 4-line unified anatomy

**Files:**
- Modify: `scripts/send_toplists_discord.py` — function `_ticker_detail_field` (lines ~238–305)

- [ ] **Step 1: Write the failing test inside `run_tests()`**

Add this test block inside the `run_tests()` function, before the `# ── Report` section:

```python
# ── Test 8: ticker card anatomy — 4 lines, separator, unified format ──────
try:
    tl  = _base_tl(top_buys=[_entry("AAPL", score=0.87)])
    payload = build_payload(tl)
    fields  = payload["embeds"][0]["fields"]
    card = next((f for f in fields if f["name"].startswith("#")), None)
    val  = card["value"] if card else ""
    _check("card_has_separator",  "────────────────────" in val, f"val={val!r}")
    _check("card_has_score_bar",  "▓" in val or "░" in val,     f"val={val!r}")
    _check("card_has_factor_emoji","👤" in val,                  f"val={val!r}")
    lines = val.split("\n")
    _check("card_four_lines", len(lines) == 4, f"lines={lines}")
except Exception:
    failures.append(f"FAIL [card_anatomy]: {traceback.format_exc()}")

# ── Test 9: mid_cap=True uses backtick rank, not medal ────────────────────
try:
    entry  = _entry("CRDO", score=0.74)
    field  = _ticker_detail_field(1, entry, mid_cap=True)
    _check("midcap_rank_backtick", "`1`" in field["value"],
           f"val={field['value']!r}")
    _check("midcap_no_medal", "🥇" not in field["value"],
           f"val={field['value']!r}")
except Exception:
    failures.append(f"FAIL [midcap_rank]: {traceback.format_exc()}")
```

Also update `total_tests = 12` → `total_tests = 16` (4 new assertions added).

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd "c:/Users/ntard/Projects/Trading dashboard/regime_trader"
python scripts/send_toplists_discord.py --run-tests
```

Expected: `FAIL [card_anatomy]` and `FAIL [midcap_rank]` in stderr.

- [ ] **Step 3: Replace `_ticker_detail_field()` with new implementation**

Replace the entire function (from `def _ticker_detail_field(` to the closing `}`) with:

```python
def _ticker_detail_field(
    rank: int,
    entry: Dict[str, Any],
    anomaly_flags: Optional[List[str]] = None,
    rank_delta: Optional[int] = None,
    buyback_conv: Optional[float] = None,
    mid_cap: bool = False,
) -> Dict[str, Any]:
    """Unified 4-line ticker card — identical anatomy for Large Cap and Mid Cap.

    Line 1: {RANK} {TICKER} — {BADGE} — {SCORE} {BAR10}  [boosts]
    Line 2: {SECTOR_EMOJI}{SECTOR} · {MARKET_CAP}
    Line 3: ────────────────────
    Line 4: 👤{v}  🏛{v}  🔄{v}  📰{v}  🌐{v}

    Args:
        mid_cap:     True → use `N` backtick rank; False → use 🥇🥈🥉 for 1–3.
        rank_delta:  shadow_rank − boosted_rank (positive = promoted by boost).
        buyback_conv: conviction boost from buyback yield (0.40 or 0.80), or None.
    """
    ticker  = entry.get("ticker", "?")
    score   = float(entry.get("final_score", 0))
    badge   = entry.get("badge", "WATCHLIST")
    factors = entry.get("factors") or {}
    sector  = (entry.get("sector") or "").strip()
    cap     = entry.get("market_cap", 0)
    boost   = float(entry.get("congress_boost", 0.0))

    # ── Line 1: rank token ────────────────────────────────────────────────
    if mid_cap:
        rank_token = f"`{rank}`"
    else:
        rank_token = _MEDAL.get(rank, f"`{rank}`")

    # ── Line 1: boost / signal tokens (space-separated, after bar) ───────
    bar_str      = _score_bar(score, width=10)
    boost_part   = f"  🏛 `+{boost:.2f}`"        if boost > 0.0              else ""
    buyback_part = f"  🔄 `+{buyback_conv:.2f}`"  if buyback_conv is not None else ""

    if rank_delta is None or rank_delta == 0:
        trend_part = ""
    elif rank_delta > 0:
        trend_part = f"  🟢+{rank_delta}"
    else:
        trend_part = f"  🔴{rank_delta}"

    ceo_tag  = "  ⚡CEO" if entry.get("ceo_buy")  else ""
    flag_tag = "  ⚠️"    if anomaly_flags          else ""

    line1 = (
        f"{rank_token} **{ticker}** — {badge} — `{score:.2f}` {bar_str}"
        f"{boost_part}{buyback_part}{trend_part}{ceo_tag}{flag_tag}"
    )

    # ── Line 2: sector · market cap ───────────────────────────────────────
    sector_label = _SECTOR_SHORT.get(sector, _SECTOR_MISC) if sector else _SECTOR_MISC
    cap_str      = f"  ·  {_fmt_cap(cap)}" if cap else ""
    line2 = f"{sector_label}{cap_str}"

    # ── Line 3: visual separator ──────────────────────────────────────────
    line3 = "────────────────────"

    # ── Line 4: factor matrix — emoji + raw value ─────────────────────────
    factor_parts = []
    for key in ("edgar", "insider", "congress", "news", "macro"):
        v = factors.get(key)
        if v is not None:
            factor_parts.append(f"{_FACTOR_EMOJI[key]}`{v:.2f}`")
    if buyback_conv is not None:
        factor_parts.append(f"🔄`{buyback_conv:.2f}`")
    line4 = "  ".join(factor_parts) if factor_parts else "—"

    value = _truncate("\n".join([line1, line2, line3, line4]), 1024)
    return {
        "name":   f"#{rank}  {ticker}",
        "value":  value,
        "inline": False,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python scripts/send_toplists_discord.py --run-tests
```

Expected: `All tests passed` (or only pre-existing failures unrelated to card anatomy).

- [ ] **Step 5: Commit**

```bash
git add scripts/send_toplists_discord.py
git commit -m "feat(discord): unified 4-line ticker card anatomy with mid_cap param"
```

---

### Task 2: Refactor `_sector_heatmap()` to structured output

**Files:**
- Modify: `scripts/send_toplists_discord.py` — function `_sector_heatmap` (lines ~218–233)

- [ ] **Step 1: Write the failing test**

Add this block inside `run_tests()`, before the `# ── Report` section:

```python
# ── Test 10: structured heatmap — top-2 tickers per sector ───────────────
try:
    entries = [
        _entry("AAPL", sector="Information Technology", score=0.87),
        _entry("MSFT", sector="Information Technology", score=0.81),
        _entry("NVDA", sector="Information Technology", score=0.79),
        _entry("JPM",  sector="Financials",             score=0.68),
    ]
    result = _sector_heatmap_structured(entries)
    tech   = result.get("🖥️ Tech", [])
    fin    = result.get("🏛️ Fin",  [])
    _check("heatmap_struct_tech_count",  len(tech) == 2,
           f"tech={tech}")
    _check("heatmap_struct_tech_ticker", tech[0][0] == "AAPL",
           f"tech={tech}")
    _check("heatmap_struct_fin_count",   len(fin) == 1,
           f"fin={fin}")
    _check("heatmap_struct_sorted_desc",
           tech[0][1] >= tech[1][1],
           f"tech scores out of order: {tech}")
except Exception:
    failures.append(f"FAIL [heatmap_structured]: {traceback.format_exc()}")
```

Also update `total_tests` to `total_tests = 20`.

- [ ] **Step 2: Run tests to verify they fail**

```bash
python scripts/send_toplists_discord.py --run-tests
```

Expected: `FAIL [heatmap_structured]` — `_sector_heatmap_structured` not yet defined.

- [ ] **Step 3: Add `_sector_heatmap_structured()` below `_sector_heatmap()`**

Insert this new function immediately after the closing of `_sector_heatmap()`:

```python
def _sector_heatmap_structured(
    entries: List[Dict],
) -> Dict[str, List[tuple]]:
    """Return {sector_label: [(ticker, score), ...]} sorted by descending score.

    At most 2 tickers per sector. Combines Large Cap + Mid Cap entries.
    Unknown/missing sectors fall back to _SECTOR_MISC.
    """
    buckets: Dict[str, List[tuple]] = {}
    for e in entries:
        raw    = (e.get("sector") or "").strip()
        label  = _SECTOR_SHORT.get(raw, _SECTOR_MISC)
        ticker = e.get("ticker", "?")
        score  = float(e.get("final_score", 0))
        buckets.setdefault(label, []).append((ticker, score))

    # Sort each bucket by descending score, keep top 2
    return {
        lbl: sorted(pairs, key=lambda x: -x[1])[:2]
        for lbl, pairs in buckets.items()
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python scripts/send_toplists_discord.py --run-tests
```

Expected: `All tests passed`.

- [ ] **Step 5: Commit**

```bash
git add scripts/send_toplists_discord.py
git commit -m "feat(discord): add _sector_heatmap_structured() returning per-sector ticker chips"
```

---

### Task 3: Refactor `build_payload()` — header, section titles, Mid Cap loop, Sector Exposure

**Files:**
- Modify: `scripts/send_toplists_discord.py` — function `build_payload` (lines ~406–620)

This is the largest task. We do it in one commit because all changes are inside one function and are tightly coupled.

- [ ] **Step 1: Write the failing tests**

Add these blocks inside `run_tests()`, before `# ── Report`:

```python
# ── Test 11: header description format ────────────────────────────────────
try:
    tl = _base_tl(vix=18.3)
    payload = build_payload(tl)
    desc = payload["embeds"][0]["description"]
    _check("header_regime_trader_tag", "[REGIME TRADER]" in desc, f"desc={desc!r}")
    _check("header_vix_value",         "18.3" in desc,            f"desc={desc!r}")
    _check("header_stable_regime",     "STABLE" in desc,          f"desc={desc!r}")
except Exception:
    failures.append(f"FAIL [header_description]: {traceback.format_exc()}")

# ── Test 12: section-title separator fields ────────────────────────────────
try:
    tl = _base_tl(top_buys=[_entry("AAPL")])
    payload = build_payload(tl)
    names = [f["name"] for f in payload["embeds"][0]["fields"]]
    _check("large_cap_section_title",
           any("Large Cap" in n for n in names), f"names={names}")
except Exception:
    failures.append(f"FAIL [section_titles]: {traceback.format_exc()}")

# ── Test 13: Mid Cap fields rendered ──────────────────────────────────────
try:
    mid = [_entry("CRDO", sector="Materials", score=0.74),
           _entry("TMDX", sector="Health Care", score=0.69)]
    tl = _base_tl(top_buys=[_entry("AAPL")], mid_caps=mid)
    payload = build_payload(tl)
    names = [f["name"] for f in payload["embeds"][0]["fields"]]
    _check("midcap_section_title",
           any("Mid Cap" in n for n in names), f"names={names}")
    _check("midcap_ticker_field",
           any("CRDO" in n for n in names),    f"names={names}")
except Exception:
    failures.append(f"FAIL [midcap_fields]: {traceback.format_exc()}")

# ── Test 14: Sector Exposure chip format ───────────────────────────────────
try:
    tl = _base_tl(top_buys=[_entry("AAPL", sector="Information Technology", score=0.87),
                              _entry("MSFT", sector="Information Technology", score=0.81)])
    payload = build_payload(tl)
    fields  = payload["embeds"][0]["fields"]
    sector_field = next((f for f in fields if "Sector" in f["name"]), None)
    _check("sector_field_exists", sector_field is not None)
    val = sector_field["value"] if sector_field else ""
    _check("sector_chip_aapl", "AAPL" in val, f"val={val!r}")
    _check("sector_chip_msft", "MSFT" in val, f"val={val!r}")
except Exception:
    failures.append(f"FAIL [sector_chips]: {traceback.format_exc()}")

# ── Test 15: truncation message format updated ────────────────────────────
try:
    fat_entry = _entry("FAT", score=0.9)
    fat_entry["factors"] = {k: 0.99 for k in ("edgar","insider","congress","news","macro")}
    tl = _base_tl(top_buys=[fat_entry] * 5)
    payload = build_payload(tl)
    fields  = payload["embeds"][0]["fields"]
    trunc_field = next((f for f in fields if "shown" in f.get("value", "")), None)
    all_five    = sum(1 for f in fields if f["name"].startswith("#")) == 5
    _check("truncation_or_all_fit",
           trunc_field is not None or all_five,
           f"fields={[f['name'] for f in fields]}")
    if trunc_field:
        _check("truncation_new_format",
               "shown — full report in logs" in trunc_field["value"],
               f"val={trunc_field['value']!r}")
except Exception:
    failures.append(f"FAIL [truncation_format]: {traceback.format_exc()}")
```

Update `total_tests = 20` → `total_tests = 30`.

- [ ] **Step 2: Run tests to verify they fail**

```bash
python scripts/send_toplists_discord.py --run-tests
```

Expected: multiple FAIL lines for `header_description`, `section_titles`, `midcap_fields`, `sector_chips`, `truncation_format`.

- [ ] **Step 3: Replace the description block in `build_payload()`**

Find this block (around line 502–514):

```python
    # ── Description: TL;DR — all critical signals visible without scrolling ─
    boost_status  = "ON 🏛" if congress_boost_on else "OFF"
    feed_note     = "  ⚠️ *feed down — redistributed*" if weights_redistributed else ""
    summary_parts = [f"**{len(top_buys)} Buy{'s' if len(top_buys) != 1 else ''}**"]
    if anomaly_count:
        summary_parts.append(f"**{anomaly_count} Anomaly{'s' if anomaly_count != 1 else ''}** ⚠️")
    summary_parts.append(f"Congress Boost: **{boost_status}**")
    summary_line = "  |  ".join(summary_parts)

    description = (
        f"{summary_line}\n"
        f"`{ticker_count} tickers`{vix_str}  ·  EDGAR-first{feed_note}"
        f"{alert_block}"
    )
```

Replace with:

```python
    # ── Description: 2-line mobile header ─────────────────────────────────
    # Line 1: brand + date
    # Line 2: VIX regime (from get_market_regime)
    feed_note   = "  ⚠️ *feed down — redistributed*" if weights_redistributed else ""
    vix_regime  = get_market_regime(float(vix_val)) if vix_val is not None else "VIX —"
    pipeline_nm = top_lists.get("pipeline", "EDGAR-first")

    description = (
        f"**[REGIME TRADER]** Daily Market Report — **{date_str}**\n"
        f"{vix_regime}{feed_note}"
        f"{alert_block}"
    )
```

- [ ] **Step 4: Replace the fields-building block in `build_payload()`**

Find the entire fields section starting with `# ── Fields: compact ticker cards` (around line 516) through the satellite blocks and old sector heatmap, ending just before `# ── Footer`. Replace everything between those two comments with:

```python
    # ── Fields ────────────────────────────────────────────────────────────
    fields: List[Dict[str, Any]] = []

    shadow_buys    = top_lists.get("shadow_top_buys") or []
    shadow_rank_of = {e.get("ticker", ""): i for i, e in enumerate(shadow_buys, 1)}

    def _ticker_fields(entries, max_n, budget, is_mid_cap):
        """Render up to max_n ticker cards within character budget."""
        result = []
        used   = 0
        added  = 0
        for i, e in enumerate(entries[:max_n], 1):
            ticker_    = e.get("ticker", "")
            shadow_r   = shadow_rank_of.get(ticker_)
            rank_delta = (shadow_r - i) if shadow_r is not None else None
            buyback_cv = buyback_conv_of.get(ticker_.upper())
            field = _ticker_detail_field(
                i, e,
                anomaly_flags=anomaly_map.get(ticker_),
                rank_delta=rank_delta,
                buyback_conv=buyback_cv,
                mid_cap=is_mid_cap,
            )
            flen = len(field["value"])
            if used + flen > budget and added > 0:
                shown = added
                total = min(max_n, len(entries))
                result.append({
                    "name":   "…",
                    "value":  f"... [{shown}/{total}] shown — full report in logs",
                    "inline": False,
                })
                break
            result.append(field)
            used  += flen
            added += 1
        return result

    # ── Large Cap section ─────────────────────────────────────────────────
    if top_buys:
        fields.append({
            "name":   "🏆 Top 5 — Large Caps",
            "value":  "​",
            "inline": False,
        })
        fields.extend(_ticker_fields(top_buys, max_n=5, budget=1900, is_mid_cap=False))

    # ── Mid Cap section ───────────────────────────────────────────────────
    mid_caps   = top_lists.get("mid_caps")   or []
    small_caps = top_lists.get("small_caps") or []

    if mid_caps:
        fields.append({
            "name":   "📈 Top 3 — Mid Caps ($2B–$10B)",
            "value":  "​",
            "inline": False,
        })
        fields.extend(_ticker_fields(mid_caps, max_n=3, budget=1900, is_mid_cap=True))

    # ── Satellite detail blocks (cyclicals + cannibals) ───────────────────
    try:
        if satellite and isinstance(satellite, dict):
            month_label = satellite.get("month", "")
            cyclicals   = satellite.get("cyclicals") or []
            cannibals   = satellite.get("cannibals") or []

            if cyclicals:
                lines = [
                    f"**{c['ticker']}** {_score_bar(c['win_rate'], 6)} "
                    f"`{c['win_rate']:.0%}` win · `{c['median_return']:+.1%}` med · `{c.get('years','?')}y`"
                    for c in cyclicals
                ]
                fields.append({
                    "name":   f"🌀  Seasonal Cyclicals — {month_label}",
                    "value":  _truncate("\n".join(lines)),
                    "inline": False,
                })

            if cannibals:
                lines = [
                    f"**{c['ticker']}** · `{c.get('buyback_yield',0):.1%}` buyback"
                    f" · P/E `{c.get('pe',0):.1f}` · `{c.get('price_vs_52w_low',0):.2f}×` vs 52w low"
                    for c in cannibals
                ]
                fields.append({
                    "name":   "🐷  Share Cannibals",
                    "value":  _truncate("\n".join(lines)),
                    "inline": False,
                })
    except Exception as exc:
        log.warning("satellite embed fields skipped: %s", exc)

    # ── Sector Exposure — structured chip grid ────────────────────────────
    all_entries = list(top_buys[:5]) + list(mid_caps[:3])
    structured  = _sector_heatmap_structured(all_entries)
    if structured:
        # Sort: descending count, then descending top score
        sorted_sectors = sorted(
            structured.items(),
            key=lambda kv: (-len(kv[1]), -(kv[1][0][1] if kv[1] else 0)),
        )
        sector_lines = []
        for lbl, pairs in sorted_sectors:
            count      = len(all_entries[0:0])  # recount from all_entries
            # recount properly
            total_in_sector = sum(
                1 for e in all_entries
                if _SECTOR_SHORT.get((e.get("sector") or "").strip(), _SECTOR_MISC) == lbl
            )
            chips = "  ".join(f"`{t}` {s:.2f}" for t, s in pairs)
            sector_lines.append(f"{lbl} ({total_in_sector})  {chips}")
        fields.append({
            "name":   "📊  Sector Exposure",
            "value":  _truncate("\n".join(sector_lines), 1024),
            "inline": False,
        })
```

- [ ] **Step 5: Update the footer in `build_payload()`**

Find:

```python
    footer_text = f"{latency_part}Cov: {coverage_pct}{gap_note}  |  Mode: {mode_str}  |  Run: {run_id}"
```

Replace with:

```python
    footer_text = f"Run: {run_id}  |  Pipeline: {pipeline_nm}"
```

Note: `pipeline_nm` was defined in the description block above (Step 3).

- [ ] **Step 6: Run the full test suite**

```bash
python scripts/send_toplists_discord.py --run-tests
```

Expected: `All tests passed (30 assertions)` (or close — verify no regressions).

- [ ] **Step 7: Smoke-test with dry-run**

```bash
python scripts/send_toplists_discord.py --dry-run 2>&1 | python -c "
import sys, json
data = json.load(sys.stdin)
embed = data['embeds'][0]
print('title:', embed['title'])
print('desc:', embed['description'][:120])
print('fields:', [(f['name'], len(f['value'])) for f in embed['fields']])
"
```

Expected: title `⚡ Alpha Pipeline [...]`, description contains `[REGIME TRADER]`, fields list shows section-title fields and ticker fields.

- [ ] **Step 8: Commit**

```bash
git add scripts/send_toplists_discord.py
git commit -m "feat(discord): mobile-first embed v2 — section headers, unified ticker cards, mid-cap tier, sector chips"
```

---

### Task 4: Commit spec + plan docs, push, trigger CI

**Files:**
- `docs/superpowers/specs/2026-05-21-discord-embed-redesign.md` (already written)
- `docs/superpowers/plans/2026-05-21-discord-embed-redesign.md` (this file)

- [ ] **Step 1: Stage and commit the docs**

```bash
git add docs/superpowers/specs/2026-05-21-discord-embed-redesign.md
git add docs/superpowers/plans/2026-05-21-discord-embed-redesign.md
git commit -m "docs: discord embed redesign spec and implementation plan"
```

- [ ] **Step 2: Push to GitHub**

```bash
git push origin main
```

Expected: push accepted, GitHub Actions triggered automatically on `main`.

- [ ] **Step 3: Verify CI triggered**

```bash
gh run list --limit 5
```

Expected: a new run appears in `queued` or `in_progress` state for the push to `main`.

- [ ] **Step 4: Monitor CI until green**

```bash
gh run watch
```

Expected: all jobs pass. If any fail, inspect with `gh run view --log-failed`.

---

## Self-Review

**Spec coverage:**
- §1 Header block → Task 3 Step 3 ✅
- §2 Section titles → Task 3 Step 4 (`🏆 Top 5` / `📈 Top 3` fields) ✅
- §3 Ticker card anatomy (4 lines, separator, boosts on Line 1, `mid_cap` param) → Task 1 ✅
- §4 Budget & truncation (`[N/5] shown — full report in logs`) → Task 3 Step 4 `_ticker_fields()` ✅
- §5 Sector Exposure structured chips → Task 2 + Task 3 Step 4 ✅
- §6 Footer `Run: {id} | Pipeline: {name}` → Task 3 Step 5 ✅
- §7 Functions unchanged list → verified: `_score_bar`, `_fmt_cap`, `get_market_regime`, etc. not touched ✅
- §8 Discord limits (12 fields max) → fields count verified in Task 3 Step 7 smoke test ✅

**Placeholder scan:** No TBD, TODO, or vague steps. All code blocks are complete.

**Type consistency:**
- `_ticker_detail_field()` signature in Task 1 matches calls in Task 3 Step 4 (`mid_cap=is_mid_cap`) ✅
- `_sector_heatmap_structured()` defined in Task 2, called in Task 3 Step 4 ✅
- `pipeline_nm` defined in Task 3 Step 3, used in Task 3 Step 5 ✅
