# Design: `_compute_catalyst` Evidence-First Narrative Rewrite

**Date:** 2026-05-31  
**Status:** Approved

---

## Problem

The current `_compute_catalyst` output ("driven by IC: 0.85 + IB: 0.72") repeats the score matrix and contains no actionable information. A trader reading the Discord brief cannot distinguish a $500k CEO purchase from a thin volume spike. The function needs to surface the underlying evidence — dollar amounts, congress names, SPY-relative returns — using data that is already present in every `entry` dict.

---

## Scope

One function rewrite in `scripts/send_toplists_discord.py`. One new module-level constant. Test 12 update in the same file's self-test block. No new API calls, no new fields required.

---

## New Module-Level Constant

```python
_NO_CATALYST = "no primary catalyst detected"
```

Added near the other module-level constants (not inside a function). The self-test references `_NO_CATALYST` directly — if the copy ever changes, the test fails loudly instead of silently passing with stale text.

---

## `_compute_catalyst(entry: dict) -> str` — Complete Replacement

**Signature:** unchanged.  
**Return:** single-line string, max 80 chars, using `" · "` as separator between signals.

### Signal priority (evaluated in order, max 3 signals total)

**1. INSIDER** — highest conviction  
Condition: `entry.get("insider_usd", 0) > 0`

USD formatting:
```python
usd = float(entry.get("insider_usd", 0) or 0)
if usd < 100_000:
    usd_str = f"${round(usd / 1000, 1)}k"   # e.g. "$12.5k"
else:
    usd_str = f"${round(usd / 1000, 0):.0f}k"  # e.g. "$150k"
```

Format with CEO:
- If `ceo_conviction_tier not in (None, "", "none")`: `f"Insider {usd_str} CEO"`
- Otherwise: `f"Insider {usd_str}"`

No `(Nd ago)` suffix. `form4_count` is a filing count, not days — displaying it as recency would be misleading. If a real `insider_recency_days` field is ever added to `top_lists.json`, the format can pick it up at that point.

**2. EARNINGS (PEAD)** — within 90-day window  
Condition: `earnings_surprise_pct is not None and earnings_surprise_days <= 90`

```python
eps_pct  = entry.get("earnings_surprise_pct")
eps_days = int(entry.get("earnings_surprise_days") or 0)
pct_fmt  = abs(eps_pct * 100)
sign     = "+" if eps_pct >= 0 else ""
verb     = "beat" if eps_pct >= 0 else "miss"
signal   = f"EPS {verb} {sign}{pct_fmt:.1f}% ({eps_days}d ago)"
```

Skip entirely when `eps_days > 90` or `eps_pct is None`.

**3. CONGRESS**  
Condition: `quiver_evidence.congress.purchases > 0`

```python
congress = (entry.get("quiver_evidence") or {}).get("congress", {})
purchases = int(congress.get("purchases", 0) or 0)
reps      = congress.get("representatives") or []
rep_str   = reps[0][:12] if reps else "members"
recency   = congress.get("recency_days")
recency_part = f" ({recency}d ago)" if recency is not None else ""
signal    = f"{purchases}x congress buy · {rep_str}{recency_part}"
```

**4. MOMENTUM**  
Condition: `abs(momentum_spy_relative) > 0.05`

```python
rel = float(entry.get("momentum_spy_relative", 0) or 0)
sign = "+" if rel >= 0 else ""
signal = f"{sign}{rel * 100:.1f}% vs SPY 12m"
```

**5. ANALYST REVISION** — fallback only  
Condition: `len(signals) == 0` AND `entry.get("analyst_revision_n", 0) >= 5`

The field is `analyst_revision_n` (not `analyst_revision_n_analysts`) — confirmed in `run_pipeline.py` result dict.

```python
n = int(entry.get("analyst_revision_n", 0) or 0)
signal = f"analyst revision signal ({n} analysts)"
```

This signal is only emitted when no other signal has fired. It is not shown alongside higher-priority signals.

### Assembly

```python
result = " · ".join(signals[:3])
if not result:
    result = _NO_CATALYST
return result[:80]
```

---

## Self-Test Update (Test 12)

Test 12 currently asserts `"driven by" in ln` for positive-signal cases. Replace with a multi-branch check that covers all 5 priority paths:

```python
_check(
    "catalyst_line_present",
    any(kw in ln for kw in ["Insider", "EPS", "congress", "vs SPY", "no primary"]),
    f"Catalyst line missing expected pattern: {ln!r}"
)
```

The zero-signal case is kept as a hard contract using `_NO_CATALYST`:

```python
e_zero = _entry("ZERO", score=0.10, insider_usd=0, ...)
cat_zero = _compute_catalyst(e_zero)
_check("zero_signal_fallback", cat_zero == _NO_CATALYST, f"cat={cat_zero!r}")
```

Test 13 (EPS tests) checks for `"EPS +15.3%"`, `"8d ago"`, and `"|"` as separator. After the rewrite:
- `"EPS beat +15.3%"` replaces `"EPS +15.3%"` — update assertion to `"EPS beat" in cat`
- `"8d ago"` — unchanged, still present
- `"|"` separator — replaced by `" · "` — update assertion to `"·" in cat`

---

## What Does Not Change

- Function signature: `_compute_catalyst(entry: dict) -> str`
- Callers: `_ticker_detail_field` (line 3), `_action_section` — both unchanged
- All other fields in `_ticker_detail_field`: line 1, line 2 (factor matrix), separators
- All other self-tests (Tests 1–11, 14+)
- No new imports required
