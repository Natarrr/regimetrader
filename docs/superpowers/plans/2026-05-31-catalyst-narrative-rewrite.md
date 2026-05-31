# Catalyst Narrative Rewrite — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `_compute_catalyst()` in `send_toplists_discord.py` with an evidence-first narrative that surfaces dollar amounts, congress names, and SPY-relative returns instead of repeating score matrix values.

**Architecture:** Single function replacement in one file. Add `_NO_CATALYST` module-level constant near existing constants (line ~111). Replace the function body (lines 200–227). Update Test 12 (checks for `"driven by"`) and Test 13 (checks for old EPS format and `"|"` separator) in the same file's self-test block. No new imports needed — all data is already in the `entry` dict.

**Tech Stack:** Python 3.11, `scripts/send_toplists_discord.py` self-test suite (`--run-tests`).

---

## File Map

| File | Change |
|---|---|
| `scripts/send_toplists_discord.py` | Add `_NO_CATALYST` constant; replace `_compute_catalyst()`; update Tests 12 and 13 |

---

## Task 1: Add `_NO_CATALYST` constant and verify tests break

**Files:**
- Modify: `scripts/send_toplists_discord.py`

- [ ] **Step 1: Add the constant**

Open `scripts/send_toplists_discord.py`. Find the block of module-level constants around line 108–111:

```python
_MEDAL: Dict[int, str] = {1: "🥇", 2: "🥈", 3: "🥉"}
_MARKET_FLAGS: Dict[str, str] = {"USA": "🇺🇸", "US": "🇺🇸", "EUROPE": "🇪🇺", "ASIA": "🇯🇵"}

_STALE_HOURS = 25
```

Add `_NO_CATALYST` immediately after `_STALE_HOURS`:

```python
_MEDAL: Dict[int, str] = {1: "🥇", 2: "🥈", 3: "🥉"}
_MARKET_FLAGS: Dict[str, str] = {"USA": "🇺🇸", "US": "🇺🇸", "EUROPE": "🇪🇺", "ASIA": "🇯🇵"}

_STALE_HOURS = 25
_NO_CATALYST = "no primary catalyst detected"
```

- [ ] **Step 2: Run the self-tests to establish baseline**

```
python scripts/send_toplists_discord.py --run-tests
```

Expected: tests pass (we haven't changed the function yet — the constant addition alone doesn't break anything). Note how many assertions pass.

---

## Task 2: Replace `_compute_catalyst()`

**Files:**
- Modify: `scripts/send_toplists_discord.py:200–227`

- [ ] **Step 3: Replace the function**

Find the existing `_compute_catalyst` function (lines 200–227). Replace it entirely with:

```python
def _compute_catalyst(entry: Dict[str, Any]) -> str:
    """Evidence-first catalyst narrative, max 80 chars, signals separated by ' · '.

    Priority order (max 3 signals emitted):
      1. INSIDER: insider_usd > 0 → "Insider $Xk [CEO]"
      2. EARNINGS (PEAD ≤90d): earnings_surprise_pct → "EPS beat/miss ±X% (Nd ago)"
      3. CONGRESS: quiver_evidence.congress.purchases > 0 → "Nx congress buy · rep"
      4. MOMENTUM: |momentum_spy_relative| > 0.05 → "±X% vs SPY 12m"
      5. ANALYST REVISION (fallback only, no other signal): analyst_revision_n ≥ 5

    Returns _NO_CATALYST when no signal fires.
    """
    signals: list[str] = []

    # 1. INSIDER
    usd = float(entry.get("insider_usd", 0) or 0)
    if usd > 0:
        if usd < 100_000:
            usd_str = f"${round(usd / 1000, 1)}k"
        else:
            usd_str = f"${round(usd / 1000, 0):.0f}k"
        ceo_tier = entry.get("ceo_conviction_tier", "none") or "none"
        ceo_part = " CEO" if ceo_tier != "none" else ""
        signals.append(f"Insider {usd_str}{ceo_part}")

    # 2. EARNINGS SURPRISE (PEAD — Bernard & Thomas 1989, ≤90d window)
    eps_pct  = entry.get("earnings_surprise_pct")
    eps_days = int(entry.get("earnings_surprise_days") or 0)
    if eps_pct is not None and eps_days <= 90:
        pct_fmt = abs(eps_pct * 100)
        sign    = "+" if eps_pct >= 0 else ""
        verb    = "beat" if eps_pct >= 0 else "miss"
        signals.append(f"EPS {verb} {sign}{pct_fmt:.1f}% ({eps_days}d ago)")

    # 3. CONGRESS
    congress   = (entry.get("quiver_evidence") or {}).get("congress", {})
    cg_buys    = int(congress.get("purchases", 0) or 0)
    if cg_buys > 0:
        reps       = congress.get("representatives") or []
        rep_str    = reps[0][:12] if reps else "members"
        recency    = congress.get("recency_days")
        rec_part   = f" ({recency}d ago)" if recency is not None else ""
        signals.append(f"{cg_buys}x congress buy · {rep_str}{rec_part}")

    # 4. MOMENTUM
    rel = float(entry.get("momentum_spy_relative", 0) or 0)
    if abs(rel) > 0.05:
        sign = "+" if rel >= 0 else ""
        signals.append(f"{sign}{rel * 100:.1f}% vs SPY 12m")

    # 5. ANALYST REVISION — fallback only (no other signal fired)
    if not signals:
        n = int(entry.get("analyst_revision_n", 0) or 0)
        if n >= 5:
            signals.append(f"analyst revision signal ({n} analysts)")

    result = " · ".join(signals[:3])
    return (result or _NO_CATALYST)[:80]
```

- [ ] **Step 4: Run the self-tests to see which fail**

```
python scripts/send_toplists_discord.py --run-tests
```

Expected failures (these are the tests we need to fix next):
- `catalyst_line_present` — still checks for `"driven by"`
- `eps_pipe_separator` — checks for `"|"` (now `" · "`)
- `eps_in_catalyst_beat` — checks for `"EPS +15.3%"` (now `"EPS beat +15.3%"`)

---

## Task 3: Update Test 12 and Test 13

**Files:**
- Modify: `scripts/send_toplists_discord.py` (self-test block, around lines 953–995)

- [ ] **Step 5: Update Test 12**

Find the Test 12 block (around line 953). The existing assertion checks:

```python
_check("catalyst_line_present", any("driven by" in ln or "no primary" in ln for ln in lines), f"lines={lines}")
```

Replace it with:

```python
_check(
    "catalyst_line_present",
    any(
        any(kw in ln for kw in ["Insider", "EPS", "congress", "vs SPY", "no primary"])
        for ln in lines
    ),
    f"Catalyst line missing expected pattern: lines={lines}",
)
```

Also add a zero-signal test immediately after the existing Test 12 block (before Test 13). Find the `except Exception:` that closes Test 12 and insert this new sub-test before the try/except boundary:

```python
        # Zero-signal entry → _NO_CATALYST
        e_zero = _entry("ZERO", score=0.10)
        e_zero["insider_usd"]           = 0.0
        e_zero["earnings_surprise_pct"] = None
        cat_zero = _compute_catalyst(e_zero)
        _check("zero_signal_fallback", cat_zero == _NO_CATALYST, f"cat_zero={cat_zero!r}")
```

- [ ] **Step 6: Update Test 13 — EPS beat format**

Find Test 13 (around line 963). The existing check:

```python
_check("eps_in_catalyst_beat",  "EPS +15.3%" in cat, f"cat={cat!r}")
```

Replace with:

```python
_check("eps_in_catalyst_beat",  "EPS beat +15.3%" in cat, f"cat={cat!r}")
```

- [ ] **Step 7: Update Test 13 — separator**

The existing check:

```python
_check("eps_pipe_separator",    "|"          in cat, f"cat={cat!r}")
```

Replace with:

```python
_check("eps_separator",         "·"          in cat, f"cat={cat!r}")
```

- [ ] **Step 8: Update Test 13 — EPS miss format**

The existing check:

```python
_check("eps_in_catalyst_miss",  "EPS -8.7%"  in cat2, f"cat2={cat2!r}")
```

Replace with:

```python
_check("eps_in_catalyst_miss",  "EPS miss -8.7%"  in cat2, f"cat2={cat2!r}")
```

- [ ] **Step 9: Run the self-tests — all must pass**

```
python scripts/send_toplists_discord.py --run-tests
```

Expected: `All tests passed (N assertions)` with no failures. If `total_assertions` count in the report line is wrong (the file hardcodes it), note that but don't change it — it's cosmetic.

- [ ] **Step 10: Commit**

```bash
git add scripts/send_toplists_discord.py
git commit -m "feat(discord): rewrite _compute_catalyst — evidence-first narrative (insider USD, EPS beat/miss, congress, SPY momentum)"
```

---

## Task 4: Smoke-check representative outputs

**Files:**
- No changes. Verification only.

- [ ] **Step 11: Verify representative catalyst strings**

```python
python - <<'EOF'
import sys
sys.path.insert(0, ".")
from scripts.send_toplists_discord import _compute_catalyst, _NO_CATALYST

# Test 1: CEO insider buy
e1 = {"insider_usd": 250000.0, "ceo_conviction_tier": "substantial",
      "earnings_surprise_pct": None, "earnings_surprise_days": 0,
      "quiver_evidence": {}, "momentum_spy_relative": 0.0,
      "analyst_revision_n": 0}
print("CEO insider:", _compute_catalyst(e1))
# Expected: "Insider $250k CEO"

# Test 2: EPS beat + congress
e2 = {"insider_usd": 0, "ceo_conviction_tier": "none",
      "earnings_surprise_pct": 0.15, "earnings_surprise_days": 12,
      "quiver_evidence": {"congress": {"purchases": 3, "sales": 0,
          "representatives": ["Nancy Pelosi"], "recency_days": 5}},
      "momentum_spy_relative": 0.08, "analyst_revision_n": 0}
print("EPS+congress:", _compute_catalyst(e2))
# Expected: "EPS beat +15.0% (12d ago) · 3x congress buy · Nancy Pelosi (5d ago)"
# (truncated to 80 chars if needed)

# Test 3: zero signal
e3 = {"insider_usd": 0, "ceo_conviction_tier": "none",
      "earnings_surprise_pct": None, "earnings_surprise_days": 0,
      "quiver_evidence": {}, "momentum_spy_relative": 0.0,
      "analyst_revision_n": 0}
print("Zero signal:", _compute_catalyst(e3))
assert _compute_catalyst(e3) == _NO_CATALYST

# Test 4: analyst revision fallback
e4 = {"insider_usd": 0, "ceo_conviction_tier": "none",
      "earnings_surprise_pct": None, "earnings_surprise_days": 0,
      "quiver_evidence": {}, "momentum_spy_relative": 0.0,
      "analyst_revision_n": 8}
print("Analyst fallback:", _compute_catalyst(e4))
assert "analyst revision signal (8 analysts)" in _compute_catalyst(e4)

# Test 5: 80-char truncation
e5 = {"insider_usd": 50000.0, "ceo_conviction_tier": "none",
      "earnings_surprise_pct": 0.20, "earnings_surprise_days": 5,
      "quiver_evidence": {"congress": {"purchases": 7, "sales": 0,
          "representatives": ["Representative A. Smith"], "recency_days": 3}},
      "momentum_spy_relative": 0.25, "analyst_revision_n": 0}
result = _compute_catalyst(e5)
print("Truncation test:", repr(result))
assert len(result) <= 80, f"Too long: {len(result)}"

print("All smoke checks passed")
EOF
```

Expected: all `print` outputs show evidence-first strings, no assertion errors.

---

## Self-Review Checklist

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| `_NO_CATALYST = "no primary catalyst detected"` constant at module level | Task 1 |
| Signal 1: `insider_usd > 0` → `"Insider $Xk [CEO]"` | Task 2 |
| USD: `< 100k → 1dp`, `≥ 100k → 0dp` | Task 2 |
| No `(Nd ago)` suffix on insider (form4_count is not days) | Task 2 |
| `ceo_conviction_tier != "none"` → append ` CEO` | Task 2 |
| Signal 2: EPS beat/miss within 90d → `"EPS beat/miss ±X% (Nd ago)"` | Task 2 |
| Signal 3: congress purchases > 0 → `"Nx congress buy · rep (Nd ago)"` | Task 2 |
| rep truncated to 12 chars, fallback to "members" | Task 2 |
| recency_days shown if not None | Task 2 |
| Signal 4: `abs(momentum_spy_relative) > 0.05` → `"±X% vs SPY 12m"` | Task 2 |
| Signal 5: analyst revision fallback ONLY when no other signal | Task 2 |
| `analyst_revision_n` (not `analyst_revision_n_analysts`) | Task 2 |
| `" · "` separator between signals | Task 2 |
| Max 3 signals, `[:80]` truncation | Task 2 |
| Returns `_NO_CATALYST` (the constant) when no signal | Task 2 |
| Test 12 updated: multi-branch `"Insider"/"EPS"/"congress"/"vs SPY"/"no primary"` | Task 3 |
| Test 12: zero-signal entry asserts `== _NO_CATALYST` | Task 3 |
| Test 13: `"EPS beat +15.3%"` replaces `"EPS +15.3%"` | Task 3 |
| Test 13: `"·"` replaces `"|"` separator check | Task 3 |
| Test 13: `"EPS miss -8.7%"` replaces `"EPS -8.7%"` | Task 3 |
| Smoke-check representative outputs | Task 4 |

All requirements covered.
