# Hedge-Fund Grade Backtest Discord Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current basic backtest Discord payload with an institutional-grade embed that explicitly shows sample size (N), alpha decay across T+5/T+10/T+20, and a capacity/liquidity tier matrix — matching the visual spec provided.

**Architecture:** All changes are confined to `scripts/backtest_signals.py`. The existing `build_backtest_discord_payload()` function is **replaced** entirely. Supporting helpers (`_market_flag`, `_failure_label`, `_alpha_decay_tag`, `_truncate_field`) are added above it. `SignalRecord` gets two optional fields (`market`, `company_name`) populated from the JSON snapshot during parsing. No new files are created.

**Tech Stack:** Python 3.11, dataclasses, existing `HorizonStats` / `SignalRecord` types, Discord Embed JSON (max 1024 chars/field, 6000 chars total embed, 25 fields).

---

## Context for the implementer

The codebase is `regime_trader` — a quantitative trading dashboard that generates daily signal lists (`top_lists.json`) stored in `logs/archive/`. Every Friday 21:00 UTC a GitHub Actions workflow (`weekly_backtest.yml`) runs `scripts/backtest_signals.py`, computes forward returns at T+5/T+10/T+20 vs SPY, and (after our recent addition) posts a Discord embed.

The current Discord payload (`build_backtest_discord_payload`) is basic. The user wants hedge-fund-grade output matching this visual spec:

```
📊 STRATEGY PERFORMANCE LOG — ACTIVE EDGE TRACKING
Dataset: N=12 signals | Tracked via Forward-Walk Validation Ledger.

🟢 HIGH BUY (N=6)        🟡 TACTICAL BUY (N=6)
Metrics     | Value       Metrics     | Value
WR          | 66.7% (4/6) WR          | 50.0% (3/6)
PF          | 2.14        PF          | 1.15
T+5 Avg     | +1.45%      T+5 Avg     | +0.80%
T+10 Avg    | +3.20%      T+10 Avg    | +1.10%
T+20 Avg    | +2.10%      T+20 Avg    | -0.45% ← Alpha Decay!
α vs SPY    | +1.85%      α vs SPY    | +0.20%

📊 Capacity & Liquidity Tiers (T+10)
• Large: +2.45% (75.0% WR, N=4)
• Mid:   +1.10% (50.0% WR, N=2)
• Small: -0.95% (33.3% WR, N=3)

⏱ Edge Trajectory — Recent BUY Signals
🇺🇸 PLTR (▓▓▓▓▓▓▓▓░░) → T+10: +5.40% | α: +4.10%
🇪🇺 SAP.DE (▓▓▓▓▓▓░░░░) → T+10: +1.20% | α: +0.90%

🚨 Risk Attribution — Top 3 Detractors
AMD (T+10: -2.10%) · Failure: EDGAR Over-reliance
```

Key data available in `SignalRecord`:
- `ticker: str` — e.g. `"PLTR"`, `"SAP.DE"`, `"7203.T"`
- `signal_date: date`
- `badge: str` — `"HIGH BUY"` or `"TACTICAL BUY"`
- `cap_tier: str` — `"large"`, `"mid"`, `"small"`
- `final_score: float` — 0.0–1.0
- `factors: Dict[str, float]` — keys: `"edgar"`, `"insider"`, `"congress"`, `"news"`, `"momentum"`
- `weights: Dict[str, float]`
- `entry_price: Optional[float]`
- `returns: Dict[int, Optional[float]]` — keys: 5, 10, 20
- `alpha: Dict[int, Optional[float]]` — keys: 5, 10, 20

`HorizonStats` fields: `horizon`, `count`, `win_rate`, `avg_return`, `median_return`, `max_drawdown`, `profit_factor`, `avg_alpha`.

`_primary_failure_factor(rec)` already exists — returns the factor name (e.g. `"edgar"`) with highest weighted contribution for a losing trade.

---

## File Map

| Path | Action | Purpose |
|---|---|---|
| `scripts/backtest_signals.py` | Modify | Add `market`/`company_name` to `SignalRecord`; add 4 helper functions; replace `build_backtest_discord_payload()` |
| `tests/test_backtest_discord.py` | Create | Unit tests for all new helpers and the payload builder |

---

## Task 1: Add `market` and `company_name` to `SignalRecord`

**Files:**
- Modify: `scripts/backtest_signals.py:96-113` (`SignalRecord` dataclass)
- Modify: `scripts/backtest_signals.py:155-216` (`_parse_snapshot` function)
- Test: `tests/test_backtest_discord.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_backtest_discord.py
import json
import tempfile
from datetime import date
from pathlib import Path
from scripts.backtest_signals import _parse_snapshot, SignalRecord


def _make_snapshot(top_buys: list, weights: dict | None = None) -> dict:
    return {
        "generated_at": "2026-05-22T14:00:00Z",
        "weights": weights or {"edgar": 0.28, "insider": 0.23, "congress": 0.22, "news": 0.15, "macro": 0.12},
        "top_buys": top_buys,
    }


def _make_entry(ticker: str, badge: str = "HIGH BUY", score: float = 0.85, **extra) -> dict:
    return {
        "ticker": ticker,
        "badge": badge,
        "final_score": score,
        "cap_tier": "large",
        "factors": {"edgar": 0.8, "insider": 0.7, "congress": 0.5, "news": 0.6, "macro": 0.5},
        **extra,
    }


def test_parse_snapshot_captures_market_from_json():
    entry = _make_entry("SAP.DE", market="EUROPE", company_name="SAP")
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(_make_snapshot([entry]), f)
        path = Path(f.name)
    records = _parse_snapshot(path)
    assert len(records) == 1
    assert records[0].market == "EUROPE"
    assert records[0].company_name == "SAP"


def test_parse_snapshot_market_defaults_to_usa():
    entry = _make_entry("AAPL")  # no market field
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
        json.dump(_make_snapshot([entry]), f)
        path = Path(f.name)
    records = _parse_snapshot(path)
    assert records[0].market == "USA"
    assert records[0].company_name == ""
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/test_backtest_discord.py::test_parse_snapshot_captures_market_from_json -v
```

Expected: `AttributeError: 'SignalRecord' object has no attribute 'market'`

- [ ] **Step 3: Add fields to `SignalRecord`**

In `scripts/backtest_signals.py`, add two optional fields at the end of `SignalRecord` (after `entry_next_day`, before the blank line before `# Filled after price download`):

```python
@dataclass
class SignalRecord:
    """One triggered signal from a historical snapshot."""
    ticker:        str
    signal_date:   date
    badge:         str
    cap_tier:      str
    final_score:   float
    factors:       Dict[str, float]
    weights:       Dict[str, float]
    strategy_era:  str
    source_file:   str
    entry_next_day: bool

    # Multi-market metadata (present in snapshots generated after 2026-05-22)
    market:        str = "USA"
    company_name:  str = ""

    # Filled after price download
    entry_price:   Optional[float] = None
    returns:       Dict[int, Optional[float]] = field(default_factory=dict)
    spy_returns:   Dict[int, Optional[float]] = field(default_factory=dict)
    alpha:         Dict[int, Optional[float]] = field(default_factory=dict)
```

- [ ] **Step 4: Populate `market` and `company_name` in `_parse_snapshot`**

Find the `records.append(SignalRecord(...))` call inside `_parse_snapshot` (around line 203). Add two keyword arguments:

```python
        records.append(SignalRecord(
            ticker        = ticker,
            signal_date   = signal_date,
            badge         = badge,
            cap_tier      = (entry.get("cap_tier") or "unknown").lower(),
            final_score   = score,
            factors       = _normalize_factors(raw_factors),
            weights       = weights,
            strategy_era  = _era_label(weights),
            source_file   = path.name,
            entry_next_day = entry_next_day,
            market        = entry.get("market", "USA"),
            company_name  = entry.get("company_name", ""),
        ))
```

- [ ] **Step 5: Run tests to verify they pass**

```
pytest tests/test_backtest_discord.py::test_parse_snapshot_captures_market_from_json tests/test_backtest_discord.py::test_parse_snapshot_market_defaults_to_usa -v
```

Expected: 2 PASSED

- [ ] **Step 6: Commit**

```bash
git add scripts/backtest_signals.py tests/test_backtest_discord.py
git commit -m "feat(backtest): add market and company_name fields to SignalRecord"
```

---

## Task 2: Add helper functions

**Files:**
- Modify: `scripts/backtest_signals.py` — insert 4 helpers before `build_backtest_discord_payload`
- Test: `tests/test_backtest_discord.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_backtest_discord.py`:

```python
from scripts.backtest_signals import (
    _market_flag, _failure_label, _alpha_decay_tag, _truncate_field
)


def test_market_flag_usa():
    assert _market_flag("USA") == "🇺🇸"

def test_market_flag_europe():
    assert _market_flag("EUROPE") == "🇪🇺"

def test_market_flag_asia():
    assert _market_flag("ASIA") == "🇯🇵"

def test_market_flag_unknown():
    assert _market_flag("UNKNOWN") == "🌐"

def test_failure_label_edgar():
    assert _failure_label("edgar") == "EDGAR Over-reliance"

def test_failure_label_momentum():
    assert _failure_label("momentum") == "Momentum Reversal"

def test_failure_label_unknown():
    assert _failure_label("xyz") == "xyz"

def test_alpha_decay_tag_no_decay():
    # T+10 > T+5 > 0: no decay
    assert _alpha_decay_tag(r5=0.01, r10=0.02, r20=0.03) == ""

def test_alpha_decay_tag_t20_negative():
    # T+20 goes negative from positive T+10: decay
    tag = _alpha_decay_tag(r5=0.02, r10=0.03, r20=-0.01)
    assert "Decay" in tag

def test_alpha_decay_tag_t20_collapses():
    # T+20 less than half of T+10: meaningful decay
    tag = _alpha_decay_tag(r5=0.02, r10=0.04, r20=0.01)
    assert "Decay" in tag

def test_truncate_field_short():
    assert _truncate_field("hello", 1024) == "hello"

def test_truncate_field_long():
    result = _truncate_field("x" * 2000, 1024)
    assert len(result) == 1024
    assert result.endswith("…")
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_backtest_discord.py::test_market_flag_usa tests/test_backtest_discord.py::test_failure_label_edgar -v
```

Expected: `ImportError` — functions don't exist yet

- [ ] **Step 3: Implement the four helpers**

Insert this block in `scripts/backtest_signals.py` immediately before the line `# ── Discord KPI notification ───`:

```python
# ── Discord payload helpers ────────────────────────────────────────────────────

_MARKET_FLAGS: Dict[str, str] = {
    "USA":    "🇺🇸",
    "EUROPE": "🇪🇺",
    "ASIA":   "🇯🇵",
}

_FAILURE_LABELS: Dict[str, str] = {
    "edgar":    "EDGAR Over-reliance",
    "insider":  "Insider Signal Fade",
    "congress": "Congress Signal Lag",
    "news":     "Sentiment Reversal",
    "momentum": "Momentum Reversal",
}


def _market_flag(market: str) -> str:
    return _MARKET_FLAGS.get(market, "🌐")


def _failure_label(factor: str) -> str:
    return _FAILURE_LABELS.get(factor, factor)


def _alpha_decay_tag(r5: float, r10: float, r20: float) -> str:
    """Return '← Alpha Decay!' if T+20 collapses vs T+10, else empty string."""
    if r10 <= 0:
        return ""
    if r20 < 0 or r20 < r10 * 0.5:
        return "  ← Alpha Decay!"
    return ""


def _truncate_field(text: str, limit: int = 1024) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_backtest_discord.py -k "flag or label or decay or truncate" -v
```

Expected: 12 PASSED

- [ ] **Step 5: Commit**

```bash
git add scripts/backtest_signals.py tests/test_backtest_discord.py
git commit -m "feat(backtest): add Discord helper functions (_market_flag, _failure_label, _alpha_decay_tag, _truncate_field)"
```

---

## Task 3: Replace `build_backtest_discord_payload`

**Files:**
- Modify: `scripts/backtest_signals.py` — replace `build_backtest_discord_payload` entirely
- Test: `tests/test_backtest_discord.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_backtest_discord.py`:

```python
from datetime import date
from dataclasses import dataclass, field as dc_field
from scripts.backtest_signals import (
    SignalRecord, HorizonStats, build_backtest_discord_payload
)


def _make_horizon_stats(horizon: int, count: int, win_rate: float,
                         avg_return: float, avg_alpha: float,
                         profit_factor: float = 1.5) -> HorizonStats:
    return HorizonStats(
        horizon=horizon, count=count, win_rate=win_rate,
        avg_return=avg_return, median_return=avg_return,
        max_drawdown=-0.05, profit_factor=profit_factor,
        avg_alpha=avg_alpha,
    )


def _make_signal_record(ticker: str, badge: str, score: float,
                         cap_tier: str = "large", market: str = "USA",
                         r10: float = 0.03, alpha10: float = 0.02) -> SignalRecord:
    rec = SignalRecord(
        ticker=ticker, signal_date=date(2026, 5, 1), badge=badge,
        cap_tier=cap_tier, final_score=score,
        factors={"edgar": 0.8, "insider": 0.6, "congress": 0.5,
                 "news": 0.5, "momentum": 0.4},
        weights={"edgar": 0.28, "insider": 0.23, "congress": 0.22,
                 "news": 0.15, "momentum": 0.12},
        strategy_era="v2", source_file="test.json",
        entry_next_day=False, market=market,
    )
    rec.entry_price = 100.0
    rec.returns  = {5: 0.01, 10: r10, 20: 0.02}
    rec.alpha    = {5: 0.005, 10: alpha10, 20: 0.015}
    rec.spy_returns = {5: 0.005, 10: 0.01, 20: 0.005}
    return rec


def _make_full_payload_inputs():
    badge_stats = {
        "HIGH BUY": [
            _make_horizon_stats(5,  6, 0.667, 0.0145, 0.010),
            _make_horizon_stats(10, 6, 0.667, 0.0320, 0.0185),
            _make_horizon_stats(20, 6, 0.500, 0.0210, 0.012),
        ],
        "TACTICAL BUY": [
            _make_horizon_stats(5,  6, 0.500, 0.0080, 0.003),
            _make_horizon_stats(10, 6, 0.500, 0.0110, 0.002),
            _make_horizon_stats(20, 6, 0.333, -0.0045, -0.001),
        ],
    }
    cap_stats = {
        "large": _make_horizon_stats(10, 4, 0.750, 0.0245, 0.015),
        "mid":   _make_horizon_stats(10, 2, 0.500, 0.0110, 0.005),
        "small": _make_horizon_stats(10, 3, 0.333, -0.0095, -0.004),
    }
    records = [
        _make_signal_record("PLTR",  "HIGH BUY",    0.92, market="USA",    r10=0.054, alpha10=0.041),
        _make_signal_record("SAP.DE","HIGH BUY",    0.78, market="EUROPE", r10=0.012, alpha10=0.009),
        _make_signal_record("AMD",   "TACTICAL BUY",0.65, market="USA",    r10=-0.021, alpha10=-0.030),
    ]
    worst = [records[2]]
    return badge_stats, cap_stats, records, worst


def test_payload_structure_has_embeds():
    badge_stats, cap_stats, records, worst = _make_full_payload_inputs()
    payload = build_backtest_discord_payload(
        report_date=date(2026, 5, 22),
        badge_stats=badge_stats, cap_stats=cap_stats,
        worst=worst, total_signals=12, records=records,
    )
    assert "embeds" in payload
    assert len(payload["embeds"]) == 1


def test_payload_title_contains_performance_log():
    badge_stats, cap_stats, records, worst = _make_full_payload_inputs()
    payload = build_backtest_discord_payload(
        report_date=date(2026, 5, 22),
        badge_stats=badge_stats, cap_stats=cap_stats,
        worst=worst, total_signals=12, records=records,
    )
    title = payload["embeds"][0]["title"]
    assert "STRATEGY PERFORMANCE LOG" in title


def test_payload_description_contains_n():
    badge_stats, cap_stats, records, worst = _make_full_payload_inputs()
    payload = build_backtest_discord_payload(
        report_date=date(2026, 5, 22),
        badge_stats=badge_stats, cap_stats=cap_stats,
        worst=worst, total_signals=12, records=records,
    )
    desc = payload["embeds"][0]["description"]
    assert "N=12" in desc


def test_payload_colour_green_when_wr_high():
    badge_stats, cap_stats, records, worst = _make_full_payload_inputs()
    payload = build_backtest_discord_payload(
        report_date=date(2026, 5, 22),
        badge_stats=badge_stats, cap_stats=cap_stats,
        worst=worst, total_signals=12, records=records,
    )
    # HIGH BUY win_rate=0.667 > 0.55 → green
    assert payload["embeds"][0]["color"] == 0x00FF00


def test_payload_colour_red_when_wr_low():
    badge_stats = {
        "HIGH BUY": [_make_horizon_stats(5, 3, 0.30, -0.02, -0.01),
                     _make_horizon_stats(10, 3, 0.30, -0.02, -0.01),
                     _make_horizon_stats(20, 3, 0.30, -0.02, -0.01)],
    }
    payload = build_backtest_discord_payload(
        report_date=date(2026, 5, 22),
        badge_stats=badge_stats, cap_stats={},
        worst=[], total_signals=3, records=[],
    )
    assert payload["embeds"][0]["color"] == 0xFF0000


def test_payload_fields_have_high_buy_block():
    badge_stats, cap_stats, records, worst = _make_full_payload_inputs()
    payload = build_backtest_discord_payload(
        report_date=date(2026, 5, 22),
        badge_stats=badge_stats, cap_stats=cap_stats,
        worst=worst, total_signals=12, records=records,
    )
    fields = payload["embeds"][0]["fields"]
    names = [f["name"] for f in fields]
    assert any("HIGH BUY" in n for n in names)
    assert any("TACTICAL BUY" in n for n in names)


def test_payload_field_contains_profit_factor():
    badge_stats, cap_stats, records, worst = _make_full_payload_inputs()
    payload = build_backtest_discord_payload(
        report_date=date(2026, 5, 22),
        badge_stats=badge_stats, cap_stats=cap_stats,
        worst=worst, total_signals=12, records=records,
    )
    fields = payload["embeds"][0]["fields"]
    hb_field = next(f for f in fields if "HIGH BUY" in f["name"])
    assert "PF" in hb_field["value"]


def test_payload_field_shows_alpha_decay():
    # TACTICAL BUY T+20 = -0.45% while T+10 = +1.10% → should flag Alpha Decay
    badge_stats, cap_stats, records, worst = _make_full_payload_inputs()
    payload = build_backtest_discord_payload(
        report_date=date(2026, 5, 22),
        badge_stats=badge_stats, cap_stats=cap_stats,
        worst=worst, total_signals=12, records=records,
    )
    fields = payload["embeds"][0]["fields"]
    tb_field = next(f for f in fields if "TACTICAL BUY" in f["name"])
    assert "Decay" in tb_field["value"]


def test_payload_cap_tier_block_present():
    badge_stats, cap_stats, records, worst = _make_full_payload_inputs()
    payload = build_backtest_discord_payload(
        report_date=date(2026, 5, 22),
        badge_stats=badge_stats, cap_stats=cap_stats,
        worst=worst, total_signals=12, records=records,
    )
    fields = payload["embeds"][0]["fields"]
    names = [f["name"] for f in fields]
    assert any("Capacity" in n or "Tier" in n for n in names)


def test_payload_edge_trajectory_shows_market_flag():
    badge_stats, cap_stats, records, worst = _make_full_payload_inputs()
    payload = build_backtest_discord_payload(
        report_date=date(2026, 5, 22),
        badge_stats=badge_stats, cap_stats=cap_stats,
        worst=worst, total_signals=12, records=records,
    )
    fields = payload["embeds"][0]["fields"]
    traj_field = next((f for f in fields if "Edge" in f["name"] or "Trajectory" in f["name"]), None)
    assert traj_field is not None
    assert "🇺🇸" in traj_field["value"] or "🇪🇺" in traj_field["value"]


def test_payload_detractors_shows_failure_label():
    badge_stats, cap_stats, records, worst = _make_full_payload_inputs()
    payload = build_backtest_discord_payload(
        report_date=date(2026, 5, 22),
        badge_stats=badge_stats, cap_stats=cap_stats,
        worst=worst, total_signals=12, records=records,
    )
    fields = payload["embeds"][0]["fields"]
    risk_field = next((f for f in fields if "Risk" in f["name"] or "Detractor" in f["name"]), None)
    assert risk_field is not None
    # AMD worst trade has edgar=0.8 → "EDGAR Over-reliance"
    assert "EDGAR" in risk_field["value"] or "Reversal" in risk_field["value"]


def test_payload_all_field_values_within_1024_chars():
    badge_stats, cap_stats, records, worst = _make_full_payload_inputs()
    payload = build_backtest_discord_payload(
        report_date=date(2026, 5, 22),
        badge_stats=badge_stats, cap_stats=cap_stats,
        worst=worst, total_signals=12, records=records,
    )
    for field in payload["embeds"][0]["fields"]:
        assert len(field["value"]) <= 1024, f"Field '{field['name']}' exceeds 1024 chars"
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_backtest_discord.py -k "payload" -v
```

Expected: Multiple FAILED — `build_backtest_discord_payload` exists but produces old output that doesn't match new assertions.

- [ ] **Step 3: Replace `build_backtest_discord_payload` in `scripts/backtest_signals.py`**

Find the existing `build_backtest_discord_payload` function (starts around line 724 after our Task 2 insertions) and replace it entirely with:

```python
def build_backtest_discord_payload(
    report_date:   date,
    badge_stats:   Dict[str, List[HorizonStats]],
    cap_stats:     Dict[str, HorizonStats],
    worst:         List[SignalRecord],
    total_signals: int,
    records:       List[SignalRecord],
) -> Dict[str, Any]:
    """Hedge-fund-grade Discord embed: N visibility, alpha decay, capacity tiers."""

    badge_order = ["HIGH BUY", "TACTICAL BUY"]
    badge_emoji = {"HIGH BUY": "🟢", "TACTICAL BUY": "🟡"}

    # ── Colour: based on HIGH BUY overall win-rate ────────────────────────────
    hb_list   = badge_stats.get("HIGH BUY", [])
    hb_by_h   = {s.horizon: s for s in hb_list}
    hb_t10    = hb_by_h.get(10)
    overall_wr = hb_t10.win_rate if hb_t10 and hb_t10.count > 0 else 0.0
    if overall_wr >= 0.55:
        colour = 0x00FF00   # green
    elif overall_wr >= 0.45:
        colour = 0xFFA500   # orange
    else:
        colour = 0xFF0000   # red

    priced = sum(1 for r in records if r.entry_price is not None)
    description = (
        f"**Dataset: N={total_signals} signals** | "
        f"{priced} priced · Forward-Walk Validation Ledger\n"
        f"────────────────────"
    )

    fields: List[Dict[str, Any]] = []

    # ── Section 1: Badge KPI table (inline, side-by-side) ────────────────────
    for badge in badge_order:
        stats_list = badge_stats.get(badge, [])
        stats_by_h = {s.horizon: s for s in stats_list}
        s5  = stats_by_h.get(5)
        s10 = stats_by_h.get(10)
        s20 = stats_by_h.get(20)
        n   = s10.count if s10 else 0
        emoji = badge_emoji.get(badge, "⚪")

        if n == 0:
            fields.append({
                "name":   f"{emoji} {badge}",
                "value":  "_No priced signals yet_",
                "inline": True,
            })
            continue

        wins  = round(s10.win_rate * n) if s10 else 0
        wr    = f"{s10.win_rate * 100:.1f}% ({wins}/{n})" if s10 else "—"
        pf    = f"{s10.profit_factor:.2f}" if s10 and s10.profit_factor != float("inf") else "∞"
        r5    = _pct_short(s5.avg_return)  if s5  else "—"
        r10   = _pct_short(s10.avg_return) if s10 else "—"
        r20_v = s20.avg_return if s20 else 0.0
        r20   = _pct_short(r20_v)          if s20 else "—"

        decay_tag = ""
        if s5 and s10 and s20:
            decay_tag = _alpha_decay_tag(s5.avg_return, s10.avg_return, s20.avg_return)

        alp   = _pct_short(s10.avg_alpha) if s10 else "—"

        value = _truncate_field(
            f"```\n"
            f"Metrics   | Value\n"
            f"----------|----------------\n"
            f"WR        | {wr}\n"
            f"PF        | {pf}\n"
            f"T+5 Avg   | {r5}\n"
            f"T+10 Avg  | {r10}\n"
            f"T+20 Avg  | {r20}{decay_tag}\n"
            f"α vs SPY  | {alp} (T+10)\n"
            f"```"
        )

        fields.append({
            "name":   f"{emoji} {badge}  (N={n})",
            "value":  value,
            "inline": True,
        })

    # ── Section 2: Capacity & Liquidity Tiers ────────────────────────────────
    tier_lines = ["────────────────────"]
    for key, label, flag in [
        ("large", "Large Caps", "🔵"),
        ("mid",   "Mid Caps",   "🟡"),
        ("small", "Small Caps", "🔴"),
    ]:
        s = cap_stats.get(key)
        if s and s.count > 0:
            wins_cap = round(s.win_rate * s.count)
            tier_lines.append(
                f"{flag} **{label}:** {_pct_short(s.avg_return)} avg "
                f"({wins_cap}/{s.count} WR {s.win_rate * 100:.1f}%) "
                f"| α {_pct_short(s.avg_alpha)}"
            )
        else:
            tier_lines.append(f"{flag} **{label}:** No data")

    fields.append({
        "name":   "📊 Capacity & Liquidity Tiers (T+10)",
        "value":  _truncate_field("\n".join(tier_lines)),
        "inline": False,
    })

    # ── Section 3: Edge Trajectory — recent BUY signals ──────────────────────
    buy_badges = {"HIGH BUY", "TACTICAL BUY"}
    recent = sorted(
        [r for r in records if r.badge in buy_badges and r.entry_price is not None],
        key=lambda r: r.signal_date,
        reverse=True,
    )[:8]

    if recent:
        traj_lines = ["────────────────────"]
        for r in recent:
            flag      = _market_flag(r.market)
            bar       = _score_bar(r.final_score, 10)
            t10       = r.returns.get(10)
            alpha10   = r.alpha.get(10)
            ret_str   = _pct_short(t10)    if t10    is not None else "pending"
            alpha_str = _pct_short(alpha10) if alpha10 is not None else "—"
            name_part = f"{r.company_name} ({r.ticker})" if r.company_name else r.ticker
            traj_lines.append(
                f"{flag} `{name_part}` ({bar}) → T+10: **{ret_str}** | α: {alpha_str}"
            )
        fields.append({
            "name":   "⏱ Edge Trajectory — Recent BUY Signals",
            "value":  _truncate_field("\n".join(traj_lines)),
            "inline": False,
        })

    # ── Section 4: Risk Attribution — Top 3 Detractors ───────────────────────
    if worst:
        risk_lines = ["────────────────────"]
        for r in worst:
            t10       = r.returns.get(10)
            ret_str   = _pct_short(t10) if t10 is not None else "—"
            pf_factor = _primary_failure_factor(r)
            pf_label  = _failure_label(pf_factor)
            pf_score  = r.factors.get(pf_factor, 0.0)
            pf_weight = r.weights.get(pf_factor, 0.0)
            flag      = _market_flag(r.market)
            risk_lines.append(
                f"{flag} `{r.ticker}` (T+10: {ret_str}) "
                f"· **{pf_label}** "
                f"(score={pf_score:.2f}, w={pf_weight:.0%})"
            )
        fields.append({
            "name":   "🚨 Risk Attribution — Top 3 Detractors",
            "value":  _truncate_field("\n".join(risk_lines)),
            "inline": False,
        })

    return {
        "embeds": [{
            "title":       "📊 STRATEGY PERFORMANCE LOG — ACTIVE EDGE TRACKING",
            "description": description,
            "color":       colour,
            "fields":      fields,
            "footer":      {
                "text": f"regime_trader · {report_date.isoformat()} · horizons: T+5 / T+10 / T+20"
            },
        }]
    }
```

- [ ] **Step 4: Run all payload tests**

```
pytest tests/test_backtest_discord.py -v
```

Expected: All tests PASS (both Task 1 + 2 + 3 tests).

- [ ] **Step 5: Run the backtest in dry-run to verify end-to-end**

```
python scripts/backtest_signals.py --dry-run --archive-dir logs/archive --verbose 2>&1 | tail -5
```

Expected: `dry-run: skipping Discord notification` in output, no exceptions.

- [ ] **Step 6: Commit**

```bash
git add scripts/backtest_signals.py tests/test_backtest_discord.py
git commit -m "feat(backtest): hedge-fund-grade Discord embed — N visibility, alpha decay, capacity tiers"
```

---

## Task 4: Push and verify

- [ ] **Step 1: Run full test suite**

```
pytest tests/ -q
```

Expected: All existing tests still pass + new test file passes.

- [ ] **Step 2: Push**

```bash
git push origin main
```

---

## Self-Review

### 1. Spec coverage

| Requirement | Task |
|---|---|
| Embed title "STRATEGY PERFORMANCE LOG — ACTIVE EDGE TRACKING" | Task 3 Step 3 |
| N={sample_size} in description | Task 3 Step 3 |
| Status color: green ≥55%, orange 45-55%, red <45% | Task 3 Step 3 |
| HIGH BUY inline field with WR, PF, T+5/10/20, Alpha | Task 3 Step 3 |
| TACTICAL BUY inline field same metrics | Task 3 Step 3 |
| Alpha Decay tag on T+20 when signal collapses | Task 3 Step 3 + Task 2 `_alpha_decay_tag` |
| Capacity & Liquidity Tiers block (Large/Mid/Small) | Task 3 Step 3 |
| Edge Trajectory up to 8 recent signals with market flag | Task 3 Step 3 |
| Market flag (🇺🇸/🇪🇺/🇯🇵) from `market` field | Task 1 + Task 2 `_market_flag` |
| Score bar in trajectory | Task 3 Step 3 (uses existing `_score_bar`) |
| Top 3 Detractors with failure label | Task 3 Step 3 + Task 2 `_failure_label` |
| 1024 char field limit enforced | Task 2 `_truncate_field` + Task 3 test |
| 4dp rounding, 2dp % display | `_pct_short` (existing) |

### 2. Placeholder scan
None found — all code blocks are complete and executable.

### 3. Type consistency
- `_market_flag(market: str) -> str` — called with `r.market` (str, default "USA") ✓
- `_failure_label(factor: str) -> str` — called with `_primary_failure_factor(r)` which returns str ✓
- `_alpha_decay_tag(r5: float, r10: float, r20: float) -> str` — called with `s.avg_return` (float) ✓
- `_truncate_field(text: str, limit: int) -> str` — called with string field values ✓
- `build_backtest_discord_payload` signature unchanged — `main()` call site needs no update ✓
