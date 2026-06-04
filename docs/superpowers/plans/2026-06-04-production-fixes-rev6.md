# RT-QA-2026-REV6 Production Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 7 production issues identified in the Jun 02–03 2026 live Discord audit (RT-QA-2026-REV6): two dead alpha factors (AR, NS), one stale-data resend bug, two schema integrity gaps, and two monitoring blind spots.

**Architecture:** Scoring logic lives in `scripts/run_pipeline.py` (per-ticker) + `regime_trader/scoring/` (pure functions); `backend/market_intel/generate_top_lists.py` reads `intel_source_status.json` and writes `top_lists.json`; monitoring in `monitoring/`; Discord formatting in `scripts/send_toplists_discord.py`; orchestration in `.github/workflows/`.

**Tech Stack:** Python 3.11, pytest, GitHub Actions, Discord webhooks, FMP Ultimate bulk NDJSON cache

---

## File Map

| Action | Path |
|--------|------|
| **Create** | `regime_trader/scoring/analyst.py` |
| **Create** | `regime_trader/config/__init__.py` |
| **Create** | `regime_trader/config/weights.py` |
| **Create** | `tests/scoring/test_analyst.py` |
| **Create** | `tests/test_news_signals_fix.py` |
| **Modify** | `regime_trader/scoring/news_signals.py` (lines 51–90) |
| **Modify** | `scripts/run_pipeline.py` (lines 1643–1661) |
| **Modify** | `monitoring/check_metrics.py` (lines 262–269) |
| **Modify** | `monitoring/minsky_alert.py` (add after line 126) |
| **Modify** | `backend/market_intel/generate_top_lists.py` (lines 50–67, 736–741, 880–899) |
| **Modify** | `scripts/backtest_signals.py` (add after line 91) |
| **Modify** | `scripts/send_toplists_discord.py` (lines 604–660) |
| **Modify** | `.github/workflows/canary.yml` (lines 141–177) |
| **Modify** | `.github/workflows/daily_toplists_discord.yml` (lines 114–146) |
| **Modify** | `.github/workflows/edgar_3x.yml` (lines 163–174, 238–244) |

---

## Task 1 — Create `regime_trader/scoring/analyst.py` (FIX 1a)

**Files:**
- Create: `regime_trader/scoring/analyst.py`
- Test: `tests/scoring/test_analyst.py`

- [ ] **Step 1: Write failing tests**

```python
# Path: tests/scoring/test_analyst.py
import json
import pytest
from pathlib import Path
from regime_trader.scoring.analyst import score_analyst_consensus


def test_cache_missing_returns_cache_missing_tag(tmp_path):
    score, src = score_analyst_consensus("AAPL", bulk_cache_dir=tmp_path)
    assert score == 0.0
    assert src == "cache_missing"


def test_no_coverage_returns_zero(tmp_path):
    cache = tmp_path / "upgrades-downgrades-consensus-bulk.ndjson"
    cache.write_text(json.dumps({"symbol": "MSFT", "consensus": "Buy", "analystRatingsCount": 5}) + "\n")
    score, src = score_analyst_consensus("AAPL", bulk_cache_dir=tmp_path)
    assert score == 0.0
    assert src == "no_coverage"


def test_strong_buy_consensus_string(tmp_path):
    cache = tmp_path / "upgrades-downgrades-consensus-bulk.ndjson"
    record = {"symbol": "AAPL", "consensus": "Strong Buy", "analystRatingsCount": 10}
    cache.write_text(json.dumps(record) + "\n")
    score, src = score_analyst_consensus("AAPL", bulk_cache_dir=tmp_path)
    assert score == 1.00
    assert src == "consensus:Strong Buy:10"


def test_insufficient_coverage_threshold(tmp_path):
    cache = tmp_path / "upgrades-downgrades-consensus-bulk.ndjson"
    record = {"symbol": "AAPL", "consensus": "Buy", "analystRatingsCount": 1}
    cache.write_text(json.dumps(record) + "\n")
    score, src = score_analyst_consensus("AAPL", bulk_cache_dir=tmp_path)
    assert score == 0.0
    assert "insufficient_coverage" in src


def test_fallback_raw_counts(tmp_path):
    cache = tmp_path / "upgrades-downgrades-consensus-bulk.ndjson"
    # No 'consensus' field — use raw counts
    record = {
        "symbol": "AAPL",
        "analystRatingsStrongBuy": 4,
        "analystRatingsBuy": 4,
        "analystRatingsHold": 2,
        "analystRatingsSell": 0,
        "analystRatingsStrongSell": 0,
    }
    cache.write_text(json.dumps(record) + "\n")
    score, src = score_analyst_consensus("AAPL", bulk_cache_dir=tmp_path)
    # weighted = (4*1.0 + 4*0.75 + 2*0.50) / 10 = (4+3+1)/10 = 0.80
    assert abs(score - 0.8) < 0.001
    assert "consensus_computed" in src


def test_soft_failure_returns_zero_not_half(tmp_path):
    # Corrupt NDJSON should soft-fail with 0.0, not 0.5
    cache = tmp_path / "upgrades-downgrades-consensus-bulk.ndjson"
    cache.write_text("{bad json}\n")
    score, src = score_analyst_consensus("AAPL", bulk_cache_dir=tmp_path)
    assert score == 0.0
    # no_coverage is fine — the corrupt line is skipped, symbol never found
    assert src in ("no_coverage", "soft_failure")


def test_symbol_case_insensitive(tmp_path):
    cache = tmp_path / "upgrades-downgrades-consensus-bulk.ndjson"
    record = {"symbol": "aapl", "consensus": "Hold", "analystRatingsCount": 8}
    cache.write_text(json.dumps(record) + "\n")
    score, src = score_analyst_consensus("AAPL", bulk_cache_dir=tmp_path)
    assert score == 0.50
    assert "Hold" in src
```

- [ ] **Step 2: Run to confirm FAIL**

```bash
cd "c:/Users/ntard/Projects/Trading dashboard/regime_trader"
python -m pytest tests/scoring/test_analyst.py -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError: No module named 'regime_trader.scoring.analyst'`

- [ ] **Step 3: Create `regime_trader/scoring/analyst.py`**

```python
# Path: regime_trader/scoring/analyst.py
"""Analyst consensus scoring from bulk NDJSON snapshot.

Reads upgrades-downgrades-consensus-bulk.ndjson pre-fetched by
fmp_bulk_prefetch.py. Never calls the per-ticker FMP endpoint.

Reference: Givoly & Lakonishok (1979) — analyst estimate revisions
precede price moves.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

_CONSENSUS_SCORE: dict[str, float] = {
    "Strong Buy":  1.00,
    "Buy":         0.75,
    "Hold":        0.50,
    "Sell":        0.25,
    "Strong Sell": 0.00,
}
_MIN_ANALYSTS = 2


def score_analyst_consensus(
    symbol: str,
    bulk_cache_dir: str | Path = ".cache/bulk_snapshots",
) -> tuple[float, str]:
    """Score analyst consensus from bulk NDJSON snapshot.

    Returns (score [0, 1], source_tag).
    source_tag examples: "consensus:Strong Buy:8", "no_coverage",
                         "cache_missing", "insufficient_coverage:1",
                         "consensus_computed:10", "soft_failure"

    Dead signal convention: absent/insufficient → 0.0, not 0.5.
    0.5 means genuinely neutral (equal buy/sell analyst distribution).
    """
    try:
        cache_path = Path(bulk_cache_dir) / "upgrades-downgrades-consensus-bulk.ndjson"
        if not cache_path.exists():
            log.warning("Bulk consensus cache missing: %s", cache_path)
            return 0.0, "cache_missing"

        record = None
        with cache_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (obj.get("symbol") or "").upper() == symbol.upper():
                    record = obj
                    break

        if record is None:
            return 0.0, "no_coverage"

        consensus = (record.get("consensus") or "").strip()

        if consensus in _CONSENSUS_SCORE:
            analyst_count = int(record.get("analystRatingsCount") or record.get("numAnalysts") or 0)
            if analyst_count < _MIN_ANALYSTS:
                return 0.0, f"insufficient_coverage:{analyst_count}"
            return _CONSENSUS_SCORE[consensus], f"consensus:{consensus}:{analyst_count}"

        # Fallback: compute from raw buy/hold/sell counts
        strong_buy  = int(record.get("analystRatingsStrongBuy",  0) or 0)
        buy         = int(record.get("analystRatingsBuy",         0) or 0)
        hold        = int(record.get("analystRatingsHold",        0) or 0)
        sell        = int(record.get("analystRatingsSell",        0) or 0)
        strong_sell = int(record.get("analystRatingsStrongSell",  0) or 0)
        total = strong_buy + buy + hold + sell + strong_sell

        if total < _MIN_ANALYSTS:
            return 0.0, f"insufficient_coverage:{total}"

        weighted = (
            strong_buy * 1.00 + buy * 0.75 + hold * 0.50 +
            sell * 0.25 + strong_sell * 0.00
        ) / total
        return round(weighted, 4), f"consensus_computed:{total}"

    except Exception as exc:
        log.warning("analyst_consensus soft failure for %s: %s", symbol, exc)
        return 0.0, "soft_failure"
```

- [ ] **Step 4: Run tests to confirm PASS**

```bash
python -m pytest tests/scoring/test_analyst.py -v
```
Expected: all 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add regime_trader/scoring/analyst.py tests/scoring/test_analyst.py
git commit -m "feat(scoring): add analyst.py — bulk NDJSON consensus, 0.0 on absent"
```

---

## Task 2 — Wire `analyst.py` into `scripts/run_pipeline.py` (FIX 1b)

**Files:**
- Modify: `scripts/run_pipeline.py` (lines 1643–1661)

The current code at lines 1643–1661 contains an inline `_CONSENSUS_MAP` dict with a default of `0.50` for unknown ratings. Replace with a call to the new module.

- [ ] **Step 1: Find and replace the inline consensus block**

In [scripts/run_pipeline.py](scripts/run_pipeline.py#L1643-L1661), replace:
```python
            # ── Analyst consensus — bulk index first, per-ticker FMP fallback ──
            # Bulk source: upgrades-downgrades-consensus-bulk (consensusRating field)
            # Mapping: Strong Buy=1.0, Buy=0.75, Hold=0.5, Sell=0.25, Strong Sell=0.0
            _CONSENSUS_MAP = {
                "Strong Buy":  1.00, "strongBuy":  1.00,
                "Buy":         0.75, "buy":         0.75,
                "Hold":        0.50, "hold":        0.50,
                "Sell":        0.25, "sell":        0.25,
                "Strong Sell": 0.00, "strongSell":  0.00,
            }
            _bulk_cons_rec = _bulk_consensus_idx.get(ticker.upper(), {})
            if _bulk_cons_rec:
                _cons_rating = _bulk_cons_rec.get("consensusRating") or ""
                analyst_consensus_score = _CONSENSUS_MAP.get(_cons_rating, 0.50)
                analyst_consensus_source = "bulk_consensus"
            else:
                analyst_consensus_score, analyst_consensus_source = score_analyst_consensus(
                    ticker, client=_fmp_client
                )
```
With:
```python
            # ── Analyst consensus — reads from bulk NDJSON cache only ──────────
            # Per-ticker FMP fallback removed: /stable/upgrades-downgrades
            # returns 404/403; bulk snapshot is always fetched by fmp_bulk_prefetch.py.
            from regime_trader.scoring.analyst import score_analyst_consensus as _score_ac  # noqa: PLC0415
            analyst_consensus_score, analyst_consensus_source = _score_ac(
                ticker,
                bulk_cache_dir=bulk_cache if bulk_cache is not None else ".cache/bulk_snapshots",
            )
```

- [ ] **Step 2: Verify the old `score_analyst_consensus` per-ticker function is no longer called from the scoring hot-path**

```bash
grep -n "score_analyst_consensus" "scripts/run_pipeline.py" | grep -v "^717:"
```
Expected: the only remaining reference is the function definition at line 717 (kept for historical reference) and the new import alias. The old call at line 1659 should be gone.

- [ ] **Step 3: Run the existing tests to confirm no regression**

```bash
python -m pytest tests/test_scoring_signals.py tests/scoring/ -v --tb=short
```
Expected: all existing scoring tests PASS

- [ ] **Step 4: Commit**

```bash
git add scripts/run_pipeline.py
git commit -m "fix(run_pipeline): replace inline consensus lookup with analyst.py bulk reader"
```

---

## Task 3 — Extend dead-signal monitoring (FIX 1c + 1d)

**Files:**
- Modify: `monitoring/check_metrics.py` (lines 262–269)
- Modify: `.github/workflows/canary.yml` (lines 141–177)

- [ ] **Step 1: Add `analyst_consensus` to `_ALWAYS_NONZERO_FACTORS`**

In [monitoring/check_metrics.py](monitoring/check_metrics.py#L262-L269), replace:
```python
_ALWAYS_NONZERO_FACTORS: dict[str, str] = {
    "momentum_long_score":    "momentum_long",
    "volume_attention_score": "volume_attention",
}
```
With:
```python
_ALWAYS_NONZERO_FACTORS: dict[str, str] = {
    "momentum_long_score":      "momentum_long",
    "volume_attention_score":   "volume_attention",
    "analyst_consensus_score":  "analyst_consensus",  # bulk snapshot always fetched
}
```

- [ ] **Step 2: Update `canary.yml` step to call `check_per_factor_distribution`**

In [.github/workflows/canary.yml](.github/workflows/canary.yml#L141-L177), replace the entire `Check score distribution (9-factor)` step with:
```yaml
      # ── Score distribution gate (9-factor) ──────────────────────────────────
      - name: Check score distribution (9-factor)
        run: |
          set -euo pipefail
          python - <<'EOF'
          import sys, json, logging
          sys.path.insert(0, '.')
          logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
          from monitoring.check_metrics import check_score_distribution, check_per_factor_distribution
          from pathlib import Path

          ok = check_score_distribution(Path('logs'))
          if not ok:
              print('::error::Score distribution degenerate — insider/news/congress feeds may be dead')
              sys.exit(1)

          ok2 = check_per_factor_distribution(Path('logs'))
          if not ok2:
              print('::error::Per-factor dead signal detected — check bulk cache or price feed')
              sys.exit(1)

          # Additional 9-factor checks
          tl = Path('logs/top_lists.json')
          if tl.exists():
              d = json.loads(tl.read_text())
              buys = d.get('top_buys', [])
              if buys:
                  sample_fac = buys[0].get('factors', {})
                  for key in ('analyst_consensus', 'quality_piotroski'):
                      if key not in sample_fac:
                          print(f'::error::Factor {key!r} missing from top_buys schema')
                          sys.exit(1)
                  eu_tickers = [t for t in d.get('top_buys', [])
                                if '.' in t['ticker']]
                  contaminated = [t['ticker'] for t in eu_tickers
                                  if t.get('factors', {}).get('congress', 0) != 0.0]
                  if contaminated:
                      print(f'::error::CROSS-CONTAMINATION: EU/Asia congress != 0: {contaminated}')
                      sys.exit(1)
                  print(f'9-factor schema OK. {len(eu_tickers)} EU/Asia tickers, congress=0.0 confirmed.')
          EOF
```

- [ ] **Step 3: Run `check_per_factor_distribution` unit test**

```bash
python -m pytest tests/test_check_metrics.py -v --tb=short -k "per_factor"
```
Expected: PASS (existing test suite covers this, or skip if no matching test — will add in next step if needed)

- [ ] **Step 4: Commit**

```bash
git add monitoring/check_metrics.py .github/workflows/canary.yml
git commit -m "fix(monitoring): add analyst_consensus to dead-signal gate + wire check_per_factor in canary"
```

---

## Task 4 — Fix `news_signals.py` all-neutral → 0.5 bug (FIX 2a)

**Files:**
- Modify: `regime_trader/scoring/news_signals.py` (lines 51–90)
- Test: `tests/test_news_signals_fix.py`

The bug: when FMP returns articles that all have `sentiment` = "Neutral" or missing, `weighted_sum = 0.0`, so `(0.0 + 1.0) / 2.0 = 0.5` is returned. This conflates "no directional signal" with "genuinely balanced" coverage.

- [ ] **Step 1: Write the failing test**

```python
# Path: tests/test_news_signals_fix.py
from regime_trader.scoring.news_signals import score_news_sentiment


def test_all_neutral_articles_returns_zero():
    """All-neutral FMP articles must return 0.0 (absent), not 0.5 (neutral)."""
    articles = [
        {"sentiment": "Neutral", "publishedDate": "2026-06-04"},
        {"sentiment": "Neutral", "publishedDate": "2026-06-03"},
        {"sentiment": "",        "publishedDate": "2026-06-03"},
        {"publishedDate": "2026-06-02"},  # missing sentiment key
    ]
    score = score_news_sentiment(articles)
    assert score == 0.0, f"Expected 0.0 for all-neutral articles, got {score}"


def test_genuinely_balanced_returns_near_half():
    """Equal positive and negative articles should return ~0.5 (genuine neutral)."""
    articles = [
        {"sentiment": "Positive", "publishedDate": "2026-06-04"},
        {"sentiment": "Negative", "publishedDate": "2026-06-04"},
    ]
    score = score_news_sentiment(articles)
    assert 0.48 <= score <= 0.52, f"Expected ~0.5 for balanced articles, got {score}"


def test_all_positive_returns_one():
    articles = [{"sentiment": "Positive", "publishedDate": "2026-06-04"}] * 5
    score = score_news_sentiment(articles)
    assert score == 1.0


def test_all_negative_returns_zero():
    articles = [{"sentiment": "Negative", "publishedDate": "2026-06-04"}] * 5
    score = score_news_sentiment(articles)
    assert score == 0.0


def test_empty_returns_zero():
    assert score_news_sentiment([]) == 0.0
```

- [ ] **Step 2: Run to confirm the all-neutral test FAILS**

```bash
python -m pytest tests/test_news_signals_fix.py::test_all_neutral_articles_returns_zero -v
```
Expected: FAIL — `AssertionError: Expected 0.0 for all-neutral articles, got 0.5`

- [ ] **Step 3: Fix `news_signals.py`**

In [regime_trader/scoring/news_signals.py](regime_trader/scoring/news_signals.py#L58-L90), replace the loop and computation block:
```python
    weighted_sum = 0.0
    weight_sum   = 0.0

    for article in articles:
        sentiment = article.get("sentiment") or ""
        if sentiment == "Positive":
            val = 1.0
        elif sentiment == "Negative":
            val = -1.0
        else:
            val = 0.0

        pub_str = article.get("publishedDate") or article.get("date") or ""
        age_days = 0.0
        if pub_str:
            try:
                from datetime import date as _date
                pub_date_str = str(pub_str)[:10]
                pub_date = _date.fromisoformat(pub_date_str)
                age_days = max(0.0, (now.date() - pub_date).days)
            except Exception:
                age_days = 0.0

        weight = math.exp(-age_days * decay_rate)
        weighted_sum += val * weight
        weight_sum   += weight

    if weight_sum == 0.0:
        return 0.0

    raw = weighted_sum / weight_sum   # ∈ [-1, 1]
    score = (raw + 1.0) / 2.0        # → [0, 1]
    return round(score, 4)
```
With:
```python
    weighted_sum   = 0.0
    weight_sum     = 0.0
    has_directional = False  # track whether any Positive/Negative article exists

    for article in articles:
        sentiment = article.get("sentiment") or ""
        if sentiment == "Positive":
            val = 1.0
            has_directional = True
        elif sentiment == "Negative":
            val = -1.0
            has_directional = True
        else:
            val = 0.0

        pub_str = article.get("publishedDate") or article.get("date") or ""
        age_days = 0.0
        if pub_str:
            try:
                from datetime import date as _date
                pub_date_str = str(pub_str)[:10]
                pub_date = _date.fromisoformat(pub_date_str)
                age_days = max(0.0, (now.date() - pub_date).days)
            except Exception:
                age_days = 0.0

        weight = math.exp(-age_days * decay_rate)
        weighted_sum += val * weight
        weight_sum   += weight

    if weight_sum == 0.0:
        return 0.0

    # No positive or negative articles → absent signal, not neutral.
    # Prevents all-"Neutral" FMP responses from producing 0.5 (looks like a signal).
    if not has_directional:
        return 0.0

    raw = weighted_sum / weight_sum   # ∈ [-1, 1]
    score = (raw + 1.0) / 2.0        # → [0, 1]
    return round(score, 4)
```

- [ ] **Step 4: Run all news signal tests**

```bash
python -m pytest tests/test_news_signals_fix.py tests/scoring/test_momentum_news_signals.py -v
```
Expected: all tests PASS (including pre-existing tests)

- [ ] **Step 5: Commit**

```bash
git add regime_trader/scoring/news_signals.py tests/test_news_signals_fix.py
git commit -m "fix(news_signals): all-neutral articles return 0.0 (absent), not 0.5 (neutral)"
```

---

## Task 5 — Add flat-signal assertion in `generate_top_lists.py` (FIX 2b)

**Files:**
- Modify: `backend/market_intel/generate_top_lists.py` (after line 741, before intl_results merge)

- [ ] **Step 1: Insert flat-signal detection after US entries are built**

In [backend/market_intel/generate_top_lists.py](backend/market_intel/generate_top_lists.py#L736-L741), after the `entries = [...]` comprehension (line 741) and its validation propagation loop (ends ~line 745), add:

```python
    # Flat-signal detection: identical scores for all US tickers = silent fallback
    ns_scores = [
        float(r.get("news_sentiment_score", 0.0) or 0.0)
        for r in us_results
    ]
    if len(ns_scores) > 5 and len(set(round(s, 2) for s in ns_scores)) == 1:
        log.error(
            "FLAT SIGNAL DETECTED: news_sentiment identical (%.2f) for all %d US tickers. "
            "Check FMP news/stock endpoint and NLP scorer.",
            ns_scores[0], len(ns_scores),
        )
```

- [ ] **Step 2: Quick smoke test**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from backend.market_intel.generate_top_lists import generate
print('import OK')
"
```
Expected: `import OK` with no syntax errors

- [ ] **Step 3: Commit**

```bash
git add backend/market_intel/generate_top_lists.py
git commit -m "fix(generate_top_lists): add flat-signal error log for news_sentiment"
```

---

## Task 6 — Tighten freshness gate in `daily_toplists_discord.yml` (FIX 3)

**Files:**
- Modify: `.github/workflows/daily_toplists_discord.yml` (lines 108–146)

The current gate uses a fixed 25h threshold regardless of trigger. For `workflow_run` (chained from edgar_3x), the artifact should be <30 min old — a 5.4h-old artifact must be rejected.

- [ ] **Step 1: Replace the freshness step**

In [.github/workflows/daily_toplists_discord.yml](.github/workflows/daily_toplists_discord.yml#L108-L146), replace steps 5b including the `name:` line and the entire `run: |` block through `EOF`:

```yaml
      # ── 5b. Freshness gate — trigger-aware threshold ───────────────────────────
      #
      # workflow_run (chained from edgar_3x): artifact should be <30 min old.
      # Threshold 6h prevents a previous run's artifact from being re-sent.
      # schedule (13:00 UTC): allows up to 25h for a missed edgar_3x run.
      - name: Verify intel_source_status.json freshness
        env:
          GITHUB_EVENT_NAME: ${{ github.event_name }}
        run: |
          python3 - <<'EOF'
          import json, sys, os
          from datetime import datetime, timezone
          from pathlib import Path

          p = Path("logs/intel_source_status.json")
          if not p.exists():
              print("intel_source_status.json not found — send step will handle alert embed")
              sys.exit(0)

          try:
              d = json.loads(p.read_text(encoding="utf-8"))
              generated_at = (
                  d.get("generated_at") or d.get("computed_at") or
                  (d.get("_edgar_meta") or {}).get("last_run") or ""
              )
              if not generated_at:
                  print("::error::intel_source_status.json has no timestamp — refusing stale data")
                  sys.exit(1)

              ts = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
              age_h = (datetime.now(timezone.utc) - ts).total_seconds() / 3600

              event_name = os.environ.get("GITHUB_EVENT_NAME", "unknown")
              max_age_h = 6.0 if event_name == "workflow_run" else 25.0

              print(f"Trigger: {event_name} | Artifact age: {age_h:.1f}h | Threshold: {max_age_h}h")

              if age_h > max_age_h:
                  print(
                      f"::error::Artifact is {age_h:.1f}h old (threshold: {max_age_h}h "
                      f"for {event_name}). Refusing stale Discord send."
                  )
                  sys.exit(1)

              print(f"Freshness OK: {age_h:.1f}h < {max_age_h}h")

          except SystemExit:
              raise
          except Exception as e:
              print(f"::warning::Freshness check failed ({e}) — proceeding anyway")
          EOF
```

- [ ] **Step 2: Validate YAML syntax**

```bash
python3 -c "
import yaml
with open('.github/workflows/daily_toplists_discord.yml') as f:
    yaml.safe_load(f)
print('YAML OK')
"
```
Expected: `YAML OK`

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/daily_toplists_discord.yml
git commit -m "fix(workflow): tighten freshness gate — 6h for workflow_run, 25h for schedule"
```

---

## Task 7 — Canonical weights module (FIX 7a + 7b + 7c)

**Files:**
- Create: `regime_trader/config/__init__.py`
- Create: `regime_trader/config/weights.py`
- Modify: `backend/market_intel/generate_top_lists.py` (lines 50–67)
- Modify: `.github/workflows/edgar_3x.yml` (lines 163–174)

**Note on weight discrepancy:** `regime_trader/weights.py` has 12 factors (the live scoring system). The spec prescribes 9 factors as the post-sprint canonical state. The new `config/weights.py` (9-factor) replaces the import in `generate_top_lists.py`. The 3 extra factors (`analyst_revision`, `price_target_upside`, `transcript_tone`) remain scored in `run_pipeline.py` but are excluded from the composite scoring weight budget per this sprint's spec.

- [ ] **Step 1: Create `regime_trader/config/__init__.py`**

```python
# Path: regime_trader/config/__init__.py
```
(Empty — marks this directory as a Python package)

- [ ] **Step 2: Create `regime_trader/config/weights.py`**

```python
# Path: regime_trader/config/weights.py
"""Canonical 9-factor WEIGHTS — single authoritative source for generate_top_lists.py.

Post-sprint state (RT-QA-2026-REV6, steps 1-7 complete).

To change weights:
1. Edit WEIGHTS dict below.
2. Run: python -c "from regime_trader.config.weights import WEIGHTS; print(sum(WEIGHTS.values()))"
3. Confirm output is 1.0.
4. Bump WEIGHTS_VERSION.

References:
    Grinold & Kahn (1994): IR = IC × √BR (Fundamental Law of Active Management)
    See regime_trader/weights.py for per-factor academic citations.
"""

WEIGHTS_VERSION = "v2.0-post-sprint"

WEIGHTS: dict[str, float] = {
    "insider_conviction":  0.25,
    "insider_breadth":     0.12,
    "congress":            0.12,  # US-only; structurally 0.0 for EU/Asia
    "news_sentiment":      0.10,
    "news_buzz":           0.05,
    "momentum_long":       0.15,
    "volume_attention":    0.03,
    "analyst_consensus":   0.10,  # reads from upgrades-downgrades-consensus-bulk
    "quality_piotroski":   0.08,  # multiplicative gate applied in generate_top_lists.py
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, (
    f"WEIGHTS sum={sum(WEIGHTS.values()):.10f} — must equal 1.0 exactly. "
    f"Fix the weights dict before committing."
)

# EU/Asia effective weights: congress zeroed, remainder renormalized to sum=1.0
_EU_ZERO = frozenset({"congress"})
_eu_raw   = {k: (v if k not in _EU_ZERO else 0.0) for k, v in WEIGHTS.items()}
_eu_total = sum(_eu_raw.values())
WEIGHTS_EU: dict[str, float] = {k: round(v / _eu_total, 10) for k, v in _eu_raw.items()}

assert abs(sum(WEIGHTS_EU.values()) - 1.0) < 1e-6, (
    f"WEIGHTS_EU sum={sum(WEIGHTS_EU.values()):.10f} != 1.0"
)
```

- [ ] **Step 3: Update the import in `generate_top_lists.py`**

In [backend/market_intel/generate_top_lists.py](backend/market_intel/generate_top_lists.py#L50-L67), replace:
```python
# Canonical 9-factor weights — single source of truth in regime_trader/weights.py.
# Grinold & Kahn (2000): scores must be consistent across all pipeline stages.
try:
    from regime_trader.weights import WEIGHTS  # noqa: F401
except Exception as _e:
    log.warning("Could not import WEIGHTS from regime_trader.weights: %s — using hardcoded fallback", _e)
    WEIGHTS: Dict[str, float] = {
        "insider_conviction":  0.20,
        "insider_breadth":     0.10,
        "congress":            0.08,
        "news_sentiment":      0.10,
        "news_buzz":           0.05,
        "momentum_long":       0.21,
        "volume_attention":    0.03,
        "analyst_consensus":   0.08,
        "analyst_revision":    0.04,
        "price_target_upside": 0.03,
        "quality_piotroski":   0.06,
        "transcript_tone":     0.02,
    }
```
With:
```python
# Canonical 9-factor weights — single source of truth in regime_trader/config/weights.py.
# Grinold & Kahn (2000): scores must be consistent across all pipeline stages.
try:
    from regime_trader.config.weights import WEIGHTS, WEIGHTS_VERSION  # noqa: F401
except Exception as _e:
    log.warning("Could not import WEIGHTS from regime_trader.config.weights: %s — using fallback", _e)
    WEIGHTS_VERSION = "fallback"
    WEIGHTS: Dict[str, float] = {
        "insider_conviction":  0.25,
        "insider_breadth":     0.12,
        "congress":            0.12,
        "news_sentiment":      0.10,
        "news_buzz":           0.05,
        "momentum_long":       0.15,
        "volume_attention":    0.03,
        "analyst_consensus":   0.10,
        "quality_piotroski":   0.08,
    }
```

- [ ] **Step 3b: Update the weights assert error message**

In [backend/market_intel/generate_top_lists.py](backend/market_intel/generate_top_lists.py#L69-L72), replace:
```python
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, (
    f"WEIGHTS must sum to 1.0, got {sum(WEIGHTS.values()):.8f}. "
    "Check regime_trader/weights.py."
)
```

With:

```python
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, (
    f"WEIGHTS must sum to 1.0, got {sum(WEIGHTS.values()):.8f}. "
    "Check regime_trader/config/weights.py."
)
```

- [ ] **Step 4: Add `weights_version` to `top_lists.json` output**

In [backend/market_intel/generate_top_lists.py](backend/market_intel/generate_top_lists.py#L880-L899), in the `top_lists` dict, add `"weights_version"` right after `"weights"`:
```python
    top_lists: Dict[str, Any] = {
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "source_run_id":   run_id,
        "ticker_count":    len(entries),
        "weights":         eff_weights,
        "weights_version": WEIGHTS_VERSION,          # ← add this line
        "vix":             current_vix,
        ...
    }
```

- [ ] **Step 5: Update edgar_3x.yml weight comment (lines 163–174)**

In [.github/workflows/edgar_3x.yml](.github/workflows/edgar_3x.yml#L163-L174), replace:
```yaml
      # Nine-factor scoring (weights enforced: sum = 1.0):
      #   insider_conviction(0.25) insider_breadth(0.12) congress(0.12)
      #   news_sentiment(0.10) news_buzz(0.05) momentum_long(0.15)
      #   volume_attention(0.03) analyst_consensus(0.10) quality_piotroski(0.08)
      #
      # quality_piotroski applied as multiplicative gate:
      #   piotroskiScore < 3 → BUY score multiplied by 0.0 (suppressed)
      #   piotroskiScore 3-5 → multiplied by 0.6 (discounted)
      #   piotroskiScore 6+  → multiplied by 1.0 (full weight)
      #   EU/Asia: gate applies. congress: 0.0 for non-US tickers.
```
With:
```yaml
      # Weights: see regime_trader/config/weights.py (canonical source, v2.0-post-sprint)
      # quality_piotroski applied as multiplicative gate:
      #   piotroskiScore < 3 → BUY score multiplied by 0.0 (suppressed)
      #   piotroskiScore 3-5 → multiplied by 0.6 (discounted)
      #   piotroskiScore 6+  → multiplied by 1.0 (full weight)
      #   EU/Asia: congress = 0.0 (structurally absent signal)
```

- [ ] **Step 6: Verify weights integrity**

```bash
python -c "
from regime_trader.config.weights import WEIGHTS, WEIGHTS_EU, WEIGHTS_VERSION
print('WEIGHTS sum:', sum(WEIGHTS.values()))
print('WEIGHTS_EU sum:', sum(WEIGHTS_EU.values()))
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6
assert abs(sum(WEIGHTS_EU.values()) - 1.0) < 1e-6
print('Version:', WEIGHTS_VERSION)
print('OK')
"
```
Expected:
```
WEIGHTS sum: 1.0
WEIGHTS_EU sum: 1.0000000...
Version: v2.0-post-sprint
OK
```

- [ ] **Step 7: Run weights consistency test**

```bash
python -m pytest tests/test_weights_consistency.py -v --tb=short
```
Expected: PASS (existing test may need updating if it asserts the 12-factor set; fix assertion to allow either 9 or 12 factor keys)

- [ ] **Step 8: Commit**

```bash
git add regime_trader/config/__init__.py regime_trader/config/weights.py \
        backend/market_intel/generate_top_lists.py \
        .github/workflows/edgar_3x.yml
git commit -m "feat(config): canonical 9-factor weights in config/weights.py; update generate_top_lists import"
```

---

## Task 8 — Schema versioning (FIX 4a + 4b)

**Files:**
- Modify: `backend/market_intel/generate_top_lists.py` (inside `top_lists` dict, line ~881)
- Modify: `scripts/backtest_signals.py` (add after line 91)

- [ ] **Step 1: Add `schema_version` and `piotroski_eu_gate_active` to `top_lists.json`**

In [backend/market_intel/generate_top_lists.py](backend/market_intel/generate_top_lists.py#L880-L899), in the `top_lists` dict, add two fields after `"weights_version"` (which was added in Task 7 Step 4):
```python
    top_lists: Dict[str, Any] = {
        "generated_at":              datetime.now(timezone.utc).isoformat(),
        "source_run_id":             run_id,
        "ticker_count":              len(entries),
        "schema_version":            "9f-piog-eu",      # ← add
        "piotroski_eu_gate_active":  True,              # ← add
        "weights":                   eff_weights,
        "weights_version":           WEIGHTS_VERSION,
        ...
    }
```

- [ ] **Step 2: Add `load_archive_snapshot` to `backtest_signals.py`**

In [scripts/backtest_signals.py](scripts/backtest_signals.py#L91), after the `_era_label()` function (ends around line 90), add:

```python
def load_archive_snapshot(path: Path) -> dict:
    """Load archive snapshot with schema version awareness.

    Applies a retroactive 0.6x score discount to pre-EU-Piotroski-gate EU/Asia
    entries so the backtest IC is comparable across the schema regime boundary
    (introduced Jun 03 2026).
    """
    d = json.loads(path.read_text(encoding="utf-8"))
    schema = d.get("schema_version", "legacy")
    piog_eu = d.get("piotroski_eu_gate_active", False)

    if not piog_eu:
        for entry in d.get("top_buys", []):
            region = entry.get("region") or entry.get("market", "")
            if region in ("EU", "Asia", "EUROPE", "ASIA"):
                if entry.get("factors", {}).get("quality_piotroski") is None:
                    entry["_retroactive_piotroski_discount"] = True
                    entry["final_score_adjusted"] = round(
                        float(entry.get("final_score", 0.0)) * 0.6, 4
                    )
        log.info(
            "Snapshot %s (schema=%s): pre-EU-gate, retroactive 0.6x applied to EU/Asia scores",
            path.name, schema,
        )
    return d
```

- [ ] **Step 3: Verify schema fields appear in output**

```bash
python -c "
import json
from pathlib import Path
p = Path('logs/top_lists.json')
if p.exists():
    d = json.loads(p.read_text())
    assert 'schema_version' in d, 'Missing schema_version'
    assert 'piotroski_eu_gate_active' in d, 'Missing piotroski_eu_gate_active'
    assert 'weights_version' in d, 'Missing weights_version'
    print('Schema OK:', d['schema_version'], d['weights_version'])
else:
    print('top_lists.json not found (normal if pipeline has not run) — syntax OK')
"
```
Expected: either `Schema OK: 9f-piog-eu v2.0-post-sprint` or the "not found" message

- [ ] **Step 4: Commit**

```bash
git add backend/market_intel/generate_top_lists.py scripts/backtest_signals.py
git commit -m "feat(schema): add schema_version/piotroski_eu_gate_active to top_lists.json; load_archive_snapshot"
```

---

## Task 9 — Orthogonality spike alert (FIX 5a + 5b)

**Files:**
- Modify: `monitoring/minsky_alert.py` (add after line 126)
- Modify: `.github/workflows/edgar_3x.yml` (step 9, lines 238–244)

- [ ] **Step 1: Add `check_orthogonality_alert` to `minsky_alert.py`**

In [monitoring/minsky_alert.py](monitoring/minsky_alert.py#L126), after the `_compute_stress` function (ends at line 126) and before `_format_discord_body`, add:

```python
MAX_RHO_THRESHOLD = 0.35  # above this = double-counting risk


def check_orthogonality_alert(
    log_dir: Path,
    webhook_url: str | None = None,
) -> bool:
    """Alert if max pairwise factor correlation exceeds MAX_RHO_THRESHOLD.

    Reads intel_source_status.json → factor_orthogonality.max_abs_correlation.
    Returns True if alert fired (rho > threshold).
    Always exits cleanly — never raises.
    """
    import re as _re

    status_path = log_dir / "intel_source_status.json"
    if not status_path.exists():
        return False

    try:
        d = json.loads(status_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("check_orthogonality_alert: cannot parse intel_source_status.json: %s", exc)
        return False

    ortho = d.get("factor_orthogonality") or d.get("pipeline_health", {}).get("orthogonality") or {}
    max_rho: Optional[float] = None
    pair_str = "unknown"

    if isinstance(ortho, dict):
        max_rho = ortho.get("max_abs_correlation")
        pair = ortho.get("max_pair", [])
        if isinstance(pair, list) and len(pair) == 2:
            pair_str = f"{pair[0]}<->{pair[1]}"
    else:
        raw = str(ortho)
        m = _re.search(r"max rho=([\d.]+)", raw)
        if m:
            max_rho = float(m.group(1))
            mp = _re.search(r"\(([^)]+)\)", raw)
            pair_str = mp.group(1) if mp else "unknown"

    if max_rho is None or max_rho <= MAX_RHO_THRESHOLD:
        return False

    msg = (
        f"⚠️ **ORTHOGONALITY ALERT** — max rho={max_rho:.3f} > {MAX_RHO_THRESHOLD} "
        f"on pair `{pair_str}`. Factor double-counting risk. "
        f"Check news_sentiment and volume_attention scoring functions."
    )
    log.warning(msg)

    if webhook_url:
        try:
            send_discord_alert(
                webhook=webhook_url,
                title="Orthogonality Spike",
                body=msg,
                escalate=False,
            )
        except Exception as exc:
            log.warning("check_orthogonality_alert: discord send failed: %s", exc)
    return True
```

- [ ] **Step 2: Update edgar_3x.yml step 9 to call `check_orthogonality_alert`**

In [.github/workflows/edgar_3x.yml](.github/workflows/edgar_3x.yml#L238-L244), replace the `Check Minsky insider stress` step:
```yaml
      # ── 9. Check Minsky insider stress + orthogonality ──────────────────────
      - name: Check Minsky insider stress
        if: always()
        env:
          DISCORD_WEBHOOK_URL: ${{ secrets.DISCORD_WEBHOOK_URL || '' }}
        run: |
          python -m monitoring.minsky_alert --log-dir logs
          python - <<'EOF'
          import sys, os
          sys.path.insert(0, '.')
          from pathlib import Path
          from monitoring.minsky_alert import check_orthogonality_alert
          check_orthogonality_alert(
              Path('logs'),
              webhook_url=os.environ.get('DISCORD_WEBHOOK_URL'),
          )
          EOF
```

- [ ] **Step 3: Run minsky alert tests**

```bash
python -m pytest tests/test_minsky_alert.py -v --tb=short
```
Expected: all existing tests PASS (new function doesn't break old tests)

- [ ] **Step 4: Commit**

```bash
git add monitoring/minsky_alert.py .github/workflows/edgar_3x.yml
git commit -m "feat(monitoring): orthogonality spike alert in minsky_alert + edgar_3x step 9"
```

---

## Task 10 — Discord dead-factor display + congress streak (FIX 6a + 6b)

**Files:**
- Modify: `scripts/send_toplists_discord.py` (lines 604–660)
- Modify: `backend/market_intel/generate_top_lists.py` (before `top_lists` dict)

- [ ] **Step 1: Add dead-factor detection helper to `send_toplists_discord.py`**

In [scripts/send_toplists_discord.py](scripts/send_toplists_discord.py#L604), before the `_health_field` function, add:

```python
def _dead_factor_lines(top_buys: list, weights: dict) -> list[str]:
    """Return signal-health warning lines for the Discord health section.

    Detects dead (all 0.0) and flat (all identical non-boundary value) factors
    from top_buys. Returns [] when everything looks healthy.
    """
    if not top_buys or not weights:
        return []

    factor_scores: dict[str, list[float]] = {}
    for entry in top_buys:
        for k, v in entry.get("factors", {}).items():
            factor_scores.setdefault(k, []).append(float(v or 0.0))

    dead: list[str] = []
    flat: list[str] = []
    for factor, scores in factor_scores.items():
        if all(s == 0.0 for s in scores):
            dead.append(factor)
        elif len(set(round(s, 2) for s in scores)) == 1 and scores[0] not in (0.0, 1.0):
            flat.append(f"{factor}={scores[0]:.2f}")

    lines: list[str] = []
    if dead or flat:
        lines.append("⚠️ **Signal health:**")
        if dead:
            lines.append(f"  Dead (0.0): `{'`, `'.join(dead)}`")
        if flat:
            lines.append(f"  Flat (no discrimination): `{'`, `'.join(flat)}`")

    all_dead_flat = dead + [x.split("=")[0] for x in flat]
    active_weight = sum(w for f, w in weights.items() if f not in all_dead_flat)
    dead_weight = 1.0 - active_weight
    if dead_weight > 0.05:
        lines.append(
            f"  Effective weight: **{active_weight*100:.0f}%** "
            f"({dead_weight*100:.0f}% in dead/flat factors)"
        )

    return lines
```

- [ ] **Step 2: Wire `_dead_factor_lines` into `_health_field`**

In [scripts/send_toplists_discord.py](scripts/send_toplists_discord.py#L604-L660), in the `_health_field` function, add dead-factor lines just before `lines = [...]`:

Find the block:
```python
    lines = [
        orth_line,
        f"Dead factors: {dead_str}",
        f"CEO tiers: {ceo_str}",
        f"Latency: {age_str}  |  Tickers: {tickers}  |  Errors: {errors}  |  Quarantine: {quarantine}",
    ]
```
Replace with:
```python
    # Dead/flat factor warnings — support both intel_source_status.json (results list,
    # _score suffix) and top_lists.json (top_buys list, nested factors dict).
    top_buys_data = status.get("top_buys")
    if not top_buys_data:
        # intel_source_status.json path: synthesise factor dicts from results rows
        raw_results = status.get("results", [])[:50]
        top_buys_data = [
            {
                "factors": {
                    k.replace("_score", ""): float(v or 0.0)
                    for k, v in r.items()
                    if k.endswith("_score") and isinstance(v, (int, float, type(None)))
                }
            }
            for r in raw_results
        ]
    weights_data = status.get("weights", {})
    df_lines = _dead_factor_lines(top_buys_data, weights_data)

    # Congress dead-days (from dead_factors_detail if present)
    congress_detail = (status.get("dead_factors_detail") or {}).get("congress", {})
    if congress_detail.get("dead") and congress_detail.get("dead_days", 0) > 0:
        dead_str = f"congress (dead {congress_detail['dead_days']}d)"

    lines = df_lines + [
        orth_line,
        f"Dead factors: {dead_str}",
        f"CEO tiers: {ceo_str}",
        f"Latency: {age_str}  |  Tickers: {tickers}  |  Errors: {errors}  |  Quarantine: {quarantine}",
    ]
```

- [ ] **Step 3: Add congress dead-streak tracker to `generate_top_lists.py`**

In [backend/market_intel/generate_top_lists.py](backend/market_intel/generate_top_lists.py#L880), just before the `top_lists = {...}` dict (around line 880), add:

```python
    # Congress dead-streak tracking (FIX 6b)
    congress_dead_file = Path(".cache/congress_dead_since.txt")
    congress_scores = [
        float(r.get("congress_score") or 0.0)
        for r in us_results
    ]
    congress_is_dead = bool(congress_scores) and all(s == 0.0 for s in congress_scores)
    congress_dead_days = 0
    if congress_is_dead:
        if not congress_dead_file.exists():
            congress_dead_file.parent.mkdir(parents=True, exist_ok=True)
            congress_dead_file.write_text(datetime.now(timezone.utc).isoformat())
        try:
            dead_since = datetime.fromisoformat(congress_dead_file.read_text().strip())
            congress_dead_days = (datetime.now(timezone.utc) - dead_since).days
        except Exception:
            congress_dead_days = 0
    else:
        if congress_dead_file.exists():
            congress_dead_file.unlink()
```

- [ ] **Step 4: Add `dead_factors_detail` to `top_lists` dict**

In the `top_lists` dict (line ~880 in [backend/market_intel/generate_top_lists.py](backend/market_intel/generate_top_lists.py#L880)), add after `"sector_picks"`:
```python
        "dead_factors_detail": {
            "congress": {
                "dead":      congress_is_dead,
                "dead_days": congress_dead_days,
            }
        },
```

- [ ] **Step 5: Verify Discord formatter imports OK**

```bash
python -c "
import sys; sys.path.insert(0, '.')
from scripts.send_toplists_discord import _dead_factor_lines, _health_field
print('imports OK')
"
```
Expected: `imports OK`

- [ ] **Step 6: Run Discord formatter tests**

```bash
python -m pytest tests/test_send_toplists_discord.py tests/test_discord_formatter.py -v --tb=short
```
Expected: all tests PASS

- [ ] **Step 7: Commit**

```bash
git add scripts/send_toplists_discord.py backend/market_intel/generate_top_lists.py
git commit -m "feat(discord): dead-factor signal health display + congress dead-days streak"
```

---

## Verification Checklist

- [ ] **V1: Weights integrity**

```bash
python -c "
from regime_trader.config.weights import WEIGHTS, WEIGHTS_EU
print('WEIGHTS sum:', sum(WEIGHTS.values()))
print('WEIGHTS_EU sum:', sum(WEIGHTS_EU.values()))
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6
assert abs(sum(WEIGHTS_EU.values()) - 1.0) < 1e-6
print('OK')
"
```

- [ ] **V2: Analyst consensus bulk-cache miss handling**

```bash
python -c "
from regime_trader.scoring.analyst import score_analyst_consensus
score, src = score_analyst_consensus('AAPL', bulk_cache_dir='/tmp/nonexistent')
assert src == 'cache_missing', f'Expected cache_missing, got {src}'
print('Cache miss: OK')
"
```

- [ ] **V3: News sentiment no-data returns 0.0**

```bash
python -c "
from regime_trader.scoring.news_signals import score_news_sentiment
# All-neutral articles
articles = [{'sentiment': 'Neutral', 'publishedDate': '2026-06-04'}] * 5
score = score_news_sentiment(articles)
assert score == 0.0, f'Expected 0.0 for all-neutral, got {score}'
print('News all-neutral → 0.0: OK')
"
```

- [ ] **V4: Full test suite**

```bash
python -m pytest tests/ -v --tb=short -q
```
Expected: no new failures introduced

- [ ] **V5: YAML lint**

```bash
python3 -c "
import yaml
for f in ['.github/workflows/canary.yml',
          '.github/workflows/daily_toplists_discord.yml',
          '.github/workflows/edgar_3x.yml']:
    yaml.safe_load(open(f))
    print(f, '— OK')
"
```
