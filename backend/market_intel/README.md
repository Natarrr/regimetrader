# Market Intel — EDGAR-first pipeline

Authoritative insider/institutional intel for `regime_trader`, sourced directly
from SEC EDGAR (Form-4, 13F-HR) with FMP as a coverage fallback.

| Layer | Module | Responsibility |
|---|---|---|
| HTTP | `edgar_ingest.py` | Ticker→CIK map, submissions, rate-limited downloads, idempotent caching |
| Parse | `edgar_parse.py` | Form-4 + 13F XML → dict records (namespace-tolerant) |
| Schema | `normalizer.py` | Canonical `InsiderEvent` / `InstitutionHolding` dataclasses |
| Score | `scorer.py` | Role-weighted P/S aggregation → 0–1 score (Spence costly signaling) |
| Fallback | `fmp_fallback.py` | FMP `/v4/insider-trading` mapper (when EDGAR empty) |
| Adapter | `adapter.py` | `fetch_intel(ticker)` — public single-entry replacement for `_run_intel_fetch` |
| CLI | `run_pipeline.py` | Scheduled-run entry: `python -m backend.market_intel.run_pipeline …` |

## Installation

```powershell
# from repo root
.\.venv\Scripts\activate    # or: source .venv/bin/activate
pip install requests pytest
```

No new dependencies beyond `requests` (already present) and `pytest` for tests.

## Environment variables

| Var | Required? | Default | Purpose |
|---|---|---|---|
| `EDGAR_USER_AGENT` | recommended | `Nathan MarketIntel n.tardy@hotmail.fr` | SEC fair-access policy: identifies the requester. **Set this in production.** |
| `SEC_USER_AGENT`   | optional alias | — | Backward-compat alias for `EDGAR_USER_AGENT` |
| `FMP_API_KEY`      | optional | empty | Enables FMP fallback when EDGAR returns nothing |
| `MARKET_INTEL_DATA_DIR` | optional | `data/raw/edgar/` | Override raw-filing cache location |

Add to `.env`:
```
EDGAR_USER_AGENT=Your Name your@email.com
FMP_API_KEY=...
```

## Quick start

```powershell
# Single ticker (writes data/raw/edgar/AAPL/<accession>/...)
python -c "from backend.market_intel import fetch_intel; import json; print(json.dumps(fetch_intel('AAPL'), indent=2, default=str))"

# Top-50 batch (writes logs/form4_summary.csv + edgar_debug_summary.json + marketintel_events.json)
python -m backend.market_intel.run_pipeline --tickers-file backend/market_intel/top50.csv --limit-forms 5

# Verbose mode (debug-level: every HTTP call, parse outcome)
python -m backend.market_intel.run_pipeline --tickers AAPL --limit-forms 1 --verbose

# Quiet mode (production cron — warnings only)
python -m backend.market_intel.run_pipeline --tickers-file top50.csv --quiet

# Import results into SQLite (idempotent — duplicates skipped)
python -m backend.market_intel.import_to_sqlite \
    --csv logs/form4_summary.csv --db data/market_intel.db

# Tests (15 unit tests, all offline)
.\.venv\Scripts\python.exe -m pytest backend/tests/market_intel/ -v
```

## Output schema

`fetch_intel(ticker)` always returns:
```json
{
  "ticker": "AAPL",
  "source": "EDGAR",
  "presence": true,
  "is_authoritative": true,
  "activity_count": 3,
  "events": [
    {
      "type": "Form-4",
      "issuer_ticker": "AAPL",
      "reporting_person": "COOK TIMOTHY D",
      "reporting_role": "CEO",
      "transaction_date": "2026-05-03",
      "transaction_code": "P",
      "shares": 1000.0,
      "price": 150.25,
      "value": 150250.0,
      "acquired_disposed": "A",
      "filing_accession": "0000320193-26-000012",
      "is_amendment": false,
      "source": "EDGAR"
    }
  ],
  "score": 0.6234,
  "score_breakdown": {
    "score": 0.6234,
    "buy_value": 300500.0,
    "sell_value": 0.0,
    "net_value": 300500.0,
    "buy_count": 1, "sell_count": 0,
    "events_in_window": 1,
    "ceo_buy": true, "amendment_count": 0
  },
  "last_updated": "2026-05-04T20:15:00+00:00",
  "errors": []
}
```

`source ∈ {"EDGAR", "FMP", "NONE"}` and `is_authoritative` is `true` only when
EDGAR actually returned data — use that flag to distinguish regulator-grade
signal from third-party aggregate.

## Scheduling

### Windows Task Scheduler (recommended for local dev)

```powershell
$A = New-ScheduledTaskAction -Execute "powershell.exe" `
     -Argument "-NoProfile -ExecutionPolicy Bypass -File $PWD\scripts\run_market_intel.ps1"
$T1 = New-ScheduledTaskTrigger -Daily -At 6:30AM
$T2 = New-ScheduledTaskTrigger -Daily -At 2:00PM
$T3 = New-ScheduledTaskTrigger -Daily -At 10:00PM
$T4 = New-ScheduledTaskTrigger -Daily -At 2:00AM
Register-ScheduledTask -Action $A -Trigger $T1,$T2,$T3,$T4 -TaskName "MarketIntel"
```

### POSIX cron

```cron
30 10 * * 1-5 /path/to/regime_trader/scripts/run_market_intel.sh top50.csv  # pre-open 06:30 ET
0  18 * * 1-5 /path/to/regime_trader/scripts/run_market_intel.sh top50.csv  # mid-day 14:00 ET
0  2  * * 2-6 /path/to/regime_trader/scripts/run_market_intel.sh top50.csv  # post-close 22:00 ET
0  6  * * *   /path/to/regime_trader/scripts/run_market_intel.sh top50.csv  # nightly backfill 02:00 ET
```

The shell wrapper uses `flock` to prevent overlapping runs; the PowerShell
wrapper checks a 60-min-old lock file. Both write to `logs/market_intel_runner.log`.

### GitHub Actions

`.github/workflows/market_intel.yml` runs the test suite on every push and
schedules four daily fetches against the top-50. Set repo secrets
`SEC_USER_AGENT` and `FMP_API_KEY` before enabling.

## Integration patch — replacing `_run_intel_fetch`

This is the exact diff against the current `streamlit_app.py` (lines ~5500–5700).
Apply with care — review each hunk; `_EXTENDED_WEIGHTS` is unchanged on purpose.

```diff
--- a/streamlit_app.py
+++ b/streamlit_app.py
@@ def _run_intel_fetch(macro_score: float, log_dir: Path) -> List[Dict]:
     log_dir.mkdir(parents=True, exist_ok=True)

+    # ── EDGAR-first authoritative layer ─────────────────────────────────────
+    # Spec says: prefer EDGAR (Form-4 / 13F regulator filings) over FMP/yfinance.
+    # fetch_intel_universe runs in parallel; EDGAR rate-limit is enforced inside.
+    from backend.market_intel.adapter import fetch_intel_universe
+    edgar_rows = fetch_intel_universe(_SCAN_UNIVERSE, max_workers=4)
+    edgar_by_sym = {r["ticker"]: r for r in edgar_rows}
+    edgar_insider_score = {
+        t: r["score"] for t, r in edgar_by_sym.items()
+        if r["source"] == "EDGAR" and r["presence"]
+    }
+    edgar_insider_presence: Set[str] = {
+        t for t, r in edgar_by_sym.items() if r["source"] == "EDGAR"
+    }
+    # 13F parsing is implemented but not yet wired into per-symbol scoring;
+    # presence is already detectable via fetch_edgar_for_ticker(include_13f=True).
+
     # ── Fetch all sources ─────────────────────────────────────────────────────
     sentiment                                          = _fetch_sentiment_live()
     insider,        insider_presence,  insider_errs   = _fetch_insider_live()
@@         comp = _geo_weighted(sub, _EXTENDED_WEIGHTS)

-        # FMP is primary; yfinance is fallback. Composite sub-dict keeps granular
-        # keys so _EXTENDED_WEIGHTS applies unchanged to the geo-weighted score.
-        insider_score = fmp_insider.get(sym,       insider.get(sym,       _NEUTRAL_SCORE))
+        # EDGAR primary -> FMP -> yfinance fallback.  Composite sub-dict keeps
+        # granular keys so _EXTENDED_WEIGHTS applies unchanged.
+        insider_score = edgar_insider_score.get(
+            sym,
+            fmp_insider.get(sym, insider.get(sym, _NEUTRAL_SCORE)),
+        )
         inst_score    = fmp_institutional.get(sym, institutional.get(sym, _NEUTRAL_SCORE))
@@         presence_flags: Dict[str, bool] = {
-            "insider":         sym in insider_presence or sym in fmp_ins_presence,
-            "institutional":   sym in inst_presence    or sym in fmp_inst_presence,
+            "insider":           sym in edgar_insider_presence
+                                 or sym in insider_presence
+                                 or sym in fmp_ins_presence,
+            "insider_edgar":     sym in edgar_insider_presence,
+            "institutional":     sym in inst_presence or sym in fmp_inst_presence,
+            "institutional_edgar": False,   # wired in next iteration (13F parser ready)
             "news":            sym in news              or sym in finnhub_news,
             "sentiment":       sym in sentiment         or sym in stocktwits,
             "finnhub_analyst": sym in finnhub_analyst,
             "macro":           True,
         }
```

The patch:

1. **EDGAR layer added as priority cascade** — `edgar_insider_score` is consulted first; FMP falls back if EDGAR has no data; yfinance falls back if FMP has no data.
2. **Composite untouched** — `_EXTENDED_WEIGHTS` applies to the same `sub` dict; the score for a symbol comes from the highest-priority source available.
3. **Canonical presence augmented** — new keys `insider_edgar` and `institutional_edgar` track EDGAR-specific presence without breaking the existing `insider` / `institutional` flags.
4. **`intel_source_status.json` augmented** — `run_pipeline.py` writes `edgar_insider`, `fmp_insider_fallback`, and `_edgar_meta` keys; the existing UI badge reader (~line 5896) handles them via the dict-shape branch already in place.

Apply during canary on 10 tickers, monitor 48h, then expand to top-50.

## Acceptance criteria

| Test | How to verify |
|---|---|
| Parse 3 Form-4 variants | `pytest backend/tests/market_intel/test_parse_form4.py -v` |
| Adapter priority logic (EDGAR > FMP > NONE) | `pytest backend/tests/market_intel/test_adapter.py -v` |
| Top-50 nightly run < 30 min | Time `run_pipeline.py` against `top50.csv` |
| EDGAR coverage ≥ 60% of top-50 after 7 days | `jq '.edgar_present_count / .ticker_count' logs/edgar_debug_summary.json` |
| Idempotent re-runs | Re-run the pipeline; confirm no `filing_downloaded` log lines, only `filing_cached` |
| Rate limit safety | Check `logs/market_intel.log` for zero `429` events |

## Architecture notes

- **Idempotence**: filings are cached on disk by accession number. Re-running
  the pipeline never re-downloads. Force a refresh by deleting the
  `data/raw/edgar/<TICKER>/<ACCESSION>/` directory.
- **Rate limiting**: `edgar_ingest._rate_wait()` enforces a global ~7 req/s cap
  using a thread-safe lock on monotonic time; safe to scale workers.
- **Amendments**: `is_amendment=True` flagged on 4/A filings; counted in
  `score_breakdown.amendment_count` so downstream consumers can deduplicate.
- **13F**: parser is implemented but not yet wired into the adapter — add
  `parse_form13f_file` results to the events list when needed.

## Canary hardening (atomic writes, circuit breaker, alert escalation)

The canary pipeline is hardened against three failure modes:

1. **Partial-file races** — every write of `intel_source_status.json` and
   `metrics.json` goes through `utils.atomic_write.atomic_write_json` (temp file
   in same dir → `os.replace`). CI readers never see half-written JSON.
2. **EDGAR cascade failures** — `edgar_ingest` runs a tiny circuit breaker:
   after `N` final HTTP failures the breaker trips open and `fetch_edgar_for_ticker`
   short-circuits (no HTTP, no `get_cik`) until a cooldown elapses. State lives in
   `.monitoring/edgar_cb.json`.
3. **Slack alert fatigue** — `monitoring/check_metrics.py` tracks consecutive
   canary failures in `.monitoring/alert_state.json`. After
   `ALERT_ESCALATE_AFTER` consecutive failures, the next Slack alert is sent
   with `escalate=True` (loud `[ESCALATE]` banner). A single passing run wipes
   the streak.

### Canary environment variables

| Var | Default | Purpose |
|---|---|---|
| `EDGAR_CB_FAIL_THRESHOLD` | `5` | Failures before the breaker opens |
| `EDGAR_CB_COOLDOWN_MIN` | `15` | Minutes the breaker stays open |
| `ALERT_ESCALATE_AFTER` | `3` | Consecutive canary failures before Slack escalates |

### Inspect / rollback runtime state

```powershell
# Inspect current circuit breaker + alert streak
type .monitoring\edgar_cb.json
type .monitoring\alert_state.json

# Force-close the breaker (operator override)
del .monitoring\edgar_cb.json

# Reset the consecutive-failure streak
del .monitoring\alert_state.json
```

`.monitoring/` is gitignored — these files are runtime state, not config.
Deleting either file is safe: the next canary run rebuilds it from defaults.

### Test the hardening

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -q
# 35 passed (atomic write × 5, CB × 7, alert state × 7, evaluate × 8, exporter+slack × 8)
```

## Why EDGAR first?

Akerlof (2001 Nobel) — *The Market for Lemons*: insider disclosures exist
specifically to neutralise information asymmetry. Form-4 is the canonical
artefact. Third-party aggregators occasionally lag, mis-classify roles, or
omit non-US issuers. Going DIY against EDGAR removes the middleman and
reduces "is FMP behind?" debugging at 06:30 ET.
