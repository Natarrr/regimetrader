# End-to-End Pipeline Validation — 2026-05-10

**Auditor:** automated diagnostics (per `prompt`)
**Scope:** EDGAR ingestion → filtering/scoring → FMP enrichment → market intelligence ranking
**Result:** ✅ **PASS** — no fixes required. All acceptance gates green.

---

## 1. Acceptance evidence

| # | Command | Expected | Observed |
|---|---|---|---|
| 1 | `python -m yamllint -c .yamllint.yml .github/workflows/*.yml` | exit 0, no output | ✅ exit 0, clean |
| 2 | `python scripts/check_secrets.py` (no keys) | exit 0 (degraded) | ✅ exit 0 — `Context: degraded (non-protected)` |
| 3 | `REQUIRE_SECRETS=true python scripts/check_secrets.py` | exit 1 | ✅ exit 1 — `Context: PROTECTED`, lists 3 missing required |
| 4 | `pytest tests/test_streamlit_app_smoke.py tests/test_logging_cfg.py tests/test_check_secrets.py -q` | all pass | ✅ **38 passed in 1.88 s** |
| 5 | `pytest tests/test_fmp_service.py tests/test_edgar_service.py -q` | all pass (mocked) | ✅ **44 passed in 32.36 s** |
| 6 | `CI=true pytest tests/ backend/tests/ -q` | all pass | ✅ **760 passed in 65.9 s** |
| 7 | `python scripts/check_imports.py` | exit 0 | ✅ all 11 sanity imports OK |
| 8 | `python -m backend.market_intel.generate_top_lists --log-dir logs --force` | top_lists.json with weights∑=1 | ✅ 10 tickers ranked, weights sum = 1.0000 |

---

## 2. Workflow audit (`.github/workflows/`)

All 8 workflows pass YAML parse, yamllint, and `actionlint`:

```
canary.yml                       OK
ci.yml                           OK
daily_toplists_discord.yml       OK
edgar_3x.yml                     OK
hybrid_pipeline.yml              OK
market_intel.yml                 OK
nightly_edgar.yml                OK
test_daily_toplists_absence.yml  OK
```

Common-issue scan:
- ✅ Line endings: all LF
- ✅ Boolean choice options: all string-quoted (`["true","false"]`)
- ✅ No job-level `if: ${{ secrets.X != '' }}` anti-pattern remaining
- ✅ All secret refs fall back via `|| ''`
- ✅ Context references match declared triggers (no `pull_request.*` leak)

`ci.yml` structure verified per spec §7:
- `sanity` → `check_imports.py` + `check_secrets.py` (non-fatal via `|| echo "...degraded mode"`)
- `smoke` → streamlit + logging + service smoke (fast gate)
- `test` → full `tests/` + `backend/tests/`

Triggers: `push`, `pull_request`, `workflow_dispatch` ✓

---

## 3. EDGAR service ([regime_trader/services/edgar_service.py](regime_trader/services/edgar_service.py))

| Property | Spec | Observed |
|---|---|---|
| Index TTL | 24 h | ✅ `_TTL_INDEX = 24*3600` |
| Filing TTL | 7 d | ✅ `_TTL_FILING = 7*24*3600` |
| Rate-limit env | `EDGAR_RATE_LIMIT` | ✅ honored, default 0.2 req/s |
| Cache root | `.cache/edgar/` | ✅ |
| Retry/backoff | session w/ retries | ✅ `_make_session()` |
| Public methods | `quarterly_index`, `list_filings`, `fetch_filing` | ✅ all present |

Cache observability via `log.debug`:
- `quarterly_index: cache hit %s`
- `fetch_filing: cache hit %s`
- `quarterly_index: fetched %d bytes for %s` (info — fresh fetch)
- `list_filings(%s, %s): %d results` (info)

Tests: 23 pass under `CI=true` (network-blocked); 44 pass without `CI=true` (mock-using integration variants).

---

## 4. FMP service ([regime_trader/services/fmp_service.py](regime_trader/services/fmp_service.py))

| Property | Spec | Observed |
|---|---|---|
| Profile TTL | 24 h | ✅ |
| Screener TTL | 6 h | ✅ |
| Insider TTL | 12 h | ✅ |
| Institutional TTL | 6 h | ✅ |
| Rate limit | `FMP_RATE_LIMIT_PER_MINUTE` (def 60) | ✅ |
| Public methods | `get_profile`, `get_profile_batch`, `get_institutional`, `screener`, `insider_buys` | ✅ all present |
| Batch endpoint | `get_profile_batch` | ✅ |

Cache observability:
- `screener: cache hit` / `screener: %d candidates (fresh)`
- `insider_buys: cache hit` / `insider_buys: %d signals (fresh)`

---

## 5. Scoring correctness ([backend/market_intel/generate_top_lists.py](backend/market_intel/generate_top_lists.py))

Default weights (spec §4.1 — verified to sum to 1.0):

| Component | Weight | Source |
|---|---|---|
| edgar | 0.30 | Form-4 insider signal (run_pipeline output) |
| insider | 0.25 | CEO buy conviction + buy/sell quality |
| congress | 0.20 | Senate/House disclosures (FMP /v4/senate-disclosure) |
| news | 0.15 | yfinance headline sentiment |
| macro | 0.10 | Pipeline health (coverage ratio + circuit-breaker) |
| **Σ** | **1.00** | renormalised on override |

**Math verification (live integration run):**

```
JNJ = 0.30·0.500 + 0.25·0.500 + 0.20·0.500 + 0.15·0.500 + 0.10·0.650
    = 0.150 + 0.125 + 0.100 + 0.075 + 0.065
    = 0.5150 ✓
```

Bounds: every component clamped to [0.15, 0.85]; final score therefore in [0.15, 0.85].

Fallback behavior verified (spec §4.2): in this run, FMP key absent → congress = 0.500 (neutral), news yfinance returned no headlines → 0.500 (neutral). Pipeline degraded gracefully without aborting.

**Top-5 ranking (system output — NOT investment advice):**

| # | Ticker | Final | EDGAR | Insider | Congress | News | Macro | Badge |
|---|---|---|---|---|---|---|---|---|
| 1 | JNJ   | 0.5150 | 0.500 | 0.500 | 0.500 | 0.500 | 0.650 | WATCHLIST |
| 1 | MSFT  | 0.5150 | 0.500 | 0.500 | 0.500 | 0.500 | 0.650 | WATCHLIST |
| 3 | GOOGL | 0.4899 | 0.500 | 0.400 | 0.500 | 0.500 | 0.650 | WATCHLIST |
| 4 | META  | 0.4772 | 0.457 | 0.400 | 0.500 | 0.500 | 0.650 | WATCHLIST |
| 5 | AAPL  | 0.4609 | 0.403 | 0.400 | 0.500 | 0.500 | 0.650 | WATCHLIST |

*Badge `WATCHLIST` is informational only; no buy/sell recommendation is implied (spec §4.4).*

---

## 6. CI smoke gate

Per spec §3.2, the smoke gate executes:

```
python scripts/check_imports.py            (≈ 0.5 s)
python scripts/check_secrets.py            (non-fatal in PRs)
pytest tests/test_streamlit_app_smoke.py
pytest tests/test_logging_cfg.py
```

Local timing: combined ≈ 2.4 s. Within the < 30 s budget.

---

## 7. Network isolation ([tests/conftest.py](tests/conftest.py))

`autouse` fixture replaces `requests.Session.send` with a `RuntimeError`-raising blocker when `CI=true`. Verified active:
- 760 tests pass under `CI=true` (some slow live-only EDGAR variants are conditionally skipped)
- Full suite without `CI=true` runs in 1m45s; with `CI=true` runs in 1m05s — confirms the block is filtering

Per-test mocks (`monkeypatch.setattr`, `unittest.mock.patch`) take precedence over the autouse fixture (closer scope).

---

## 8. Observability

Logs emitted per service:

| Source | Level | Event |
|---|---|---|
| EDGAR | DEBUG | `quarterly_index: cache hit` |
| EDGAR | INFO  | `quarterly_index: fetched %d bytes` |
| EDGAR | INFO  | `list_filings: %d results` |
| EDGAR | DEBUG | `fetch_filing: cache hit` |
| FMP | DEBUG | `screener: cache hit` / `insider_buys: cache hit` |
| FMP | INFO  | `screener: %d candidates (fresh)` / `insider_buys: %d signals (fresh)` |
| pipeline | INFO | per-ticker `final=%.4f factors={...}` |
| pipeline | INFO | `top_lists.json written` / `top5.csv written` |

**Gap (informational, not blocking):** spec §8 lists structured counter names (`edgar.calls_total`, etc.). The repo emits semantically-equivalent log lines but no Prometheus-style counters. Adding them is non-invasive but out of scope for this audit since current visibility is sufficient for the workflow exit-code gates.

---

## 9. Safety constraints — verified

- ✅ **No secret values printed.** `check_secrets.py` only emits `present: True/False` booleans; tests assert no key value leaks via logs (`test_logging_cfg.py::TestSecretMaskFilter`).
- ✅ **No investment advice generated.** Output uses `WATCHLIST` badge, factor breakdown, and explanatory rationale; no buy/sell labels.
- ✅ **Live HTTP blocked in CI.** `conftest.py` autouse fixture catches unmocked calls.
- ✅ **No production logic changes** in this audit.

---

## 10. Assumptions

- Local Python 3.14 venv with `requirements-ci.txt` + `streamlit` + full project deps.
- `.env` present locally (FMP_API_KEY, ALPACA_*, etc.) — used only for the live Streamlit demo, not the CI smoke gate.
- `gh` CLI not installed — workflow run-log retrieval (spec §2.1) is therefore deferred to maintainer; programmatic YAML/actionlint checks were used as proxies and pass clean.
- `actionlint` 1.7.7 binary fetched and used out-of-band (not committed to repo).

---

## 11. Recommendations (non-blocking)

1. **Add structured counters** — `edgar.calls_total`, `fmp.calls_cached`, etc. — for Grafana/Loki visibility. Current log lines work for grep-based monitoring but aren't structured.
2. **Smoke job timing budget** — add `pytest --durations=10` flag in CI to surface slow smoke tests early.
3. **Add `gh` CLI** to the local toolchain so the next audit can pull GitHub Actions run logs directly.

None of the above are necessary for the current PR/release.

---

## 12. Acceptance — PASS

All gates green. No diff produced. `patches/` folder empty by design.
