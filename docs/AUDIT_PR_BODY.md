# docs(audit): end-to-end pipeline validation 20260510

## Summary

Validates the complete EDGAR → scoring → FMP → ranking pipeline end-to-end. **All acceptance gates pass; no production code changes were made.**

This PR adds three audit-trail documents under `docs/`:

| File | Purpose |
|---|---|
| `docs/AUDIT_DIAGNOSTICS.md` | Full report: command outputs, math verification, observability inventory |
| `docs/AUDIT_GIT_COMMANDS.txt` | Git commands used to produce this PR |
| `docs/AUDIT_PR_BODY.md` | This document |

## Validation results

| Gate | Result |
|---|---|
| `yamllint -c .yamllint.yml .github/workflows/*.yml` | ✅ exit 0, clean |
| `python scripts/check_secrets.py` (no keys) | ✅ exit 0 (degraded mode) |
| `REQUIRE_SECRETS=true python scripts/check_secrets.py` | ✅ exit 1, lists 3 missing required |
| `pytest tests/test_streamlit_app_smoke.py tests/test_logging_cfg.py tests/test_check_secrets.py -q` | ✅ 38 passed |
| `pytest tests/test_fmp_service.py tests/test_edgar_service.py -q` | ✅ 44 passed (mocked) |
| `CI=true pytest tests/ backend/tests/ -q` | ✅ 760 passed |
| `python scripts/check_imports.py` | ✅ 11/11 imports OK |
| `python -m backend.market_intel.generate_top_lists --log-dir logs --force` | ✅ 10 tickers ranked |

### Math correctness

Live integration run produced:

```
JNJ = 0.30·0.500 + 0.25·0.500 + 0.20·0.500 + 0.15·0.500 + 0.10·0.650
    = 0.5150  ✓
```

Weights `{edgar:0.30, insider:0.25, congress:0.20, news:0.15, macro:0.10}` sum to **1.0000** as required by spec §4.

### Pipeline output (system output — NOT investment advice)

| # | Ticker | Final | Badge |
|---|---|---|---|
| 1 | JNJ   | 0.5150 | WATCHLIST |
| 1 | MSFT  | 0.5150 | WATCHLIST |
| 3 | GOOGL | 0.4899 | WATCHLIST |
| 4 | META  | 0.4772 | WATCHLIST |
| 5 | AAPL  | 0.4609 | WATCHLIST |

`WATCHLIST` is informational only; no buy/sell label is implied.

## Test plan

- [ ] CI run on this branch passes all 3 jobs (`sanity`, `smoke`, `test`)
- [ ] No regressions in `pytest tests/` or `pytest backend/tests/`
- [ ] `yamllint -c .yamllint.yml .github/workflows/*.yml` returns no output
- [ ] Reviewer reads `docs/AUDIT_DIAGNOSTICS.md` end-to-end

## Safety

- ✅ No secret values printed anywhere in logs/test output.
- ✅ Output is system telemetry only — no investment recommendation.
- ✅ All CI tests mock external HTTP via `tests/conftest.py` autouse fixture (active when `CI=true`).
- ✅ No changes to scoring logic, EDGAR/FMP services, or workflows.

## Rollback

If the audit branch causes any unexpected behavior:

```bash
git revert --no-edit <commit-sha>
git push
```

Since the change is documentation-only, rollback is risk-free.

## Recommendations (non-blocking, follow-up tickets)

1. Add structured counters (`edgar.calls_total`, `fmp.calls_cached`, etc.) for Grafana visibility.
2. Add `--durations=10` to CI smoke pytest invocation.
3. Install `gh` CLI in dev toolchain to enable workflow log retrieval in future audits.
