# FMP Unification — Quiver & Finnhub Deprecation Design

**Date:** 2026-05-22  
**Author:** Lead Quantitative Core & Infrastructure Engineer  
**Status:** Approved for implementation

---

## Goal

Replace all Quiver Quantitative and Finnhub dependencies with a single FMP Ultimate client. Centralize all 5 Markowitz factors across USA, EUROPE, and ASIA markets under one authenticated session. The 5-factor scoring model is immutable:

```
final_score = 0.28·edgar + 0.23·insider + 0.22·congress + 0.15·news + 0.12·momentum
```

---

## Architecture

A new `FMPClient` service class centralizes all FMP interactions with per-bucket TTL caching, a configurable rate limiter (`FMP_MAX_RPS`), and plan-restriction detection. Existing scorer functions (`score_congress`, `score_insider_value`, `score_news_*`, `score_edgar`, `score_momentum`) are **untouched** — only the data fetch layer changes. `FMPFetcher` is made market-agnostic. `QuiverClient` and all Finnhub fetch functions are deleted.

---

## File Map

| File | Action | Purpose |
|---|---|---|
| `regime_trader/services/fmp_client.py` | **Create** | Unified FMP service: congress, insider, news, quote |
| `regime_trader/services/quiver_client.py` | **Delete** | Replaced entirely by FMPClient |
| `regime_trader/fetchers/fmp_fetcher.py` | **Modify** | Remove EUROPE hard-coding; make market-agnostic |
| `scripts/run_pipeline.py` | **Modify** | Replace Quiver/Finnhub fetch functions; wire FMPClient |
| `regime_trader/scanners/discovery_scanner.py` | **Modify** | Replace `_enrich_with_quiver` with no-op stub |
| `.github/workflows/nightly_edgar.yml` | **Modify** | Remove `QUIVER_API_KEY`, `FINNHUB_API_KEY` |
| `.github/workflows/edgar_3x.yml` | **Modify** | Remove `QUIVER_API_KEY`, `FINNHUB_API_KEY` |
| `.github/workflows/canary.yml` | **Modify** | Remove `QUIVER_API_KEY`, `FINNHUB_API_KEY` |
| `tests/test_quiver_client.py` | **Replace** → `tests/test_fmp_client.py` | Same structural coverage, FMP mocks |
| `tests/test_congress_fetcher.py` | **Modify** | Update S3→FMP fallback test (was S3→Quiver) |

---

## FMPClient Design

### Location
`regime_trader/services/fmp_client.py`

### TTL Cache (per bucket)

| Bucket | TTL | Endpoint |
|---|---|---|
| `congress` | 12h | `/api/v4/senate-trading`, `/api/v4/house-trades` |
| `insider` | 12h | `/api/v4/insider-trading` |
| `news` | 2h | `/api/v3/stock_news` |
| `quote` | 5 min | `/api/v3/quote` |

Cache stored under `.cache/fmp/<bucket>/<ticker>.json`. `_cache_read` accepts `bypass_cache: bool = False` — when `True`, skips disk and forces a live call. The 5-minute TTL on `quote` is sufficient for the 21h UTC nightly run; `bypass_cache` is available for edge cases.

### Rate Limiter

```python
_DEFAULT_MAX_RPS = 20          # safe for Ultimate (cap: 50 req/s)
_MAX_RPS = float(os.getenv("FMP_MAX_RPS", _DEFAULT_MAX_RPS))
_MIN_DELAY = 1.0 / _MAX_RPS   # 0.05s at 20 rps
```

Enforced via `time.monotonic()` tracking on the shared session. At 20 rps: 640 calls (160 tickers × 4 factors) ≈ 32s total. CI test environments mock the session — the rate limiter is never exercised in unit tests.

### Public API

```python
class FMPClient:
    def __init__(self, api_key: str | None = None, cache_root: Path | None = None)

    # Congress factor
    def get_congress_trades(self, ticker: str, lookback_days: int = 180) -> dict
    # Returns: {"purchases": int, "sales": int, "total": int, "recency_days": int}
    # Uses disclosureDate (not transactionDate) for recency_days calculation.
    # Returns {} for non-US tickers (senate/house endpoint returns [] → score_congress({}) = 0.0)

    # Insider factor
    def get_insider_purchases(self, ticker: str, lookback_days: int = 180) -> tuple[float, int]
    # Returns: (total_purchases_usd, days_since_most_recent)
    # Filters acquistionOrDisposition == "A" only.
    # total_purchases_usd = sum(securitiesTransacted * price) for each qualifying record.
    # Returns (0.0, 0) on empty or non-US ticker.

    # News factor
    def get_news_score(self, ticker: str) -> float
    # Returns: float in [0.0, 1.0]
    # Formula: 0.60 * (positive_count / total) + 0.40 * min(1.0, total / 50)
    # Falls back to _score_news_yfinance(ticker) if response is [] OR buzz_norm fails
    # due to zero article count. This covers EU/Asia tickers with thin FMP coverage.

    # Quote / momentum
    def get_quote(self, ticker: str, bypass_cache: bool = False) -> dict
    # Returns raw FMP quote dict: {price, marketCap, volume, avgVolume, eps, ...}
    # bypass_cache=True forces a live call regardless of TTL.
    # Accepts international suffixes natively (SAP.DE, 7203.T) on Ultimate plan.
```

---

## Factor Behavior Matrix

| Factor | FMP Endpoint | International Suffix | Non-US Behavior |
|---|---|---|---|
| Congress (0.22) | `/api/v4/senate-trading` + `/api/v4/house-trades` | Accepted (returns `[]`) | `{}` → `score_congress({})` = 0.0 |
| Insider (0.23) | `/api/v4/insider-trading` | Accepted (often empty) | `(0.0, 0)` → `score_insider_value` = 0.0 |
| News (0.15) | `/api/v3/stock_news` | Native (SAP.DE, etc.) | Empty → fallback to yfinance |
| Quote/Momentum (0.12) | `/api/v3/quote` | Native, no translation | Global close data direct |
| EDGAR (0.28) | SEC EDGAR direct | USA only | EU/Asia: `score_edgar(0)` = 0.0 (existing behavior) |

**Key invariant:** No scorer function receives `None` or crashes on a missing factor. Every fetch function returns a typed default (`{}`, `(0.0, 0)`, `0.0`, `[]`) on empty or failed responses.

---

## Congress Date Handling

FMP exposes both `transactionDate` (the actual trade date, non-public at the time) and `disclosureDate` (when the filing was published and the market gained knowledge).

**`recency_days` is computed from `disclosureDate`**, not `transactionDate`. This is correct for quantitative models: alpha decay starts from the moment the information is public, not from the secret transaction.

---

## Insider Parsing

```python
total_purchases_usd = sum(
    float(r["securitiesTransacted"]) * float(r["price"])
    for r in records
    if r.get("acquistionOrDisposition") == "A"
    and float(r.get("price") or 0) > 0
)
```

This produces the same `(total_purchases_usd, days_since_most_recent)` tuple that `score_insider_value()` already consumes — no scorer changes needed.

---

## News Scoring with Fallback

```python
def get_news_score(self, ticker: str) -> float:
    articles = self._get_news(ticker)        # cached 2h
    if not articles:
        return _score_news_yfinance(ticker)  # immediate fallback on empty []
    positive = sum(1 for a in articles if a.get("sentiment") == "Positive")
    total = len(articles)
    buzz_norm = min(1.0, total / 50.0)
    if total == 0 or buzz_norm == 0:
        return _score_news_yfinance(ticker)  # fallback on zero-volume days
    return round(0.60 * (positive / total) + 0.40 * buzz_norm, 4)
```

The double guard (empty array AND zero buzz) ensures EU/Asia tickers on low-volume days do not silently score 0.0 when yfinance can provide a better estimate.

---

## run_pipeline.py Changes (surgical)

### Renamed functions

| Old name | New name | Change |
|---|---|---|
| `fetch_quiver_insider_all` | `fetch_fmp_insider_all` | Calls `FMPClient.get_insider_purchases` per ticker |
| `score_news_finnhub` | `score_news_fmp` | Calls `FMPClient.get_news_score` |
| `fetch_congress_buys` | unchanged | Quiver fallback block replaced with FMPClient fallback |

### Import change
```python
# Remove:
from regime_trader.services.quiver_client import QuiverClient as _QuiverClient

# Add:
from regime_trader.services.fmp_client import FMPClient as _FMPClient
```

### Env var references removed
- `QUIVER_API_KEY` — all 4 references removed
- `FINNHUB_API_KEY` — all references removed (news and insider fallback)

---

## FMPFetcher Refactor (market-agnostic)

Current `FMPFetcher` hard-codes `market=MarketEnum.EUROPE`. After refactor:

```python
class FMPFetcher(BaseMarketFetcher):
    def __init__(self, api_key: str, market: MarketEnum) -> None:
        self._api_key = api_key
        self._market = market

    @property
    def market(self) -> MarketEnum:
        return self._market
```

The orchestrator in `run_pipeline.py` instantiates one `FMPFetcher` per market:

```python
FMPFetcher(api_key=fmp_key, market=MarketEnum.EUROPE)
FMPFetcher(api_key=fmp_key, market=MarketEnum.ASIA)
```

The FMP Ultimate plan accepts `SAP.DE`, `ASML.AS`, `7203.T` natively via `/api/v3/quote` — no suffix translation layer needed.

---

## Discovery Scanner Stub

```python
def _enrich_with_quiver(result_dicts: list[dict]) -> list[dict]:
    """Quiver deprecated — stub returns empty quiver dict for all results."""
    for r in result_dicts:
        r.setdefault("quiver", {})
    return result_dicts
```

The Streamlit Stock Picker UI already handles `quiver: {}` gracefully (it was the behavior when `QUIVER_API_KEY` was absent). Zero visual regression.

---

## Test Strategy

### `tests/test_fmp_client.py` (replaces `tests/test_quiver_client.py`)

Structural mirror of the old test file — same test classes, FMP mock responses:

| Class | Tests |
|---|---|
| `TestFMPClientCongress` | returns dict on success, caches, empty on error, empty on no key, disclosureDate used for recency |
| `TestFMPClientInsider` | returns tuple on success, filters A only, securitiesTransacted×price math, empty on error |
| `TestFMPClientNews` | returns float on success, fallback on empty [], fallback on zero-article day |
| `TestFMPClientQuote` | returns dict on success, bypass_cache forces live call, empty on error |
| `TestFMPClientCongressByTicker` | aggregates purchases, aggregates sales, non-US returns {}, recency uses disclosureDate |
| `TestEnrichWithQuiver` (stub) | returns quiver={} always, no network calls ever |
| `TestCIIsolation` | client constructible, rate limiter uses FMP_MAX_RPS env |

### `tests/test_congress_fetcher.py` — 1 test updated

`test_s3_403_falls_back_to_quiver` → `test_s3_403_falls_back_to_fmp`: patches `FMPClient.get_congress_trades` instead of `QuiverClient.congress_by_ticker`. All 11 other tests in this file are unchanged.

### All other 1054 tests — zero changes

They mock at the scorer level (`score_congress`, `score_insider_value`, etc.) — completely isolated from the data fetch layer being replaced.

---

## Environment Variable Changes

### Removed
```
QUIVER_API_KEY   — delete from .env, all workflow files, all code references
FINNHUB_API_KEY  — delete from .env, all workflow files, all code references
```

### Retained
```
FMP_API_KEY      — single source of truth for all API authentication
FMP_MAX_RPS      — optional, defaults to 20 (Ultimate plan: 50 req/s max)
```

---

## Verification

After implementation, the end-to-end check:

1. `pytest tests/` — 1085 tests pass (updated count: replaces 19 Quiver tests with ~20 FMP tests)
2. `python scripts/run_pipeline.py --verbose` — log shows `FMPClient insider: N tickers`, `FMPClient congress: N transactions`, `FMPClient news: fmp` (not `finnhub`/`yfinance`) for US tickers
3. `grep -r "QUIVER\|QuiverClient\|FINNHUB\|finnhub" scripts/ regime_trader/ tests/` — zero matches (except comments in test fixtures if any)
4. `git diff HEAD -- .env` — `QUIVER_API_KEY` and `FINNHUB_API_KEY` lines absent
