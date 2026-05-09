# FMP Service

`regime_trader/services/fmp_service.py` — centralised, rate-limited, file-cached
FMP / yfinance data service.

## Why a service layer?

Before this service existed, every module that needed market-cap data made its own
direct HTTP call to FMP. With 50+ callers and a 60 req/min free-tier cap, this
caused silent throttling and stale data. The service centralises all calls behind:

1. A **token-bucket rate limiter** (configurable per process).
2. A **file cache** under `.cache/fmp/` with per-bucket TTLs.
3. A **shared `requests.Session`** with retry/backoff.

## Public API

```python
from regime_trader.services.fmp_service import default_fmp

# Single ticker profile (24 h cache)
profile = default_fmp.get_profile("AAPL")
# → {"symbol": "AAPL", "mktCap": 3e12, "price": 185.0, ...} | None

# Batch market caps (parallel, 24 h cache per symbol)
caps = default_fmp.get_profile_batch(["AAPL", "MSFT", "NVDA"])
# → {"AAPL": 3e12, "MSFT": 2e12, "NVDA": 2.5e12}

# Equity screener via yfinance (6 h cache)
picks = default_fmp.screener(cap_min=200_000_000, limit=50)
# → [{"sym": "NVDA", "volume_spike": 1.8, "price_change_pct": 3.2, ...}, ...]

# Insider buy signals via yfinance (12 h cache)
buys = default_fmp.insider_buys(lookback_days=90, limit=100)
# → [{"sym": "TSLA", "key_value_usd": 5e6, "normalized_pct_mcap": 0.06, ...}, ...]

# Institutional accumulation via yfinance 13F (6 h cache)
inst = default_fmp.get_institutional("AAPL")
# → {"sym": "AAPL", "accumulation_score": 0.42, "major_fund_count": 2, ...} | None
```

## Cache TTLs

| Bucket | TTL | Cache path |
| ------ | --- | ---------- |
| `profile` | 24 h | `.cache/fmp/profile/<SYM>.json` |
| `screener` | 6 h | `.cache/fmp/screener/screener_<params>.json` |
| `insider` | 12 h | `.cache/fmp/insider/insider_<params>.json` |
| `institutional` | 6 h | `.cache/fmp/institutional/<SYM>.json` |

TTLs are checked at read time against the file's `mtime`. To force a refresh,
delete the relevant cache file or use the `clear_cache()` helper (not yet
implemented — delete manually: `rm -rf .cache/fmp/`).

## Rate limiting

```
env var: FMP_RATE_LIMIT_PER_MINUTE   (default: 60)
```

The token bucket enforces a minimum inter-call interval of `60 / rate` seconds
**per process**. For multi-process deployments, reduce the rate proportionally.

Adjust for your FMP plan tier:

| FMP Plan | Safe limit | Env setting |
| -------- | ---------- | ----------- |
| Free | 60 req/min | (default) |
| Starter | 300 req/min | `FMP_RATE_LIMIT_PER_MINUTE=300` |
| Professional | 1500 req/min | `FMP_RATE_LIMIT_PER_MINUTE=1500` |

## API migration notes

FMP v3 and v4 endpoints were retired August 31, 2025. This service uses:
- `stable/profile` — for market-cap and company metadata
- **yfinance** — for screener, insider buys, and institutional accumulation

If FMP re-enables any v3/v4 endpoints, update the relevant method to call
`self._get_json(...)` instead of the yfinance path; the caching and rate
limiting are endpoint-agnostic.

## Using your own instance

```python
from regime_trader.services.fmp_service import FmpService

# High-throughput instance for batch jobs
batch_svc = FmpService(rate_per_minute=300)

# Test instance with custom cache
from pathlib import Path
test_svc = FmpService(rate_per_minute=999, cache_root=Path("/tmp/test_cache"))
```
