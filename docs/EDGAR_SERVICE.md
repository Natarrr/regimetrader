# EDGAR Service

`regime_trader/services/edgar_service.py` — SEC EDGAR bulk-index fetcher with
polite rate limiting and file cache.

## Why index-based, not scraping?

SEC's quarterly full-index files (`company.idx`) list every filing for every
company in a quarter in a single fixed-width text file. Downloading the index
once and filtering locally is:

- **Faster** than per-filing HTTP requests.
- **Polite** — one request retrieves 500k+ filing references.
- **Stable** — index files are published by SEC and never change once posted.
- **Cacheable** — a 24 h TTL covers all intra-day use cases.

## Public API

```python
from regime_trader.services.edgar_service import default_edgar

# Parse a full quarterly index (24 h cache)
rows = default_edgar.quarterly_index(year=2026, quarter=1)
# → [{"company_name": "APPLE INC", "form_type": "4", "cik": "320193",
#     "date_filed": "2026-03-01", "filename": "edgar/data/..."}, ...]

# Filter rows client-side:
apple_forms = [r for r in rows if "APPLE" in r["company_name"] and r["form_type"] == "4"]

# List filings for a specific CIK (via SEC Atom feed, 24 h cache)
filings = default_edgar.list_filings(cik="0000320193", form_type="4", max_results=20)
# → [{"cik": "320193", "form_type": "4", "date_filed": "2026-03-01",
#     "url": "https://www.sec.gov/.../form4.htm"}, ...]

# Fetch a specific filing document (7-day cache)
text = default_edgar.fetch_filing("https://www.sec.gov/Archives/edgar/data/320193/form4.htm")
# → "<html>...</html>" | None
```

## Cache TTLs

| Bucket | TTL | Cache path |
| ------ | --- | ---------- |
| `index` | 24 h | `.cache/edgar/index/<year>_Q<quarter>.txt` |
| `filings` | 24 h | `.cache/edgar/filings/<cik>_<form>_<n>.txt` |
| `docs` | 7 days | `.cache/edgar/docs/<safe_url>.txt` |

## Rate limiting

```
env var: EDGAR_RATE_LIMIT   (default: 0.2 req/sec)
```

SEC's `robots.txt` and developer guidance recommend a maximum of **10 req/sec**
for automated access. The default of 0.2 req/sec (one request every 5 seconds)
is deliberately conservative. Increase only if you have confirmed SEC approval
for higher rates.

| Use case | Safe limit | Env setting |
| -------- | ---------- | ----------- |
| Batch nightly job | 0.2 req/s | (default) |
| Interactive research | 1.0 req/s | `EDGAR_RATE_LIMIT=1.0` |
| Registered bulk downloader | up to 10 req/s | `EDGAR_RATE_LIMIT=10` |

## CIK lookup

SEC uses 10-digit zero-padded CIK numbers. Common ways to find a CIK:

```bash
# Via SEC EDGAR company search
curl "https://www.sec.gov/cgi-bin/browse-edgar?company=apple&CIK=&type=4&action=getcompany&output=atom"

# Via SEC's company facts endpoint (returns CIK for known tickers)
curl "https://data.sec.gov/submissions/CIK0000320193.json" | python -m json.tool | head -20
```

## EDGAR User-Agent requirement

SEC requires a descriptive `User-Agent` header including a contact email:

```
User-Agent: regime-trader-research contact@example.com
```

The service sets this header on all requests. **Update the email** in
`edgar_service.py → _HEADERS` to your team's actual contact address before
running at scale. Requests without a valid User-Agent may be blocked.

## Example: find all Form 4 (insider) filings for Apple in Q1 2026

```python
from regime_trader.services.edgar_service import default_edgar

# Option 1: quarterly index (efficient — one HTTP call for all companies)
rows = default_edgar.quarterly_index(2026, 1)
apple_4s = [r for r in rows if "APPLE" in r["company_name"] and r["form_type"] == "4"]

# Option 2: direct CIK lookup (SEC Atom feed)
filings = default_edgar.list_filings("0000320193", form_type="4", max_results=10)

for f in filings[:3]:
    doc = default_edgar.fetch_filing(f["url"])
    if doc:
        print(f["date_filed"], len(doc), "bytes")
```
