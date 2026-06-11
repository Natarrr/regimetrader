# Path: tools/build_universes.py
"""Build config/universe_eu.csv and config/universe_apac.csv (v3.0 step 6).

Sources, in order:
  1. stable/company-screener per exchange (probe-verified live 2026-06-10) —
     unknown exchange names return 0 rows harmlessly.
  2. config/ticker_registry.json lists (europe/europe_mid/asia/asia_mid) —
     static fallback, always available.

Every candidate is then validated:
  - get_region(ticker) must equal the target region (suffix purity at the
    door — wrong-exchange screener rows self-clean here);
  - 60-day median ADV (close × volume) above the regional liquidity floor:
    EU >= $5M; APAC >= $10M (large) / $3M (mid & small);
  - cap tier from market cap: large >= $10B, mid $2–10B, small < $2B.

Output schema: ticker,name,sector,cap_tier,exchange,adv_usd
Target >= 80 names per region so (sector × cap_tier) buckets usually clear
MIN_BUCKET_SIZE=5 (the cap_tier-only fallback chain handles the rest).

Run:
    python tools/build_universes.py [--region EU|APAC|both]
        [--out-dir config] [--max-candidates 250] [--no-screener]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config.weights import get_region  # noqa: E402

BASE = "https://financialmodelingprep.com/stable"
API_KEY = os.environ.get("FMP_API_KEY", "")

# Unknown names cost one empty response each — breadth comes from the registry.
EU_EXCHANGES = ["XETRA", "EURONEXT", "LSE", "SIX", "OSE", "STO", "CPH", "HEL"]
APAC_EXCHANGES = ["TSE", "HKSE", "KSC", "KOE", "NSE", "SES", "SET", "JKSE"]

ADV_FLOOR = {
    "EU": {"large": 5e6, "mid": 5e6, "small": 5e6},
    "APAC": {"large": 10e6, "mid": 3e6, "small": 3e6},
}
REGION_KEY = {"EU": "EU", "APAC": "ASIA"}            # get_region() vocabulary
REGISTRY_LISTS = {"EU": ("europe", "europe_mid"), "APAC": ("asia", "asia_mid")}

_session = requests.Session()
_session.headers["User-Agent"] = "regime-trader-universe-builder/1.0"


def _get(path: str, params: dict) -> list:
    try:
        r = _session.get(f"{BASE}/{path}", params={**params, "apikey": API_KEY},
                         timeout=15)
        if r.status_code != 200:
            return []
        body = r.json()
        return body if isinstance(body, list) else []
    except Exception as exc:
        print(f"  WARN {path}: {exc}")
        return []
    finally:
        time.sleep(0.06)  # stay far below FMP_MAX_RPS


def _screener_candidates(region: str) -> dict:
    """ticker → {name, sector, exchange} from company-screener."""
    out: dict = {}
    exchanges = EU_EXCHANGES if region == "EU" else APAC_EXCHANGES
    for exchange in exchanges:
        rows = _get("company-screener", {
            "exchange": exchange, "marketCapMoreThan": 2_000_000_000,
            "isEtf": "false", "isFund": "false", "limit": 100,
        })
        print(f"  screener {exchange}: {len(rows)} rows")
        for row in rows:
            sym = (row.get("symbol") or "").strip()
            if sym and sym not in out:
                out[sym] = {
                    "name": row.get("companyName") or "",
                    "sector": row.get("sector") or "Unknown",
                    "exchange": row.get("exchangeShortName")
                    or row.get("exchange") or exchange,
                }
    return out


def _registry_candidates(region: str, registry_path: Path) -> dict:
    out: dict = {}
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  WARN registry unreadable: {exc}")
        return out
    for key in REGISTRY_LISTS[region]:
        for item in registry.get(key, []) or []:
            sym = (item if isinstance(item, str)
                   else (item.get("ticker") or item.get("symbol") or "")).strip()
            if sym and sym not in out:
                out[sym] = {"name": "", "sector": "Unknown", "exchange": ""}
    return out


def _enrich(ticker: str, meta: dict) -> dict | None:
    """Quote (mcap/sector) + 60d median ADV. None when below floor/no data."""
    quote_rows = _get("quote", {"symbol": ticker})
    quote = quote_rows[0] if quote_rows else {}
    mcap = float(quote.get("marketCap") or 0.0)
    if mcap < 2e9:
        return None
    cap_tier = "large" if mcap >= 10e9 else "mid"

    prices = _get("historical-price-eod/full", {"symbol": ticker, "limit": 60})
    dollar_vol = [
        float(p.get("close") or 0.0) * float(p.get("volume") or 0.0)
        for p in prices
        if p.get("close") and p.get("volume")
    ]
    if len(dollar_vol) < 30:
        return None
    adv = statistics.median(dollar_vol)
    return {
        "ticker": ticker,
        "name": meta.get("name") or quote.get("name") or "",
        "sector": meta.get("sector") if meta.get("sector") not in (None, "", "Unknown")
        else (quote.get("sector") or "Unknown"),
        "cap_tier": cap_tier,
        "exchange": meta.get("exchange") or quote.get("exchange") or "",
        "adv_usd": int(adv),
    }


def _stratified(candidates: dict, cap: int) -> list:
    """Round-robin candidates by exchange suffix before applying the cap.

    A plain alphabetical cut is suffix-biased: digit-first tickers (KR 6-digit,
    HK 4-digit) monopolize the cap and whole markets (JP .T, IN .NS) vanish.
    Interleaving keeps every exchange represented at any cap size.
    """
    from collections import defaultdict
    groups: dict = defaultdict(list)
    for ticker in sorted(candidates):
        suffix = ticker.rsplit(".", 1)[-1] if "." in ticker else ""
        groups[suffix].append(ticker)
    queues = [groups[k] for k in sorted(groups)]
    out: list = []
    while len(out) < cap and any(queues):
        for q in queues:
            if q and len(out) < cap:
                out.append(q.pop(0))
    return out


def build_region(region: str, out_dir: Path, max_candidates: int,
                 use_screener: bool, registry_path: Path) -> Path:
    print(f"\n== {region} universe ==")
    candidates: dict = {}
    if use_screener:
        candidates.update(_screener_candidates(region))
    for sym, meta in _registry_candidates(region, registry_path).items():
        candidates.setdefault(sym, meta)

    target_region = REGION_KEY[region]
    in_region = {t: m for t, m in candidates.items()
                 if get_region(t) == target_region}
    dropped = len(candidates) - len(in_region)
    print(f"  candidates: {len(candidates)} ({dropped} dropped by "
          f"get_region != {target_region})")

    rows = []
    for ticker in _stratified(in_region, max_candidates):
        meta = in_region[ticker]
        enriched = _enrich(ticker, meta)
        if enriched is None:
            continue
        floor = ADV_FLOOR[region][enriched["cap_tier"]]
        if enriched["adv_usd"] < floor:
            continue
        rows.append(enriched)

    rows.sort(key=lambda r: r["adv_usd"], reverse=True)
    out_path = out_dir / ("universe_eu.csv" if region == "EU"
                          else "universe_apac.csv")
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["ticker", "name", "sector", "cap_tier",
                            "exchange", "adv_usd"])
        writer.writeheader()
        writer.writerows(rows)

    status = "OK" if len(rows) >= 80 else "BELOW TARGET (>=80)"
    print(f"  wrote {out_path}: {len(rows)} names — {status}")
    return out_path


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--region", choices=["EU", "APAC", "both"],
                        default="both")
    parser.add_argument("--out-dir", type=Path, default=Path("config"))
    parser.add_argument("--max-candidates", type=int, default=600)
    parser.add_argument("--no-screener", action="store_true",
                        help="registry seed only (offline-ish mode)")
    parser.add_argument("--registry", type=Path,
                        default=Path("config/ticker_registry.json"))
    args = parser.parse_args(argv)

    if not API_KEY:
        print("ERROR: FMP_API_KEY not set (add to .env or export)")
        return 1
    regions = ["EU", "APAC"] if args.region == "both" else [args.region]
    for region in regions:
        build_region(region, args.out_dir, args.max_candidates,
                     use_screener=not args.no_screener,
                     registry_path=args.registry)
    return 0


if __name__ == "__main__":
    sys.exit(main())
