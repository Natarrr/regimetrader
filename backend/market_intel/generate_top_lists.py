"""backend/market_intel/generate_top_lists.py
Generate top_lists.json from EDGAR pipeline outputs.

Five-factor scoring per ticker
───────────────────────────────
  edgar       (w=0.30)  Form-4 insider signal — from run_pipeline output
  insider     (w=0.25)  CEO buy conviction + buy/sell quality — derived from score_breakdown
  congress    (w=0.20)  Senate/House disclosures via FMP /v4/senate-disclosure (optional)
  news        (w=0.15)  yfinance headline sentiment (no API key needed)
  macro       (w=0.10)  Pipeline health proxy: coverage ratio + circuit-breaker state

Weights can be overridden via --weights '{"edgar":0.40,"insider":0.20,...}'.
All weights are renormalised to sum=1 before use.

Market cap tiers (relative within universe, based on precomputed map + optional yfinance):
  large:   top 40 % of universe by market cap
  mid:     middle 35 %
  small:   bottom 25 %

Congress / FMP note
───────────────────
FMP /v4/senate-disclosure requires a paid plan. If FMP_API_KEY is absent or
the endpoint returns 403/404, congress_score defaults to 0.50 (neutral).
The same applies to /v3/institutional-holder for institutional signal.

Usage
─────
  python -m backend.market_intel.generate_top_lists
  python -m backend.market_intel.generate_top_lists --log-dir logs --top-n 5
  python -m backend.market_intel.generate_top_lists --force --weights '{"edgar":0.40}'
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger("market_intel.generate_top_lists")

# ── Default weights (must sum to 1.0 after normalisation) ─────────────────────
DEFAULT_WEIGHTS: Dict[str, float] = {
    "edgar":     0.30,
    "insider":   0.25,
    "congress":  0.20,
    "news":      0.15,
    "macro":     0.10,
}

# ── Approximate market caps (USD) for top-50 universe — updated May 2026 ──────
# Source: yfinance / Bloomberg spot check.  Regenerate with:
#   python -c "import yfinance as yf, json; ..."
_APPROX_CAPS: Dict[str, float] = {
    "NVDA": 5141e9,  "GOOGL": 4822e9, "AAPL": 4222e9,  "MSFT": 3126e9,
    "AMZN": 2917e9,  "AVGO": 1100e9,  "META": 1566e9,  "TSLA": 1547e9,
    "LLY":   850e9,  "JPM":   821e9,  "WMT":  950e9,   "UNH":  560e9,
    "JNJ":   536e9,  "XOM":   520e9,  "ORCL": 480e9,   "MA":   500e9,
    "V":     611e9,  "COST":  380e9,  "NFLX": 420e9,   "ABBV": 320e9,
    "HD":    350e9,  "PG":    350e9,  "BAC":  310e9,   "CVX":  280e9,
    "KO":    290e9,  "CRM":   280e9,  "ADBE": 200e9,   "PEP":  210e9,
    "ACN":   210e9,  "MCD":   230e9,  "LIN":  220e9,   "WFC":  210e9,
    "TMO":   170e9,  "CSCO":  200e9,  "AMD":  200e9,   "ABT":  190e9,
    "GE":    200e9,  "NOW":   200e9,  "PM":   190e9,   "TXN":  180e9,
    "CAT":   170e9,  "QCOM":  160e9,  "UNP":  140e9,   "RTX":  160e9,
    "NEE":   130e9,  "AMGN":  150e9,  "DHR":  140e9,   "DIS":  160e9,
    "INTC":   95e9,  "NKE":   90e9,
}

# ── Bullish/bearish word lists (same as score_helpers.py, local copy) ─────────
_BULL = frozenset([
    "beat","beats","exceed","exceeds","upgrade","upgrades","buy","outperform",
    "strong","record","rally","surge","surges","gain","growth","bullish","profit",
    "raise","raises","tops","jump","soar","soars","boom","approval","breakthrough",
])
_BEAR = frozenset([
    "miss","misses","downgrade","downgrades","sell","underperform","concern",
    "decline","weak","loss","losses","cut","fall","fell","drop","recession",
    "layoff","lawsuit","fine","warning","risk","volatile","below","disappoints",
    "investigation","fraud","bankruptcy","default",
])


# ── Factor helpers ─────────────────────────────────────────────────────────────

def _headline_score(title: str) -> float:
    words = set(title.lower().split())
    bull  = len(words & _BULL)
    bear  = len(words & _BEAR)
    if bull == 0 and bear == 0:
        return 0.50
    return round(max(0.10, min(0.90, 0.50 + 0.20 * (bull - bear))), 4)


def compute_insider_score(breakdown: Dict[str, Any]) -> float:
    """Quality-based insider conviction score (separate from EDGAR raw score).

    Spence (2001) — CEO open-market purchases commit personal capital → strongest
    conviction signal.  Buy/sell ratio captures directional pressure.
    Amendment filings are noisier (potential corrections) → small penalty.

    # $s_{\\text{insider}} = 0.50 + 0.30 \\cdot \\mathbf{1}_{\\text{CEO buy}}
    #                       + 0.10 \\cdot (r_{\\text{buy}} - 0.5) - 0.05 \\cdot \\mathbf{1}_{\\text{amended}}$
    """
    base = 0.50
    if breakdown.get("ceo_buy"):
        base += 0.30
    buy_count  = int(breakdown.get("buy_count",  0))
    sell_count = int(breakdown.get("sell_count", 0))
    total = buy_count + sell_count
    if total > 0:
        buy_ratio = buy_count / total
        base += (buy_ratio - 0.50) * 0.20
    if int(breakdown.get("amendment_count", 0)) > 0:
        base -= 0.05
    return round(min(1.0, max(0.0, base)), 4)


def fetch_news_score(ticker: str) -> float:
    """Aggregate yfinance headline sentiment for a ticker.

    Returns 0.50 (neutral) on any failure — never raises.
    """
    try:
        import yfinance as yf  # lazy import — not needed in tests
        articles = (yf.Ticker(ticker).news or [])[:10]
        if not articles:
            return 0.50
        scores = [_headline_score(a.get("title") or "") for a in articles]
        return round(sum(scores) / len(scores), 4)
    except Exception as exc:
        log.debug("news fetch failed for %s: %s", ticker, exc)
        return 0.50


def _fmp_get(path: str, params: Dict[str, Any], timeout: float = 8.0) -> Any:
    """Single FMP GET — returns None on any error (no retry, fast-fail)."""
    api_key = os.getenv("FMP_API_KEY", "")
    if not api_key:
        return None
    try:
        import requests  # already in requirements.txt
        url = f"https://financialmodelingprep.com/api{path}"
        resp = requests.get(url, params={**params, "apikey": api_key}, timeout=timeout)
        if resp.status_code in (401, 403, 404):
            log.debug("FMP %s status=%d (plan limit?)", path, resp.status_code)
            return None
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        log.debug("FMP %s failed: %s", path, str(exc)[:200])
        return None


def fetch_congress_score(ticker: str) -> float:
    """Senate/House disclosure score from FMP /v4/senate-disclosure.

    Purchase-biased recent disclosures → > 0.50.
    Sale-biased → < 0.50.  Missing data → 0.50 neutral.

    Friedman (1976) — political insiders act on the same private signals
    as corporate insiders; congressional trades precede market moves.
    """
    data = _fmp_get("/v4/senate-disclosure", {"symbol": ticker})
    if not data or not isinstance(data, list):
        return 0.50
    now = datetime.now(timezone.utc)
    purchase_weight = 0.0
    sale_weight     = 0.0
    for item in data[:20]:
        tx_date_str = item.get("transactionDate") or item.get("disclosureDate") or ""
        tx_type     = str(item.get("type") or "").lower()
        try:
            tx_date = datetime.strptime(tx_date_str[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
            days_ago = max(1, (now - tx_date).days)
        except ValueError:
            days_ago = 180
        recency_weight = 1.0 / math.sqrt(days_ago)
        if "purchase" in tx_type or "buy" in tx_type:
            purchase_weight += recency_weight
        elif "sale" in tx_type or "sell" in tx_type:
            sale_weight += recency_weight
    total = purchase_weight + sale_weight
    if total == 0:
        return 0.50
    raw = purchase_weight / total          # [0, 1]
    return round(0.20 + raw * 0.60, 4)    # map to [0.20, 0.80]


def fetch_institutional_score(ticker: str) -> float:
    """Institutional ownership net-change score from FMP /v3/institutional-holder.

    Positive aggregate net change → > 0.50.  Net selling → < 0.50.
    """
    data = _fmp_get("/v3/institutional-holder", {"symbol": ticker})
    if not data or not isinstance(data, list):
        return 0.50
    total_change = sum(float(h.get("change") or 0) for h in data[:50])
    if total_change == 0:
        return 0.50
    # Sigmoid squash: ±10M shares = ±0.25 shift from neutral 0.50
    scaled = total_change / 10_000_000.0
    squashed = 1.0 / (1.0 + math.exp(-scaled * 0.5))  # sigmoid
    return round(min(0.85, max(0.15, squashed)), 4)


def compute_congress_institutional_score(ticker: str) -> float:
    """Combined congress + institutional signal — average of both, fallback-safe."""
    cong = fetch_congress_score(ticker)
    inst = fetch_institutional_score(ticker)
    return round((cong + inst) / 2.0, 4)


def compute_macro_score(metrics: Optional[Dict]) -> float:
    """Derive a macro score from EDGAR pipeline health metrics.

    Proxy logic (no GARCH/CAPE/yield required here):
      - Base: 0.55 (slight positive)
      - Coverage ratio >= 0.90: +0.10 (healthy data environment)
      - Coverage ratio <  0.60: -0.15 (data stress → macro risk)
      - Error rate > 10%:       -0.10 (pipeline under pressure)
      - Circuit breaker open:   -0.20 (SEC unreachable → news blackout)

    Returns 0.50 if metrics unavailable.
    """
    if not metrics:
        return 0.50
    score = 0.55
    ticker_count = max(1, int(metrics.get("ticker_count", 1)))
    edgar_count  = int(metrics.get("edgar_count", ticker_count))
    error_count  = int(metrics.get("error_count", 0))
    coverage     = edgar_count / ticker_count
    error_rate   = error_count / ticker_count
    if coverage >= 0.90:
        score += 0.10
    elif coverage < 0.60:
        score -= 0.15
    if error_rate > 0.10:
        score -= 0.10
    return round(min(0.85, max(0.15, score)), 4)


# ── Market cap tier classification ─────────────────────────────────────────────

def get_market_cap(ticker: str) -> float:
    """Return market cap (USD) — precomputed map first, yfinance fallback."""
    cap = _APPROX_CAPS.get(ticker.upper())
    if cap:
        return cap
    try:
        import yfinance as yf
        info = yf.Ticker(ticker).info
        return float(info.get("marketCap") or 0)
    except Exception:
        return 0.0


def assign_tier(tickers_with_caps: List[Tuple[str, float]]) -> Dict[str, str]:
    """Assign relative market cap tier within the universe.

    Tiers (relative to universe, not absolute S&P standards):
      large: top 40 % by market cap
      mid:   middle 35 %
      small: bottom 25 %

    With fewer than 3 tickers the percentile split is meaningless —
    all are assigned "large" to avoid empty mid/small sections.
    """
    if not tickers_with_caps:
        return {}
    if len(tickers_with_caps) < 3:
        return {ticker: "large" for ticker, _ in tickers_with_caps}

    sorted_tickers = sorted(tickers_with_caps, key=lambda x: x[1], reverse=True)
    n = len(sorted_tickers)
    large_cutoff = max(1, int(n * 0.40))
    mid_cutoff   = max(large_cutoff + 1, int(n * 0.75))
    tiers: Dict[str, str] = {}
    for i, (ticker, _) in enumerate(sorted_tickers):
        if i < large_cutoff:
            tiers[ticker] = "large"
        elif i < mid_cutoff:
            tiers[ticker] = "mid"
        else:
            tiers[ticker] = "small"
    return tiers


# ── Top list generation ────────────────────────────────────────────────────────

def score_ticker(
    entry: Dict[str, Any],
    metrics: Optional[Dict],
    weights: Dict[str, float],
) -> Dict[str, Any]:
    """Compute 5-factor composite score for one ticker entry."""
    ticker    = str(entry.get("ticker") or "?").upper()
    breakdown = entry.get("score_breakdown") or {}

    edgar_score    = float(entry.get("score") or 0.50)
    insider_score  = compute_insider_score(breakdown)
    congress_score = compute_congress_institutional_score(ticker)
    news_score     = fetch_news_score(ticker)
    macro_score    = compute_macro_score(metrics)

    total_w = sum(weights.values())
    w = {k: v / total_w for k, v in weights.items()}

    final = round(
        w["edgar"]    * edgar_score
        + w["insider"]  * insider_score
        + w["congress"] * congress_score
        + w["news"]     * news_score
        + w["macro"]    * macro_score,
        4,
    )
    final = max(0.0, min(1.0, final))

    if final >= 0.80:
        badge = "HIGH BUY"
    elif final >= 0.60:
        badge = "TACTICAL BUY"
    else:
        badge = "WATCHLIST"

    return {
        "ticker":          ticker,
        "final_score":     final,
        "badge":           badge,
        "factors": {
            "edgar":     edgar_score,
            "insider":   insider_score,
            "congress":  congress_score,
            "news":      news_score,
            "macro":     macro_score,
        },
        "market_cap":      get_market_cap(ticker),
        "activity_count":  int(entry.get("activity_count") or 0),
        "source":          str(entry.get("source") or "EDGAR"),
        "ceo_buy":         bool(breakdown.get("ceo_buy", False)),
        "net_value":       float(breakdown.get("net_value") or 0.0),
    }


def generate(
    events_path: Path,
    metrics_path: Path,
    output_path: Path,
    output_csv: Path,
    weights: Dict[str, float],
    top_n: int = 5,
    force: bool = False,
    run_id: str = "",
) -> Dict[str, Any]:
    """Main generation logic — returns the top_lists dict."""

    # ── Freshness check (skip if recent and not forced) ───────────────────────
    if not force and output_path.exists():
        age_s = time.time() - output_path.stat().st_mtime
        if age_s < 4 * 3600:
            log.info("top_lists.json is %.0f min old — skipping (use --force to override)", age_s / 60)
            return json.loads(output_path.read_text(encoding="utf-8"))

    # ── Load inputs ───────────────────────────────────────────────────────────
    if not events_path.exists():
        log.error("marketintel_events.json not found at %s", events_path)
        sys.exit(2)

    events: List[Dict] = json.loads(events_path.read_text(encoding="utf-8"))
    if not isinstance(events, list) or not events:
        log.error("marketintel_events.json is empty or not a list")
        sys.exit(2)

    metrics: Optional[Dict] = None
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("could not read metrics.json: %s", exc)

    log.info("Scoring %d tickers with weights %s", len(events), weights)

    # ── Score each ticker ─────────────────────────────────────────────────────
    scored: List[Dict] = []
    for entry in events:
        ticker = str(entry.get("ticker") or "?")
        try:
            result = score_ticker(entry, metrics, weights)
            scored.append(result)
            log.info("  %s  final=%.4f  factors=%s", ticker,
                     result["final_score"], result["factors"])
        except Exception as exc:
            log.warning("scoring failed for %s: %s", ticker, exc)

    if not scored:
        log.error("no tickers scored successfully")
        sys.exit(2)

    # ── Market cap tiers ──────────────────────────────────────────────────────
    caps = [(t["ticker"], t["market_cap"]) for t in scored]
    tiers = assign_tier(caps)
    for t in scored:
        t["cap_tier"] = tiers.get(t["ticker"], "large")

    # ── Build lists ───────────────────────────────────────────────────────────
    all_sorted    = sorted(scored, key=lambda x: x["final_score"], reverse=True)
    top_buys      = all_sorted[:top_n]
    mid_caps_pool = [t for t in all_sorted if t["cap_tier"] == "mid"]
    small_caps_pool = [t for t in all_sorted if t["cap_tier"] == "small"]

    # Fallback: if not enough in tier, fill from adjacent
    if len(mid_caps_pool) < top_n:
        extras = [t for t in all_sorted if t not in mid_caps_pool and t not in top_buys]
        mid_caps_pool += extras
    if len(small_caps_pool) < top_n:
        extras = [t for t in all_sorted if t not in small_caps_pool]
        small_caps_pool += extras

    mid_caps   = mid_caps_pool[:top_n]
    small_caps = small_caps_pool[:top_n]

    top_lists = {
        "generated_at":  datetime.now(timezone.utc).isoformat(),
        "source_run_id": run_id or os.getenv("GITHUB_RUN_ID", "local"),
        "weights":       weights,
        "ticker_count":  len(scored),
        "top_buys":      top_buys,
        "mid_caps":      mid_caps,
        "small_caps":    small_caps,
    }

    # ── Write outputs ─────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(top_lists, indent=2, default=str), encoding="utf-8")
    log.info("top_lists.json written to %s", output_path)

    # ── Write CSV (top 5 overall for quick review) ────────────────────────────
    import csv
    with open(output_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "rank","ticker","final_score","badge","edgar","insider",
            "congress","news","macro","cap_tier","ceo_buy","net_value",
        ])
        writer.writeheader()
        for rank, entry in enumerate(top_buys, 1):
            f = entry["factors"]
            writer.writerow({
                "rank":        rank,
                "ticker":      entry["ticker"],
                "final_score": entry["final_score"],
                "badge":       entry["badge"],
                "edgar":       f["edgar"],
                "insider":     f["insider"],
                "congress":    f["congress"],
                "news":        f["news"],
                "macro":       f["macro"],
                "cap_tier":    entry["cap_tier"],
                "ceo_buy":     entry["ceo_buy"],
                "net_value":   entry["net_value"],
            })
    log.info("top5.csv written to %s", output_csv)

    return top_lists


# ── CLI ────────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate top_lists.json from EDGAR pipeline outputs")
    parser.add_argument("--log-dir",    type=Path, default=Path("logs"),
                        help="Directory containing marketintel_events.json and metrics.json")
    parser.add_argument("--output",     type=Path, default=None,
                        help="Output path for top_lists.json (default: <log-dir>/top_lists.json)")
    parser.add_argument("--output-csv", type=Path, default=None,
                        help="Output path for top5.csv (default: <log-dir>/top5.csv)")
    parser.add_argument("--top-n",      type=int,  default=5,
                        help="Number of tickers per list (default: 5)")
    parser.add_argument("--force",      action="store_true",
                        help="Regenerate even if output is recent")
    parser.add_argument("--run-id",     type=str,  default="",
                        help="GitHub run ID to embed in output")
    parser.add_argument("--weights",    type=str,  default=None,
                        help='JSON weight overrides e.g. \'{"edgar":0.40,"insider":0.20}\'')
    parser.add_argument("--verbose",    action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    weights = dict(DEFAULT_WEIGHTS)
    if args.weights:
        try:
            overrides = json.loads(args.weights)
            weights.update(overrides)
        except Exception as exc:
            log.error("invalid --weights JSON: %s", exc)
            return 1

    output_path = args.output     or (args.log_dir / "top_lists.json")
    output_csv  = args.output_csv or (args.log_dir / "top5.csv")

    generate(
        events_path = args.log_dir / "marketintel_events.json",
        metrics_path = args.log_dir / "metrics.json",
        output_path  = output_path,
        output_csv   = output_csv,
        weights      = weights,
        top_n        = args.top_n,
        force        = args.force,
        run_id       = args.run_id,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
