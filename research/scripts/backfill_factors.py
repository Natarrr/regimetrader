# Path: research/scripts/backfill_factors.py
"""Backfill factor history → research/data/backfill/factor_scores.ndjson.

Reconstructs per-(ticker, snapshot_date) factor scores plus a SPY-relative
21-day forward-return label, so research.scripts.ic_engine can measure realised
rank-IC without waiting ~3 months for the live archive to accumulate 60+
snapshots.

QUANT-INTEGRITY SCOPE (deliberate deviation from the 2026-06-06 spec).
The spec proposed reusing the *current* analyst/fundamental snapshot as a proxy
for every past date. That injects look-ahead bias (CLAUDE.md §3: anchor to
filingDate, never leak future data). We refuse to do that. This backfill
reconstructs only the factors that are genuinely point-in-time from daily price
history:

    momentum_long     12-1m SPY-relative return  [Jegadeesh & Titman, 1993]
    volume_attention  20d rolling volume z-score  [WorldQuant 101 Alphas, ts_rank]

Fundamental / insider / analyst factors are emitted ABSENT for past dates (they
need a point-in-time fundamentals source the FMP client does not yet expose).
ic_metrics drops them pairwise, so they surface as "investigate" — an honest
"not yet measurable", never a look-ahead-contaminated IC.

De-overlap at the source: snapshots are sampled 21 trading days apart so the
21-day forward windows do not overlap (López de Prado 2018, ch. 7).

Run:
    python -m research.scripts.backfill_factors --tickers-file config/universe.csv
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:  # load FMP_API_KEY from .env when run standalone (mirrors build_universes.py)
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ── Pure quant helpers (point-in-time; carry the look-ahead-bias risk) ────────


def sample_snapshot_indices(
    n: int, spacing: int = 21, horizon: int = 21, count: int = 52,
) -> List[int]:
    """Indices of snapshot dates, ``spacing`` apart, newest first then reversed.

    The newest usable index is ``n-1-horizon`` so every snapshot still has a
    full forward window. Stepping back by ``spacing`` (>= horizon) guarantees
    non-overlapping forward returns at the source.
    """
    last = n - 1 - horizon
    if last < 0:
        return []
    idxs: List[int] = []
    i = last
    while i >= 0 and len(idxs) < count:
        idxs.append(i)
        i -= spacing
    return sorted(idxs)


def _ratio(series: Sequence[Optional[float]], i: int, j: int) -> Optional[float]:
    """series[j]/series[i]-1, or None if either point is missing/zero."""
    if i < 0 or j < 0 or i >= len(series) or j >= len(series):
        return None
    a, b = series[i], series[j]
    if a is None or b is None or a == 0:
        return None
    return b / a - 1.0


def forward_excess_return(
    closes: Sequence[Optional[float]],
    spy_closes: Sequence[Optional[float]],
    idx: int,
    horizon: int = 21,
) -> Optional[float]:
    """SPY-relative excess forward return over ``horizon`` trading days."""
    asset = _ratio(closes, idx, idx + horizon)
    spy = _ratio(spy_closes, idx, idx + horizon)
    if asset is None or spy is None:
        return None
    return asset - spy


def momentum_excess(
    closes: Sequence[Optional[float]],
    spy_closes: Sequence[Optional[float]],
    idx: int,
    lookback: int = 252,
    skip: int = 21,
) -> Optional[float]:
    """12-1m SPY-relative momentum: return from idx-lookback to idx-skip.

    Skipping the most recent month avoids the short-term reversal that
    contaminates raw momentum [Jegadeesh & Titman, 1993].
    """
    asset = _ratio(closes, idx - lookback, idx - skip)
    spy = _ratio(spy_closes, idx - lookback, idx - skip)
    if asset is None or spy is None:
        return None
    return asset - spy


def volume_zscore(
    volumes: Sequence[float], idx: int, window: int = 20,
) -> Optional[float]:
    """Rolling z-score of volume over the trailing ``window`` days.

    Cross-sectional ranking happens later in ic_metrics (spearman), so the raw
    z-score is sufficient here. [WorldQuant 101 Alphas — ts_rank of volume.]
    """
    if idx - window < 0:
        return None
    hist = [v for v in volumes[idx - window:idx] if v is not None]
    if len(hist) < window:
        return None
    mu = statistics.mean(hist)
    sd = statistics.pstdev(hist)
    if sd == 0:
        return 0.0
    return (volumes[idx] - mu) / sd


def anchor_filing(
    filings: Sequence[Dict[str, Any]],
    d: date,
    date_field: str = "filingDate",
) -> Optional[Dict[str, Any]]:
    """Most recent filing observable on or before ``d`` (look-ahead guard).

    Anchors strictly to ``filingDate`` — a statement whose fiscal period ended
    before ``d`` but was *filed* after ``d`` was not observable yet and is
    excluded (CLAUDE.md §3: never use fiscal_period_end).
    """
    best: Optional[Dict[str, Any]] = None
    best_dt: Optional[date] = None
    for f in filings:
        raw = f.get(date_field)
        if not raw:
            continue
        try:
            fdt = date.fromisoformat(str(raw)[:10])
        except ValueError:
            continue
        if fdt <= d and (best_dt is None or fdt > best_dt):
            best, best_dt = f, fdt
    return best


# ── FMP orchestration (thin shell over the pure helpers + FMPClient) ──────────


def reconstruct_ticker(
    client: Any,
    ticker: str,
    spy_map: Dict[str, float],
    *,
    horizon: int = 21,
    lookback: int = 252,
    skip: int = 21,
    spacing: int = 21,
    count: int = 52,
    vol_window: int = 20,
) -> List[Dict[str, Any]]:
    """Reconstruct price-derived factor records for one ticker."""
    from src.services.fmp_client import fmp_prices_to_arrays
    need = lookback + horizon + spacing * count + vol_window + 5
    rows = client.get_historical_prices(ticker, limit=max(400, need))
    closes, volumes, dates = fmp_prices_to_arrays(rows)
    if len(closes) < lookback + horizon + 2:
        return []
    spy_aligned: List[Optional[float]] = [spy_map.get(dt) for dt in dates]

    out: List[Dict[str, Any]] = []
    for idx in sample_snapshot_indices(len(closes), spacing, horizon, count):
        asset_fwd = _ratio(closes, idx, idx + horizon)
        spy_fwd = _ratio(spy_aligned, idx, idx + horizon)
        if asset_fwd is None or spy_fwd is None:
            continue   # SPY non-trading on this calendar slot → unmeasurable
        out.append({
            "ticker": ticker,
            "snapshot_date": dates[idx],
            "momentum_long": momentum_excess(closes, spy_aligned, idx, lookback, skip),
            "volume_attention": volume_zscore(volumes, idx, vol_window),
            "forward_return_21d": asset_fwd,
            "spy_return_21d": spy_fwd,
        })
    return out


def _load_universe_tickers(csv_path: Path) -> List[str]:
    import csv
    with csv_path.open(encoding="utf-8", newline="") as fh:
        return [r["ticker"].strip().upper()
                for r in csv.DictReader(fh) if r.get("ticker", "").strip()]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tickers-file", type=Path, default=Path("config/universe.csv"))
    parser.add_argument("--out", type=Path,
                        default=Path("research/data/backfill/factor_scores.ndjson"))
    parser.add_argument("--count", type=int, default=52)
    args = parser.parse_args(argv)

    from src.services.fmp_client import FMPClient, fmp_prices_to_arrays
    client = FMPClient()
    if not client._api_key:   # noqa: SLF001 — research script, explicit guard
        parser.error("FMP_API_KEY not set (add to .env or export)")

    spy_rows = client.get_historical_prices("SPY", limit=400 + args.count * 21)
    spy_closes, _, spy_dates = fmp_prices_to_arrays(spy_rows)
    spy_map = dict(zip(spy_dates, spy_closes))
    if not spy_map:
        parser.error("could not fetch SPY history — aborting backfill")

    tickers = _load_universe_tickers(args.tickers_file)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with args.out.open("w", encoding="utf-8") as fh:
        for i, ticker in enumerate(tickers, 1):
            recs = reconstruct_ticker(client, ticker, spy_map, count=args.count)
            for rec in recs:
                fh.write(json.dumps(rec) + "\n")
            total += len(recs)
            print(f"  [{i}/{len(tickers)}] {ticker}: {len(recs)} snapshots")

    print(f"backfill -> {args.out}  [{total} records from {len(tickers)} tickers]")
    print("Next: python -m research.scripts.ic_engine")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
