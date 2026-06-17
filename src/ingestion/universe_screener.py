# Path: src/ingestion/universe_screener.py
"""Hybrid dynamic universe — curated core anchor + dynamically screened satellite.

The frozen CSV universe is the root cause of ticker stagnation: the same ~160
names are re-scored every run, so capital never rotates to fresh opportunities
(QuantConnect Lean's coarse/fine *dynamic universe selection* is the industry
answer). This module keeps the curated CSV as a stable **core anchor** and adds
a rotating **satellite** sleeve of names with surging trading attention that are
*not* already in the core.

Satellite selection is a cross-sectional **volume-velocity rank**
[WorldQuant 101 Alphas — ts_rank/rank of volume]: short-window vs long-window
mean volume, ranked across candidates. Pure pandas/numpy, no look-ahead (volume
only), normalized within the region's own peer group (CLAUDE.md §3).

Gated by the ``UNIVERSE_DYNAMIC`` env flag (default off → byte-identical legacy
behavior, same convention as ``SCORING_V3_SHADOW``).
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Region → screener exchanges (mirrors tools/build_universes.py vocabulary).
_REGION_EXCHANGES: Dict[str, List[str]] = {
    "US": ["NASDAQ", "NYSE"],
    "EU": ["XETRA", "EURONEXT", "LSE", "SIX", "OSE", "STO", "CPH", "HEL"],
    "ASIA": ["TSE", "HKSE", "KSC", "KOE", "NSE", "SES", "SET", "JKSE"],
}
_MIN_MARKET_CAP = 2_000_000_000   # $2B floor — same as build_universes.py


# ── Pure vectorized selection (the quant heart; fully unit-tested) ────────────


def compute_volume_velocity(
    volume_panel: pd.DataFrame, short: int = 5, long: int = 20,
) -> pd.Series:
    """Per-ticker volume velocity = short-window / long-window mean volume − 1.

    A positive value means recent volume is running hot versus its own baseline
    — emerging attention. NaN (not inf) when the long-window baseline is zero,
    so dead names never masquerade as infinite velocity.
    """
    short_mean = volume_panel.tail(short).mean()
    long_mean = volume_panel.tail(long).mean().replace(0, np.nan)
    return short_mean / long_mean - 1.0


def cross_sectional_rank(series: pd.Series) -> pd.Series:
    """Percentile rank across the cross-section [WorldQuant 101 Alphas — rank]."""
    return series.rank(pct=True)


def select_satellite(
    velocity: pd.Series, core: Set[str], k: int,
) -> List[str]:
    """Top-k velocity names excluding anything already in the core."""
    s = velocity.dropna()
    s = s[~s.index.isin(core)]
    return list(s.sort_values(ascending=False).head(k).index)


def merge_universe(
    core_rows: Sequence[Dict[str, Any]],
    satellite_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Union core ∪ satellite, core taking precedence, tagged by ``origin``."""
    core_ids = {r["ticker"] for r in core_rows}
    out: List[Dict[str, Any]] = []
    for r in core_rows:
        rr = dict(r)
        rr["origin"] = "core"
        out.append(rr)
    for r in satellite_rows:
        if r["ticker"] in core_ids:
            continue
        rr = dict(r)
        rr["origin"] = "satellite"
        out.append(rr)
    return out


# ── FMP orchestration (thin shell; gated by UNIVERSE_DYNAMIC) ─────────────────


def _dynamic_enabled() -> bool:
    return os.getenv("UNIVERSE_DYNAMIC", "").lower() in ("1", "true", "yes")


def _load_core(tickers_file: Path) -> List[Dict[str, Any]]:
    from src.ingestion.run_pipeline import load_tickers  # lazy → no import cycle
    return load_tickers(tickers_file)


def _screen_candidates(client: Any, region: str) -> Dict[str, Dict[str, Any]]:
    """{ticker: {sector, cap_tier}} from the screener across region exchanges."""
    out: Dict[str, Dict[str, Any]] = {}
    for exchange in _REGION_EXCHANGES.get(region, _REGION_EXCHANGES["US"]):
        for row in client.get_company_screener(
                exchange=exchange, market_cap_more_than=_MIN_MARKET_CAP,
                limit=100):
            sym = (row.get("symbol") or "").strip().upper()
            if not sym or sym in out:
                continue
            mcap = float(row.get("marketCap") or 0.0)
            out[sym] = {
                "sector": row.get("sector") or "Unknown",
                "cap_tier": "large" if mcap >= 10e9 else "mid",
            }
    return out


def _volume_panel(
    client: Any, tickers: Sequence[str], long: int,
) -> pd.DataFrame:
    from src.services.fmp_client import fmp_prices_to_arrays
    cols: Dict[str, pd.Series] = {}
    for t in tickers:
        rows = client.get_historical_prices(t, limit=long + 10)
        _, volumes, dates = fmp_prices_to_arrays(rows)
        if volumes:
            cols[t] = pd.Series(volumes, index=dates)
    return pd.DataFrame(cols) if cols else pd.DataFrame()


def _append_churn(
    log_dir: Optional[Path], events: Sequence[Dict[str, Any]],
) -> None:
    """Append-only churn audit (convention: logs/kill_switch_audit.ndjson)."""
    if not log_dir or not events:
        return
    path = Path(log_dir) / "universe_churn.ndjson"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")


def resolve_universe(
    tickers_file: Path,
    region: str = "US",
    *,
    client: Any = None,
    log_dir: Optional[Path] = None,
    satellite_k: Optional[int] = None,
    short: int = 5,
    long: int = 20,
) -> List[Dict[str, Any]]:
    """Return the scoring universe: core only, or core+satellite when enabled.

    Flag off (default) → exactly ``load_tickers(tickers_file)`` (legacy, safe).
    """
    core_rows = _load_core(tickers_file)
    if not _dynamic_enabled():
        return core_rows

    if client is None:
        from src.services.fmp_client import FMPClient
        client = FMPClient()
    if not getattr(client, "_api_key", ""):
        log.warning("UNIVERSE_DYNAMIC set but FMP key absent — core universe only.")
        return core_rows

    k = satellite_k if satellite_k is not None else int(
        os.getenv("UNIVERSE_SATELLITE_K", "20"))
    candidates = _screen_candidates(client, region)
    panel = _volume_panel(client, list(candidates), long)
    if panel.empty:
        log.warning("Dynamic universe: no volume data screened — core only.")
        return core_rows

    velocity = compute_volume_velocity(panel, short, long)
    core_ids = {r["ticker"] for r in core_rows}
    chosen = select_satellite(velocity, core_ids, k)

    today = date.today().isoformat()
    sat_rows: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    for t in chosen:
        meta = candidates.get(t, {})
        sat_rows.append({"ticker": t, "sector": meta.get("sector", "Unknown"),
                         "cap_tier": meta.get("cap_tier", "mid")})
        events.append({
            "date": today, "ticker": t, "action": "added",
            "reason": f"volume_velocity={float(velocity[t]):.3f} "
                      f"(top-{k} surging attention, absent from core)",
            "velocity": round(float(velocity[t]), 4),
        })
    _append_churn(log_dir, events)

    merged = merge_universe(core_rows, sat_rows)
    log.info("Dynamic universe: %d core + %d satellite = %d names (region=%s)",
             len(core_rows), len(sat_rows), len(merged), region)
    return merged
