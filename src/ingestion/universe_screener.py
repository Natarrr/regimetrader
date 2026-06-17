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

# ── Small/mid satellite (feeds the SMID leverage sleeve only) ──────────────────
# The $2B core/dynamic floor structurally excludes small caps, which is why the
# small/mid-cap desk renders empty. This dedicated tranche screens the leverage
# sleeve's own band ($300M–$10B, cook _SMID_CAP_MIN/_MAX) with a liquidity floor,
# tagged origin="smid_satellite" so the scorer isolates it (own cap_tier
# normalization bucket; exempt from the main low-coverage gate). Gated by the
# UNIVERSE_SMID_SATELLITE flag (default off → legacy-safe, same convention as
# UNIVERSE_DYNAMIC).
_SMID_SCREEN_MIN_CAP = 300_000_000      # $300M — sleeve floor
_SMID_SCREEN_MAX_CAP = 10_000_000_000   # $10B  — sleeve ceiling
_SMID_SCREEN_MIN_VOLUME = 200_000       # liquidity floor — drop illiquid micro-caps
_SMID_MID_THRESHOLD = 2_000_000_000     # < $2B → "small", else "mid"


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


def _smid_enabled() -> bool:
    return os.getenv("UNIVERSE_SMID_SATELLITE", "").lower() in ("1", "true", "yes")


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


def _screen_smid_candidates(
    client: Any, region: str,
) -> Dict[str, Dict[str, Any]]:
    """{ticker: {sector, cap_tier, market_cap}} in the SMID band ($300M–$10B).

    Screens ``market_cap_more_than=$300M`` with a volume floor (the screener has
    no upper-cap parameter) and band-filters client-side to ≤ $10B, tagging
    small (< $2B) vs mid."""
    out: Dict[str, Dict[str, Any]] = {}
    for exchange in _REGION_EXCHANGES.get(region, _REGION_EXCHANGES["US"]):
        for row in client.get_company_screener(
                exchange=exchange,
                market_cap_more_than=_SMID_SCREEN_MIN_CAP,
                volume_more_than=_SMID_SCREEN_MIN_VOLUME,
                limit=100):
            sym = (row.get("symbol") or "").strip().upper()
            if not sym or sym in out:
                continue
            mcap = float(row.get("marketCap") or 0.0)
            if not (_SMID_SCREEN_MIN_CAP <= mcap <= _SMID_SCREEN_MAX_CAP):
                continue
            out[sym] = {
                "sector": row.get("sector") or "Unknown",
                "cap_tier": "small" if mcap < _SMID_MID_THRESHOLD else "mid",
                "market_cap": mcap,
            }
    return out


def _resolve_smid_satellite(
    client: Any, region: str, existing: Set[str],
    *, log_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Dedicated small/mid tranche for the SMID leverage sleeve.

    Balances small and mid candidates so true small caps are represented (a
    plain size sort would fill the quota with the $2B–$10B end). Tagged
    origin='smid_satellite'; cook re-ranks survivors by leverage_score, so this
    only needs to supply a sane, liquid candidate pool."""
    k = int(os.getenv("UNIVERSE_SMID_K", "30"))
    candidates = _screen_smid_candidates(client, region)
    smalls = [(s, m) for s, m in candidates.items()
              if s not in existing and m["cap_tier"] == "small"]
    mids = [(s, m) for s, m in candidates.items()
            if s not in existing and m["cap_tier"] == "mid"]
    half = max(1, k // 2)
    chosen = smalls[:half] + mids[:max(0, k - len(smalls[:half]))]

    today = date.today().isoformat()
    rows: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    for sym, meta in chosen:
        rows.append({
            "ticker": sym,
            "sector": meta.get("sector", "Unknown"),
            "cap_tier": meta.get("cap_tier", "small"),
            "origin": "smid_satellite",
        })
        events.append({
            "date": today, "ticker": sym, "action": "added",
            "reason": f"smid_satellite mcap=${meta.get('market_cap', 0.0) / 1e9:.2f}B "
                      f"(small/mid leverage-sleeve candidate, absent from core)",
        })
    _append_churn(log_dir, events)
    return rows


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

    Both flags off (default) → exactly ``load_tickers(tickers_file)`` (legacy,
    safe). ``UNIVERSE_DYNAMIC`` adds the volume-velocity satellite;
    ``UNIVERSE_SMID_SATELLITE`` adds the small/mid leverage-sleeve tranche.
    """
    core_rows = _load_core(tickers_file)
    dynamic, smid = _dynamic_enabled(), _smid_enabled()
    if not dynamic and not smid:
        return core_rows

    if client is None:
        from src.services.fmp_client import FMPClient
        client = FMPClient()
    if not getattr(client, "_api_key", ""):
        log.warning(
            "UNIVERSE_DYNAMIC/SMID set but FMP key absent — core universe only.")
        return core_rows

    rows: List[Dict[str, Any]] = list(core_rows)

    if dynamic:
        k = satellite_k if satellite_k is not None else int(
            os.getenv("UNIVERSE_SATELLITE_K", "20"))
        candidates = _screen_candidates(client, region)
        panel = _volume_panel(client, list(candidates), long)
        if panel.empty:
            log.warning("Dynamic universe: no volume data screened — satellite skipped.")
        else:
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
            rows = merge_universe(core_rows, sat_rows)
            log.info("Dynamic universe: %d core + %d satellite = %d names (region=%s)",
                     len(core_rows), len(sat_rows), len(rows), region)

    if smid:
        existing = {r["ticker"] for r in rows}
        smid_rows = _resolve_smid_satellite(
            client, region, existing, log_dir=log_dir)
        rows.extend(smid_rows)
        log.info("SMID satellite: +%d small/mid names (region=%s)",
                 len(smid_rows), region)

    return rows
