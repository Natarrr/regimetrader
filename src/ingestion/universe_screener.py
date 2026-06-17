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
_SMID_SCREEN_MIN_VOLUME = 200_000       # share-volume floor — drop illiquid micro-caps
_SMID_MID_THRESHOLD = 2_000_000_000     # < $2B → "small", else "mid"

# ADV dollar-volume gate (price × volume) — the real liquidity guard against
# market-impact traps [Amihud 2002]; a share count alone misprices a $2 stock vs
# a $200 stock. Applied only when the screener row carries price+volume; rows
# lacking the fields fall back to the server-side share-volume floor (never
# dropped on absent data). Soft-beta α tilts candidate ranking toward higher-
# leverage (directional-velocity) names without a hard, noisy beta exclusion.
_SMID_MIN_DOLLAR_VOL = float(os.getenv("SMID_MIN_DOLLAR_VOL", "3000000"))  # $3M/day
_SMID_BETA_ALPHA = float(os.getenv("SMID_BETA_ALPHA", "0.15"))            # soft tilt


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
    """{ticker: {sector, cap_tier, market_cap, adv_usd, beta}} in the SMID band.

    Screens the SMALL ($300M–$2B) and MID ($2B–$10B) bands SEPARATELY. FMP's
    company-screener returns rows market-cap-DESCENDING within a ceiling, so a
    single $300M–$10B screen at limit=100 only ever surfaces the $6–10B top of the
    band — true small caps never enter the candidate pool (confirmed by the
    2026-06-17 dry-run: all 30 SMID names landed $6.17–9.90B, zero < $2B).
    Splitting the ceiling forces each tranche into its own 100-row window.

    Each band carries ``is_actively_trading`` (shell guard) + a share-volume floor,
    then an ADV dollar-volume gate (price × volume ≥ ``_SMID_MIN_DOLLAR_VOL``) to
    drop market-impact traps [Amihud 2002]. ``adv_usd`` and ``beta`` ride along for
    the downstream soft-beta leverage rank. Rows lacking price/volume are kept on
    the server-side share floor — absent data is never used to reject."""
    bands = (
        (_SMID_SCREEN_MIN_CAP, _SMID_MID_THRESHOLD),    # small: $300M–$2B
        (_SMID_MID_THRESHOLD, _SMID_SCREEN_MAX_CAP),    # mid:   $2B–$10B
    )
    out: Dict[str, Dict[str, Any]] = {}
    for exchange in _REGION_EXCHANGES.get(region, _REGION_EXCHANGES["US"]):
        for lo_cap, hi_cap in bands:
            for row in client.get_company_screener(
                    exchange=exchange,
                    market_cap_more_than=lo_cap,
                    market_cap_lower_than=hi_cap,
                    volume_more_than=_SMID_SCREEN_MIN_VOLUME,
                    is_actively_trading=True,
                    limit=100):
                sym = (row.get("symbol") or "").strip().upper()
                if not sym or sym in out:
                    continue
                mcap = float(row.get("marketCap") or 0.0)
                if not (_SMID_SCREEN_MIN_CAP <= mcap <= _SMID_SCREEN_MAX_CAP):
                    continue
                price = float(row.get("price") or 0.0)
                volume = float(row.get("volume") or 0.0)
                adv_usd = price * volume if price > 0 and volume > 0 else None
                if adv_usd is not None and adv_usd < _SMID_MIN_DOLLAR_VOL:
                    continue   # illiquid → market-impact trap; drop
                out[sym] = {
                    "sector": row.get("sector") or "Unknown",
                    "cap_tier": "small" if mcap < _SMID_MID_THRESHOLD else "mid",
                    "market_cap": mcap,
                    "adv_usd": adv_usd,
                    "beta": float(row["beta"]) if row.get("beta") is not None else None,
                }
    return out


def _leverage_rank(
    group: List[tuple], alpha: float = _SMID_BETA_ALPHA,
) -> List[tuple]:
    """Order ``(sym, meta)`` by ADV liquidity with a soft-beta leverage tilt.

    key = adv_usd · (1 + α·clip(z(beta), −2, 2)). ADV dollar-volume is the
    liquidity anchor; beta is a *soft* boost toward higher-leverage (directional-
    velocity) names — never a hard exclusion, because beta is a noisy estimate.
    Rows with no ADV/beta keep a neutral key, so a field-less screener row (or a
    test fixture) sorts in stable insertion order rather than being penalised."""
    if not group:
        return []
    betas = np.array(
        [m.get("beta") if m.get("beta") is not None else np.nan for _, m in group],
        dtype=float)
    finite = betas[np.isfinite(betas)]
    if finite.size >= 2 and finite.std() > 0:
        z = np.where(np.isfinite(betas), (betas - finite.mean()) / finite.std(), 0.0)
        boost = 1.0 + alpha * np.clip(z, -2.0, 2.0)
    else:
        boost = np.ones(len(group))
    keyed = [
        (sym, meta, (float(meta["adv_usd"]) if meta.get("adv_usd") else 0.0) * b)
        for (sym, meta), b in zip(group, boost)
    ]
    keyed.sort(key=lambda x: -x[2])   # stable → ties keep insertion order
    return [(sym, meta) for sym, meta, _ in keyed]


def _resolve_smid_satellite(
    client: Any, region: str, existing: Set[str],
    *, log_dir: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Dedicated small/mid tranche for the SMID leverage sleeve.

    Balances small and mid candidates so true small caps are represented (a
    plain size sort would fill the quota with the $2B–$10B end), ranking within
    each tier by ADV liquidity + soft-beta leverage (``_leverage_rank``). Tagged
    origin='smid_satellite'; cook re-ranks survivors by leverage_score, so this
    only needs to supply a sane, liquid, leverage-tilted candidate pool."""
    k = int(os.getenv("UNIVERSE_SMID_K", "30"))
    candidates = _screen_smid_candidates(client, region)
    smalls = _leverage_rank([(s, m) for s, m in candidates.items()
                             if s not in existing and m["cap_tier"] == "small"])
    mids = _leverage_rank([(s, m) for s, m in candidates.items()
                           if s not in existing and m["cap_tier"] == "mid"])
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
        adv = meta.get("adv_usd")
        adv_txt = f"adv=${adv / 1e6:.1f}M" if adv else "adv=n/a"
        beta = meta.get("beta")
        beta_txt = f"beta={beta:.2f}" if beta is not None else "beta=n/a"
        events.append({
            "date": today, "ticker": sym, "action": "added",
            "reason": f"smid_satellite mcap=${meta.get('market_cap', 0.0) / 1e9:.2f}B "
                      f"{adv_txt} {beta_txt} "
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
