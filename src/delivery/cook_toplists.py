#!/usr/bin/env python3
# Path: src/delivery/cook_toplists.py
"""Merge US top_lists.json + INTL top_lists_intl.json into a unified combined payload.

Schema transformation for INTL StrategyEngine output:
  composite_score  -> final_score
  region_applied   -> market ("EUROPE" or "ASIA" via ticker_registry.json)
  factor_snapshots -> factors (congress: 0.0 injected -- audit check E)
  badge            -> computed from final_score thresholds
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from src.config.weights import get_region
from src.risk.regime import (
    RiskRegime,
    apply_capitulation_filter,
    get_regime,
    is_panic,
    vix_multiplier,
)

try:
    from backend.market_intel.portfolio_optimizer import (
        run_optimizer as _run_optimizer,
        build_large_cap_anchors as _build_anchors,
    )
    from src.risk.exit_rules import enrich_with_exit_anchors as _enrich_exits
    _EXTENSIONS_AVAILABLE = True
except ImportError:
    _EXTENSIONS_AVAILABLE = False

_LARGE_CAP_THRESHOLD = 10_000_000_000
_MID_CAP_MIN         =  2_000_000_000
_MID_CAP_MAX         = 10_000_000_000
_SMALL_CAP_MIN       =    300_000_000
_SMALL_CAP_MAX       =  2_000_000_000

# ── SMID leverage sleeve (US-only) ────────────────────────────────────────────
# Composite re-rank of the already-vetted US buy list toward the small/mid-cap
# leverage profile: final_score carries the full 9-factor alpha (VIX overlay,
# Piotroski gate and momentum-regime dampening already applied upstream);
# momentum_long is re-emphasized per [Jegadeesh & Titman, 1993] (12-1m momentum
# persists strongest outside mega-caps; size premium [Banz, 1981]) and
# quality_piotroski per [Piotroski, 2000] (F-score alpha concentrates in
# small/neglected names).
_SMID_LEVERAGE_WEIGHTS = {
    "final_score":       0.50,
    "momentum_long":     0.30,
    "quality_piotroski": 0.20,
}
assert abs(sum(_SMID_LEVERAGE_WEIGHTS.values()) - 1.0) < 1e-6

_SMID_CAP_MIN = _SMALL_CAP_MIN         # $300M inclusive
_SMID_CAP_MAX = _LARGE_CAP_THRESHOLD   # $10B inclusive (large bracket starts > $10B)
_SMID_TOP_N   = 3
# Post-earnings-announcement drift: positive surprises drift upward for roughly
# 60 days post-announcement [Bernard & Thomas, 1989]. Boost is applied ONLY on
# affirmative evidence — absent earnings meta must never read as bearish.
# Mirrored in send_discord._SMID_PEAD_WINDOW_DAYS (flag must agree with boost).
_SMID_PEAD_WINDOW_DAYS = 60
_SMID_PEAD_BOOST       = 1.10

_BADGE_THRESHOLDS = [(0.80, "HIGH BUY"), (0.60, "TACTICAL BUY"), (0.00, "WATCHLIST")]

# ── Selection churn (rotate over-tenured leaders out of the displayed desks) ───
# Breaks the "always the same top-3" trap: a name that holds a top-N desk slot for
# >= max_tenure consecutive runs sits out the next run so a fresh contender shows.
# Centralized here (one cook job, one state file) so US/EU/Asia rotate from a
# single persisted tenure map — avoids cross-job state coordination. Gated by the
# UNIVERSE_CHURN flag (default off → selection is byte-identical to legacy).
_WHALE_FLOW_MIN      = 0.80   # mirror send_discord._WHALE_FLOW_MIN
_WHALE_NPR_SPIKE_MIN = 0.30   # mirror send_discord._WHALE_NPR_SPIKE_MIN
_CHURN_STATE_PATH = Path("logs/universe_state.json")
_CHURN_AUDIT_PATH = Path("logs/universe_churn.ndjson")


def _has_whale_signal(entry: dict) -> bool:
    """Mirror send_discord._whale_signal — extreme top-decile 13F inflow or an
    insider acquired/disposed spike. Whale names are EXEMPT from churn cooldown:
    the high-conviction accumulation signal the desk exists to surface must never
    be rotated out (consistent with the send-side target-gate exemption)."""
    flow = (entry.get("factors") or {}).get("inst_flow_13f")
    try:
        if flow is not None and float(flow) >= _WHALE_FLOW_MIN:
            return True
    except (TypeError, ValueError):
        pass
    npr = entry.get("insider_npr") or {}
    try:
        return float(npr.get("spike") or 0.0) >= _WHALE_NPR_SPIKE_MIN
    except (TypeError, ValueError):
        return False


def _churn_regional_desks(
    regions: dict,
    desk_n: int,
    max_tenure: int,
    state_path: Path = _CHURN_STATE_PATH,
    audit_path: Path = _CHURN_AUDIT_PATH,
) -> tuple[dict, list]:
    """Rotate over-tenured leaders out of each region's displayed top-`desk_n`.

    `regions` maps a region label → (primary_list, overflow_list); the candidate
    pool per region is primary+overflow (score-ranked by cooled_top_n). A name that
    has held a top-`desk_n` slot for >= max_tenure runs is rotated out for one run
    (returned in `cooled_entries`) UNLESS it shows an active whale signal. Tenure
    persists in `state_path`; rotation events append to `audit_path`. Returns
    (filtered_regions, cooled_entries) — cooled names are stripped from primary AND
    overflow so the send-side ≥3 backfill cannot resurrect them this run.
    """
    from src.scoring.churn import cooled_top_n, update_tenure  # noqa: PLC0415
    try:
        prev_state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        prev_state = {}
    # Whale names are never on cooldown — drop them from the tenure map before the
    # cooldown test so they stay eligible every run.
    whale_tickers = {
        e.get("ticker")
        for primary, overflow in regions.values()
        for e in (primary + overflow)
        if _has_whale_signal(e)
    }
    eff_state = {t: v for t, v in prev_state.items() if t not in whale_tickers}

    selected_all: list = []
    cooled_entries: list = []
    events_all: list = []
    filtered: dict = {}
    for label, (primary, overflow) in regions.items():
        pool = [e for e in (primary + overflow)
                if isinstance(e.get("final_score"), (int, float))]
        selected, events = cooled_top_n(pool, eff_state, max_tenure, n=desk_n)
        cooled = {ev["ticker"] for ev in events if ev["action"] == "cooldown"}
        sel_set = {e.get("ticker") for e in selected}
        selected_all += [e["ticker"] for e in selected]
        cooled_entries += [e for e in pool if e.get("ticker") in cooled]
        events_all += [{**ev, "region": label} for ev in events]
        # Promote the post-churn top-N into primary (rotated-in names included) and
        # keep the remaining non-cooled names as overflow — so MVO, sector exposure
        # and the audit see the rotated desk, and the send-side ≥3 backfill stays
        # consistent. Cooled names are dropped from both (returned for WATCH).
        filtered[label] = (
            selected,
            [e for e in pool
             if e.get("ticker") not in sel_set and e.get("ticker") not in cooled],
        )

    new_state = update_tenure(
        eff_state, selected_all, [e.get("ticker") for e in cooled_entries])
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(new_state, indent=2), encoding="utf-8")
    if events_all:
        with audit_path.open("a", encoding="utf-8") as fh:
            for ev in events_all:
                fh.write(json.dumps(ev) + "\n")
        n_cool = sum(1 for ev in events_all if ev["action"] == "cooldown")
        print(f"[COOK] Universe churn: {n_cool} cooldown rotation(s), "
              f"max_tenure={max_tenure}")
    return filtered, cooled_entries


def _apply_sector_count_cap(
    entries: list, max_per_sector: int = 2
) -> tuple[list, list]:
    """Return (primary, overflow) — primary keeps ≤ max_per_sector per sector, sorted by score."""
    sorted_entries = sorted(entries, key=lambda e: e.get("final_score", 0), reverse=True)
    sector_counts: dict[str, int] = {}
    primary, overflow = [], []
    for entry in sorted_entries:
        sector = entry.get("factors", {}).get("sector", "Unknown")
        if sector_counts.get(sector, 0) < max_per_sector:
            primary.append(entry)
            sector_counts[sector] = sector_counts.get(sector, 0) + 1
        else:
            overflow.append(entry)
    return primary, overflow


def _segment_by_market_cap(entries: list) -> tuple[list, list, list]:
    """Return (large_cap >$10B, mid_cap $2B-$10B, small_cap $300M-$2B). Others excluded.

    INTL entries are excluded from all brackets: their market_cap arrives in
    listing currency (FMP quote.marketCap has no FX normalization — JPY for .T,
    GBp for .L, …), so treating it as USD would misclassify brackets and skew
    MVO allocations. Lift this guard only once FX normalization exists upstream.
    """
    large, mid, small = [], [], []
    for entry in entries:
        if entry.get("pipeline") == "INTL":
            continue
        cap = entry.get("market_cap", 0) or 0
        if cap > _LARGE_CAP_THRESHOLD:
            large.append(entry)
        elif _MID_CAP_MIN <= cap <= _MID_CAP_MAX:
            mid.append(entry)
        elif _SMALL_CAP_MIN <= cap < _MID_CAP_MIN:
            small.append(entry)
    return large, mid, small


def _build_smid_leverage_pool(entries: list, top_n: int = _SMID_TOP_N) -> list:
    """Top-N US small/mid-cap entries re-ranked by the leverage composite.

    leverage_score = (0.50*final_score + 0.30*momentum_long
                      + 0.20*quality_piotroski) * pead_boost

    pead_boost = 1.10 iff earnings_surprise_pct > 0 and
    0 < earnings_surprise_days <= 60 [Bernard & Thomas, 1989]; 1.00 otherwise
    (data absence is NEVER bearish). leverage_score is a RANKING key, not a
    calibrated probability — it may exceed 1.0 (max 1.10) and is deliberately
    not gated by audit check A, which validates final_score only.

    INTL entries are excluded (same FX-normalization guard as
    _segment_by_market_cap). Returns COPIES ({**e, "leverage_score": ...}) —
    candidates are shared by reference with top_buys_usa/usa_overflow and
    must not be mutated.
    """
    w = _SMID_LEVERAGE_WEIGHTS
    scored: list = []
    seen: set = set()   # dedup — a name may appear in several source lists
    for entry in entries:
        if entry.get("pipeline") == "INTL":
            continue
        ticker = entry.get("ticker")
        if ticker in seen:
            continue
        cap = entry.get("market_cap", 0) or 0
        if not (_SMID_CAP_MIN <= cap <= _SMID_CAP_MAX):
            continue
        seen.add(ticker)
        factors = entry.get("factors") or {}
        base = (
            w["final_score"]         * float(entry.get("final_score") or 0.0)
            + w["momentum_long"]     * float(factors.get("momentum_long") or 0.0)
            + w["quality_piotroski"] * float(factors.get("quality_piotroski") or 0.0)
        )
        boost = 1.0
        pct = entry.get("earnings_surprise_pct")
        days = int(entry.get("earnings_surprise_days") or 0)
        if pct is not None and float(pct) > 0 and 0 < days <= _SMID_PEAD_WINDOW_DAYS:
            boost = _SMID_PEAD_BOOST
        scored.append({**entry, "leverage_score": round(base * boost, 4)})
    scored.sort(key=lambda e: (-e["leverage_score"],
                               -float(e.get("final_score") or 0.0),
                               e.get("ticker", "")))
    return scored[:top_n]


def _badge(score: float) -> str:
    for threshold, label in _BADGE_THRESHOLDS:
        if score >= threshold:
            return label
    return "WATCHLIST"


def _build_registry_map(registry_path: Path) -> dict:
    """Return {ticker: {"market", "cap_tier", "sector"}} from ticker_registry.json."""
    if not registry_path.exists():
        return {}
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    mapping: dict = {}
    for entry in registry.get("europe", []) + registry.get("europe_mid", []):
        mapping[entry["ticker"]] = {
            "market":   "EUROPE",
            "cap_tier": entry.get("cap_tier", "large"),
            "sector":   entry.get("sector", ""),
        }
    for entry in registry.get("asia", []) + registry.get("asia_mid", []):
        mapping[entry["ticker"]] = {
            "market":   "ASIA",
            "cap_tier": entry.get("cap_tier", "large"),
            "sector":   entry.get("sector", ""),
        }
    return mapping


_MARKET_BY_REGION = {"EU": "EUROPE", "ASIA": "ASIA"}


def _cap_tier_from_mcap(mcap: float) -> str:
    """3-way cap tier from market cap (matches FMPFetcher._v3_cap_tier)."""
    if mcap >= 10e9:
        return "large"
    if mcap >= 2e9:
        return "mid"
    return "small"


def _resolve_intl_meta(raw: dict, ticker_market_map: dict) -> dict | None:
    """Per-ticker {market, cap_tier, sector} for an INTL entry.

    The curated registry is authoritative when it lists the ticker. Otherwise —
    for SMID-satellite small-caps and the broad CSV-sourced core, which are
    legitimately scored but absent from ticker_registry.json — derive the market
    from the ticker suffix (get_region: EU→EUROPE, ASIA→ASIA) and the cap tier
    from the scored market cap. Returns None ONLY when the ticker cannot be
    placed in an EU/ASIA pool (US/unknown suffix) — the geographic-leakage guard
    that previously dropped ALL unregistered names is preserved for that case.
    """
    ticker = raw.get("ticker", "")
    if ticker in ticker_market_map:
        return ticker_market_map[ticker]
    market = _MARKET_BY_REGION.get(get_region(ticker))
    if market is None:
        return None
    mcap = float(raw.get("market_cap") or 0.0)
    return {
        "market": market,
        "cap_tier": _cap_tier_from_mcap(mcap) if mcap > 0 else "large",
        "sector": raw.get("sector", ""),
    }


def _normalize_intl_entry(raw: dict, ticker_market_map: dict, vix: float) -> dict:
    """Convert StrategyEngine entry to audit_payload-compatible format.

    Applies VIX dampening to final_score. congress is forced to 0.0 (audit check E).
    """
    ticker = raw.get("ticker", "")
    # Registry is authoritative when present; otherwise the metadata is derived
    # from the ticker suffix + scored market cap (_resolve_intl_meta), so
    # satellite small-caps and CSV-core names are NOT dropped. None only for
    # US/unknown suffixes (geographic-leakage guard) — caller drops those first.
    _registry_meta = _resolve_intl_meta(raw, ticker_market_map)
    if _registry_meta is None:
        raise ValueError(
            f"{ticker!r} not placeable in EU/ASIA (US/unknown suffix)")
    market = _registry_meta["market"]
    cap_tier = _registry_meta.get("cap_tier", "large")
    composite_score = float(raw.get("composite_score", 0.0))
    # Single dampening point for INTL — vix_multiplier carries the Crash tier
    # (×0.20 at VIX ≥ 40) so US and INTL stay symmetric in every regime.
    # apply_capitulation_filter() no longer multiplies (filter + badge only).
    composite_score = round(composite_score * vix_multiplier(vix), 4)
    factor_snapshots = raw.get("factor_snapshots", {})
    # Strip any congress value from snapshots then pin to 0.0 (cannot be overridden)
    factors = {k: v for k, v in factor_snapshots.items() if k != "congress"}
    factors["congress"] = 0.0
    return {
        "ticker":          ticker,
        "sector":          (raw.get("sector") or _registry_meta.get("sector") or "").strip(),
        "final_score":     composite_score,
        "badge":           _badge(composite_score),
        "market":          market,
        "cap_tier":        cap_tier,
        "factors":         factors,
        "pipeline":        raw.get("pipeline", "INTL"),
        "weight_coverage": raw.get("weight_coverage", 0.0),
        # Forward raw prices for exit-anchor enrichment (src/risk/exit_rules)
        "target_price":    raw.get("target_price"),
        "current_price":   raw.get("current_price"),
        "price_to_book":   raw.get("price_to_book"),  # raw P/B ratio for 🎯 line
        "beta_30d":        raw.get("beta_30d"),       # P2.1 — CAPITULATION low-beta gate
        # Forward analyst meta for badge lines
        "analyst_consensus_source":    raw.get("analyst_consensus_source", "none"),
        "analyst_revision_score":      float(raw.get("analyst_revision_score") or 0.0),
        "analyst_revision_n_analysts": int(raw.get("analyst_revision_n_analysts") or 0),
        "price_target_upside_score":   raw.get("price_target_upside_score"),  # None = no analyst target
        "quality_piotroski_score":     float(raw.get("quality_piotroski_score") or 0.0),
        "earnings_surprise_pct":       raw.get("earnings_surprise_pct"),
        "earnings_surprise_days":      int(raw.get("earnings_surprise_days") or 0),
        "insider_usd":                 float(raw.get("insider_usd") or 0.0),
        "market_cap":                  float(raw.get("market_cap") or 0.0),
        # Absolute 12-1m return in the listing market — INTL has no SPY-relative
        # momentum, so momentum_spy_relative is intentionally NOT set (the old
        # mapping rendered absolute returns as "vs SPY 12m" in catalysts).
        "return_12_1m":                float(raw.get("return_12_1m") or 0.0),
        # Recent run-up for the freshness/extension gate (send_discord). Absence
        # stays None (unknown) — never coerced to 0.0, which would read as a real
        # flat move and wrongly keep an unscored name on the actionable list.
        "return_5d":                   raw.get("return_5d"),
        "return_21d":                  raw.get("return_21d"),
        "analyst_consensus_score":     factor_snapshots.get("analyst_consensus"),
    }


def _leg_as_of(path: Path, embedded) -> datetime:
    """Truthful 'data as-of' timestamp for an input leg.

    Prefers the leg's own embedded ``generated_at`` (the real compute time
    stamped by the producer) and falls back to the file's mtime — so a leg
    that did NOT re-run surfaces its true age rather than the cook timestamp.
    """
    if embedded:
        try:
            return datetime.fromisoformat(str(embedded).replace("Z", "+00:00"))
        except ValueError:
            pass
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)


def cook(
    us_input: Path,
    intl_input: Path,
    registry: Path,
    output: Path,
    mvo_enabled: bool = True,
    sector_count_cap: int = 2,
) -> None:
    # ── US payload ────────────────────────────────────────────────────────────
    us_data = json.loads(us_input.read_text(encoding="utf-8"))
    # Support both top_buys_usa (new) and top_buys (legacy) key names
    top_buys_usa = us_data.get("top_buys_usa") or us_data.get("top_buys", [])
    vix = us_data.get("vix")
    vix_regime = us_data.get("vix_regime", "Unknown")
    kill_switch = us_data.get("kill_switch", False)
    us_ticker_count = us_data.get("ticker_count", len(top_buys_usa))

    if vix is None:
        sys.exit(
            "[COOK] ERROR: US payload missing 'vix' — cannot apply macro overlay. Aborting."
        )

    # ── INTL payload ──────────────────────────────────────────────────────────
    intl_raw: list = json.loads(intl_input.read_text(encoding="utf-8"))
    ticker_market = _build_registry_map(registry)

    top_buys_europe: list = []
    top_buys_asia: list = []
    eu_mid_small: list = []
    asia_mid_small: list = []
    for raw_entry in intl_raw:
        # Honor scored metadata: registry names use the registry; satellite
        # small-caps + CSV-core names (absent from the registry) are placed by
        # suffix + scored cap. Only genuinely unplaceable (US/unknown) tickers
        # are dropped — the geographic-leakage guard for the combined audit.
        if _resolve_intl_meta(raw_entry, ticker_market) is None:
            print(f"[COOK] WARN: {raw_entry.get('ticker')!r} not placeable in "
                  f"EU/ASIA (US/unknown suffix) — entry dropped")
            continue
        normalized = _normalize_intl_entry(raw_entry, ticker_market, vix)
        cap = normalized["cap_tier"]
        if normalized["market"] == "EUROPE":
            if cap in ("mid", "small"):
                eu_mid_small.append(normalized)
            else:
                top_buys_europe.append(normalized)
        elif normalized["market"] == "ASIA":
            if cap in ("mid", "small"):
                asia_mid_small.append(normalized)
            else:
                top_buys_asia.append(normalized)

    # Snapshot the scored intl breadth BEFORE the capitulation move and sector
    # cap mutate the lists — ticker_count reports scored coverage, and under
    # CAPITULATION the regional lists are emptied into watchlist.
    intl_scored_count = (len(top_buys_europe) + len(top_buys_asia)
                         + len(eu_mid_small) + len(asia_mid_small))

    # ── Capitulation regime gate ──────────────────────────────────────────────
    regime = get_regime(vix)
    top_buys_usa    = apply_capitulation_filter(top_buys_usa,    vix)
    top_buys_europe = apply_capitulation_filter(top_buys_europe, vix)
    top_buys_asia   = apply_capitulation_filter(top_buys_asia,   vix)
    eu_mid_small    = apply_capitulation_filter(eu_mid_small,    vix)
    asia_mid_small  = apply_capitulation_filter(asia_mid_small,  vix)
    eu_mid_small    = sorted(eu_mid_small,   key=lambda x: -x["final_score"])[:1]
    asia_mid_small  = sorted(asia_mid_small, key=lambda x: -x["final_score"])[:1]
    vix_regime      = regime.value

    # Under CAPITULATION, survivors are badged WATCHLIST by apply_capitulation_filter.
    # Move them OUT of top_buys_* (buy-signal lists) into a dedicated watchlist key
    # so the Discord embed generator does not render them as "Active Buy Signals."
    watchlist: list = []
    if regime == RiskRegime.CAPITULATION:
        for _lst in (top_buys_usa, top_buys_europe, top_buys_asia):
            watchlist.extend(_lst)
        top_buys_usa    = []
        top_buys_europe = []
        top_buys_asia   = []
        eu_mid_small    = []
        asia_mid_small  = []
        print(
            f"[COOK] CAPITULATION REGIME (VIX={vix:.1f}) — "
            f"top_buys_* emptied; {len(watchlist)} structural anchor(s) moved to watchlist."
        )

    # ── ATR / Batch Floor enrichment ──────────────────────────────────────────
    if _EXTENSIONS_AVAILABLE:
        for entry in top_buys_usa + top_buys_europe + top_buys_asia:
            _enrich_exits(entry, entry.get("atr_14"))

    # ── Sector count cap ─────────────────────────────────────────────────────
    top_buys_usa,    usa_overflow    = _apply_sector_count_cap(top_buys_usa,    sector_count_cap)
    top_buys_europe, eu_overflow     = _apply_sector_count_cap(top_buys_europe, sector_count_cap)
    top_buys_asia,   asia_overflow   = _apply_sector_count_cap(top_buys_asia,   sector_count_cap)

    # ── SMID leverage sleeve — candidates AFTER capitulation gate and sector
    # cap; overflow is included so the sleeve is NOT sector-constrained. The
    # dedicated US mid_caps/small_caps lists (from generate_top_lists, fed by the
    # small/mid satellite) are added so the sleeve is no longer starved by the
    # large-cap-dominated top buys — _build_smid_leverage_pool dedups by ticker
    # and band-filters to $300M–$10B. Under CAPITULATION top_buys_usa/usa_overflow
    # are already emptied and the cap-segmented lists are withheld ⇒ pool [].
    smid_candidates = list(top_buys_usa) + list(usa_overflow)
    if regime != RiskRegime.CAPITULATION:
        smid_candidates += (list(us_data.get("mid_caps") or [])
                            + list(us_data.get("small_caps") or []))
    top_buys_smid = _build_smid_leverage_pool(smid_candidates)

    # ── Selection churn ───────────────────────────────────────────────────────
    # Rotate over-tenured leaders out of the displayed desks so the top-N is not
    # byte-identical every run. Applied AFTER the SMID sleeve is built (the SMID
    # leverage desk is intentionally exempt) and BEFORE MVO so portfolio
    # construction sees the rotated desks. Skipped under CAPITULATION (no desks).
    if (regime != RiskRegime.CAPITULATION
            and os.getenv("UNIVERSE_CHURN", "").lower() in ("1", "true", "yes")):
        _max_tenure = int(os.getenv("UNIVERSE_CHURN_MAX_TENURE", "3"))
        _desk_n = int(os.getenv("UNIVERSE_CHURN_DESK_N", "3"))
        _filtered, _cooled = _churn_regional_desks(
            {
                "USA":    (top_buys_usa, usa_overflow),
                "EUROPE": (top_buys_europe, eu_overflow),
                "ASIA":   (top_buys_asia, asia_overflow),
            },
            desk_n=_desk_n, max_tenure=_max_tenure,
        )
        top_buys_usa,    usa_overflow  = _filtered["USA"]
        top_buys_europe, eu_overflow   = _filtered["EUROPE"]
        top_buys_asia,   asia_overflow = _filtered["ASIA"]
        watchlist.extend(_cooled)   # cooled names sit out the desk → WATCH section

    # ── 3-tier capital allocation ─────────────────────────────────────────────
    mvo_pools: dict = {}
    if _EXTENSIONS_AVAILABLE and mvo_enabled:
        all_candidates = top_buys_usa + top_buys_europe + top_buys_asia
        large_entries, mid_entries, small_entries = _segment_by_market_cap(all_candidates)

        if large_entries:
            mvo_pools["large_cap_anchors"] = {
                "bracket": "LARGE_CAP_ANCHOR",
                "cap_range": ">$10B",
                "positions": _build_anchors(large_entries),
            }

        if len(mid_entries) >= 2:
            mid_weights, mid_method = _run_optimizer(
                [e["ticker"] for e in mid_entries],
                [e["final_score"] for e in mid_entries],
                [e.get("factors", {}).get("sector", "Unknown") for e in mid_entries],
                vix=vix, mode="sharpe", position_ceiling=0.35,
            )
            mvo_pools["mid_cap"] = {
                "method": mid_method, "bracket": "MID_CAP", "cap_range": "$2B-$10B",
                "positions": [
                    {
                        "ticker":       e["ticker"],
                        "allocation":   round(mid_weights.get(e["ticker"], 0.0), 4),
                        "final_score":  e["final_score"],
                        "price_target": e.get("target_price") or e.get("price_target"),
                        "exit_anchors": e.get("exit_anchors", {}),
                    }
                    for e in mid_entries if mid_weights.get(e["ticker"], 0.0) > 0.001
                ],
            }

        if len(small_entries) >= 2:
            # ADV liquidity gate is currently INERT: no upstream producer
            # emits adv_20d_usd yet, and run_optimizer additionally requires
            # portfolio_value_usd (not passed here) before it caps anything.
            # Forward-compat plumbing only — do not read this as a live control.
            adv_map = {
                e["ticker"]: e.get("adv_20d_usd")
                for e in small_entries if e.get("adv_20d_usd")
            }
            small_weights, small_method = _run_optimizer(
                [e["ticker"] for e in small_entries],
                [e["final_score"] for e in small_entries],
                [e.get("factors", {}).get("sector", "Unknown") for e in small_entries],
                vix=vix, mode="min_variance",
                position_floor=0.10, position_ceiling=0.25,
                adv_20d_map=adv_map or None,
            )
            mvo_pools["small_cap"] = {
                "method": small_method, "bracket": "SMALL_CAP", "cap_range": "$300M-$2B",
                "positions": [
                    {
                        "ticker":       e["ticker"],
                        "allocation":   round(small_weights.get(e["ticker"], 0.0), 4),
                        "final_score":  e["final_score"],
                        "price_target": e.get("target_price") or e.get("price_target"),
                        "exit_anchors": e.get("exit_anchors", {}),
                    }
                    for e in small_entries if small_weights.get(e["ticker"], 0.0) > 0.001
                ],
            }

    # ── Data-freshness anchor ─────────────────────────────────────────────────
    # `generated_at` (below) records cook time; it is NOT the age of the market
    # data. `data_as_of` is the OLDEST input leg's real timestamp, so a leg that
    # failed to re-run (e.g. stale top_lists_intl.json) trips the DATA STALE
    # banner in send_discord instead of silently reading "0.0h".
    data_as_of = min(
        _leg_as_of(us_input, us_data.get("generated_at")),
        _leg_as_of(intl_input, None),  # INTL artifact is a bare list — mtime only
    ).isoformat()

    # ── Write combined output ─────────────────────────────────────────────────
    combined = {
        "top_buys_usa":    top_buys_usa,
        "top_buys_europe": top_buys_europe,
        "top_buys_asia":   top_buys_asia,
        # US SMID leverage sleeve, ranked by leverage_score (always present —
        # [] under CAPITULATION or when no US name falls in the $300M–$10B band)
        "top_buys_smid":   top_buys_smid,
        "watchlist":       watchlist,          # populated under CAPITULATION regime
        "usa_overflow":    usa_overflow,
        "eu_overflow":     eu_overflow,
        "asia_overflow":   asia_overflow,
        "mvo_pools":       mvo_pools,
        "eu_mid_small":    eu_mid_small,
        "asia_mid_small":  asia_mid_small,
        "vix":             vix,
        "vix_regime":      vix_regime,
        "kill_switch":     kill_switch,
        # Market-regime nowcast inputs + FMP bulk telemetry (Discord brief).
        "spy_momentum_regime": us_data.get("spy_momentum_regime"),
        "spy_return_63d":      us_data.get("spy_return_63d"),
        "bulk_coverage":       us_data.get("bulk_coverage"),
        # Scored coverage (universe semantics): US universe count + every
        # registered intl ticker that was scored — independent of the sector
        # cap and the CAPITULATION watchlist move.
        "ticker_count":    us_ticker_count + intl_scored_count,
        "data_as_of":      data_as_of,
        "generated_at":    datetime.now(timezone.utc).isoformat(),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    watchlist_note = f" | {len(watchlist)} watchlist" if watchlist else ""
    print(
        f"[COOK] Combined payload -> {output} "
        f"({len(top_buys_usa)} US + {len(top_buys_europe)} EU "
        f"+ {len(top_buys_asia)} Asia + {len(top_buys_smid)} smid{watchlist_note})"
    )


def _fetch_vix_for_on_demand() -> float:
    """Live ^VIX quote via FMPClient for the INTL on-demand route.

    The US route lifts VIX from top_lists_us.json; StrategyEngine output
    carries no macro state, so it is fetched here. Failures propagate
    (FMPEndpointError / ValueError) — the safety gate requires a real VIX
    and a silent default could mask a kill-switch regime.
    """
    from src.services.fmp_client import FMPClient
    quote = FMPClient().get_quote("^VIX")
    vix = float(quote.get("price") or quote.get("previousClose") or 0.0)
    if vix <= 0.0:
        raise ValueError(
            "FMP ^VIX quote returned no usable price — cannot run safety gate"
        )
    return vix


def cook_on_demand(
    ticker: str,
    us_input: Path,
    intl_input: Path,
    registry: Path,
    output: Path,
) -> dict:
    """ChatOps single-ticker payload: emit an on_demand_ticker block.

    Routes by ticker shape — dotted tickers (SAP.DE, 7203.T) must be in
    ticker_registry.json and read from the INTL StrategyEngine output; bare
    tickers read from the US top_lists_us.json. The payload deliberately
    contains NO top_buys_* / mvo_pools keys so bulk consumers can never
    mistake it for a daily artifact.

    Raises ValueError (unknown/unscored ticker, missing macro state) or
    FileNotFoundError (missing input artifact); cook() is untouched.
    """
    ticker = ticker.strip().upper()
    is_intl = "." in ticker

    if is_intl:
        ticker_market = _build_registry_map(Path(registry))
        if ticker not in ticker_market:
            raise ValueError(
                f"{ticker!r} has an international suffix but is not in "
                f"ticker_registry.json — on-demand INTL scoring only covers "
                f"registered tickers"
            )
        intl_path = Path(intl_input)
        if not intl_path.exists():
            raise FileNotFoundError(f"INTL input not found: {intl_path}")
        intl_raw: list = json.loads(intl_path.read_text(encoding="utf-8"))
        raw_entry = next(
            (r for r in intl_raw if r.get("ticker") == ticker), None)
        if raw_entry is None:
            raise ValueError(
                f"{ticker!r} not present in {intl_path} — INTL fetch/scoring "
                f"produced no row for it"
            )
        vix = _fetch_vix_for_on_demand()
        kill_switch = is_panic(vix)
        vix_regime = get_regime(vix).value
        entry = _normalize_intl_entry(raw_entry, ticker_market, vix)
        pipeline = "INTL"
        # StrategyEngine scores Σ(w·s)/Σ(w_available) per ticker — absolute
        # by construction, no peer group involved.
        scoring_mode = "absolute"
        source_run_id = None
        on_demand_as_of = _leg_as_of(intl_path, None)
    else:
        us_path = Path(us_input)
        if not us_path.exists():
            raise FileNotFoundError(f"US input not found: {us_path}")
        us_data = json.loads(us_path.read_text(encoding="utf-8"))
        bucket = us_data.get("top_buys_usa") or us_data.get("top_buys", [])
        entry = next((e for e in bucket if e.get("ticker") == ticker), None)
        if entry is None:
            raise ValueError(
                f"{ticker!r} not present in {us_path} — run generate_top_lists "
                f"--single-ticker {ticker} first"
            )
        vix = us_data.get("vix")
        if vix is None:
            raise ValueError(
                "US payload missing 'vix' — safety gate requires macro state"
            )
        kill_switch = us_data.get("kill_switch", False)
        vix_regime = us_data.get("vix_regime") or get_regime(float(vix)).value
        pipeline = "US"
        scoring_mode = us_data.get("scoring_mode", "absolute")
        source_run_id = us_data.get("source_run_id")
        on_demand_as_of = _leg_as_of(us_path, us_data.get("generated_at"))
        if "weight_coverage" not in entry:
            # Display-only coverage: share of canonical US weight whose factor
            # survived the schema gate (missing_sources counts None AND 0.0).
            from src.config.weights import WEIGHTS as _US_WEIGHTS
            _missing = set(
                (entry.get("validation_metadata") or {}).get("missing_sources") or []
            )
            entry["weight_coverage"] = round(
                sum(w for f, w in _US_WEIGHTS.items() if f not in _missing), 4
            )

    combined = {
        "on_demand": True,
        "on_demand_ticker": {
            "ticker":       ticker,
            "pipeline":     pipeline,
            "scoring_mode": scoring_mode,
            "entry":        entry,
        },
        "vix":           vix,
        "vix_regime":    vix_regime,
        "kill_switch":   kill_switch,
        "ticker_count":  1,
        "source_run_id": source_run_id,
        "data_as_of":    on_demand_as_of.isoformat(),
        "generated_at":  datetime.now(timezone.utc).isoformat(),
    }

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(
        f"[COOK] On-demand payload -> {output} "
        f"({ticker} via {pipeline}, score={entry.get('final_score')}, "
        f"vix={vix}, kill_switch={kill_switch})"
    )
    return combined


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge US + INTL top_lists artifacts into a combined payload"
    )
    parser.add_argument(
        "--us-input",
        default="logs/top_lists_us.json",
        help="Path to US top_lists.json (9-factor, from run_pipeline.py)",
    )
    parser.add_argument(
        "--intl-input",
        default="logs/top_lists_intl.json",
        help="Path to INTL top_lists_intl.json (6-factor StrategyEngine output)",
    )
    parser.add_argument(
        "--registry",
        default="config/ticker_registry.json",
        help="Path to ticker_registry.json for EU/Asia region classification",
    )
    parser.add_argument(
        "--output",
        default="logs/top_lists.json",
        help="Destination for combined payload (overwrites US input by default)",
    )
    parser.add_argument(
        "--on-demand-ticker",
        default=None,
        help="ChatOps single-ticker mode: emit an on_demand_ticker payload for "
             "exactly this ticker (US route reads --us-input, INTL route reads "
             "--intl-input + --registry) instead of the merged daily toplists",
    )
    args = parser.parse_args()

    us_input = Path(args.us_input)
    intl_input = Path(args.intl_input)
    registry = Path(args.registry)
    output = Path(args.output)

    if args.on_demand_ticker:
        # Only the routed input is required — the other artifact does not
        # exist in an on-demand workflow run.
        try:
            cook_on_demand(
                ticker=args.on_demand_ticker,
                us_input=us_input,
                intl_input=intl_input,
                registry=registry,
                output=output,
            )
            return 0
        except (ValueError, FileNotFoundError) as exc:
            print(f"[COOK] ERROR: {exc}", file=sys.stderr)
            return 1

    if not us_input.exists():
        print(f"[COOK] ERROR: US input not found: {us_input}", file=sys.stderr)
        return 1
    if not intl_input.exists():
        print(f"[COOK] ERROR: INTL input not found: {intl_input}", file=sys.stderr)
        return 1

    cook(us_input, intl_input, registry, output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
