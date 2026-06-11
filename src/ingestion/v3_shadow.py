# Path: src/ingestion/v3_shadow.py
"""src/ingestion/v3_shadow.py
v3.0 shadow-mode wiring (migration step 5; plan "Scoring Engine v3.0").

SCORING_V3_SHADOW=1 makes the US pipeline emit `final_score_v3`,
`pillar_*_score`, `weight_coverage_v3` ALONGSIDE the untouched v2.2 fields:
v2.2 stays authoritative until cutover. Shadow failures must never break
the production path — they degrade to unavailable factors and log loudly.

Two halves:
    compute_v3_raw_columns()  — called inside _score_ticker with data already
        in scope (EDGAR P-transactions, ratios record, revision/PEAD inputs)
        plus two cached client calls (cash-flow/EV for fcf_yield, 13F flow).
    apply_v3_shadow()         — called after the v2.2 final-score block;
        builds the engine input rows, runs score_universe_v3(region="US"),
        and merges the v3 output columns back.

Column-name notes:
    analyst_revision_score_v3 / congress_score_v3 are separate keys because
    the v2.2 row already carries same-named factors with DIFFERENT math
    (v2.2 revision damps toward 0; v3 damps toward 0.5).
    congress surge multiplier is wired as identity for now —
    fetch_congress_buys() aggregates counts without transaction dates, so
    the 30d/180d acceleration split needs payload plumbing first (planned
    alongside cutover; congress_surge_multiplier() is implemented + tested).
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.config.factor_matrix import FACTOR_MATRIX_V3
from src.scoring.alt_signals import score_insider_alpha, score_inst_flow_13f
from src.scoring.consensus_signals import (
    score_analyst_revision,
    score_pead_surprise,
)
from src.scoring.engine_v3 import score_universe_v3
from src.scoring.fundamental_signals import score_fcf_yield, score_quality_dupont

log = logging.getLogger(__name__)


def v3_shadow_enabled() -> bool:
    return os.getenv("SCORING_V3_SHADOW", "") == "1"


def _g(row: Optional[Dict], *names: str) -> Optional[float]:
    """TTM-suffix-tolerant field getter (FMP field-drift defense, risk #3)."""
    for name in names:
        value = (row or {}).get(name)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _purchase_windows(p_transactions: List[Dict]) -> tuple[int, int, float, float]:
    """(p_count_30d, p_count_31_180d, usd_buy_csuite, usd_buy_total).

    P-code only by construction — the caller passes p_transactions, sales
    never reach this module [Cohen, Malloy & Pomorski 2012].
    """
    today = datetime.now(timezone.utc).date()
    p30 = p31_180 = 0
    usd_total = usd_csuite = 0.0
    for tx in p_transactions or []:
        try:
            age = (today - datetime.fromisoformat(str(tx.get("date", ""))[:10]).date()).days
        except ValueError:
            age = 180  # unparseable date — count as oldest in-window (conservative)
        if age <= 30:
            p30 += 1
        elif age <= 180:
            p31_180 += 1
        else:
            continue  # outside lookback entirely
        value = float(tx.get("value") or 0.0)
        usd_total += value
        if tx.get("is_ceo"):
            usd_csuite += value
    return p30, p31_180, usd_csuite, usd_total


def compute_v3_raw_columns(
    *,
    ticker: str,
    fmp_client: Any,
    ratios_row: Optional[Dict],
    p_transactions: List[Dict],
    conviction_score: float,
    breadth_score: float,
    revision_pct: Optional[float],
    n_analysts: int,
    eps_surprise_pct: Optional[float],
    eps_surprise_days: int,
    congress_score: float,
) -> Dict[str, Any]:
    """v3 raw factor columns from _score_ticker-scope data (US shadow)."""
    cols: Dict[str, Any] = {}

    # ── P1.1 quality_dupont (bulk record first; cached refetch otherwise) ──
    ratios = ratios_row or fmp_client.get_ratios_ttm(ticker) or {}
    cols["quality_dupont_score"] = score_quality_dupont(
        roa=_g(ratios, "returnOnAssetsTTM", "returnOnAssets"),
        npm=_g(ratios, "netProfitMarginTTM", "netProfitMargin"),
        asset_turnover=_g(ratios, "assetTurnoverTTM", "assetTurnover"),
        debt_to_equity=_g(ratios, "debtToEquityRatioTTM", "debtToEquityRatio",
                          "debtEquityRatioTTM"),
    )

    # ── P1.2 fcf_yield (cached endpoints; new to the US path) ──────────────
    try:
        cf_rows = fmp_client.get_cash_flow_statements(ticker, limit=4) or []
        fcf_ttm = sum(float(r.get("freeCashFlow") or 0.0) for r in cf_rows)
        ev = fmp_client.get_enterprise_value(ticker)
        cols["fcf_yield_score"] = (
            score_fcf_yield(fcf_ttm, ev) if ev else 0.0
        )
    except Exception as exc:  # shadow must never break the v2.2 path
        log.warning("v3 shadow fcf_yield %s soft-fail: %s", ticker, exc)
        cols["fcf_yield_score"] = 0.0

    # ── P2 consensus (inputs already fetched by the v2.2 path) ─────────────
    cols["analyst_revision_score_v3"] = score_analyst_revision(
        revision_pct, n_analysts)
    cols["pead_surprise_score"] = score_pead_surprise(
        eps_surprise_pct, eps_surprise_days)
    # price_target_upside_score: reused directly from the v2.2 row (same
    # scorer, now hardened in the client with pairing/GBX/ratio guards).

    # ── P3 alternative flow ────────────────────────────────────────────────
    p30, p31_180, usd_csuite, usd_total = _purchase_windows(p_transactions)
    cols["insider_alpha_score"] = score_insider_alpha(
        conviction=conviction_score,
        breadth_residual=breadth_score,
        p_count_30d=p30,
        p_count_31_180d=p31_180,
        usd_buy_csuite=usd_csuite,
        usd_buy_total=usd_total,
    )
    cols["congress_score_v3"] = congress_score  # surge mult: see module docstring
    try:
        summary = fmp_client.get_institutional_ownership(ticker)
        cols["inst_flow_13f_score"] = score_inst_flow_13f(summary)
    except Exception as exc:
        log.warning("v3 shadow 13F %s soft-fail: %s — unavailable", ticker, exc)
        cols["inst_flow_13f_score"] = None

    return cols


# Engine factor name → v2.2 row column carrying its raw v3 value.
_V3_INPUT_MAP: Dict[str, str] = {
    "quality_dupont": "quality_dupont_score",
    "fcf_yield": "fcf_yield_score",
    "quality_piotroski": "quality_piotroski_score",
    "analyst_revision": "analyst_revision_score_v3",
    "pead_surprise": "pead_surprise_score",
    "price_target_upside": "price_target_upside_score",
    "insider_alpha": "insider_alpha_score",
    "congress": "congress_score_v3",
    "inst_flow_13f": "inst_flow_13f_score",
}
assert set(_V3_INPUT_MAP) == set(FACTOR_MATRIX_V3["US"]), (
    "_V3_INPUT_MAP drifted from the US factor matrix"
)

_V3_OUTPUT_KEYS = (
    "final_score_v3",
    "weight_coverage_v3",
    "_low_coverage_v3",
    "pillar_fundamental_score",
    "pillar_consensus_score",
    "pillar_alternative_score",
)


def apply_v3_shadow(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Score the US universe under v3 and merge v3 fields onto v2.2 rows.

    v2.2 keys are never modified. Engine inputs are built on copies so the
    name collisions (analyst_revision/congress v2.2-vs-v3 math) cannot bleed.
    """
    if not results:
        return results
    inputs = []
    for r in results:
        row = {
            "ticker": r.get("ticker"),
            "sector": r.get("sector", "Unknown"),
            "cap_tier": r.get("cap_tier", "large"),
            "market": r.get("market", "USA"),
            "quality_piotroski_raw": r.get("quality_piotroski_raw"),
        }
        for factor, column in _V3_INPUT_MAP.items():
            row[f"{factor}_score"] = r.get(column)
        inputs.append(row)

    try:
        scored = score_universe_v3(inputs, "US")
    except ValueError as exc:
        log.error("v3 shadow scoring failed: %s — v2.2 output unaffected", exc)
        return results

    for r, s in zip(results, scored):
        for key in _V3_OUTPUT_KEYS:
            r[key] = s.get(key)
        for key in ("_factor_blackout", "_contamination_masked"):
            if key in s:
                r[key] = s[key]
    return results
