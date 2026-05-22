"""regime_trader/ui/portfolio_advisor_engine.py
Hybrid scoring engine for the Portfolio Advisor page.

Reads scores from logs/intel_source_status.json (no new API calls).
Optionally generates a 2-sentence Claude narrative per position.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

log = logging.getLogger(__name__)

_ROOT               = Path(__file__).parent.parent.parent
_STATUS_PATH        = _ROOT / "logs" / "intel_source_status.json"
_TOP_LISTS_PATH     = _ROOT / "logs" / "top_lists.json"

_KILL_SWITCH_REGIMES = {"Crash", "Panic"}


@dataclass
class PositionAdvice:
    ticker:           str
    revolut_ticker:   str
    net_qty:          float
    avg_cost:         float
    currency:         str
    source:           str
    signal:           str          # ADD | HOLD | REDUCE | EXIT
    final_score:      Optional[float]
    factors:          Dict[str, float]
    signal_age_days:  Optional[int]
    swap_candidate:   Optional[Dict[str, Any]]
    narrative:        Optional[str]   # Claude 2-sentence text, populated later
    not_in_universe:  bool
    market_value:     float = 0.0    # filled by UI from live price

    # Evidence pass-through fields (populated from intel_source_status.json)
    news_source:             str   = "none"
    insider_usd:             float = 0.0
    momentum_spy_relative:   float = 0.0
    volume_spike:            float = 1.0
    quiver_evidence:         Dict[str, Any] = field(default_factory=dict)


# ── Core logic (pure, testable) ───────────────────────────────────────────────

def compute_signal(score: float, regime: str) -> str:
    if regime in _KILL_SWITCH_REGIMES:
        return "EXIT"
    if score >= 0.65:
        return "ADD"
    if score >= 0.45:
        return "HOLD"
    if score >= 0.30:
        return "REDUCE"
    return "EXIT"


def compute_health_score(positions: List[Dict[str, Any]]) -> float:
    """Weighted average final_score by market_value across all positions."""
    total_value = sum(p.get("market_value", 0.0) for p in positions)
    if total_value <= 0:
        return 0.0
    return sum(
        p.get("final_score", 0.0) * p.get("market_value", 0.0)
        for p in positions
    ) / total_value


def _signal_age_days(status: Dict[str, Any]) -> Optional[int]:
    raw = status.get("computed_at")
    if not raw:
        return None
    try:
        ts = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return (datetime.now(timezone.utc) - ts).days
    except Exception:
        return None


def find_swap_candidate(
    ticker: str,
    sector: str,
    held_tickers: Set[str],
    top_lists: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Return the top-scored unowned ticker in the same sector, or None."""
    all_entries = (
        top_lists.get("top_buys", []) +
        top_lists.get("mid_caps", []) +
        top_lists.get("small_caps", [])
    )
    candidates = [
        e for e in all_entries
        if e.get("sector") == sector
        and e.get("ticker") != ticker
        and e.get("ticker") not in held_tickers
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda e: e.get("final_score", 0.0))


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_status() -> Dict[str, Any]:
    try:
        return json.loads(_STATUS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception as exc:
        log.warning("intel_source_status.json load failed: %s", exc)
        return {}


def _load_top_lists() -> Dict[str, Any]:
    try:
        return json.loads(_TOP_LISTS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except Exception as exc:
        log.warning("top_lists.json load failed: %s", exc)
        return {}


def _build_score_index(status: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build ticker -> scored_row lookup from intel_source_status.json."""
    return {r["ticker"]: r for r in status.get("results", []) if r.get("ticker")}


# ── Public API ────────────────────────────────────────────────────────────────

def build_advice(
    positions: List[Dict[str, Any]],
    regime: str,
) -> List[PositionAdvice]:
    """Score and signal all positions. Returns PositionAdvice list."""
    status     = _load_status()
    top_lists  = _load_top_lists()
    score_idx  = _build_score_index(status)
    age_days   = _signal_age_days(status)
    held       = {p["ticker"] for p in positions}

    advice_list = []
    for pos in positions:
        ticker   = pos["ticker"]
        row      = score_idx.get(ticker)

        if row is None:
            advice_list.append(PositionAdvice(
                ticker          = ticker,
                revolut_ticker  = pos.get("revolut_ticker", ticker),
                net_qty         = pos["net_qty"],
                avg_cost        = pos["avg_cost"],
                currency        = pos.get("currency", "USD"),
                source          = pos.get("source", "revolut"),
                signal          = "—",
                final_score     = None,
                factors         = {},
                signal_age_days = age_days,
                swap_candidate  = None,
                narrative       = None,
                not_in_universe = True,
            ))
            continue

        final_score = float(
            row.get("edgar_score", 0) * 0.28 +
            row.get("insider_score", 0) * 0.23 +
            row.get("congress_score", 0) * 0.22 +
            row.get("news_score", 0) * 0.15 +
            row.get("momentum_score", 0) * 0.12
        )
        signal = compute_signal(final_score, regime)
        swap   = find_swap_candidate(ticker, row.get("sector", ""), held, top_lists) \
                 if signal in ("REDUCE", "EXIT") else None

        advice_list.append(PositionAdvice(
            ticker          = ticker,
            revolut_ticker  = pos.get("revolut_ticker", ticker),
            net_qty         = pos["net_qty"],
            avg_cost        = pos["avg_cost"],
            currency        = pos.get("currency", "USD"),
            source          = pos.get("source", "revolut"),
            signal          = signal,
            final_score     = round(final_score, 4),
            factors         = {
                "edgar":    round(float(row.get("edgar_score",   0)), 4),
                "insider":  round(float(row.get("insider_score", 0)), 4),
                "congress": round(float(row.get("congress_score",0)), 4),
                "news":     round(float(row.get("news_score",    0)), 4),
                "macro":    round(float(row.get("momentum_score",0)), 4),
            },
            signal_age_days       = age_days,
            swap_candidate        = swap,
            narrative             = None,
            not_in_universe       = False,
            news_source           = row.get("news_source", "none"),
            insider_usd           = float(row.get("insider_usd", 0.0)),
            momentum_spy_relative = float(row.get("momentum_spy_relative", 0.0)),
            volume_spike          = float(row.get("volume_spike", 1.0)),
            quiver_evidence       = row.get("quiver_evidence", {}),
        ))

    return advice_list
