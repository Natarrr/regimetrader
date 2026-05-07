"""backend/utils/triggers.py
Minsky precondition trigger logic with full provenance.

Returns a dict (not just an int) so the UI can display raw values alongside
the trigger booleans -- making the "1/3 conditions met" count auditable.

Thresholds (Engle/Shiller/Friedman):
    persistence >= 0.98   -- GARCH volatility clustering (Engle 2003)
    cape_percentile >= 95 -- Shiller CAPE extreme overvaluation (Shiller 2013)
    yield_spread < 0 bps  -- yield curve inversion (Friedman 1968)
"""
from __future__ import annotations

import datetime
import logging
from typing import Dict

logger = logging.getLogger(__name__)

# ── Thresholds (single source of truth) ──────────────────────────────────────

PERSISTENCE_THRESHOLD      = 0.98
CAPE_PERCENTILE_THRESHOLD  = 95.0
YIELD_SPREAD_THRESHOLD_BPS = 0.0


def compute_minsky_conditions(
    persistence: float,
    cape_pct: float,
    yield_bps: float,
) -> Dict:
    """Return provenance dict for Minsky precondition evaluation.

    Args:
        persistence: GJR-GARCH P = alpha + beta + gamma/2
        cape_pct:    Shiller CAPE percentile vs 40-year history [0-100]
        yield_bps:   10Y-2Y treasury spread in basis points

    Returns:
        Dict with raw values, trigger booleans, conditions_met count, and
        alert_level label.  Store this dict in your regime/session context
        and display raw values in the UI for full audit traceability.

    Example:
        >>> result = compute_minsky_conditions(0.9446665, 98.72, 52.0)
        >>> result["conditions_met"]
        1
        >>> result["persistence_trigger"]
        False
    """
    persistence_trigger = bool(persistence >= PERSISTENCE_THRESHOLD)
    cape_trigger        = bool(cape_pct    >= CAPE_PERCENTILE_THRESHOLD)
    yield_trigger       = bool(yield_bps   <  YIELD_SPREAD_THRESHOLD_BPS)

    conditions_met = int(persistence_trigger) + int(cape_trigger) + int(yield_trigger)
    alert_level    = ["CLEAR", "WATCH", "WARNING", "CRITICAL"][conditions_met]

    result = {
        "timestamp":           datetime.datetime.utcnow().isoformat(),
        "persistence":         round(float(persistence), 6),
        "persistence_trigger": persistence_trigger,
        "persistence_thresh":  PERSISTENCE_THRESHOLD,
        "cape_pct":            round(float(cape_pct), 2),
        "cape_trigger":        cape_trigger,
        "cape_thresh":         CAPE_PERCENTILE_THRESHOLD,
        "yield_bps":           round(float(yield_bps), 1),
        "yield_trigger":       yield_trigger,
        "yield_thresh":        YIELD_SPREAD_THRESHOLD_BPS,
        "conditions_met":      conditions_met,
        "alert_level":         alert_level,
    }

    logger.debug(
        "Minsky triggers: P=%.4f(trig=%s) CAPE=%.1f(trig=%s) spread=%.1fbps(trig=%s) -> %d/3 %s",
        persistence, persistence_trigger,
        cape_pct, cape_trigger,
        yield_bps, yield_trigger,
        conditions_met, alert_level,
    )
    return result


def minsky_ui_line(trace: Dict) -> str:
    """Format a one-line summary string for display in the Streamlit UI.

    Example output:
        "Minsky: 1/3 -- persistence 0.9447 (ok), CAPE 98.72% (TRIG), yield +52.0bps (ok)"
    """
    def _fmt(value, trigger: bool) -> str:
        return "TRIG" if trigger else "ok"

    return (
        f"Minsky {trace['conditions_met']}/3 [{trace['alert_level']}]"
        f"  --  persistence {trace['persistence']:.4f} ({_fmt(trace['persistence'], trace['persistence_trigger'])})"
        f"  |  CAPE {trace['cape_pct']:.1f}% ({_fmt(trace['cape_pct'], trace['cape_trigger'])})"
        f"  |  yield {trace['yield_bps']:+.1f}bps ({_fmt(trace['yield_bps'], trace['yield_trigger'])})"
    )
