"""backend/automation/alert_handler.py
Deterministic alert rules and portfolio action handlers -- Minsky (FIH).

Three Minsky preconditions map to four escalation levels with explicit portfolio
actions. All trigger thresholds are top-of-file constants -- change them here
and the logic throughout the module updates automatically.

Persistence formula (Engle 2003 Nobel):
  P = alpha + beta + gamma/2  ->  triggers when P >= 0.98

CAPE percentile (Shiller 2013 Nobel):
  Triggers when CAPE percentile >= 95 (top 5% of historical valuations)

Yield spread (Friedman 1968 Nobel):
  Triggers when 10Y-2Y spread < 0 bps (yield curve inversion)

Alert levels:
  0/3  CLEAR    -- normal monitoring
  1/3  WATCH    -- tighten risk limits, -10% leverage, intraday monitoring
  2/3  WARNING  -- trim cyclical 20%, buy protective puts 10% notional
  3/3  CRITICAL -- de-risk to defensive, notify stakeholders, continuous monitoring

Production wiring:
  Replace the stub functions (reduce_leverage, buy_protective_puts, etc.) with
  calls to your execution layer (Alpaca, IBKR, internal order management).
  Replace notify_team() with Slack/PagerDuty/email integrations.

Run standalone:
    python -m backend.automation.alert_handler
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Deterministic trigger thresholds ─────────────────────────────────────────

PERSISTENCE_THRESHOLD   = 0.98    # GARCH alpha + beta + gamma/2 (Engle 2003)
CAPE_PERCENTILE_THRESHOLD = 95.0  # Shiller CAPE vs 40-yr history (Shiller 2013)
YIELD_SPREAD_THRESHOLD_BPS = 0.0  # 10Y-2Y in bps -- inversion (Friedman 1968)

# ── Alert level labels ────────────────────────────────────────────────────────

ALERT_LEVELS: Dict[int, str] = {
    0: "CLEAR",
    1: "WATCH",
    2: "WARNING",
    3: "CRITICAL",
}

# ── Notification message templates ───────────────────────────────────────────

NOTIFY_MESSAGES: Dict[int, str] = {
    1: "MINSKY WATCH: 1/3 conditions met -- tightening risk limits and monitoring intraday.",
    2: "MINSKY WARNING: 2/3 conditions met -- trimming cyclical exposure and buying protection.",
    3: "MINSKY CRITICAL: 3/3 conditions met -- IMMEDIATE DE-RISK to defensive allocation. All stakeholders notified.",
}


# ── Portfolio data class ──────────────────────────────────────────────────────

@dataclass
class Portfolio:
    """Minimal portfolio representation for alert handler actions.

    Replace with your actual portfolio object in production; the action
    stubs call methods on this object by convention.
    """
    value: float
    leverage: float = 1.0
    cyclical_weight: float = 0.40   # fraction of portfolio in cyclical sectors
    hedge_notional: float = 0.0     # current protective-put notional outstanding
    mode: str = "normal"            # "normal" | "intraday" | "continuous"
    actions_taken: List[str] = field(default_factory=list)


# ── Core trigger evaluation ───────────────────────────────────────────────────

def evaluate_minsky_conditions(
    persistence: float,
    cape_percentile: float,
    yield_spread_bps: float,
) -> int:
    """Engle/Shiller/Friedman -- Count breached Minsky preconditions (0–3).

    Each condition is binary (triggered / not triggered):
      persistence_trigger = persistence >= PERSISTENCE_THRESHOLD
      cape_trigger        = cape_percentile >= CAPE_PERCENTILE_THRESHOLD
      yield_trigger       = yield_spread_bps < YIELD_SPREAD_THRESHOLD_BPS

    Args:
        persistence:      GJR-GARCH P = alpha + beta + gamma/2.
        cape_percentile:  Current CAPE percentile vs 40-year history.
        yield_spread_bps: 10Y-2Y treasury spread in basis points.

    Returns:
        Integer 0–3 representing how many conditions are triggered.
    """
    persistence_trigger = persistence >= PERSISTENCE_THRESHOLD
    cape_trigger        = cape_percentile >= CAPE_PERCENTILE_THRESHOLD
    yield_trigger       = yield_spread_bps < YIELD_SPREAD_THRESHOLD_BPS

    conditions_met = int(persistence_trigger) + int(cape_trigger) + int(yield_trigger)

    logger.debug(
        "Minsky eval: P=%.4f(trig=%s), CAPE=%.1f(trig=%s), spread=%.1fbps(trig=%s) -> %d/3",
        persistence, persistence_trigger,
        cape_percentile, cape_trigger,
        yield_spread_bps, yield_trigger,
        conditions_met,
    )
    return conditions_met


def get_alert_level(conditions_met: int) -> str:
    """Return the alert level label for a given conditions count."""
    return ALERT_LEVELS.get(conditions_met, "CRITICAL")


# ── Portfolio action dispatcher ───────────────────────────────────────────────

def handle_conditions(conditions_met: int, portfolio: Portfolio) -> List[str]:
    """Minsky (FIH) -- Execute deterministic risk actions based on conditions count.

    Dispatch table:
      0/3 CLEAR    -- no action
      1/3 WATCH    -- intraday monitoring + reduce leverage 10%
      2/3 WARNING  -- trim cyclical 20% + buy protective puts 10% notional
      3/3 CRITICAL -- full de-risk to defensive + stakeholder notification

    Args:
        conditions_met: Integer 0–3 from evaluate_minsky_conditions().
        portfolio:      Portfolio object to mutate.

    Returns:
        List of action description strings (for logging / audit trail).
    """
    level = get_alert_level(conditions_met)
    actions: List[str] = [f"Alert level: {level} ({conditions_met}/3 conditions)"]

    if conditions_met == 0:
        logger.info("[ALERT] CLEAR -- no Minsky thresholds breached")

    elif conditions_met == 1:
        set_monitoring(portfolio, "intraday")
        reduce_leverage(portfolio, 0.10)
        actions += [
            "Monitoring escalated to intraday",
            f"Leverage reduced 10% -> {portfolio.leverage:.3f}",
        ]
        logger.warning("[ALERT] WATCH: %s", NOTIFY_MESSAGES[1])
        notify_team(NOTIFY_MESSAGES[1], level="WARNING")

    elif conditions_met == 2:
        trim_exposure(portfolio, "cyclical", 0.20)
        hedge_notional = 0.10 * portfolio.value
        buy_protective_puts(portfolio, notional=hedge_notional)
        actions += [
            "Cyclical exposure trimmed 20%",
            f"Protective puts purchased: ${hedge_notional:,.0f} notional",
        ]
        logger.error("[ALERT] WARNING: %s", NOTIFY_MESSAGES[2])
        notify_team(NOTIFY_MESSAGES[2], level="ERROR")

    elif conditions_met == 3:
        de_risk_to_defensive(portfolio)
        actions += [
            "Portfolio de-risked to defensive allocation (leverage=0)",
            f"Stakeholders notified: '{NOTIFY_MESSAGES[3]}'",
        ]
        logger.critical("[ALERT] CRITICAL: %s", NOTIFY_MESSAGES[3])
        notify_team(NOTIFY_MESSAGES[3], level="CRITICAL")

    portfolio.actions_taken.extend(actions)
    return actions


# ── Portfolio action stubs ────────────────────────────────────────────────────
# Replace with real execution layer calls in production.

def set_monitoring(portfolio: Portfolio, frequency: str) -> None:
    """Escalate monitoring cadence ('intraday', 'hourly', 'continuous')."""
    portfolio.mode = frequency
    logger.warning("[ACTION] Monitoring frequency set to: %s", frequency)


def reduce_leverage(portfolio: Portfolio, fraction: float) -> None:
    """Reduce portfolio leverage by fraction (e.g. 0.10 = reduce by 10%).

    Production: submit market orders to close leveraged positions proportionally.
    """
    old = portfolio.leverage
    portfolio.leverage = round(old * (1.0 - fraction), 6)
    logger.warning("[ACTION] Leverage %s -> %s (reduced %.0f%%)",
                   old, portfolio.leverage, fraction * 100)


def trim_exposure(portfolio: Portfolio, segment: str, fraction: float) -> None:
    """Trim segment exposure by fraction.

    Production: identify cyclical sector ETFs / positions and sell fraction.
    """
    if segment == "cyclical":
        old = portfolio.cyclical_weight
        portfolio.cyclical_weight = round(old * (1.0 - fraction), 6)
        logger.error("[ACTION] Cyclical weight %s -> %s (trimmed %.0f%%)",
                     old, portfolio.cyclical_weight, fraction * 100)
    else:
        logger.error("[ACTION] trim_exposure: unknown segment '%s'", segment)


def buy_protective_puts(portfolio: Portfolio, notional: float) -> None:
    """Buy protective index puts for the given notional.

    Production: route to options desk / broker API with delta-neutral sizing.
    """
    portfolio.hedge_notional += notional
    logger.error("[ACTION] Bought protective puts: $%.0f notional (total hedge: $%.0f)",
                 notional, portfolio.hedge_notional)


def de_risk_to_defensive(portfolio: Portfolio) -> None:
    """Full de-risk: close risk positions, move to treasuries + cash.

    Production: flatten all equity and leveraged positions, move to short-duration
    treasuries and cash equivalents.
    """
    portfolio.leverage = 0.0
    portfolio.cyclical_weight = 0.0
    portfolio.mode = "continuous"
    logger.critical("[ACTION] DE-RISKED to defensive allocation (leverage=0, cyclical=0)")


def notify_team(message: str, level: str = "WARNING") -> None:
    """Send alert to stakeholders.

    Production: wire to Slack via webhook, PagerDuty API, or SMTP.
    Replace the logger call below with your notification provider.
    """
    logger.critical("[NOTIFY|%s] %s", level, message)
    # Example Slack webhook (replace URL and uncomment in production):
    # import requests
    # requests.post(
    #     os.getenv("SLACK_WEBHOOK_URL", ""),
    #     json={"text": f"[{level}] {message}"},
    #     timeout=5,
    # )


# ── Standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.DEBUG, format="%(levelname)-8s %(name)s : %(message)s")

    scenarios = [
        {"persistence": 0.95, "cape_percentile": 80.0,  "yield_spread_bps":  50.0,  "label": "Calm (0/3)"},
        {"persistence": 0.99, "cape_percentile": 80.0,  "yield_spread_bps":  50.0,  "label": "High Vol (1/3)"},
        {"persistence": 0.99, "cape_percentile": 96.0,  "yield_spread_bps":  50.0,  "label": "Val+Vol (2/3)"},
        {"persistence": 0.99, "cape_percentile": 96.0,  "yield_spread_bps": -30.0,  "label": "FULL MINSKY (3/3)"},
    ]

    print(f"\n{'='*70}")
    print(f"  MINSKY ALERT HANDLER -- SCENARIO DEMO")
    print(f"{'='*70}")

    for s in scenarios:
        pf = Portfolio(value=1_000_000)
        n = evaluate_minsky_conditions(s["persistence"], s["cape_percentile"], s["yield_spread_bps"])
        act = handle_conditions(n, pf)
        print(f"\n  Scenario: {s['label']}")
        print(f"    Conditions: {n}/3  |  Level: {get_alert_level(n)}")
        for a in act:
            print(f"    > {a}")

    print(f"\n{'='*70}\n")
    sys.exit(0)
