"""decision_matrix/engine/config.py
All magic numbers in one place.  Change thresholds here and the logic
updates automatically throughout the engine.
"""

# ── Risk scoring ──────────────────────────────────────────────────────────────

RISK_THRESHOLDS = {
    "low":      30.0,
    "moderate": 55.0,
    "elevated": 75.0,
}

MAX_COMPONENT_RISK = 40.0    # max score per component (Regime/Concentration/Intel)
MAX_TOTAL_RISK_RAW = 110.0   # Regime(40) + Concentration(30) + Intel(30) + headroom

# ── Portfolio constraints ─────────────────────────────────────────────────────

SECTOR_CAP        = 0.25   # fraction of MV — breach triggers concentration warning
DAILY_DD_LIMIT    = 0.03   # daily drawdown limit (fraction)
MAX_POSITIONS     = 15     # informational cap shown in brief

# ── Volatility ────────────────────────────────────────────────────────────────

ATR_MULT_DEFAULT   = 3.0   # ATR stop multiplier in Bull/Neutral
ATR_MULT_DEFENSIVE = 2.0   # ATR stop multiplier in Bear/Crash

ATR_ALERT_THRESHOLD = 0.20  # ATR % above 30-day mean that triggers alert

# ── Minsky thresholds (Engle/Shiller/Friedman) ───────────────────────────────

PERSISTENCE_THRESHOLD      = 0.98   # GARCH alpha + beta + gamma/2
CAPE_PERCENTILE_THRESHOLD  = 95.0   # Shiller CAPE percentile vs 40-yr history
YIELD_SPREAD_THRESHOLD_BPS = 0.0    # 10Y-2Y bps — inversion

# ── Action urgency (lower int = higher urgency in sorted tables) ──────────────

ACTION_URGENCY = {
    "SELL":     0,
    "TRIM":     1,
    "BUY MORE": 2,
    "ADD":      3,
    "HOLD":     4,
}

# ── Regime action overrides ───────────────────────────────────────────────────
# block: actions that are forbidden in this regime
# force: what to convert them to

REGIME_OVERRIDES = {
    "Crash": {"block": {"BUY MORE", "ADD"}, "force": "SELL",  "reason": "Crash regime -- capital preservation"},
    "Panic": {"block": {"BUY MORE", "ADD"}, "force": "SELL",  "reason": "Panic regime -- reduce all exposure"},
    "Bear":  {"block": {"BUY MORE", "ADD"}, "force": "HOLD",  "reason": "Bear regime -- no new longs"},
}

# ── Conviction grade thresholds ───────────────────────────────────────────────
# (min_score, grade_label, hex_color)

CONVICTION_GRADES = [
    (0.70, "A", "#00c851"),
    (0.50, "B", "#ffbb33"),
    (0.00, "C", "#ff4444"),
]

# ── Regime risk scores (0-40) ─────────────────────────────────────────────────

REGIME_RISK_MAP = {
    "Bull":    10,
    "Euphoria": 20,
    "Mania":   35,
    "Neutral": 20,
    "Bear":    35,
    "Panic":   40,
    "Crash":   40,
    "Unknown": 25,
}
