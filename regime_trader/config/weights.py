# Path: regime_trader/config/weights.py
#
# WEIGHTS — canonical source for all scoring weight sets.
# Version: v2.2-global (2026-06)
#
# PATCH v2.2 (2026-06):
# WEIGHTS_GLOBAL updated now that FMP Ultimate covers insider/news/analyst
# globally. The previous v2.1 redistribution was based on the assumption that
# insider/news/analyst were absent for EU/Asia — that was wrong.
#
# Only TWO factors are absent vs US:
#   congress (0.22)        — structurally absent (no STOCK Act equivalent)
#   transcript_tone (0.00) — FMP earning-call-transcript-latest US-only
#
# Total freed: 0.22, redistributed per academic evidence:
#   analyst_consensus  +0.10 (Givoly & Lakonishok 1979; stronger outside US)
#   news_sentiment     +0.03 (Tetlock 2007; global news corpus via FMP)
#   momentum_long      +0.02 (Rouwenhorst 1998 EU momentum premium)
#   volume_attention   +0.02
#   quality_piotroski  +0.05 (Piotroski 2000; accounting-identity, universal)
#
# insider_conviction stays at 0.30 (same as US — MAR Art.19 = same quality
# as SEC Form 4 per Seyhun 1998; no redistribution needed).
#
# Weights sum check enforced at module load time via assert.
# Any modification must maintain sum == 1.0.

WEIGHTS_VERSION = "v2.2-global"

# ── US universe (Sprint v2.3: analyst_consensus and quality_piotroski activated) ────
# analyst_consensus (0.10) and quality_piotroski (0.08) funded from congress:
# Congress weight reduced 0.22 → 0.04 (congress is US-structural; IC/breadth/quality
# decay faster than quality/consensus signals at this allocation).
WEIGHTS_US: dict[str, float] = {
    "insider_conviction": 0.30,
    "insider_breadth":    0.15,
    "congress":           0.04,
    "news_sentiment":     0.10,
    "news_buzz":          0.05,
    "momentum_long":      0.15,
    "volume_attention":   0.03,
    "analyst_consensus":  0.10,
    "quality_piotroski":  0.08,
}
assert abs(sum(WEIGHTS_US.values()) - 1.0) < 1e-6, (
    f"WEIGHTS_US sums to {sum(WEIGHTS_US.values()):.8f}, not 1.0"
)

# ── Global universe — EU / Asia ────────────────────────────────────────────────
#
# congress = 0.00  (structural absence — STOCK Act is US-only)
# transcript_tone = 0.00 (FMP earning-call-transcript-latest US-only)
#
# Net changes vs WEIGHTS_US (v2.3 sprint — activating 4 wired-but-zeroed factors):
#   insider_conviction  0.30 → 0.28  (−0.02 donor — MAR Art.19 parity maintained)
#   insider_breadth     0.15 → 0.14  (−0.01 donor)
#   news_buzz           0.05 → 0.04  (−0.01 donor — lowest IC)
#   volume_attention    0.05 → 0.04  (−0.01 donor)
#   analyst_revision    0.00 → 0.02  (+0.02 — Chan, Jegadeesh & Lakonishok 1996)
#   price_target_upside 0.00 → 0.03  (+0.03 — Brav & Lehavy 2003)
#   congress            0.22 → 0.00  (structurally absent)
#   transcript_tone     —   → 0.00  (structurally absent)
WEIGHTS_GLOBAL: dict[str, float] = {
    "insider_conviction":  0.28,   # −0.02 vs US — MAR Art.19 parity maintained
    "insider_breadth":     0.14,   # −0.01 vs US
    "congress":            0.00,   # structurally absent outside US
    "news_sentiment":      0.13,   # +0.03 — global news corpus via FMP
    "news_buzz":           0.04,   # −0.01 donor
    "momentum_long":       0.17,   # +0.02 — Rouwenhorst 1998 EU premium
    "volume_attention":    0.04,   # −0.01 donor
    "analyst_consensus":   0.10,   # +0.10 — stronger signal in less-covered markets
    "quality_piotroski":   0.05,   # +0.05 — accounting-identity, universal
    "analyst_revision":    0.02,   # activated — Chan, Jegadeesh & Lakonishok 1996
    "price_target_upside": 0.03,   # activated — Brav & Lehavy 2003
    "transcript_tone":     0.00,   # structurally absent outside US
}
assert abs(sum(WEIGHTS_GLOBAL.values()) - 1.0) < 1e-6, (
    f"WEIGHTS_GLOBAL sums to {sum(WEIGHTS_GLOBAL.values()):.8f}, not 1.0"
)

# ── European universe — Quality-Core Model ─────────────────────────────────────
# quality_piotroski + analyst_revision + price_target_upside >= 0.55 (quality pillar)
# Piotroski 2000; Chan, Jegadeesh & Lakonishok 1996; Brav & Lehavy 2003
WEIGHTS_EU: dict[str, float] = {
    "insider_conviction":  0.12,
    "insider_breadth":     0.06,
    "congress":            0.00,   # structurally absent
    "news_sentiment":      0.05,
    "news_buzz":           0.02,
    "momentum_long":       0.08,   # Rouwenhorst 1998 (moderated for EU)
    "volume_attention":    0.02,
    "analyst_consensus":   0.07,
    "quality_piotroski":   0.28,   # dominant pillar — Piotroski 2000
    "analyst_revision":    0.15,   # Chan, Jegadeesh & Lakonishok 1996
    "price_target_upside": 0.15,   # Brav & Lehavy 2003
    "transcript_tone":     0.00,   # structurally absent outside US
}
assert abs(sum(WEIGHTS_EU.values()) - 1.0) < 1e-6, (
    f"WEIGHTS_EU sums to {sum(WEIGHTS_EU.values()):.8f}, not 1.0"
)

# ── Asian universe — Liquidity & Momentum Model ────────────────────────────────
# momentum_long + volume_attention >= 0.35 (liquidity/momentum pillar)
# Rouwenhorst 1998; Gervais & Odean 2001; Givoly & Lakonishok 1979; Tetlock 2007
WEIGHTS_ASIA: dict[str, float] = {
    "insider_conviction":  0.15,
    "insider_breadth":     0.07,
    "congress":            0.00,   # structurally absent
    "news_sentiment":      0.15,   # Tetlock 2007
    "news_buzz":           0.08,
    "momentum_long":       0.25,   # Rouwenhorst 1998 — APAC momentum premium
    "volume_attention":    0.10,   # Gervais & Odean 2001
    "analyst_consensus":   0.12,   # Givoly & Lakonishok 1979
    "quality_piotroski":   0.03,
    "analyst_revision":    0.03,
    "price_target_upside": 0.02,
    "transcript_tone":     0.00,   # structurally absent outside US
}
assert abs(sum(WEIGHTS_ASIA.values()) - 1.0) < 1e-6, (
    f"WEIGHTS_ASIA sums to {sum(WEIGHTS_ASIA.values()):.8f}, not 1.0"
)

# ── Convenience alias (legacy callers expecting WEIGHTS get US set) ────────────
WEIGHTS = WEIGHTS_US

# ── Piotroski F-Score gate (Piotroski 2000, JAR) ─────────────────────────────
# Applied as a multiplicative gate on the final BUY score after weighted sum.
#   F-Score < suppress_below → multiplier 0.0  (BUY suppressed)
#   F-Score < discount_below → multiplier discount_factor
#   F-Score ≥ discount_below → multiplier 1.0  (full weight)
# missing_score is used when the endpoint returns no data (conservative default).
PIOTROSKI_GATE: dict[str, float] = {
    "suppress_below":  3,
    "discount_below":  6,
    "discount_factor": 0.6,
    "missing_score":   3,
}

# ── Region classifier ─────────────────────────────────────────────────────────
_EU_SUFFIXES: frozenset[str] = frozenset({
    ".PA", ".DE", ".L", ".AS", ".MI", ".MC", ".BR",
    ".VX", ".LS", ".OL", ".ST", ".HE", ".CO", ".F", ".BE",
})

_ASIA_SUFFIXES: frozenset[str] = frozenset({
    ".T", ".HK", ".KS", ".KQ", ".SS", ".SZ",
    ".NS", ".BO", ".SI", ".BK", ".JK",
})

_GLOBAL_SUFFIXES: frozenset[str] = _EU_SUFFIXES | _ASIA_SUFFIXES


def get_region(ticker: str) -> str:
    """Return 'EU', 'ASIA', or 'US' for a given ticker symbol.

    Uses suffix matching only — no external lookup required.
    Unrecognised suffixes default to 'US' (conservative).
    """
    upper = ticker.upper()
    dot_idx = upper.rfind(".")
    if dot_idx == -1:
        return "US"
    suffix = upper[dot_idx:]
    if suffix in _EU_SUFFIXES:
        return "EU"
    if suffix in _ASIA_SUFFIXES:
        return "ASIA"
    return "US"


def get_weights(ticker: str) -> dict[str, float]:
    """Return the correct weight set for ticker (copy — callers may not mutate)."""
    region = get_region(ticker)
    if region == "EU":
        return dict(WEIGHTS_EU)
    if region == "ASIA":
        return dict(WEIGHTS_ASIA)
    return dict(WEIGHTS_US)


def is_congress_eligible(ticker: str) -> bool:
    """True only for US tickers where congress signal is structurally available."""
    return get_region(ticker) == "US"
