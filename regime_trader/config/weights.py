# Path: regime_trader/config/weights.py
#
# WEIGHTS — canonical source for all scoring weight sets.
# Version: v2.1-global (2026-06)
#
# Two weight sets:
#   WEIGHTS_US     — full 9-factor set, US tickers only (EDGAR + S3 congress feeds)
#   WEIGHTS_GLOBAL — 8-factor set, EU/Asia tickers (congress structurally absent)
#
# Design rationale for WEIGHTS_GLOBAL
# ─────────────────────────────────────
# FMP Ultimate stable/ routes provide all factors globally EXCEPT congress:
#   insider-trading/search  — MAR-compliant EU disclosures + EDINET (JP) partial
#   upgrades-downgrades-consensus-bulk — global analyst coverage
#   ratios-ttm-bulk         — global Piotroski components (accounting-identity based)
#   news/stock              — global news corpus
#   historical-price-eod/full — global price/volume history
#
# congress (0.22 in WEIGHTS_US) has NO equivalent outside the US STOCK Act /
# S3 Stock Watcher feeds. Setting it to 0.0 and renormalising is the only
# academically honest approach — there is no proxy signal.
#
# Redistribution of the freed 0.22:
#   → insider_conviction  +0.08  (largest alpha source per Seyhun 1998;
#                                  EU MAR Art.19 disclosures carry similar signal)
#   → analyst_consensus   +0.07  (Givoly & Lakonishok 1979 estimate-revision
#                                  signal is stronger outside the US where fewer
#                                  retail investors process analyst upgrades)
#   → momentum_long       +0.04  (Jegadeesh & Titman 1993; EU momentum premium
#                                  documented by Rouwenhorst 1998)
#   → quality_piotroski   +0.03  (Piotroski 2000 F-Score is accounting-identity
#                                  based — universally applicable)
#
# Weights sum check is enforced by assert at module load time.
# Any modification must maintain sum == 1.0.

WEIGHTS_VERSION = "v2.1-global"

# ── US universe ───────────────────────────────────────────────────────────────
WEIGHTS_US: dict[str, float] = {
    "insider_conviction": 0.30,
    "insider_breadth":    0.15,
    "congress":           0.22,
    "news_sentiment":     0.10,
    "news_buzz":          0.05,
    "momentum_long":      0.15,
    "volume_attention":   0.03,
    "analyst_consensus":  0.00,   # wired sprint step 2
    "quality_piotroski":  0.00,   # wired sprint step 6
}
assert abs(sum(WEIGHTS_US.values()) - 1.0) < 1e-6, (
    f"WEIGHTS_US sums to {sum(WEIGHTS_US.values()):.8f}, not 1.0"
)

# ── Global universe — EU / Asia (congress absent, 0.22 redistributed) ─────────
#
# congress = 0.0  (structural absence — NOT a soft zero / missing data)
# The freed 0.22 is redistributed to the 4 factors with strongest
# cross-market empirical support (citations above).
#
# Net changes vs WEIGHTS_US:
#   insider_conviction  0.30 → 0.38  (+0.08)
#   analyst_consensus   0.00 → 0.07  (+0.07)
#   momentum_long       0.15 → 0.19  (+0.04)
#   quality_piotroski   0.00 → 0.03  (+0.03)
#   congress            0.22 → 0.00  (-0.22)
#   all others          unchanged
WEIGHTS_GLOBAL: dict[str, float] = {
    "insider_conviction": 0.38,   # +0.08 — MAR Art.19 carries similar alpha
    "insider_breadth":    0.15,   # unchanged
    "congress":           0.00,   # structurally absent outside US
    "news_sentiment":     0.10,   # unchanged
    "news_buzz":          0.05,   # unchanged
    "momentum_long":      0.19,   # +0.04 — Rouwenhorst 1998 EU momentum premium
    "volume_attention":   0.03,   # unchanged
    "analyst_consensus":  0.07,   # +0.07 — stronger signal in less-covered markets
    "quality_piotroski":  0.03,   # +0.03 — accounting-identity, universal
}
assert abs(sum(WEIGHTS_GLOBAL.values()) - 1.0) < 1e-6, (
    f"WEIGHTS_GLOBAL sums to {sum(WEIGHTS_GLOBAL.values()):.8f}, not 1.0"
)

# ── Convenience alias (legacy callers expecting WEIGHTS get US set) ────────────
WEIGHTS = WEIGHTS_US

# ── Region classifier ─────────────────────────────────────────────────────────
# Determines which weight set applies to a given ticker.
# Matching order: suffix check → exchange prefix → default US.
#
# EU suffixes  : .PA .DE .L .AS .MI .MC .BR .VX .LS .OL .ST .HE .CO .F .BE
# Asia suffixes: .T .HK .KS .KQ .SS .SZ .NS .BO .SI .BK .JK
#
# Tickers without a recognised suffix are assumed US (NASDAQ/NYSE/OTC).
# This is intentional — unknown = US is the conservative default.

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

    Examples
    --------
    >>> get_region("SAP.DE")
    'EU'
    >>> get_region("7203.T")
    'ASIA'
    >>> get_region("AAPL")
    'US'
    >>> get_region("005930.KS")
    'ASIA'
    """
    upper = ticker.upper()
    dot_idx = upper.rfind(".")
    if dot_idx == -1:
        return "US"
    suffix = upper[dot_idx:]   # e.g. ".DE", ".T"
    if suffix in _EU_SUFFIXES:
        return "EU"
    if suffix in _ASIA_SUFFIXES:
        return "ASIA"
    return "US"


def get_weights(ticker: str) -> dict[str, float]:
    """Return the correct weight set for *ticker*.

    US tickers  → WEIGHTS_US   (full 9-factor, congress included)
    EU/Asia     → WEIGHTS_GLOBAL (8-factor, congress = 0.0)

    The returned dict is a copy — callers may not mutate the canonical sets.
    """
    region = get_region(ticker)
    if region == "US":
        return dict(WEIGHTS_US)
    return dict(WEIGHTS_GLOBAL)


def is_congress_eligible(ticker: str) -> bool:
    """True only for US tickers where congress signal is structurally available."""
    return get_region(ticker) == "US"
