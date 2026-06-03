# regime_trader/weights.py
"""
Canonical WEIGHTS definition — single source of truth (12-factor schema).

Import this in generate_top_lists.py, run_pipeline.py, and any test that
validates weight distribution.

Enforced constraint: sum(WEIGHTS.values()) == 1.0 (asserted at import time).

Weight constraints (enforced by tests):
  momentum_long  must be the largest weight (strongest empirical IC)
  congress       must be <= 0.10 (sparse US-only binary signal)
  volume_attention must be <= 0.05 (attention tilt, not alpha factor)

EU/Asia effective weight stack (insider=0, congress=0, news=0,
analyst_consensus=0, analyst_revision=0, transcript_tone=0):
  momentum_long(0.21) + volume_attention(0.03) +
  quality_piotroski(0.06) + price_target_upside(0.03)
  = 0.33 coverage, renormalized to 1.0 for EU/Asia

Academic citations:
  insider_conviction:  Seyhun (1988), Lakonishok & Lee (2001)
  insider_breadth:     Cohen, Malloy & Pomorski (2012) — routine vs opportunistic
  congress:            Eggers & Hainmueller (2013) — abnormal congressional returns
  news_sentiment:      Tetlock (2007) — media pessimism predicts returns
  news_buzz:           Da, Engelberg & Gao (2011) — investor attention proxy
  momentum_long:       Jegadeesh & Titman (1993) — 12-1 month momentum
  volume_attention:    Gervais, Kaniel & Mingelgrin (2001) — high-volume premium
  analyst_consensus:   Givoly & Lakonishok (1979) — revision momentum
  analyst_revision:    Chan, Jegadeesh & Lakonishok (1996) — estimate revision
  price_target_upside: Brav & Lehavy (2003) — analyst price target drift
  quality_piotroski:   Piotroski (2000) — F-Score 9-signal financial health gate
  transcript_tone:     Matsumoto et al. (2011) — management guidance tone
"""

WEIGHTS: dict[str, float] = {
    "insider_conviction":  0.20,
    "insider_breadth":     0.10,
    "congress":            0.08,
    "news_sentiment":      0.10,
    "news_buzz":           0.05,
    "momentum_long":       0.21,
    "volume_attention":    0.03,
    "analyst_consensus":   0.08,
    "analyst_revision":    0.04,
    "price_target_upside": 0.03,
    "quality_piotroski":   0.06,
    "transcript_tone":     0.02,
}

# ── Invariant: enforced at import time ──────────────────────────────────────
_weight_sum = sum(WEIGHTS.values())
assert abs(_weight_sum - 1.0) < 1e-6, (
    f"WEIGHTS sum = {_weight_sum:.8f}, must equal 1.0. "
    f"Fix the dict before committing."
)

# Piotroski gate thresholds
PIOTROSKI_GATE: dict[str, float] = {
    "suppress_below": 3,    # piotroskiScore < 3  → BUY multiplier 0.0
    "discount_below": 6,    # piotroskiScore 3-5  → BUY multiplier 0.6
    "discount_factor": 0.6,
    "missing_score":   3,   # treat missing data as score=3 (discounted)
}

# VIX regime thresholds (aligned with _apply_vix_overlay kill-switch)
VIX_THRESHOLDS: dict[str, float] = {
    "kill_switch": 30.0,    # VIX ≥ 30 → all BUY signals suppressed
    "bear":        20.0,    # VIX 20-29 → Bear regime, graduated caution
}

# Congress score for non-US tickers — hard zero, never 0.5
CONGRESS_SCORE_NON_US: float = 0.0

# Analyst consensus score when no coverage available — neutral
ANALYST_CONSENSUS_NO_COVERAGE: float = 0.50
