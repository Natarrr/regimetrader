# regime_trader/weights.py
"""
Canonical WEIGHTS definition — single source of truth.

Import this in generate_top_lists.py, run_pipeline.py, and any test that
validates weight distribution.

Enforced constraint: sum(WEIGHTS.values()) == 1.0 (asserted at import time).

Changes from 12-factor schema (removed analyst_revision, price_target_upside,
transcript_tone — sell-side triplet correlation risk per Grinold-Kahn 2000):
  congress:          0.08 → 0.12  (freed weight redistributed)
  insider_conviction 0.15 → 0.25  (Seyhun 1988: strongest individual alpha source)
  insider_breadth    0.12 → 0.12  (unchanged)
  analyst_consensus  0.04 → 0.10  (NEW weight: upgrades-downgrades-consensus-bulk)
  quality_piotroski  0.08 → 0.08  (unchanged; now sourced from financial-scores-bulk)

EU/Asia effective weight stack (insider=0, congress=0):
  news_sentiment(0.10) + news_buzz(0.05) + momentum_long(0.15) +
  volume_attention(0.03) + analyst_consensus(0.10) + quality_piotroski(0.08)
  = 0.51 coverage (was 0.33 before analyst_consensus wired via bulk)

quality_piotroski scoring — multiplicative gate applied AFTER linear combination:
  piotroskiScore 0-2  → BUY multiplier = 0.0  (suppressed)
  piotroskiScore 3-5  → BUY multiplier = 0.6  (discounted)
  piotroskiScore 6-9  → BUY multiplier = 1.0  (full)
  SELL signals: gate does NOT apply (asymmetric protection)
  EU/Asia: gate applies (Piotroski is accounting-identity based, exchange-agnostic)
  Missing data: treat as piotroskiScore=3 (discounted, not suppressed)

analyst_consensus scoring:
  Strong Buy  = 1.00
  Buy         = 0.75
  Hold        = 0.50
  Sell        = 0.25
  Strong Sell = 0.00
  No coverage = 0.50 (neutral — not penalised for absence unlike congress)
  Source: upgrades-downgrades-consensus-bulk → consensusRating field

Academic citations:
  insider_conviction:  Seyhun (1988), Lakonishok & Lee (2001)
  insider_breadth:     Cohen, Malloy & Pomorski (2012) — routine vs opportunistic
  congress:            Eggers & Hainmueller (2013) — abnormal congressional returns
  news_sentiment:      Tetlock (2007) — media pessimism predicts returns
  news_buzz:           Da, Engelberg & Gao (2011) — investor attention proxy
  momentum_long:       Jegadeesh & Titman (1993) — 12-1 month momentum
  volume_attention:    Gervais, Kaniel & Mingelgrin (2001) — high-volume premium
  analyst_consensus:   Givoly & Lakonishok (1979) — revision momentum
  quality_piotroski:   Piotroski (2000) — F-Score 9-signal financial health gate
"""

WEIGHTS: dict[str, float] = {
    "insider_conviction":  0.25,
    "insider_breadth":     0.12,
    "congress":            0.12,
    "news_sentiment":      0.10,
    "news_buzz":           0.05,
    "momentum_long":       0.15,
    "volume_attention":    0.03,
    "analyst_consensus":   0.10,
    "quality_piotroski":   0.08,
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
