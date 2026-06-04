# Path: regime_trader/config/weights.py
"""Canonical 9-factor WEIGHTS — single authoritative source for generate_top_lists.py.

Post-sprint state (RT-QA-2026-REV6, steps 1-7 complete).

To change weights:
1. Edit WEIGHTS dict below.
2. Run: python -c "from regime_trader.config.weights import WEIGHTS; print(sum(WEIGHTS.values()))"
3. Confirm output is 1.0.
4. Bump WEIGHTS_VERSION.

References:
    Grinold & Kahn (1994): IR = IC × √BR (Fundamental Law of Active Management)
    See regime_trader/weights.py for per-factor academic citations.
"""

WEIGHTS_VERSION = "v2.0-post-sprint"

WEIGHTS: dict[str, float] = {
    "insider_conviction":  0.25,
    "insider_breadth":     0.12,
    "congress":            0.12,  # US-only; structurally 0.0 for EU/Asia
    "news_sentiment":      0.10,
    "news_buzz":           0.05,
    "momentum_long":       0.15,
    "volume_attention":    0.03,
    "analyst_consensus":   0.10,  # reads from upgrades-downgrades-consensus-bulk
    "quality_piotroski":   0.08,  # multiplicative gate applied in generate_top_lists.py
}

assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, (
    f"WEIGHTS sum={sum(WEIGHTS.values()):.10f} — must equal 1.0 exactly. "
    f"Fix the weights dict before committing."
)

# EU/Asia effective weights: congress zeroed, remainder renormalized to sum=1.0
_EU_ZERO = frozenset({"congress"})
_eu_raw   = {k: (v if k not in _EU_ZERO else 0.0) for k, v in WEIGHTS.items()}
_eu_total = sum(_eu_raw.values())
WEIGHTS_EU: dict[str, float] = {k: round(v / _eu_total, 10) for k, v in _eu_raw.items()}

assert abs(sum(WEIGHTS_EU.values()) - 1.0) < 1e-6, (
    f"WEIGHTS_EU sum={sum(WEIGHTS_EU.values()):.10f} != 1.0"
)
