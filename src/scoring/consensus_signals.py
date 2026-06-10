# Path: src/scoring/consensus_signals.py
"""src/scoring/consensus_signals.py
v3.0 Market Sentiment & Consensus Momentum scorers (pillar P2).

Theory:
    Chan, Jegadeesh & Lakonishok (1996), "Momentum Strategies", JF 51(5):
        EPS estimate revisions predict drift; the second derivative
        (revision_velocity) captures acceleration ahead of level chasers.

    Bernard & Thomas (1989), "Post-Earnings-Announcement Drift", JAR 27:
        SUE predicts returns 60–90 days post-announcement; the decay window
        here mirrors that horizon — exhausted drift is None, never bearish.

All three factors are SIGNED (centered at 0.5): damping pulls toward the
neutral 0.5 prior, NOT toward 0.0 — thin analyst coverage must not read as
a bearish signal. Unavailability is always None (reweighted downstream).
"""
from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)

_BASE_EPS = 1e-4  # |estimate| below this is a degenerate revision base


# ── Analyst revision (centralized from run_pipeline / fmp_fetcher inline) ─────

def score_analyst_revision(
    revision_pct: Optional[float],
    n_analysts: int,
) -> Optional[float]:
    """EPS estimate revision momentum [Chan, Jegadeesh & Lakonishok 1996].

    score = 0.5 + (clip(rev, −0.30, +0.30) / 0.60) · min(1, n/10)

    The analyst-count damping scales the DEVIATION from 0.5 — a thin-coverage
    positive revision shrinks toward neutral, never toward bearish 0.0
    (the v2.2 inline mapping damped the whole score toward 0, structurally
    penalizing small/mid caps for coverage they never had).

    None when revision_pct is None or n_analysts < 3 (coverage gate).
    """
    if revision_pct is None or n_analysts < 3:
        return None
    clipped = max(-0.30, min(0.30, float(revision_pct)))
    damp = min(1.0, n_analysts / 10.0)
    return 0.5 + (clipped / 0.60) * damp


# ── Revision velocity (ASIA P2) ───────────────────────────────────────────────

def score_revision_velocity(estimates: list[dict]) -> Optional[float]:
    """Second derivative of the EPS revision path.

    rev_now  = (e0 − e2) / |e2|
    rev_prev = (e1 − e3) / |e3|
    vel      = rev_now − rev_prev
    score    = 0.5 + (clip(vel, −0.30, +0.30) / 0.60) · min(1, n/8)

    estimates: analyst-estimates rows, newest-first, with estimatedEpsAvg
    and numberAnalystEstimatedEps. None when fewer than 4 usable rows or
    either base estimate is near zero (degenerate percentage).
    """
    if not estimates or len(estimates) < 4:
        return None
    try:
        e = [float(row.get("estimatedEpsAvg")) for row in estimates[:4]]
    except (TypeError, ValueError):
        return None
    if abs(e[2]) < _BASE_EPS or abs(e[3]) < _BASE_EPS:
        return None

    rev_now = (e[0] - e[2]) / abs(e[2])
    rev_prev = (e[1] - e[3]) / abs(e[3])
    vel = rev_now - rev_prev

    n = int(estimates[0].get("numberAnalystEstimatedEps") or 0)
    damp = min(1.0, n / 8.0)
    clipped = max(-0.30, min(0.30, vel))
    return 0.5 + (clipped / 0.60) * damp


# ── PEAD surprise (US P2) ─────────────────────────────────────────────────────

def score_pead_surprise(
    surprise_pct: Optional[float],
    days_since: Optional[int],
) -> Optional[float]:
    """Post-earnings announcement drift [Bernard & Thomas 1989].

    base  = clip(surprise, −0.50, +0.50) + 0.50
    decay = 1.0 if days ≤ 60 else (90 − days)/30
    score = 0.5 + (base − 0.5) · decay

    None when surprise is unavailable or the report is older than 90 days —
    exhausted drift carries no information and must NOT read as bearish.
    days are anchored to the announcement date (never fiscal period end).
    """
    if surprise_pct is None or days_since is None:
        return None
    days = int(days_since)
    if days < 0 or days > 90:
        return None

    base = max(-0.50, min(0.50, float(surprise_pct))) + 0.50
    decay = 1.0 if days <= 60 else (90 - days) / 30.0
    return 0.5 + (base - 0.5) * decay
