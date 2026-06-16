"""src/scoring/news_signals.py
Orthogonal news signals: directional sentiment (recency-decayed) and buzz.

Theory:
    Tetlock (2007), "Giving Content to Investor Sentiment: The Role of Media
    in the Stock Market", Journal of Finance 62(3) pp. 1139–1168:
        Negative media content strongly predicts downward price pressure with
        a half-life of ~3-5 days. Sentiment must be recency-weighted: articles
        from J-60 carry essentially no predictive power for current prices.

    Barber & Odean (2008): volume of coverage (buzz) is an attention signal
    independent of sentiment direction. High buzz predicts buying from
    attention-driven investors, not sustained alpha.

    Separation rationale: mixing sentiment and buzz in a single score (60/40)
    dilutes the directional signal — a mega-cap with 200 articles/week saturates
    the buzz component even when sentiment is neutral, inflating the combined
    score. Orthogonal decomposition preserves both signals with independent weights.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone

log = logging.getLogger(__name__)

_LN2 = math.log(2)


def score_news_sentiment(
    articles: list[dict],
    decay_half_life_days: float = 5.0,
) -> float:
    """Recency-weighted directional sentiment in [0, 1].

    Formula:
        For each article:
            sentiment_value = +1 (Positive), -1 (Negative), 0 (Neutral/missing)
            age_days        = max(0, days since publishedDate)
            weight          = exp(-age_days * ln(2) / half_life)
                              # half-life=5d: weight halves every 5 days
        raw   = Σ(sentiment_value × weight) / Σ(weight)  → [-1, 1]
        score = (raw + 1) / 2                              → [0,  1]

    Returns 0.0 if articles is empty or all weights round to zero (dead signal,
    not neutral). Consistent with insider/congress dead-signal treatment.

    Reference: Tetlock (2007), Journal of Finance 62(3). Half-life ~3-5 days.
    """
    if not articles:
        return 0.0

    now = datetime.now(timezone.utc)
    half_life = max(0.1, float(decay_half_life_days))
    decay_rate = _LN2 / half_life

    weighted_sum    = 0.0
    weight_sum      = 0.0
    has_directional = False  # track whether any Positive/Negative article exists

    for article in articles:
        sentiment = article.get("sentiment") or ""
        if sentiment == "Positive":
            val = 1.0
            has_directional = True
        elif sentiment == "Negative":
            val = -1.0
            has_directional = True
        else:
            val = 0.0

        pub_str = article.get("publishedDate") or article.get("date") or ""
        age_days = 0.0
        if pub_str:
            try:
                from datetime import date as _date
                pub_date_str = str(pub_str)[:10]
                pub_date = _date.fromisoformat(pub_date_str)
                age_days = max(0.0, (now.date() - pub_date).days)
            except Exception:
                age_days = 0.0

        weight = math.exp(-age_days * decay_rate)
        weighted_sum += val * weight
        weight_sum   += weight

    if weight_sum == 0.0:
        return 0.0

    # No positive or negative articles → absent signal, not neutral.
    # Prevents all-"Neutral" FMP responses from producing 0.5 (looks like a signal).
    if not has_directional:
        return 0.0

    raw = weighted_sum / weight_sum   # ∈ [-1, 1]
    score = (raw + 1.0) / 2.0        # → [0, 1]
    return round(score, 4)


def score_transcript_tone(text: str | None) -> float | None:
    """Earnings call guidance tone scorer in [0, 1], or None when no signal.

    Classifies management guidance language into three tones:
      raised guidance   → 0.80 (bullish — explicit upward revision)
      reaffirmed        → 0.55 (neutral-positive — no change, confident tone)
      lowered guidance  → 0.20 (bearish — explicit downward revision)
      no guidance found → None (SIGNED: absent ≠ bearish; weight redistributes)

    The caller (run_pipeline.score_transcript_tone) handles FMP fetch and
    wraps the result in (float|None, source_str) for pipeline compatibility.

    Reference: [Huang et al., 2018 — "The Predictive Power of Conference
    Call Sentiment", Journal of Financial Economics]
    """
    if not text:
        return None

    t = text.lower()

    raise_phrases = [
        "raising guidance", "raised guidance", "increase our guidance",
        "raising our full-year", "above the high end", "raising our outlook",
        "above our guidance", "raising revenue guidance",
    ]
    lower_phrases = [
        "lowering guidance", "lowered guidance", "reduce our guidance",
        "below our guidance", "revising guidance lower", "lowering our outlook",
        "below the low end", "headwinds",
    ]
    maintain_phrases = [
        "reaffirming guidance", "reaffirm", "maintaining guidance", "on track to",
        "comfortable with our guidance", "reiterate", "confident in our",
    ]

    cnt_raise = sum(t.count(p) for p in raise_phrases)
    cnt_lower = sum(t.count(p) for p in lower_phrases)
    cnt_maint = sum(t.count(p) for p in maintain_phrases)

    if cnt_raise + cnt_lower + cnt_maint == 0:
        return None

    if cnt_raise > cnt_lower and cnt_raise > cnt_maint:
        return 0.80
    if cnt_lower > cnt_raise and cnt_lower > cnt_maint:
        return 0.20
    return 0.55


def score_news_buzz(articles: list[dict]) -> float:
    """Pure attention/buzz signal in [0, 1].

    Formula:
        n_recent = count of articles published in last 7 days
        score    = min(1.0, log1p(n_recent) / log1p(50))

    Returns 0.0 if no recent articles (dead signal, not neutral).

    Reference: Barber & Odean (2008), Review of Financial Studies 21(2).
    """
    if not articles:
        return 0.0

    now = datetime.now(timezone.utc)
    n_recent = 0
    _BUZZ_WINDOW_DAYS = 7  # Barber & Odean (2008): 7-day attention window

    for article in articles:
        pub_str = article.get("publishedDate") or article.get("date") or ""
        if not pub_str:
            continue
        try:
            from datetime import date as _date
            pub_date = _date.fromisoformat(str(pub_str)[:10])
            if (now.date() - pub_date).days <= _BUZZ_WINDOW_DAYS:
                n_recent += 1
        except Exception:
            continue

    if n_recent == 0:
        return 0.0

    return round(min(1.0, math.log1p(n_recent) / math.log1p(50)), 4)
