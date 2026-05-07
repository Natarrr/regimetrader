"""backend/utils/score_helpers.py
Five-factor scoring helpers for the market-intel pipeline.

Deliberately standalone: no imports from streamlit_app.py so this module
can be used by backend routers, diagnostics scripts, and tests without
pulling in the full Streamlit dependency graph.

Factors
───────
  macro         : commodity-regime conviction (already computed upstream)
  institutional : institutional ownership / accumulation direction
  insider       : key-executive open-market purchases
  news          : headline sentiment (yfinance titles, word-list scoring)
  regime        : HMM regime multiplier applied at aggregation time

All factor scores are normalised to [0, 1] before aggregation.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# ── Tunable weights ────────────────────────────────────────────────────────────
# Sum = 1.00.  "regime" weight applies to the regime multiplier contribution
# (already encoded as a number ≥ 0, not a [0,1] score — see aggregate_scores).

WEIGHTS: Dict[str, float] = {
    "macro":         0.25,
    "institutional": 0.20,
    "insider":       0.20,
    "news":          0.20,
    "regime":        0.15,
}

# Badge thresholds
BADGE_HIGH_BUY     = 0.80
BADGE_TACTICAL_BUY = 0.60

# Alert thresholds (for caller logic)
ALERT_INSIDER_RISE     = 0.15   # insider_score 7-day delta that triggers alert
ALERT_INST_DROP        = 0.15   # institutional_score 30-day delta that triggers alert
ALERT_MACRO_REDUCE     = 0.28   # macro_score absolute floor
ALERT_SCORE_DROP       = 0.20   # final_score 1-day delta that triggers alert

# Headline word lists (duplicated here to avoid importing streamlit_app.py)
_BULL_WORDS: frozenset = frozenset([
    "beat", "beats", "exceed", "exceeds", "exceeded", "upgrade", "upgrades",
    "upgraded", "buy", "outperform", "outperforms", "strong", "record", "rally",
    "rallies", "surge", "surges", "surged", "gain", "gains", "growth", "bullish",
    "profit", "profits", "revenue", "raise", "raises", "raised", "tops",
    "topped", "jump", "jumps", "jumped", "soar", "soars", "soared", "boom",
    "positive", "breakthrough", "approval", "approved", "expands", "expansion",
])
_BEAR_WORDS: frozenset = frozenset([
    "miss", "misses", "missed", "downgrade", "downgrades", "downgraded",
    "sell", "underperform", "underperforms", "concern", "concerns", "decline",
    "declines", "declined", "weak", "weakness", "loss", "losses", "cut",
    "cuts", "cutting", "fall", "falls", "fell", "drop", "drops", "dropped",
    "recession", "layoff", "layoffs", "lawsuit", "fine", "fined", "warning",
    "warn", "warns", "risk", "risks", "volatile", "volatility", "below",
    "disappoints", "disappointing", "disappointed", "halt", "halted",
    "investigation", "probe", "fraud", "bankruptcy", "default",
])


# ── Utility ────────────────────────────────────────────────────────────────────

def safe_float(x, default: float = 0.5) -> float:
    """Return float(x) clamped to [0, 1], or default on any error."""
    try:
        if x is None:
            return default
        v = float(x)
        return max(0.0, min(1.0, v))
    except Exception:
        return default


def _headline_score(title: str) -> float:
    words = set(title.lower().split())
    bull  = len(words & _BULL_WORDS)
    bear  = len(words & _BEAR_WORDS)
    if bull == 0 and bear == 0:
        return 0.50
    return round(max(0.10, min(0.90, 0.50 + 0.20 * (bull - bear))), 4)


# ── Fetchers ───────────────────────────────────────────────────────────────────

def fetch_insider_data(ticker: str) -> float:
    """Return insider buying conviction score in [0, 1].  0.5 = neutral.

    Layer 1: yfinance insider_transactions — parses buy/sell counts and
             identifies CEO/CFO purchases (score floor 0.85 if present).
    Layer 2: falls back to 0.5 with an explicit WARNING log so the caller
             knows it's using the default, not real data.
    """
    import pandas as _pd
    try:
        import yfinance as _yf
        txns = _yf.Ticker(ticker).insider_transactions
        if txns is None or (hasattr(txns, "empty") and txns.empty):
            logger.warning("[INSIDER %s] insider_transactions empty — neutral 0.50", ticker)
            return 0.50

        _cols = {str(c).strip().lower(): str(c).strip() for c in txns.columns}
        text_col = next((_cols[k] for k in ("text", "transaction") if k in _cols), None)
        pos_col  = next((_cols[k] for k in ("position",) if k in _cols), None)
        val_col  = next((_cols[k] for k in ("value",) if k in _cols), None)

        buy_count = sell_count = 0
        ceo_cfo_buy = 0.0

        for _, row in txns.iterrows():
            raw_text = row.get(text_col, "") if text_col else ""
            raw_pos  = row.get(pos_col,  "") if pos_col  else ""
            raw_val  = row.get(val_col,   0) if val_col  else 0

            text = "" if _pd.isna(raw_text) else str(raw_text).upper().strip()
            pos  = "" if _pd.isna(raw_pos)  else str(raw_pos).upper().strip()
            val  = 0.0
            try:
                val = abs(float(raw_val or 0))
            except (TypeError, ValueError):
                pass

            is_buy  = "PURCHASE" in text or "ACQUI" in text or text.startswith("BUY")
            is_sell = "SALE" in text or "SELL" in text

            if is_buy:
                buy_count += 1
                if any(k in pos for k in ("CEO", "CFO", "CHIEF EXECUTIVE", "CHIEF FINANCIAL")):
                    ceo_cfo_buy += val
            elif is_sell:
                sell_count += 1

        total = buy_count + sell_count
        if total == 0:
            logger.warning("[INSIDER %s] no parseable buy/sell rows — neutral 0.50", ticker)
            return 0.50

        score = round(0.30 + 0.60 * (buy_count / total), 4)
        if ceo_cfo_buy > 0:
            score = max(score, 0.85)
        logger.debug("[INSIDER %s] score=%.3f buy=%d sell=%d ceo_buy=%.0f",
                     ticker, score, buy_count, sell_count, ceo_cfo_buy)
        return score

    except Exception:
        logger.exception("[INSIDER %s] unhandled error — neutral 0.50", ticker)
        return 0.50


def fetch_institutional_score(ticker: str) -> float:
    """Return institutional ownership / accumulation score in [0, 1].  0.5 = neutral.

    Layer 1: yfinance institutional_holders — net flow from pctHeld × pctChange.
    Layer 2: yfinance major_holders — institutionsPercentHeld static score.
    Layer 3: returns 0.50 with explicit WARNING log.
    """
    import pandas as _pd
    try:
        import yfinance as _yf
        ih = _yf.Ticker(ticker).institutional_holders
        if ih is not None and not (hasattr(ih, "empty") and ih.empty):
            ih.columns = [str(c).strip() for c in ih.columns]
            _cols = {c.lower(): c for c in ih.columns}
            pct_held_col   = next(
                (_cols[k] for k in ("pctheld", "% out", "pct held", "pctout") if k in _cols),
                None,
            )
            pct_change_col = next(
                (_cols[k] for k in ("pctchange", "% change", "pct change") if k in _cols),
                None,
            )
            if pct_held_col:
                top        = ih.head(20)
                pct_held   = _pd.to_numeric(top[pct_held_col], errors="coerce").fillna(0.0)
                if pct_change_col:
                    pct_change = _pd.to_numeric(top[pct_change_col], errors="coerce").fillna(0.0)
                    net_flow   = float((pct_held * pct_change).sum())
                    score = round(max(0.20, min(0.90, 0.55 + 5.0 * net_flow)), 4)
                    logger.debug("[INST %s] net_flow=%.4f score=%.3f", ticker, net_flow, score)
                    return score
                else:
                    total_held = float(pct_held.sum())
                    score = round(max(0.20, min(0.90, 0.30 + total_held)), 4)
                    logger.debug("[INST %s] total_held=%.4f score=%.3f", ticker, total_held, score)
                    return score
            logger.warning("[INST %s] institutional_holders columns not usable: %s",
                           ticker, list(ih.columns))
    except Exception:
        logger.exception("[INST %s] institutional_holders error", ticker)

    # Layer 2 — major_holders
    try:
        import yfinance as _yf
        import pandas as _pd
        mh = _yf.Ticker(ticker).major_holders
        if mh is not None and not (hasattr(mh, "empty") and mh.empty):
            mh.index = [str(i).strip() for i in mh.index]
            _mh_idx = {i.lower(): i for i in mh.index}
            for _key in ("institutionspercentheld",
                         "% held by institutions",
                         "institutions percent held"):
                if _key in _mh_idx:
                    inst_pct = float(
                        _pd.to_numeric(mh.loc[_mh_idx[_key]].iloc[0], errors="coerce") or 0.0
                    )
                    score = round(max(0.20, min(0.90, 0.30 + inst_pct * 0.65)), 4)
                    logger.debug("[INST %s] major_holders inst_pct=%.3f score=%.3f",
                                 ticker, inst_pct, score)
                    return score
        logger.warning("[INST %s] major_holders empty or key missing — neutral 0.50", ticker)
    except Exception:
        logger.exception("[INST %s] major_holders error", ticker)

    logger.warning("[INST %s] all layers failed — neutral 0.50", ticker)
    return 0.50


def fetch_news_sentiment_for_ticker(ticker: str, max_items: int = 10) -> float:
    """Return headline sentiment score in [0, 1].  0.5 = neutral.

    Pulls yfinance news titles and scores each headline with the
    bull/bear word list. Returns the mean across up to max_items titles.
    Logs a WARNING if yfinance returns no news (common for small-caps).
    """
    try:
        import yfinance as _yf
        news = _yf.Ticker(ticker).news or []
        if not news:
            logger.warning("[NEWS %s] yfinance returned no news — neutral 0.50", ticker)
            return 0.50

        scores = []
        for item in news[:max_items]:
            content = item.get("content", {})
            title   = (
                content.get("title", "") if isinstance(content, dict)
                else item.get("title", "")
            )
            if title:
                scores.append(_headline_score(title))

        if not scores:
            logger.warning("[NEWS %s] no parseable headlines — neutral 0.50", ticker)
            return 0.50

        result = round(sum(scores) / len(scores), 4)
        logger.debug("[NEWS %s] score=%.3f from %d headlines", ticker, result, len(scores))
        return result

    except Exception:
        logger.exception("[NEWS %s] unhandled error — neutral 0.50", ticker)
        return 0.50


# ── Core aggregation ───────────────────────────────────────────────────────────

def aggregate_scores(
    macro_score:          float,
    institutional_score:  float,
    insider_score:        float,
    news_score:           float,
    regime_mult:          float,
    weights:              Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Combine all five factors into a final_score with transparent breakdown.

    regime_mult is a multiplier (e.g. 1.20 for Bull, 0.65 for Bear).
    It is normalised to [0, 1] before weighting so the formula is purely additive.
    Capped at 1.50 to prevent extreme regimes from inflating the score.

    Returns a dict suitable for direct update() onto a pick dict:
        {macro, institutional, insider, news, regime_mult,
         regime_score, final_score, badge, badge_clr, score_breakdown}
    """
    if weights is None:
        weights = WEIGHTS

    m    = safe_float(macro_score,         0.50)
    inst = safe_float(institutional_score, 0.50)
    ins  = safe_float(insider_score,       0.50)
    news = safe_float(news_score,          0.50)
    reg  = safe_float(regime_mult,         1.00)

    # Normalise regime_mult → [0, 1]: treat 1.0 as 0.50 neutral
    # Range of STOCK_REGIME_MULT: 0.00 (Crash) → 1.20 (Bull) → 1.15 (Euphoria)
    # Map [0.0, 1.50] linearly to [0.0, 1.0]
    reg_capped = min(reg, 1.50)
    reg_score  = round(reg_capped / 1.50, 4)

    final = round(
        weights["macro"]         * m
        + weights["institutional"] * inst
        + weights["insider"]       * ins
        + weights["news"]          * news
        + weights["regime"]        * reg_score,
        4,
    )
    final = max(0.0, min(1.0, final))

    if final >= BADGE_HIGH_BUY:
        badge, badge_clr = "HIGH BUY",     "#00c851"
    elif final >= BADGE_TACTICAL_BUY:
        badge, badge_clr = "TACTICAL BUY", "#ffbb33"
    else:
        badge, badge_clr = "WATCHLIST",    "#9e9e9e"

    return {
        "macro_score":         round(m,    4),
        "institutional_score": round(inst, 4),
        "insider_score":       round(ins,  4),
        "news_score":          round(news, 4),
        "regime_mult":         round(reg,  4),
        "final_score":         final,
        "badge":               badge,
        "badge_clr":           badge_clr,
        "score_breakdown": {
            "macro":         round(weights["macro"]         * m,         4),
            "institutional": round(weights["institutional"] * inst,      4),
            "insider":       round(weights["insider"]       * ins,       4),
            "news":          round(weights["news"]          * news,      4),
            "regime":        round(weights["regime"]        * reg_score, 4),
        },
    }


# ── Alert helpers ──────────────────────────────────────────────────────────────

def check_alerts(
    ticker:         str,
    current:        Dict[str, float],
    previous:       Optional[Dict[str, float]] = None,
) -> List[str]:
    """Return a list of alert strings for a ticker given current and (optionally) previous scores."""
    from typing import List
    alerts: List[str] = []

    if current.get("macro_score", 0.5) < ALERT_MACRO_REDUCE:
        alerts.append(
            f"{ticker}: Macro Reduce — macro_score {current['macro_score']:.2f} < {ALERT_MACRO_REDUCE}"
        )

    if previous:
        delta_score = current.get("final_score", 0.5) - previous.get("final_score", 0.5)
        if delta_score < -ALERT_SCORE_DROP:
            alerts.append(
                f"{ticker}: Momentum Deterioration — final_score fell {delta_score:.2f} in 1 day"
            )

        delta_ins = current.get("insider_score", 0.5) - previous.get("insider_score", 0.5)
        if delta_ins > ALERT_INSIDER_RISE:
            alerts.append(
                f"{ticker}: Insider Accumulation — insider_score rose +{delta_ins:.2f} in 7 days"
            )

        delta_inst = current.get("institutional_score", 0.5) - previous.get("institutional_score", 0.5)
        if delta_inst < -ALERT_INST_DROP:
            alerts.append(
                f"{ticker}: Institutional Selling — institutional_score fell {delta_inst:.2f} in 30 days"
            )

    return alerts
