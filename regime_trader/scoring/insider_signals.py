"""regime_trader/scoring/insider_signals.py
Orthogonal insider signal decomposition.

Theory — Cohen, Malloy & Pomorski (2012), "Decoding Inside Information"
(Journal of Finance 67:3, pp. 1009–1043):
    Only "opportunistic" insider trades (P-code, irregular timing) carry
    measurable alpha (~7% annualised forward 1-month return).  "Routine"
    transactions (A=award, F=tax withholding, M=exercise, regular calendar
    patterns) have near-zero alpha and should be excluded from any signal.

    Two orthogonal dimensions:
      1. Conviction  — dollar magnitude + officer seniority (CEO premium).
         Captures: "how much skin in the game?"
      2. Breadth     — consensus across distinct insiders (P vs S ratio).
         Captures: "how many insiders agree?"
         Reference: Lakonishok & Lee (2001), Seyhun (1998) — concordance
         among insiders independently predicts forward returns.

    A single CEO buying $10M → high conviction, low breadth.
    Eight directors each buying $50k → low conviction, high breadth.
    Both are genuine alpha signals; they are designed to be uncorrelated.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone, timedelta
from typing import Any

log = logging.getLogger(__name__)

# Fix #7: relative CEO purchase significance thresholds (basis points of market cap).
# Rationale: Cohen, Malloy & Pomorski (2012) — opportunistic CEO trades are only
# predictive when the purchase is *substantial relative to the firm's size*, not in
# absolute USD. A $10M buy on a $100B mega-cap is immaterial window dressing;
# the same $10M on a $1B mid-cap signals genuine conviction.
_CEO_BPS_MODEST      = 0.5   # < 0.5 bps → no bonus (immaterial)
_CEO_BPS_SUBSTANTIAL = 1.0   # 0.5–1.0 bps → modest; 1.0–5.0 bps → substantial
_CEO_BPS_EXCEPTIONAL = 5.0   # ≥ 5.0 bps → exceptional (capped by annual comp check)
_CEO_FALLBACK_USD    = 50_000  # absolute floor when market_cap unavailable


def _ceo_purchase_significance(
    ceo_purchase_usd: float,
    market_cap: float,
    ceo_annual_comp: float | None = None,
) -> float:
    """CEO conviction multiplier in [1.0, 1.15] based on relative purchase size.

    Primary metric (always): basis points of market cap acquired.
        bps = ceo_purchase_usd / market_cap * 10_000
        < 0.5 bps  → 1.00 (no bonus, immaterial)
        0.5–1.0    → 1.05 (modest)
        1.0–5.0    → 1.10 (substantial)
        ≥ 5.0      → 1.15 (exceptional), reduced to 1.10 if purchase
                           also < 50% of annual comp (window dressing filter).

    Fallback when market_cap <= 0: use absolute USD threshold $50k.

    Scale invariance: a $10M buy on $1B (1000 bps) scores 1.15;
    the same $10M on $100B (1 bp) scores 1.10 — not 1.15.

    NOTE on calibration: on a large-cap universe (S&P 500-style), the bps
    thresholds are structurally rarely activated — most CEO purchases on
    mega-caps are < 0.5 bps because market caps dwarf practical purchase sizes.
    This is correct behavior: the multiplier detects *relatively significant*
    CEO conviction, which by definition rarely occurs on mega-caps. For
    small/mid-cap universes, the multiplier activates more frequently.
    Empirical: on a 160-ticker S&P 500-style universe, ~99% of tickers receive
    'none' tier. To activate the multiplier broadly, expand universe to mid/small caps.
    """
    if ceo_purchase_usd <= 0:
        return 1.0

    if market_cap > 0:
        bps = ceo_purchase_usd / market_cap * 10_000
        if bps < _CEO_BPS_MODEST:
            return 1.0
        elif bps < _CEO_BPS_SUBSTANTIAL:
            return 1.05
        elif bps < _CEO_BPS_EXCEPTIONAL:
            return 1.10
        else:
            # Exceptional tier: require purchase ≥ 50% of annual comp when known.
            if ceo_annual_comp is not None and ceo_purchase_usd < 0.5 * ceo_annual_comp:
                return 1.10  # large bps but small vs comp → cap at substantial
            return 1.15
    else:
        # market_cap unavailable — fallback to absolute threshold (raised from $25k)
        log.warning(
            "CEO significance: market_cap unavailable, falling back to absolute USD threshold"
        )
        return 1.10 if ceo_purchase_usd >= _CEO_FALLBACK_USD else 1.0


def score_insider_conviction(
    key_purchases_usd: float,
    market_cap: float,
    days_since_most_recent: int = 0,
    ceo_purchase_usd: float = 0.0,
    ceo_annual_comp: float | None = None,
) -> float:
    """Dollar-weighted conviction signal in [0, 1].

    Extends score_insider_value() with a CEO/CFO premium scaled by purchase
    size relative to market cap (Fix #7 — Cohen, Malloy & Pomorski 2012).

    Formula:
        pct  = key_purchases_usd / market_cap
        raw  = log(1 + pct * 10000) / log(1 + 100)
        base = 0.30 + 0.60 * raw                      # floor at 0.30
        multiplier = _ceo_purchase_significance(...)   # Fix #7: relative bps
        base = min(0.95, base * multiplier)
        Recency decay to 0.5 for purchases > 30 days old.

    Returns 0.0 if key_purchases_usd <= 0 (dead signal, not neutral).
    """
    if key_purchases_usd <= 0 or market_cap <= 0:
        return 0.0

    pct = key_purchases_usd / market_cap
    raw = min(1.0, math.log1p(pct * 10000) / math.log1p(100))
    base = round(0.30 + 0.60 * raw, 6)

    # CEO/CFO premium — Fix #7: relative to market cap, not absolute USD.
    multiplier = _ceo_purchase_significance(ceo_purchase_usd, market_cap, ceo_annual_comp)
    if multiplier > 1.0:
        base = min(0.95, base * multiplier)

    # Recency decay: purchases older than 30 days decay toward 0.5
    if days_since_most_recent > 30:
        decay = max(0.70, 1.0 - 0.30 * min(days_since_most_recent - 30, 150) / 150)
        base = 0.5 + (base - 0.5) * decay

    return round(base, 4)


def score_insider_breadth(
    p_transactions: list[dict[str, Any]],
    s_transactions: list[dict[str, Any]],
    lookback_days: int = 90,
) -> float:
    """Insider consensus breadth signal in [0, 1].

    Measures concordance among distinct insiders — independent of dollar size.
    Reference: Lakonishok & Lee (2001), Seyhun (1998).

    Formula:
        Filter both lists to lookback_days window from today.
        n_buyers  = distinct insider_id with >= 1 P-code in window
        n_sellers = distinct insider_id with >= 1 S-code in window

        if n_buyers + n_sellers == 0: return 0.0  (dead signal)

        buyer_ratio   = n_buyers / (n_buyers + n_sellers)
        breadth_scale = min(1.0, log(1 + n_total) / log(1 + 10))
        base          = 0.7 * buyer_ratio + 0.3 * breadth_scale

        Recency decay to 0.5 if most recent P-transaction > 30 days old.

    Returns 0.0 if no transactions in window (dead signal, not neutral).
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).date().isoformat()

    def _in_window(txs: list[dict]) -> list[dict]:
        result = []
        for tx in txs:
            d = str(tx.get("date", "") or "")[:10]
            if d >= cutoff:
                result.append(tx)
        return result

    p_recent = _in_window(p_transactions)
    s_recent = _in_window(s_transactions)

    if not p_recent and not s_recent:
        return 0.0

    # Distinct insiders (use insider_id; fall back to title if absent)
    buyer_ids  = {tx.get("insider_id") or tx.get("title", f"buyer_{i}")
                  for i, tx in enumerate(p_recent)}
    seller_ids = {tx.get("insider_id") or tx.get("title", f"seller_{i}")
                  for i, tx in enumerate(s_recent)}

    n_buyers  = len(buyer_ids)
    n_sellers = len(seller_ids)
    n_total   = n_buyers + n_sellers

    if n_total == 0:
        return 0.0

    buyer_ratio   = n_buyers / n_total
    breadth_scale = min(1.0, math.log1p(n_total) / math.log1p(10))
    base          = 0.7 * buyer_ratio + 0.3 * breadth_scale

    # Recency decay from most recent P-transaction
    if p_recent:
        dates = [str(tx.get("date", "") or "")[:10] for tx in p_recent if tx.get("date")]
        if dates:
            most_recent = max(dates)
            try:
                from datetime import date as _date
                days_ago = (datetime.now(timezone.utc).date() - _date.fromisoformat(most_recent)).days
                if days_ago > 30:
                    decay = max(0.70, 1.0 - 0.30 * min(days_ago - 30, 150) / 150)
                    base = 0.5 + (base - 0.5) * decay
            except Exception:
                pass

    return round(base, 4)


def log_conviction_breadth_correlation(results: list[dict[str, Any]]) -> None:
    """Log Pearson r between conviction and breadth cross-sectionally.

    Per spec: r must be < 0.4.  If >= 0.4, the signals are not orthogonal
    — log ERROR so ops can investigate.
    """
    pairs = [
        (
            float(r.get("insider_conviction_score", 0.0) or 0.0),
            float(r.get("insider_breadth_score", 0.0) or 0.0),
        )
        for r in results
        if float(r.get("insider_conviction_score", 0.0) or 0.0) > 0.0
        and float(r.get("insider_breadth_score", 0.0) or 0.0) > 0.0
    ]
    if len(pairs) < 5:
        log.info("conviction↔breadth correlation: insufficient pairs (%d)", len(pairs))
        return
    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]
    n  = len(pairs)
    mx, my = sum(xs) / n, sum(ys) / n
    num   = sum((x - mx) * (y - my) for x, y in pairs)
    denom = math.sqrt(
        sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)
    )
    if denom == 0:
        log.info("conviction↔breadth correlation: undefined (zero variance in one signal)")
        return
    r = num / denom
    if r >= 0.4:
        log.error(
            "ORTHOGONALITY CHECK FAILED: conviction↔breadth Pearson r=%.3f >= 0.4 — "
            "signals are correlated, decomposition is not working as intended.",
            r,
        )
    else:
        log.info("conviction↔breadth Pearson r=%.3f (< 0.4 ✓ orthogonal)", r)
