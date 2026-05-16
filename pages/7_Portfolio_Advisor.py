"""pages/7_Portfolio_Advisor.py
Daily portfolio advice — Revolut-first, table + expand drawer.

Signals: ADD / HOLD / REDUCE / EXIT derived from 5-factor quant scores.
Claude narrative: 2-sentence explanation per position (gated by ANTHROPIC_API_KEY,
cached in st.session_state to avoid repeated API calls).

Run via sidebar: "💼 Portfolio Advisor"
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from regime_trader.ui.portfolio_advisor_engine import (
    PositionAdvice,
    build_advice,
    compute_health_score,
)

log = logging.getLogger(__name__)

_ROOT                   = Path(__file__).parent.parent
_REVOLUT_PORTFOLIO_PATH = _ROOT / "data" / "revolut_portfolio.json"
_ANTHROPIC_KEY          = os.getenv("ANTHROPIC_API_KEY", "")
_HAS_CLAUDE             = bool(_ANTHROPIC_KEY)

_SIGNAL_COLOR = {
    "ADD":    "#00d26a",
    "HOLD":   "#60a5fa",
    "REDUCE": "#f5a623",
    "EXIT":   "#ff4d6d",
    "—":      "#888888",
}

_SIGNAL_ICON = {
    "ADD":    "➕",
    "HOLD":   "=",
    "REDUCE": "⬇️",
    "EXIT":   "🚫",
    "—":      "—",
}


def _load_revolut() -> Optional[Dict[str, Any]]:
    try:
        return json.loads(_REVOLUT_PORTFOLIO_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as exc:
        log.warning("revolut_portfolio.json: %s", exc)
        return None


def _load_regime() -> str:
    try:
        import yfinance as yf
        from regime_trader.models.regime_detector import vix_rule
        df = yf.download("^VIX", period="2d", interval="1d", progress=False, auto_adjust=True)
        if df.empty:
            return "Unknown"
        vix = float(df["Close"].squeeze().dropna().iloc[-1])
        return vix_rule(vix)
    except Exception:
        return "Unknown"


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_live_prices(tickers: tuple[str, ...]) -> Dict[str, float]:
    if not tickers:
        return {}
    try:
        import yfinance as yf
        ticker_list = list(tickers)
        raw = yf.download(
            ticker_list, period="2d", interval="1d",
            progress=False, auto_adjust=True,
            **({"group_by": "ticker"} if len(ticker_list) > 1 else {})
        )
        prices: Dict[str, float] = {}
        for t in ticker_list:
            try:
                col = raw[t]["Close"] if len(ticker_list) > 1 else raw["Close"]
                prices[t] = float(col.dropna().iloc[-1])
            except Exception:
                pass
        return prices
    except Exception:
        return {}


def _get_claude_narrative(ticker: str, advice: PositionAdvice, regime: str) -> str:
    """Generate or retrieve cached Claude 2-sentence narrative."""
    cache_key = f"_advisor_narrative_{ticker}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    if not _HAS_CLAUDE:
        return ""

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=_ANTHROPIC_KEY)
        f = advice.factors
        prompt = (
            f"You are a quantitative analyst. In exactly 2 sentences, explain why "
            f"{ticker} signals {advice.signal} in the current {regime} regime. "
            f"Factor scores: Edgar={f.get('edgar', 0):.2f}, Insider={f.get('insider', 0):.2f}, "
            f"Congress={f.get('congress', 0):.2f}, News={f.get('news', 0):.2f}, "
            f"Macro={f.get('macro', 0):.2f}. Overall score: {advice.final_score:.3f if advice.final_score is not None else 'N/A'}. "
            f"Be specific about which factor drives the signal."
        )
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=120,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip()
        st.session_state[cache_key] = text
        return text
    except Exception as exc:
        log.warning("Claude narrative failed for %s: %s", ticker, exc)
        return ""


def _factor_bar(score: float, width: int = 10) -> str:
    filled = min(width, max(0, round(score * width)))
    return "█" * filled + "░" * (width - filled)


def _render_regime_banner(regime: str) -> None:
    _COLOR = {
        "Bull":    "#00FFA3",
        "Neutral": "#60A5FA",
        "Bear":    "#FFB347",
        "Panic":   "#FF6B6B",
        "Crash":   "#FF2222",
    }
    color = _COLOR.get(regime, "#9E9E9E")
    st.markdown(
        f'<div style="background:{color}18;border:1px solid {color};border-radius:8px;'
        f'padding:10px 20px;margin:8px 0;">'
        f'<span style="color:{color};font-size:1.1em;">Regime: <strong>{regime}</strong></span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render() -> None:
    st.title("💼 Portfolio Advisor")
    st.caption("Daily buy/sell/hold signals on your Revolut positions. Scores from last pipeline run.")

    # ── Load Revolut data ──────────────────────────────────────────────────────
    revolut_data = _load_revolut()
    if revolut_data is None:
        st.info(
            "No Revolut portfolio found. Upload your statement in the **Portfolio Sync** tab "
            "(Dashboard → Portfolio Sync)."
        )
        return

    positions   = revolut_data.get("positions", [])
    imported_at = revolut_data.get("imported_at", "—")

    if not positions:
        st.warning("Imported portfolio has no positions.")
        return

    # ── Detect regime ──────────────────────────────────────────────────────────
    with st.spinner("Detecting regime…"):
        regime = _load_regime()

    _render_regime_banner(regime)
    st.caption(f"Revolut portfolio · Imported: **{imported_at}** · {len(positions)} positions")

    # ── Build advice ───────────────────────────────────────────────────────────
    with st.spinner("Scoring positions…"):
        advice_list = build_advice(positions, regime)

    # ── Fetch live prices ──────────────────────────────────────────────────────
    scored_tickers = tuple(a.ticker for a in advice_list if not a.not_in_universe)
    with st.spinner("Fetching live prices…"):
        prices = _fetch_live_prices(scored_tickers)

    # Attach market_value
    for adv in advice_list:
        price = prices.get(adv.ticker)
        if price is not None:
            adv.market_value = adv.net_qty * price

    # ── Portfolio health summary ───────────────────────────────────────────────
    health_positions = [
        {"ticker": a.ticker, "final_score": a.final_score or 0.0, "market_value": a.market_value}
        for a in advice_list if a.final_score is not None
    ]
    health_score = compute_health_score(health_positions)

    signal_counts = {s: sum(1 for a in advice_list if a.signal == s)
                     for s in ("ADD", "HOLD", "REDUCE", "EXIT")}

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Health Score", f"{health_score:.2f}",
              help="Weighted avg factor score by position value. ≥0.65 = healthy.")
    c2.metric("➕ ADD",    signal_counts.get("ADD",    0))
    c3.metric("= HOLD",   signal_counts.get("HOLD",   0))
    c4.metric("⬇️ REDUCE", signal_counts.get("REDUCE", 0))
    c5.metric("🚫 EXIT",  signal_counts.get("EXIT",   0))

    if not _HAS_CLAUDE:
        st.caption("💡 Set ANTHROPIC_API_KEY in .env to enable Claude narratives per position.")

    st.divider()

    # ── Position table — sorted by urgency ────────────────────────────────────
    sort_order = {"EXIT": 0, "REDUCE": 1, "ADD": 2, "HOLD": 3, "—": 4}
    advice_list.sort(key=lambda a: sort_order.get(a.signal, 9))

    for adv in advice_list:
        signal_color = _SIGNAL_COLOR.get(adv.signal, "#888")
        signal_icon  = _SIGNAL_ICON.get(adv.signal, "—")
        score_str    = f"{adv.final_score:.3f}" if adv.final_score is not None else "N/A"
        price        = prices.get(adv.ticker)
        cost_basis   = adv.net_qty * adv.avg_cost
        unreal_pl    = (adv.market_value - cost_basis) if price is not None else None
        unreal_pct   = (unreal_pl / cost_basis * 100) if (unreal_pl is not None and cost_basis > 0) else None

        age_str  = f"Signal: {adv.signal_age_days}d old" if adv.signal_age_days is not None else ""
        age_warn = adv.signal_age_days is not None and adv.signal_age_days > 30

        header = (
            f"**{adv.ticker}**"
            + (f" *(Revolut: {adv.revolut_ticker})*" if adv.revolut_ticker != adv.ticker else "")
            + "  ·  "
            + f"<span style='color:{signal_color};font-weight:700;'>{signal_icon} {adv.signal}</span>"
            + f"  ·  Score: **{score_str}**"
            + (f"  ·  {unreal_pl:+,.2f} ({unreal_pct:+.1f}%)" if unreal_pl is not None else "")
            + (f"  ·  ⚠️ {age_str}" if age_warn else (f"  ·  {age_str}" if age_str else ""))
            + ("  ·  *not in universe*" if adv.not_in_universe else "")
        )

        with st.expander(header, expanded=False):
            if adv.not_in_universe:
                st.caption("This ticker is not in the scoring universe — no factor data available.")
            else:
                # Factor bars
                f = adv.factors
                factor_data = [
                    {"Factor": "📋 Edgar",    "Weight": "30%", "Score": f"{f.get('edgar',    0):.3f}", "Bar": _factor_bar(f.get('edgar',    0))},
                    {"Factor": "🏦 Insider",  "Weight": "25%", "Score": f"{f.get('insider',  0):.3f}", "Bar": _factor_bar(f.get('insider',  0))},
                    {"Factor": "🏛️ Congress", "Weight": "20%", "Score": f"{f.get('congress', 0):.3f}", "Bar": _factor_bar(f.get('congress', 0))},
                    {"Factor": "📰 News",     "Weight": "15%", "Score": f"{f.get('news',     0):.3f}", "Bar": _factor_bar(f.get('news',     0))},
                    {"Factor": "📈 Macro",    "Weight": "10%", "Score": f"{f.get('macro',    0):.3f}", "Bar": _factor_bar(f.get('macro',    0))},
                ]
                st.dataframe(pd.DataFrame(factor_data), use_container_width=True, hide_index=True)

                # Claude narrative
                if _HAS_CLAUDE and adv.signal != "—":
                    with st.spinner(f"Generating narrative for {adv.ticker}…"):
                        narrative = _get_claude_narrative(adv.ticker, adv, regime)
                    if narrative:
                        st.markdown(f"> {narrative}")

                # Swap candidate
                if adv.swap_candidate:
                    swap = adv.swap_candidate
                    st.info(
                        f"🔄 **Consider rotating into {swap['ticker']}** "
                        f"(score {swap.get('final_score', 0):.2f}, {swap.get('badge', '')}, "
                        f"same sector)"
                    )

            # Raw position data (always shown)
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Qty",        f"{adv.net_qty:.4f}")
            col_b.metric("Avg Cost",   f"{adv.avg_cost:.2f} {adv.currency}")
            col_c.metric("Live Price", f"{price:.2f}" if price is not None else "—")
