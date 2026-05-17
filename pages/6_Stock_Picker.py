"""pages/6_Stock_Picker.py
Monthly stock pick leaderboard — sector picks + cap-tier picks.
Reads logs/top_lists.json (produced by edgar_3x pipeline). Zero API calls.

Run via sidebar: "📅 Stock Picker"
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

log = logging.getLogger(__name__)

_ROOT       = Path(__file__).parent.parent
_TOP_LISTS  = _ROOT / "logs" / "top_lists.json"

_BADGE_COLOR = {
    "HIGH BUY":     "#00d26a",
    "TACTICAL BUY": "#f5a623",
    "WATCHLIST":    "#888888",
}

_SECTOR_EMOJI = {
    "Energy":                   "⚡",
    "Materials":                "🪨",
    "Communication Services":   "📡",
    "Healthcare":               "🏥",
    "Information Technology":   "💻",
}


@st.cache_data(ttl=3600, show_spinner=False)
def _load_top_lists() -> Optional[Dict[str, Any]]:
    try:
        return json.loads(_TOP_LISTS.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as exc:
        log.warning("top_lists.json load failed: %s", exc)
        return None


def _score_bar(score: float, width: int = 10) -> str:
    filled = round(max(0.0, min(1.0, score)) * width)
    return "█" * filled + "░" * (width - filled)


def _render_ticker_table(entries: List[Dict[str, Any]], show_watchlist: bool = False) -> None:
    if not entries:
        st.caption("No tickers in this category.")
        return

    rows = []
    for i, e in enumerate(entries, 1):
        badge  = e.get("badge", "WATCHLIST")
        if badge == "WATCHLIST" and not show_watchlist:
            continue
        score  = e.get("final_score", 0.0)
        f      = e.get("factors", {})
        rows.append({
            "#":        i,
            "Ticker":   e.get("ticker", "?"),
            "Cap":      e.get("cap_tier", "?").capitalize(),
            "Score":    f"{score:.3f}",
            "Bar":      _score_bar(score),
            "Badge":    badge,
            "CEO Buy":  "✅" if e.get("ceo_buy") else "",
            "Edgar":    f"{f.get('edgar',0):.2f}",
            "Insider":  f"{f.get('insider',0):.2f}",
            "Congress": f"{f.get('congress',0):.2f}",
            "News":     f"{f.get('news',0):.2f}",
            "Momentum": f"{f.get('momentum',0):.2f}",
        })

    if not rows:
        st.caption("No HIGH BUY or TACTICAL BUY tickers. Toggle 'Show Watchlist' to see all.")
        return

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_ticker_list_with_evidence(entries: List[Dict[str, Any]], show_watchlist: bool = False) -> None:
    """Render entries as expandable rows with evidence sub-sections."""
    if not entries:
        st.caption("No tickers in this category.")
        return

    shown = 0
    for i, e in enumerate(entries, 1):
        badge = e.get("badge", "WATCHLIST")
        if badge == "WATCHLIST" and not show_watchlist:
            continue
        shown += 1
        score  = e.get("final_score", 0.0)
        f      = e.get("factors", {})
        ticker = e.get("ticker", "?")
        ceo    = "✅ CEO Buy" if e.get("ceo_buy") else ""
        label  = f"**{i}. {ticker}** — {badge}  Score: {score:.3f}  {ceo}"

        with st.expander(label, expanded=False):
            # Factor mini-table
            factor_rows = [
                {"Factor": "📋 Edgar",    "W": "28%", "Score": f"{f.get('edgar',    0):.3f}"},
                {"Factor": "🏦 Insider",  "W": "23%", "Score": f"{f.get('insider',  0):.3f}"},
                {"Factor": "🏛️ Congress", "W": "22%", "Score": f"{f.get('congress', 0):.3f}"},
                {"Factor": "📰 News",     "W": "15%", "Score": f"{f.get('news',     0):.3f}"},
                {"Factor": "📈 Momentum", "W": "12%", "Score": f"{f.get('momentum', 0):.3f}"},
            ]
            st.dataframe(pd.DataFrame(factor_rows), use_container_width=True, hide_index=True)

            # Evidence sub-section
            insider_usd          = float(e.get("insider_usd", 0.0))
            news_source          = e.get("news_source", "none")
            momentum_spy_rel     = float(e.get("momentum_spy_relative", 0.0))
            volume_spike         = float(e.get("volume_spike", 1.0))
            qe                   = e.get("quiver_evidence", {})
            cong                 = qe.get("congress", {})
            cong_net             = cong.get("net", 0)
            cong_buys            = cong.get("purchases", 0)
            cong_sales           = cong.get("sales", 0)
            cong_days            = cong.get("recency_days")
            cong_reps            = cong.get("representatives", [])

            has_insider  = insider_usd > 0 or f.get("insider", 0) > 0
            has_congress = (cong_buys > 0 or cong_sales > 0) or f.get("congress", 0) > 0
            has_news     = news_source != "none" or f.get("news", 0) > 0
            has_momentum = (momentum_spy_rel != 0.0 or volume_spike != 1.0) or f.get("momentum", 0) > 0
            has_edgar    = f.get("edgar", 0) > 0

            evidence_lines = []
            if has_insider:
                evidence_lines.append(f"🏦 **Insider** · ${insider_usd:,.0f}")
            if has_congress:
                days_str = f" · {cong_days}d ago" if cong_days is not None else ""
                reps_str = ", ".join(cong_reps[:2]) if cong_reps else "—"
                evidence_lines.append(
                    f"🏛️ **Congress** · Net {cong_net:+d} ({cong_buys} buys, {cong_sales} sells){days_str} · [{reps_str}]"
                )
            if has_news:
                src = {"finnhub": "Finnhub", "yfinance": "yfinance"}.get(news_source, news_source)
                evidence_lines.append(f"📰 **News** · Source: {src} · Score: {f.get('news', 0):.2f}")
            if has_momentum:
                evidence_lines.append(
                    f"📈 **Momentum** · {momentum_spy_rel*100:+.1f}% vs SPY · {volume_spike:.1f}× avg vol"
                )
            if has_edgar:
                ceo_str = " · CEO Buy ✅" if e.get("ceo_buy") else ""
                evidence_lines.append(f"📋 **EDGAR** · Score {f.get('edgar', 0):.2f}{ceo_str}")

            if evidence_lines:
                st.markdown("\n\n".join(evidence_lines))
            else:
                st.caption("No evidence data available for this ticker.")

    if shown == 0:
        st.caption("No HIGH BUY or TACTICAL BUY tickers. Toggle 'Show Watchlist' to see all.")


def render() -> None:
    st.title("📅 Stock Picker")
    st.caption("Monthly pick leaderboard powered by the 5-factor scoring engine. Informational only.")

    # ── Load data ──────────────────────────────────────────────────────────────
    col_ref, col_ts = st.columns([1, 6])
    if col_ref.button("↻ Refresh", key="sp_refresh"):
        _load_top_lists.clear()
        st.rerun()

    data = _load_top_lists()

    if data is None:
        st.error(
            "**⚠️ No data** — `logs/top_lists.json` not found.\n\n"
            "Run the edgar_3x pipeline to generate picks:\n"
            "```\npython -m backend.market_intel.generate_top_lists --force\n```"
        )
        return

    generated_at = data.get("generated_at", "—")
    ticker_count = data.get("ticker_count", 0)
    col_ts.caption(f"Pipeline ran: **{generated_at}** · {ticker_count} tickers scored")

    show_watchlist = st.toggle("Show WATCHLIST tickers", value=False, key="sp_show_watchlist")

    st.divider()

    # ── Section 1: Sector Picks ────────────────────────────────────────────────
    st.subheader("Sector Picks — Top 3 per Sector")

    sector_picks: Dict[str, List] = data.get("sector_picks", {})

    if not sector_picks:
        st.warning(
            "Sector picks not in this snapshot. Re-run the pipeline with the updated "
            "`generate_top_lists.py` to populate sector data."
        )
    else:
        for sector, emoji in _SECTOR_EMOJI.items():
            picks = sector_picks.get(sector, [])
            label = f"{emoji} {sector} ({len(picks)} picks)"
            with st.expander(label, expanded=True):
                _render_ticker_list_with_evidence(picks, show_watchlist=show_watchlist)

    st.divider()

    # ── Section 2: Cap-Tier Overview ──────────────────────────────────────────
    st.subheader("Cap-Tier Overview")

    col_tb, col_mc, col_sc = st.columns(3)

    with col_tb:
        st.markdown("**🏆 Top Buys**")
        _render_ticker_table(data.get("top_buys", []), show_watchlist=show_watchlist)

    with col_mc:
        st.markdown("**⬡ Mid Caps**")
        _render_ticker_table(data.get("mid_caps", []), show_watchlist=show_watchlist)

    with col_sc:
        st.markdown("**◆ Small Caps**")
        _render_ticker_table(data.get("small_caps", []), show_watchlist=show_watchlist)
