"""pages/1_Monetary_Pulse.py
Friedman (1968) · Kuznets (1971) · Prescott (2004) — Monetary Regime Dashboard.

Directly calls backend quant_models without going through FastAPI.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import plotly.graph_objects as go

from backend.data.fred_service import (
    fetch_10y_yield,
    fetch_2y_yield,
    fetch_m2_velocity,
    fetch_real_gdp,
)
from backend.quant_models.monetary_pulse import (
    yield_spread,
    is_inverted,
    m2_velocity_trend,
    hp_filter_trend,
    monetary_regime,
)

st.set_page_config(page_title="Monetary Pulse", page_icon="💰", layout="wide")

st.title("💰 Monetary Pulse")
st.caption("Friedman (1968 Nobel) · Kuznets (1971 Nobel) · Prescott (2004 Nobel)")

_CHART_BG = "#050505"
_CARD_BG  = "#121212"
_GREEN    = "#00FFA3"
_RED      = "#FF6B6B"
_GOLD     = "#FFD700"
_PURPLE   = "#A78BFA"
_GRID     = "#1E1E1E"


@st.cache_data(ttl=3600, show_spinner=False)
def _load() -> dict:
    """Friedman (1968 Nobel) + Kuznets (1971 Nobel) + Prescott (2004 Nobel).

    Fetches FRED macro series and computes monetary regime snapshot.
    Yield spread: $S_t = r_{10Y,t} - r_{2Y,t}$ (basis points).
    HP filter: $\\min_{\\tau} \\sum_t (y_t - \\tau_t)^2 + \\lambda \\sum_t (\\Delta^2 \\tau_t)^2$
    """
    gs10 = fetch_10y_yield()
    gs2  = fetch_2y_yield()
    m2v  = fetch_m2_velocity()
    gdp  = fetch_real_gdp()

    spread        = yield_spread(gs10, gs2)
    _, cycle      = hp_filter_trend(gdp)
    spread_latest = float(spread.iloc[-1])
    m2v_latest    = float(m2v.iloc[-1])

    return dict(
        gs10=gs10, gs2=gs2, m2v=m2v,
        spread=spread, cycle=cycle,
        spread_latest=spread_latest,
        m2v_latest=m2v_latest,
        m2v_trend=m2_velocity_trend(m2v),
        hp_cycle=float(cycle.iloc[-1]),
        inverted=is_inverted(spread),
        regime=monetary_regime(spread, m2v),
    )


with st.spinner("Fetching FRED data…"):
    try:
        d = _load()
    except Exception as exc:
        st.error(f"FRED fetch failed: {exc}")
        st.stop()

# ── KPI row ────────────────────────────────────────────────────────────────────
_REGIME_ICON = {"EASING": "🟢", "NEUTRAL": "🟡", "TIGHTENING": "🔴"}
icon = _REGIME_ICON.get(d["regime"], "⚪")

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "Yield Spread (10Y − 2Y)",
    f"{d['spread_latest']:.1f} bps",
    delta="⚠️ INVERTED" if d["inverted"] else "Normal",
    delta_color="inverse" if d["inverted"] else "normal",
)
c2.metric("M2 Velocity", f"{d['m2v_latest']:.4f}", delta=d["m2v_trend"])
c3.metric("HP GDP Cycle", f"{d['hp_cycle']:.2f}")
c4.metric("Monetary Regime", f"{icon} {d['regime']}")

st.divider()

# ── Yield curve chart ──────────────────────────────────────────────────────────
fig_yield = go.Figure()
fig_yield.add_trace(go.Scatter(
    x=d["gs10"].index, y=d["gs10"].values,
    name="10Y Yield", line=dict(color=_GREEN, width=1.5),
))
fig_yield.add_trace(go.Scatter(
    x=d["gs2"].index, y=d["gs2"].values,
    name="2Y Yield", line=dict(color=_RED, width=1.5),
))
fig_yield.add_trace(go.Scatter(
    x=d["spread"].index, y=d["spread"].values,
    name="Spread (bps)", yaxis="y2",
    fill="tozeroy",
    line=dict(color=_GOLD, width=1),
    fillcolor="rgba(255,215,0,0.08)",
))
fig_yield.add_hline(y=0, yref="y2", line_dash="dash", line_color="#555", line_width=1)
fig_yield.update_layout(
    title="US Treasury Yields & 10Y−2Y Spread",
    template="plotly_dark",
    paper_bgcolor=_CHART_BG, plot_bgcolor=_CARD_BG,
    yaxis=dict(title="Yield (%)", gridcolor=_GRID),
    yaxis2=dict(title="Spread (bps)", overlaying="y", side="right", gridcolor=_GRID, zeroline=False),
    legend=dict(bgcolor=_CARD_BG, bordercolor=_GRID),
    height=370,
    hovermode="x unified",
)
st.plotly_chart(fig_yield, width="stretch")

# ── M2 Velocity + HP Cycle ─────────────────────────────────────────────────────
col_m2, col_gdp = st.columns(2)

with col_m2:
    fig_m2 = go.Figure()
    fig_m2.add_trace(go.Scatter(
        x=d["m2v"].index, y=d["m2v"].values,
        name="M2 Velocity",
        line=dict(color=_GREEN, width=1.5),
        fill="tozeroy", fillcolor="rgba(0,255,163,0.05)",
    ))
    fig_m2.update_layout(
        title="M2 Money Velocity (FRED M2V)",
        template="plotly_dark",
        paper_bgcolor=_CHART_BG, plot_bgcolor=_CARD_BG,
        yaxis=dict(title="Velocity", gridcolor=_GRID),
        height=300,
    )
    st.plotly_chart(fig_m2, width="stretch")

with col_gdp:
    cycle_pos = d["cycle"].clip(lower=0)
    cycle_neg = d["cycle"].clip(upper=0)

    fig_gdp = go.Figure()
    fig_gdp.add_trace(go.Scatter(
        x=d["cycle"].index, y=cycle_pos.values,
        name="Expansion", fill="tozeroy",
        line=dict(color=_GREEN, width=0),
        fillcolor="rgba(0,255,163,0.2)",
    ))
    fig_gdp.add_trace(go.Scatter(
        x=d["cycle"].index, y=cycle_neg.values,
        name="Contraction", fill="tozeroy",
        line=dict(color=_RED, width=0),
        fillcolor="rgba(255,107,107,0.2)",
    ))
    fig_gdp.add_trace(go.Scatter(
        x=d["cycle"].index, y=d["cycle"].values,
        name="Cycle",
        line=dict(color=_PURPLE, width=1.5),
    ))
    fig_gdp.add_hline(y=0, line_dash="dash", line_color="#555", line_width=1)
    fig_gdp.update_layout(
        title="HP-Filter GDP Cycle Component (λ=1600)",
        template="plotly_dark",
        paper_bgcolor=_CHART_BG, plot_bgcolor=_CARD_BG,
        yaxis=dict(title="Cycle", gridcolor=_GRID),
        height=300,
        showlegend=False,
    )
    st.plotly_chart(fig_gdp, width="stretch")

# ── Interpretation ─────────────────────────────────────────────────────────────
with st.expander("Model notes"):
    st.markdown("""
**Friedman (1968)** — The 10Y−2Y spread is the canonical leading indicator.
Inversion precedes recessions by 12–18 months on average.

**Kuznets (1971)** — M2 velocity $V = GDP / M2$ measures the turnover rate of money.
Falling velocity signals deflationary pressure or hoarding.

**Prescott (2004)** — The HP filter ($\\lambda=1600$ for quarterly data) decomposes
GDP into trend and cycle. Positive cycle = above-trend expansion; negative = contraction.
    """)
