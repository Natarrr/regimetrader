"""pages/3_Valuation_Radar.py
Shiller (2013) · Thaler (2017) — CAPE, Excess CAPE Yield & Valuation Danger Zone.

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
import pandas as pd

from backend.data.fred_service import fetch_10y_yield, fetch_cpi
from backend.quant_models.valuation_radar import (
    fetch_shiller_cape_series,
    cape_percentile,
    excess_cape_yield,
    real_yield,
    is_valuation_danger_zone,
)

_CHART_BG = "#050505"
_CARD_BG  = "#121212"
_GREEN    = "#00FFA3"
_RED      = "#FF6B6B"
_GOLD     = "#FFD700"
_PURPLE   = "#A78BFA"
_GRID     = "#1E1E1E"

_CAPE_HIGH   = 30.0
_CAPE_BUBBLE = 40.0


@st.cache_data(ttl=3600, show_spinner=False)
def _load() -> dict:
    """Shiller (2013 Nobel) + Thaler (2017 Nobel) — CAPE & Excess CAPE Yield.

    CAPE: $CAPE_t = P_t / (\\frac{1}{10} \\sum_{k=0}^{9} E_{t-k})$
    ECY: $ECY = \\frac{1}{CAPE} - r_{real,10Y}$ (negative = equities expensive vs bonds)
    Real yield: $r_{real} \\approx r_{nominal} - \\pi$ (Fisher approximation)
    """
    cape_series = fetch_shiller_cape_series()
    if cape_series.empty:
        raise ValueError("CAPE series unavailable from all sources.")

    current_cape = float(cape_series.iloc[-1])

    try:
        gs10        = fetch_10y_yield()
        cpi         = fetch_cpi()
        nominal_10y = float(gs10.iloc[-1])
        cpi_12m     = float(cpi.pct_change(12).dropna().iloc[-1] * 100)
        real_10y    = real_yield(nominal_10y, cpi_12m)
    except Exception:
        real_10y = 0.015

    pct = cape_percentile(cape_series, current_cape)
    ecy = excess_cape_yield(current_cape, real_10y)

    return dict(
        cape_series=cape_series,
        current_cape=current_cape,
        pct=pct,
        ecy=ecy,
        real_10y=real_10y,
        danger=is_valuation_danger_zone(pct),
    )


def render() -> None:
    """Render the Valuation Radar page.

    Shiller (2013 Nobel) — CAPE smooths cyclical earnings noise to reveal
    whether equities are priced for long-run disappointment.
    """
    st.title("📈 Valuation Radar")
    st.caption("Shiller (2013 Nobel) · Thaler (2017 Nobel)")

    with st.spinner("Loading Shiller CAPE data…"):
        try:
            d = _load()
        except Exception as exc:
            st.error(f"Valuation data failed: {exc}")
            return

    danger_icon = "🔴 DANGER ZONE" if d["danger"] else "🟢 Normal"
    ecy_icon    = "🔴 Expensive" if d["ecy"] < 0 else "🟢 Attractive"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Shiller CAPE", f"{d['current_cape']:.1f}x",
              delta=("🔴 Extreme" if d["current_cape"] > _CAPE_BUBBLE
                     else ("🟡 Elevated" if d["current_cape"] > _CAPE_HIGH else "🟢 Normal")))
    c2.metric("CAPE Percentile", f"{d['pct']:.1f}th",
              delta=danger_icon,
              delta_color="inverse" if d["danger"] else "normal")
    c3.metric("Excess CAPE Yield", f"{d['ecy']*100:.2f}%",
              delta=ecy_icon,
              delta_color="inverse" if d["ecy"] < 0 else "normal")
    c4.metric("Real 10Y Yield", f"{d['real_10y']*100:.2f}%")

    st.divider()

    col_chart, col_gauge = st.columns([3, 1])

    with col_chart:
        series    = d["cape_series"].dropna()
        cutoff    = pd.Timestamp.now() - pd.DateOffset(years=40)
        series_40 = series[series.index >= cutoff]

        fig_cape = go.Figure()
        fig_cape.add_hrect(y0=_CAPE_BUBBLE, y1=series_40.max() * 1.1,
                           fillcolor="rgba(255,107,107,0.07)", line_width=0,
                           annotation_text="Bubble Territory",
                           annotation_font_color=_RED,
                           annotation_position="top left")
        fig_cape.add_hrect(y0=_CAPE_HIGH, y1=_CAPE_BUBBLE,
                           fillcolor="rgba(255,215,0,0.05)", line_width=0,
                           annotation_text="Elevated",
                           annotation_font_color=_GOLD,
                           annotation_position="top left")
        fig_cape.add_trace(go.Scatter(
            x=series_40.index, y=series_40.values,
            name="CAPE", line=dict(color=_PURPLE, width=1.5),
            fill="tozeroy", fillcolor="rgba(167,139,250,0.05)",
        ))
        fig_cape.add_hline(y=_CAPE_HIGH, line_dash="dot", line_color=_GOLD, line_width=1)
        fig_cape.add_hline(y=_CAPE_BUBBLE, line_dash="dot", line_color=_RED, line_width=1)
        fig_cape.add_hline(y=d["current_cape"], line_dash="dash", line_color=_GREEN,
                           line_width=1.5,
                           annotation_text=f"Current: {d['current_cape']:.1f}x",
                           annotation_font_color=_GREEN)

        for dt, crash_label in [("2000-03-01", "Dot-com peak"), ("2007-10-01", "2008 crisis")]:
            ts  = pd.Timestamp(dt)
            idx = series_40.index.searchsorted(ts)
            if idx < len(series_40):
                x_str = str(series_40.index[idx].date())
                fig_cape.add_shape(
                    type="line", x0=x_str, x1=x_str, y0=0, y1=1,
                    xref="x", yref="paper",
                    line=dict(color="#555", width=1, dash="dot"),
                )
                fig_cape.add_annotation(
                    x=x_str, y=0.98, yref="paper",
                    text=crash_label, showarrow=False,
                    font=dict(color="#999", size=10),
                    xanchor="left", yanchor="top",
                )

        fig_cape.update_layout(
            title="Shiller CAPE — 40-Year History",
            template="plotly_dark",
            paper_bgcolor=_CHART_BG, plot_bgcolor=_CARD_BG,
            yaxis=dict(title="CAPE (P/E10)", gridcolor=_GRID),
            legend=dict(bgcolor=_CARD_BG),
            height=420,
            hovermode="x unified",
        )
        st.plotly_chart(fig_cape, use_container_width=True)

    with col_gauge:
        gauge_color = _RED if d["pct"] > 95 else (_GOLD if d["pct"] > 75 else _GREEN)
        fig_pct = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=d["pct"],
            delta={"reference": 75, "valueformat": ".1f",
                   "prefix": "vs 75th: ", "suffix": "pts"},
            number={"suffix": "th pct", "font": {"color": gauge_color}},
            title={"text": "CAPE Percentile<br><span style='font-size:0.8em'>vs 40-yr history</span>",
                   "font": {"color": "#E0E0E0", "size": 14}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#E0E0E0"},
                "bar": {"color": gauge_color},
                "bgcolor": _CARD_BG,
                "threshold": {
                    "line": {"color": _RED, "width": 3},
                    "thickness": 0.75, "value": 95,
                },
                "steps": [
                    {"range": [0, 75],   "color": "#1A3A2A"},
                    {"range": [75, 95],  "color": "#3A3A1A"},
                    {"range": [95, 100], "color": "#3A1A1A"},
                ],
            },
        ))
        fig_pct.update_layout(
            paper_bgcolor=_CHART_BG, font=dict(color="#E0E0E0"),
            height=280, margin=dict(t=30, b=10, l=10, r=10),
        )
        st.plotly_chart(fig_pct, use_container_width=True)

        ecy_val = d["ecy"] * 100
        fig_ecy = go.Figure(go.Bar(
            x=["Earnings Yield (1/CAPE)", "Real 10Y Yield", "Excess CAPE Yield"],
            y=[100 / d["current_cape"], d["real_10y"] * 100, ecy_val],
            marker_color=[_PURPLE, _GOLD, _RED if ecy_val < 0 else _GREEN],
            text=[f"{v:.2f}%" for v in [100 / d["current_cape"],
                                         d["real_10y"] * 100, ecy_val]],
            textposition="outside",
        ))
        fig_ecy.update_layout(
            title="Equity vs Bond Yield",
            template="plotly_dark",
            paper_bgcolor=_CHART_BG, plot_bgcolor=_CARD_BG,
            yaxis=dict(title="%", gridcolor=_GRID, zeroline=True, zerolinecolor="#555"),
            height=280,
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig_ecy, use_container_width=True)

    with st.expander("Model notes"):
        st.markdown(r"""
**Shiller (2013)** — CAPE smooths earnings over 10 years to remove cyclical noise:
$CAPE_t = P_t / \overline{E}_{10}$. Historically > 30x precedes corrections.

**Thaler (2017)** — Excess CAPE Yield measures equity risk premium vs bonds:
$ECY = 1/CAPE - r_{real,10Y}$. When $ECY < 0$, equities are expensive relative to bonds (Minsky precondition 2).

**Danger Zone**: CAPE percentile > 95th (top 5% of all monthly readings since 1880).
        """)
