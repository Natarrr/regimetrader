"""pages/4_Contagion_Web.py
Leontief (1973) · Tirole (2014) — Inter-Sector Shock Propagation via I-O Matrix.

Directly calls backend quant_models without going through FastAPI.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from backend.quant_models.contagion_web import (
    SECTORS,
    build_io_matrix,
    leontief_inverse,
    shock_propagation,
    critical_nodes,
    total_gdp_impact,
)

st.set_page_config(page_title="Contagion Web", page_icon="🕸️", layout="wide")

st.title("🕸️ Contagion Web")
st.caption("Leontief (1973 Nobel) · Tirole (2014 Nobel)")

_CHART_BG = "#050505"
_CARD_BG  = "#121212"
_GREEN    = "#00FFA3"
_RED      = "#FF6B6B"
_GOLD     = "#FFD700"
_PURPLE   = "#A78BFA"
_GRID     = "#1E1E1E"


@st.cache_data(show_spinner=False)
def _build_matrix() -> tuple:
    """Leontief (1973 Nobel) — Compute I-O matrix and Leontief inverse.

    Leontief inverse: $L = (I - A)^{-1}$, where $x = L \\cdot d$.
    Output multiplier for sector $i$: $m_i = \\sum_j L_{ij}$.
    """
    A     = build_io_matrix()
    L     = leontief_inverse(A)
    nodes = critical_nodes(L, top_n=5)
    mults = L.sum(axis=1)
    return A, L, nodes, mults


A, L, top_nodes, multipliers = _build_matrix()

# ── Controls ───────────────────────────────────────────────────────────────────
st.subheader("Shock Simulation")
col_s, col_p, col_b = st.columns([2, 2, 1])
sector    = col_s.selectbox("Shocked Sector", SECTORS, index=SECTORS.index("Energy"))
shock_pct = col_p.slider("Demand Shock (%)", min_value=-100, max_value=-1, value=-20, step=1)
run       = col_b.button("▶ Simulate", type="primary", width="stretch")

if run or "contagion_result" in st.session_state:
    if run:
        impacts  = shock_propagation(A, {sector: shock_pct / 100.0})
        gdp_imp  = total_gdp_impact(impacts)
        st.session_state["contagion_result"] = (sector, shock_pct, impacts, gdp_imp)
    else:
        sector, shock_pct, impacts, gdp_imp = st.session_state["contagion_result"]

    # ── KPI row ────────────────────────────────────────────────────────────────
    shocked_impact = impacts.get(sector, 0.0)
    other_impacts  = [v for k, v in impacts.items() if k != sector]
    max_spill      = min(other_impacts)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("GDP Impact (weighted)", f"{gdp_imp:.2f}%",
              delta_color="inverse")
    c2.metric(f"{sector} Output Impact", f"{shocked_impact:.2f}%",
              delta_color="inverse")
    c3.metric("Worst Spillover", f"{max_spill:.2f}%",
              delta=min(impacts, key=impacts.get),
              delta_color="inverse")
    c4.metric("Critical Node #1", top_nodes[0],
              delta=f"×{multipliers[SECTORS.index(top_nodes[0])]:.2f} multiplier")

    st.divider()

    col_bar, col_heat = st.columns([3, 2])

    # ── Sector impact bar chart ────────────────────────────────────────────────
    with col_bar:
        sorted_items = sorted(impacts.items(), key=lambda x: x[1])
        s_names = [k for k, _ in sorted_items]
        s_vals  = [v for _, v in sorted_items]
        colors  = [_RED if v < 0 else _GREEN for v in s_vals]

        fig_bar = go.Figure(go.Bar(
            y=s_names, x=s_vals,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.2f}%" for v in s_vals],
            textposition="outside",
        ))
        fig_bar.add_vline(x=0, line_color="#555", line_width=1)
        fig_bar.update_layout(
            title=f"Output Impact per Sector — {sector} shock {shock_pct}%",
            template="plotly_dark",
            paper_bgcolor=_CHART_BG, plot_bgcolor=_CARD_BG,
            xaxis=dict(title="% Output Change", gridcolor=_GRID, zeroline=False),
            yaxis=dict(gridcolor=_GRID),
            height=420,
            margin=dict(l=120, r=80, t=50, b=30),
        )
        st.plotly_chart(fig_bar, width="stretch")

    # ── Leontief matrix heatmap ────────────────────────────────────────────────
    with col_heat:
        short = [s.replace("_", "\n") for s in SECTORS]
        fig_heat = go.Figure(go.Heatmap(
            z=L,
            x=short, y=short,
            colorscale="Viridis",
            colorbar=dict(title="Multiplier", tickfont=dict(size=9)),
            hoverongaps=False,
            hovertemplate="From %{y} → %{x}<br>Multiplier: %{z:.3f}<extra></extra>",
        ))
        fig_heat.update_layout(
            title="Leontief Inverse L = (I−A)⁻¹",
            template="plotly_dark",
            paper_bgcolor=_CHART_BG, plot_bgcolor=_CARD_BG,
            height=420,
            xaxis=dict(tickfont=dict(size=8)),
            yaxis=dict(tickfont=dict(size=8)),
        )
        st.plotly_chart(fig_heat, width="stretch")

else:
    # ── Pre-run: show critical nodes by default ────────────────────────────────
    st.info("Select a sector and click **▶ Simulate** to run the shock.")

# ── Critical Nodes (always visible) ───────────────────────────────────────────
st.divider()
st.subheader("Critical Nodes — Output Multiplier Ranking")

mult_df_items = sorted(
    zip(SECTORS, multipliers.tolist()),
    key=lambda x: -x[1]
)
c_names = [s for s, _ in mult_df_items]
c_mults = [m for _, m in mult_df_items]
bar_colors = [
    _RED if i < 3 else (_GOLD if i < 5 else _PURPLE)
    for i in range(len(c_names))
]

fig_nodes = go.Figure(go.Bar(
    x=c_names, y=c_mults,
    marker_color=bar_colors,
    text=[f"{m:.3f}×" for m in c_mults],
    textposition="outside",
))
fig_nodes.update_layout(
    title="Sector Output Multipliers (row-sum of Leontief inverse)",
    template="plotly_dark",
    paper_bgcolor=_CHART_BG, plot_bgcolor=_CARD_BG,
    yaxis=dict(title="Output Multiplier", gridcolor=_GRID),
    xaxis=dict(tickangle=-30),
    height=320,
    showlegend=False,
)
st.plotly_chart(fig_nodes, width="stretch")

col_n1, col_n2, col_n3 = st.columns(3)
for i, col in enumerate([col_n1, col_n2, col_n3]):
    sname = top_nodes[i]
    midx  = SECTORS.index(sname)
    col.metric(f"#{i+1} Critical Node", sname,
               delta=f"×{multipliers[midx]:.3f} total output multiplier")

with st.expander("Model notes"):
    st.markdown("""
**Leontief (1973)** — The I-O model captures all indirect supply-chain dependencies.
$\\Delta x = (I-A)^{-1} \\cdot \\Delta d$ — total output change from a demand shock $\\Delta d$.

**Tirole (2014)** — Systemic risk arises from interconnected balance sheets.
Sectors with high multipliers ($m_i = \\sum_j L_{ij}$) amplify shocks across the economy.

The calibrated 11×11 matrix approximates BEA 2022 Use Tables across GICS sectors.
A shock to a **critical node** (high multiplier) propagates further than one to a peripheral sector.
    """)
