"""pages/2_Volatility_Brain.py
Engle (2003) · Merton (1997) — GJR-GARCH Volatility & Distance-to-Default.

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

from backend.data.market_service import MarketData
from backend.quant_models.volatility_brain import (
    fit_gjr_garch,
    volatility_regime,
    merton_distance_to_default,
)
from utils.volatility import annualise_vol_from_condvar

st.set_page_config(page_title="Volatility Brain", page_icon="📊", layout="wide")

st.title("📊 Volatility Brain")
st.caption("Engle (2003 Nobel) · Merton (1997 Nobel)")

_CHART_BG = "#050505"
_CARD_BG  = "#121212"
_GREEN    = "#00FFA3"
_RED      = "#FF6B6B"
_GOLD     = "#FFD700"
_PURPLE   = "#A78BFA"
_GRID     = "#1E1E1E"


@st.cache_data(ttl=1800, show_spinner=False)
def _run_garch(sym: str, yrs: int) -> dict:
    """Engle (2003 Nobel) — GJR-GARCH(1,1) persistence measure.

    Persistence: $P = \\alpha + \\beta + \\gamma/2$.
    Leverage effect: $\\gamma > 0$ means negative shocks amplify volatility.
    CLUSTERING regime when $P > 0.98$.
    """
    md   = MarketData()
    bars = md.get_historical_bars(symbol=sym, years_back=yrs)
    if len(bars) < 100:
        raise ValueError(f"Only {len(bars)} bars — need ≥ 100 for GARCH fit.")
    log_ret = np.log(bars["Close"] / bars["Close"].shift(1)).dropna().values
    result  = fit_gjr_garch(log_ret)
    result["vol_regime"]  = volatility_regime(result["persistence"])
    result["log_returns"] = log_ret
    result["symbol"]      = sym.upper()
    return result


@st.cache_data(ttl=1800, show_spinner=False)
def _run_merton(sym: str, face_debt: float, rfr: float) -> dict:
    """Merton (1997 Nobel) — Distance-to-Default structural credit model.

    Solves the system: $E = V N(d_1) - F e^{-rT} N(d_2)$,
    $\\sigma_E = (V/E) N(d_1) \\sigma_V$.
    Default probability: $P(D) = N(-d_2)$.
    """
    md   = MarketData()
    bars = md.get_historical_bars(symbol=sym, years_back=3)
    log_ret    = np.log(bars["Close"] / bars["Close"].shift(1)).dropna().values
    equity_vol = float(np.std(log_ret) * np.sqrt(252))
    price      = float(bars["Close"].iloc[-1])
    try:
        import yfinance as yf
        shares = yf.Ticker(sym).info.get("sharesOutstanding", 1e9)
    except Exception:
        shares = 1e9
    equity_value = price * shares
    return merton_distance_to_default(equity_value, face_debt, rfr, equity_vol)


# ── Inputs ─────────────────────────────────────────────────────────────────────
col_sym, col_yrs, col_btn = st.columns([3, 1, 1])
symbol = col_sym.text_input("Ticker Symbol", value="SPY", placeholder="SPY, AAPL…").upper().strip()
years  = col_yrs.number_input("Lookback (yrs)", min_value=2, max_value=10, value=5)
run    = col_btn.button("▶ Run GARCH", type="primary", use_container_width=True)

_cached_sym  = st.session_state.get("garch_symbol", "")
_cached_yrs  = st.session_state.get("garch_years",  -1)
_stale       = (symbol != _cached_sym or years != _cached_yrs)

if run or ("garch_result" in st.session_state and not _stale):
    if run or _stale:
        with st.spinner(f"Fitting GJR-GARCH on {symbol}…"):
            try:
                g = _run_garch(symbol, years)
                st.session_state["garch_result"] = g
                st.session_state["garch_symbol"] = symbol
                st.session_state["garch_years"]  = years
            except Exception as exc:
                st.error(f"GARCH failed: {exc}")
                st.stop()
    else:
        g = st.session_state["garch_result"]

    # ── KPI row ────────────────────────────────────────────────────────────────
    regime_icon = "🔴" if g["vol_regime"] == "CLUSTERING" else "🟢"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Persistence (α+β+γ/2)", f"{g['persistence']:.4f}",
              delta=f"{regime_icon} {g['vol_regime']}",
              delta_color="inverse" if g["vol_regime"] == "CLUSTERING" else "normal")
    c2.metric("Annual Vol (latest)", f"{g['latest_conditional_vol_ann']*100:.2f}%")
    c3.metric("Beta (β)", f"{g['beta']:.4f}")
    c4.metric("Leverage (γ)", f"{g['gamma']:.4f}")

    st.divider()

    col_left, col_right = st.columns([1, 2])

    # ── Parameter bar ──────────────────────────────────────────────────────────
    with col_left:
        fig_params = go.Figure(go.Bar(
            x=["ω (omega)", "α (alpha)", "γ (gamma)", "β (beta)"],
            y=[g["omega"], g["alpha"], g["gamma"], g["beta"]],
            marker_color=[_GOLD, _GREEN, _RED, _PURPLE],
            text=[f"{v:.6f}" for v in [g["omega"], g["alpha"], g["gamma"], g["beta"]]],
            textposition="outside",
        ))
        fig_params.update_layout(
            title="GJR-GARCH(1,1) Parameters",
            template="plotly_dark",
            paper_bgcolor=_CHART_BG, plot_bgcolor=_CARD_BG,
            yaxis=dict(gridcolor=_GRID, showticklabels=False),
            height=320,
            margin=dict(t=40, b=20),
        )
        st.plotly_chart(fig_params, use_container_width=True)

        # Persistence gauge
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=g["persistence"],
            number={"font": {"color": _RED if g["persistence"] > 0.98 else _GREEN}},
            title={"text": "Persistence", "font": {"color": "#E0E0E0", "size": 14}},
            gauge={
                "axis": {"range": [0.80, 1.0], "tickcolor": "#E0E0E0", "tickfont": {"size": 10}},
                "bar": {"color": _RED if g["persistence"] > 0.98 else _GREEN},
                "bgcolor": _CARD_BG,
                "threshold": {
                    "line": {"color": _RED, "width": 3},
                    "thickness": 0.75,
                    "value": 0.98,
                },
                "steps": [
                    {"range": [0.80, 0.98], "color": "#1A3A2A"},
                    {"range": [0.98, 1.00], "color": "#3A1A1A"},
                ],
            },
        ))
        fig_gauge.update_layout(
            paper_bgcolor=_CHART_BG, font=dict(color="#E0E0E0"), height=220,
            margin=dict(t=30, b=10, l=20, r=20),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Conditional volatility time series ─────────────────────────────────────
    with col_right:
        h_t_pct2 = g.get("h_t_pct2")
        if h_t_pct2 is not None:
            # Use canonical converter: h_t_pct2 is daily variance in %-pt²
            # annualise_vol_from_condvar returns annualised vol in plain %-pt (e.g. 10.25)
            ann_vol = annualise_vol_from_condvar(h_t_pct2, units="percent").to_numpy()
            fig_cv = go.Figure()
            fig_cv.add_trace(go.Scatter(
                y=ann_vol,
                mode="lines",
                name="Cond. Vol (Ann. %)",
                line=dict(color=_GREEN, width=1),
                fill="tozeroy",
                fillcolor="rgba(0,255,163,0.06)",
            ))
            fig_cv.add_hline(
                y=float(g["latest_conditional_vol_ann"] * 100),
                line_dash="dash", line_color=_GOLD, line_width=1,
                annotation_text=f"Latest: {g['latest_conditional_vol_ann']*100:.2f}%",
                annotation_font_color=_GOLD,
            )
            fig_cv.update_layout(
                title=f"{g['symbol']} — Conditional Volatility (Annualised %)",
                template="plotly_dark",
                paper_bgcolor=_CHART_BG, plot_bgcolor=_CARD_BG,
                yaxis=dict(title="Vol %", gridcolor=_GRID),
                xaxis=dict(title="Trading days", gridcolor=_GRID),
                height=560,
                margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig_cv, use_container_width=True)

    # ── Merton D2D (optional) ──────────────────────────────────────────────────
    st.divider()
    with st.expander("🏦 Merton Distance-to-Default (optional)"):
        st.caption("Requires total face value of liabilities from the balance sheet.")
        m_col1, m_col2, m_col3 = st.columns(3)
        face_debt = m_col1.number_input("Face Value of Debt ($)", min_value=1e6,
                                         value=5e10, step=1e9, format="%.0f")
        rfr       = m_col2.number_input("Risk-Free Rate", min_value=0.0, max_value=0.2,
                                         value=0.045, step=0.005, format="%.3f")
        run_merton = m_col3.button("Compute D2D", use_container_width=True)

        if run_merton:
            with st.spinner("Running Merton model…"):
                try:
                    mrt = _run_merton(symbol, face_debt, rfr)
                    d2d_icon = "🔴" if mrt["d2d"] < 1.5 else ("🟡" if mrt["d2d"] < 3.0 else "🟢")
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("Distance-to-Default", f"{mrt['d2d']:.3f} σ", delta=d2d_icon)
                    mc2.metric("P(Default)", f"{mrt['prob_default']*100:.3f}%")
                    mc3.metric("Implied Asset Vol", f"{mrt['asset_vol']*100:.2f}%")
                except Exception as exc:
                    st.error(f"Merton failed: {exc}")

else:
    st.info("Enter a ticker and click **▶ Run GARCH** to begin.")

with st.expander("Model notes"):
    st.markdown("""
**Engle (2003)** — GJR-GARCH(1,1) extends GARCH with a leverage term $\\gamma$:
$\\sigma^2_t = \\omega + \\alpha \\varepsilon^2_{t-1} + \\gamma \\varepsilon^2_{t-1} \\mathbf{1}[\\varepsilon_{t-1}<0] + \\beta \\sigma^2_{t-1}$

Persistence $P = \\alpha + \\beta + \\gamma/2 > 0.98$ signals volatility clustering (Minsky precondition 1).

**Merton (1997)** — Treats equity as a call option on firm assets:
$D2D = (\\ln(V/F) + (r - \\sigma_V^2/2)T) / (\\sigma_V \\sqrt{T})$

$D2D < 1.5\\sigma$ = distress zone.
    """)
