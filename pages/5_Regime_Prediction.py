"""pages/5_Regime_Prediction.py
Lucas (1995) · Sargent (2011) + Minsky — Composite Laureate Regime & Crisis Alert.

Combines HMM regime, monetary pulse, and volatility into a 4-state laureate label.
Fires Minsky Moment alert when all 3 preconditions breach simultaneously.

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

from backend.quant_models.prediction_controller import (
    classify_regime,
    combined_position_scale,
    minsky_moment,
)
from backend.quant_models.valuation_radar import fetch_shiller_cape_series, cape_percentile
from backend.quant_models.volatility_brain import fit_gjr_garch, volatility_regime
from backend.quant_models.monetary_pulse import (
    yield_spread, is_inverted, monetary_regime, m2_velocity_trend,
)
# TODO: MarketData import removed — backend.data.market_service does not exist; wire up correct service
MarketData = None  # placeholder until re-wired
from backend.data.fred_service import fetch_10y_yield, fetch_2y_yield, fetch_m2_velocity

_HMM_ERR: str = ""
try:
    from analysis.feature_engineer import FeatureEngineer
    from hmm_engine.classifier import RegimeClassifier
    _HMM_OK = True
except Exception as _e:
    _HMM_OK = False
    _HMM_ERR = f"{type(_e).__name__}: {_e}"

_CHART_BG = "#050505"
_CARD_BG  = "#121212"
_GREEN    = "#00FFA3"
_RED      = "#FF6B6B"
_GOLD     = "#FFD700"
_PURPLE   = "#A78BFA"
_BLUE     = "#60A5FA"
_GRID     = "#1E1E1E"

_REGIME_COLORS = {
    "BULL":       _GREEN,
    "OVERHEATED": _GOLD,
    "FRAGILE":    _PURPLE,
    "CRASH":      _RED,
}
_REGIME_ICONS = {
    "BULL":       "🟢",
    "OVERHEATED": "🟡",
    "FRAGILE":    "🟣",
    "CRASH":      "🔴",
}
_MINSKY_COLORS = {
    "CRITICAL": _RED,
    "WARNING":  _GOLD,
    "WATCH":    _PURPLE,
    "CLEAR":    _GREEN,
}
_LAUREATE_SCALE_HINT = {
    "BULL": "1.00", "OVERHEATED": "0.70", "FRAGILE": "0.40", "CRASH": "0.00",
}
_TICKER_MAP: dict[str, str] = {
    "SPY":       "SPY",
    "Nifty 500": "^CRSLDX",
    "VIX":       "^VIX",
}


@st.cache_data(ttl=3600, show_spinner=False)
def _load_macro() -> dict:
    """Friedman (1968 Nobel) + Kuznets (1971 Nobel) — Macro regime inputs.

    Yield spread: $S_t = r_{10Y,t} - r_{2Y,t}$ (bps).
    """
    gs10   = fetch_10y_yield()
    gs2    = fetch_2y_yield()
    m2v    = fetch_m2_velocity()
    spread = yield_spread(gs10, gs2)
    return dict(
        spread=spread,
        spread_latest=float(spread.iloc[-1]),
        inverted=is_inverted(spread),
        mon_regime=monetary_regime(spread, m2v),
        m2v_trend=m2_velocity_trend(m2v),
    )


@st.cache_data(ttl=3600, show_spinner=False)
def _load_cape() -> dict:
    """Shiller (2013 Nobel) — CAPE percentile for Minsky precondition 2."""
    series       = fetch_shiller_cape_series()
    current_cape = float(series.iloc[-1]) if not series.empty else 30.0
    pct          = cape_percentile(series, current_cape) if not series.empty else 75.0
    return dict(cape=current_cape, pct=pct)


@st.cache_data(ttl=1800, show_spinner=False)
def _load_garch(sym: str) -> dict:
    """Engle (2003 Nobel) — GJR-GARCH persistence for Minsky precondition 1.

    Persistence: $P = \\alpha + \\beta + \\gamma/2 > 0.98$ triggers CLUSTERING.
    """
    md      = MarketData()
    bars    = md.get_historical_bars(symbol=sym, years_back=5)
    log_ret = np.log(bars["Close"] / bars["Close"].shift(1)).dropna().values
    result  = fit_gjr_garch(log_ret)
    result["vol_regime"] = volatility_regime(result["persistence"])
    return result


def _run_hmm(sym: str) -> dict:
    """Lucas (1995 Nobel) + Sargent (2011 Nobel) — HMM causal forward-filter regime.

    Rational expectations: agents price all available information into current prices.
    Regime transitions: $q_t \\sim \\text{HMM}(\\pi, A, B)$ (forward algorithm — no look-ahead).
    """
    if not _HMM_OK:
        return {"hmm_label": "Unknown", "position_scale": 1.0, "is_uncertain": False,
                "regime_probs": [], "label_sequence": [], "color_map": {}}

    bars             = MarketData().get_historical_bars(symbol=sym, years_back=3)
    features, returns, _ = FeatureEngineer().build(bars)
    clf              = RegimeClassifier()
    clf.fit(features, returns)
    state            = clf.predict_current(features[-20:])
    seq              = clf.predict_sequence(features)

    confirmed_col  = seq["confirmed_label"].tolist()
    raw_col        = seq["raw_label"].tolist()
    label_sequence = [
        str(c) if c is not None else (str(r) if r is not None else "Unknown")
        for c, r in zip(confirmed_col, raw_col)
    ]
    return {
        "hmm_label":      state.confirmed_label or state.raw_label or "Unknown",
        "position_scale": float(state.position_scale),
        "is_uncertain":   bool(state.is_uncertain),
        "regime_probs":   ([float(p) for p in state.regime_probs]
                           if state.regime_probs is not None else []),
        "label_sequence": label_sequence,
        "color_map":      dict(state.color_map),
    }


def render() -> None:
    """Render the Regime Prediction page.

    Lucas (1995 Nobel) + Sargent (2011 Nobel) — rational expectations and
    policy regime changes drive asset allocation decisions.
    """
    st.title("🎯 Regime Prediction")
    st.caption(
        "Lucas (1995 Nobel) · Sargent (2011 Nobel) · Minsky Financial Instability Hypothesis"
    )

    st.markdown("**Quick select**")
    qs_cols = st.columns(len(_TICKER_MAP) + 1)
    for idx, (label, _) in enumerate(_TICKER_MAP.items()):
        if qs_cols[idx].button(label, key=f"_rp_qs_{label}"):
            st.session_state["_rp_qs_symbol"] = label

    col_sym, col_btn = st.columns([4, 1])
    _default = _TICKER_MAP.get(st.session_state.get("_rp_qs_symbol", "SPY"), "SPY")
    _raw     = col_sym.text_input(
        "Ticker Symbol (or pick above)",
        value=_default,
        placeholder="SPY, QQQ, ^VIX, ^CRSLDX…",
        key="_rp_ticker",
    )
    symbol       = _TICKER_MAP.get(_raw.strip(), _raw.strip().upper())
    _display_name = next((k for k, v in _TICKER_MAP.items() if v == symbol), symbol)
    run          = col_btn.button("▶ Analyze", type="primary", key="_rp_run")

    if symbol in ("^CRSLDX", "^VIX"):
        st.caption(
            f"ℹ️ **{_display_name}** selected — CAPE and yield-curve signals remain "
            "US-based (Shiller CAPE + FRED); GARCH and HMM run on the selected ticker."
        )

    if not _HMM_OK:
        st.warning(f"⚠️ HMM engine not importable — regime will show 'Unknown'. "
                   f"Minsky alert still runs on macro + valuation data.  [{_HMM_ERR}]")

    if run or "_rp_regime_result" in st.session_state:
        if run:
            col_p1, col_p2, col_p3 = st.columns(3)
            col_p1.info("📡 Loading macro data…")
            col_p2.info("📈 Fitting GARCH…")
            col_p3.info("🤖 Running HMM…")

            with st.spinner("Fetching all data sources…"):
                try:
                    macro = _load_macro()
                except Exception as exc:
                    macro = {"spread_latest": 0.0, "inverted": False,
                             "mon_regime": "NEUTRAL", "m2v_trend": "STABLE", "spread": None}
                    st.warning(f"Macro data partial: {exc}")

                try:
                    cape_d = _load_cape()
                except Exception as exc:
                    cape_d = {"cape": 30.0, "pct": 75.0}
                    st.warning(f"CAPE partial: {exc}")

                try:
                    garch = _load_garch(symbol)
                except Exception as exc:
                    garch = {"persistence": 0.95, "latest_conditional_vol_ann": 0.15,
                             "vol_regime": "STABLE"}
                    st.warning(f"GARCH partial: {exc}")

                try:
                    hmm = _run_hmm(symbol)
                except Exception as exc:
                    hmm = {"hmm_label": "Unknown", "position_scale": 1.0,
                           "is_uncertain": False, "regime_probs": None, "label_sequence": []}
                    st.warning(f"HMM partial: {exc}")

            laureate    = classify_regime(hmm["hmm_label"], macro["mon_regime"],
                                          garch["vol_regime"])
            minsky      = minsky_moment(
                garch_persistence=garch["persistence"],
                cape_percentile=cape_d["pct"],
                yield_spread_bps=macro["spread_latest"],
            )
            final_scale = combined_position_scale(hmm.get("position_scale", 1.0), laureate)

            st.session_state["_rp_regime_result"] = dict(
                symbol=symbol, display_name=_display_name,
                macro=macro, cape_d=cape_d, garch=garch, hmm=hmm,
                laureate=laureate, minsky=minsky, final_scale=final_scale,
            )
            col_p1.empty()
            col_p2.empty()
            col_p3.empty()

        r           = st.session_state["_rp_regime_result"]
        macro       = r["macro"]
        cape_d      = r["cape_d"]
        garch       = r["garch"]
        hmm         = r["hmm"]
        laureate    = r["laureate"]
        minsky      = r["minsky"]
        sym         = r["symbol"]
        _disp       = r.get("display_name", sym)
        final_scale = r.get("final_scale",
                             combined_position_scale(hmm.get("position_scale", 1.0), laureate))

        regime_color = _REGIME_COLORS.get(laureate, _BLUE)
        regime_icon  = _REGIME_ICONS.get(laureate, "⚪")
        st.markdown(
            f"""<div style="background:{regime_color}22; border:2px solid {regime_color};
            border-radius:8px; padding:16px 24px; margin:12px 0;">
            <span style="font-size:2em; color:{regime_color};">{regime_icon} {laureate}</span>
            <span style="color:#E0E0E0; margin-left:16px; font-size:1.1em;">
            Laureate Regime · {_disp}</span>
            <span style="color:#888; float:right; font-size:0.9em;">
            Position scale: {final_scale:.2f}
            {'· ⚠️ Uncertain' if hmm.get('is_uncertain') else ''}</span>
            </div>""",
            unsafe_allow_html=True,
        )

        minsky_color = _MINSKY_COLORS.get(minsky.alert_level, _GREEN)
        minsky_icons = {"CRITICAL": "🚨", "WARNING": "⚠️", "WATCH": "👁️", "CLEAR": "✅"}
        minsky_icon  = minsky_icons.get(minsky.alert_level, "✅")
        st.markdown(
            f"""<div style="background:{minsky_color}22; border:1px solid {minsky_color};
            border-radius:8px; padding:12px 24px; margin:6px 0;">
            <span style="font-size:1.3em; color:{minsky_color};">
            {minsky_icon} Minsky Alert: <strong>{minsky.alert_level}</strong>
            ({minsky.conditions_met}/3 conditions)</span><br>
            <span style="color:#B0B0B0; font-size:0.9em;">{minsky.narrative}</span>
            </div>""",
            unsafe_allow_html=True,
        )

        st.divider()

        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("HMM Label", hmm.get("hmm_label", "Unknown"),
                  delta=f"HMM scale {hmm.get('position_scale', 1.0):.2f}",
                  delta_color="off")
        c2.metric("Monetary", macro["mon_regime"],
                  delta="⚠️ INVERTED" if macro["inverted"] else macro["m2v_trend"],
                  delta_color="inverse" if macro["inverted"] else "off")
        c3.metric("Volatility", garch["vol_regime"],
                  delta=f"P={garch['persistence']:.4f}",
                  delta_color="inverse" if garch["vol_regime"] == "CLUSTERING" else "normal")
        c4.metric("CAPE Percentile", f"{cape_d['pct']:.1f}th",
                  delta_color="inverse" if cape_d["pct"] > 95 else "normal")
        c5.metric("Yield Spread", f"{macro['spread_latest']:.1f} bps",
                  delta_color="inverse" if macro["inverted"] else "normal")
        c6.metric("Policy Scale", f"{final_scale:.2f}",
                  delta=f"laureate ×{_LAUREATE_SCALE_HINT.get(laureate, '?')}",
                  delta_color="inverse" if final_scale < 0.4 else "normal")

        st.divider()

        col_m, col_regime_hist = st.columns([1, 2])

        with col_m:
            st.subheader("Minsky Conditions")
            cond_data = [
                ("GARCH Persistence",  minsky.garch_persistence,  0.98,
                 "≥ 0.98",  minsky.garch_persistence >= 0.98),
                ("CAPE Percentile",    minsky.cape_percentile,    95.0,
                 "≥ 95th",  minsky.cape_percentile >= 95.0),
                ("Yield Spread (bps)", minsky.yield_spread_bps,   0.0,
                 "< 0 (inverted)", minsky.yield_spread_bps < 0.0),
            ]
            for label, val, _threshold, desc, triggered in cond_data:
                icon = "🔴" if triggered else "🟢"
                st.markdown(
                    f"""<div style="background:{'#3A1A1A' if triggered else '#1A3A2A'};
                    border-radius:6px; padding:10px 14px; margin:6px 0;">
                    <strong style="color:{'#FF6B6B' if triggered else '#00FFA3'};">
                    {icon} {label}</strong><br>
                    <span style="color:#E0E0E0;">Value: {val:.2f} &nbsp; Trigger: {desc}</span>
                    </div>""",
                    unsafe_allow_html=True,
                )

            fig_conds = go.Figure(go.Indicator(
                mode="gauge+number",
                value=minsky.conditions_met,
                number={"suffix": "/3", "font": {"color": minsky_color}},
                title={"text": "Conditions Met", "font": {"color": "#E0E0E0", "size": 13}},
                gauge={
                    "axis": {"range": [0, 3], "tickvals": [0, 1, 2, 3],
                             "tickcolor": "#E0E0E0"},
                    "bar": {"color": minsky_color},
                    "bgcolor": _CARD_BG,
                    "steps": [
                        {"range": [0, 1], "color": "#1A3A2A"},
                        {"range": [1, 2], "color": "#3A3A1A"},
                        {"range": [2, 3], "color": "#3A1A1A"},
                    ],
                },
            ))
            fig_conds.update_layout(
                paper_bgcolor=_CHART_BG, font=dict(color="#E0E0E0"),
                height=200, margin=dict(t=20, b=10, l=10, r=10),
            )
            st.plotly_chart(fig_conds, use_container_width=True)

        with col_regime_hist:
            label_seq = hmm.get("label_sequence", [])
            if label_seq:
                st.subheader(f"{_disp} — HMM Regime History")
                _LABEL_COLORS = {
                    "Bull": _GREEN, "Euphoria": _GREEN, "Mania": _GREEN,
                    "Neutral": _BLUE, "Unknown": "#888",
                    "Bear": _RED, "Panic": _RED, "Crash": _RED,
                }
                unique_labels = sorted(set(label_seq))
                fig_hist = go.Figure()
                for lbl in unique_labels:
                    y_vals = [i if label_seq[i] == lbl else None
                              for i in range(len(label_seq))]
                    fig_hist.add_trace(go.Scatter(
                        x=list(range(len(label_seq))),
                        y=y_vals,
                        mode="markers",
                        name=lbl,
                        marker=dict(color=_LABEL_COLORS.get(lbl, _PURPLE),
                                    size=4, symbol="square"),
                    ))
                fig_hist.update_layout(
                    template="plotly_dark",
                    paper_bgcolor=_CHART_BG, plot_bgcolor=_CARD_BG,
                    yaxis=dict(tickvals=list(range(len(set(label_seq)))),
                               gridcolor=_GRID, showticklabels=False),
                    xaxis=dict(title="Trading days", gridcolor=_GRID),
                    height=320,
                    legend=dict(bgcolor=_CARD_BG, orientation="h"),
                    margin=dict(t=10, b=40),
                )
                st.plotly_chart(fig_hist, use_container_width=True)

                probs = hmm.get("regime_probs")
                if probs is not None and hasattr(probs, "__len__") and len(probs) > 0:
                    fig_probs = go.Figure(go.Bar(
                        x=[f"State {i}" for i in range(len(probs))],
                        y=list(probs),
                        marker_color=[_GREEN, _GOLD, _RED, _PURPLE, _BLUE][:len(probs)],
                        text=[f"{p:.1%}" for p in probs],
                        textposition="outside",
                    ))
                    fig_probs.update_layout(
                        title="Current Regime Probabilities (Forward Filter)",
                        template="plotly_dark",
                        paper_bgcolor=_CHART_BG, plot_bgcolor=_CARD_BG,
                        yaxis=dict(title="Probability", gridcolor=_GRID, range=[0, 1]),
                        height=240,
                        margin=dict(t=40, b=20),
                    )
                    st.plotly_chart(fig_probs, use_container_width=True)
            else:
                st.info("HMM regime history will appear here after analysis.")

    else:
        st.info("Enter a ticker and click **▶ Analyze** to compute the composite regime "
                "and Minsky alert.")
        st.markdown("""
**What this page computes:**
1. **HMM Regime** — fits a Hidden Markov Model on 3 years of daily bars and returns the current confirmed state (Bull / Neutral / Bear)
2. **Monetary Regime** — from live FRED yield curve & M2 velocity data
3. **Volatility Regime** — GJR-GARCH persistence on 5 years of daily returns
4. **Laureate Label** — combines the three into: `BULL | OVERHEATED | FRAGILE | CRASH`
5. **Minsky Alert** — fires when all 3 preconditions breach simultaneously
        """)

    with st.expander("Model notes"):
        st.markdown(r"""
**Lucas (1995)** — Rational expectations: regime transitions occur when the
consensus forecast of fundamentals shifts. All three signals are processed simultaneously.

**Sargent (2011)** — Policy regime changes: the four-state label maps to position sizing.
`BULL` → full allocation · `FRAGILE` → reduce · `CRASH` → de-risk.

**Minsky** — Financial Instability Hypothesis: stability is destabilising.
The alert fires when GARCH persistence > 0.98 AND CAPE percentile > 95 AND yield curve inverts.
All three must breach simultaneously for a CRITICAL alert.

HMM uses the **causal forward algorithm** (no Viterbi) — backtests are free of look-ahead bias.
        """)
