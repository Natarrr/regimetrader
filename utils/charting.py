"""utils/charting.py
Bloomberg-terminal chart utilities for the regime_trader dashboard.

apply_pro_theme   — overlay dark layout on any Plotly figure (call after build).
macro_heatmap_fig — RdYlGn divergent Z-score heatmap (US / EU / Asia x indicators).
dcf_waterfall_fig — Classic DCF -> macro adjustment -> ML-DCF waterfall.
"""
from __future__ import annotations

from typing import Any, Dict, List

import plotly.graph_objects as go

# ── Bloomberg terminal palette ─────────────────────────────────────────────────
_GREEN  = "#00FFA3"   # buy / positive / expanding
_RED    = "#FF3366"   # sell / negative / contracting
_BLUE   = "#00BFFF"   # totals / neutral accent
_GRID   = "#2A2A2A"   # subtle grid lines
_TEXT   = "#E0E0E0"   # primary readable text
_DIM    = "#AAAAAA"   # axis labels, secondary text
_FONT   = "Courier New, JetBrains Mono, monospace"


def apply_pro_theme(
    fig: go.Figure,
    *,
    title: str | None = None,
    height: int | None = None,
    x_title: str | None = None,
    y_title: str | None = None,
    hovermode: str = "x unified",
) -> go.Figure:
    """Overlay Bloomberg terminal dark theme on any Plotly figure.

    Uses update_layout + update_xaxes/update_yaxes so existing axis properties
    (tickformat, range, secondary axes, etc.) are preserved.
    """
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=40 if title else 20, b=20),
        font=dict(color=_DIM, family=_FONT, size=9),
        hovermode=hovermode,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=8),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    if title:
        fig.update_layout(
            title=dict(text=title, font=dict(size=10, color="#CCCCCC"), x=0.01)
        )
    if height:
        fig.update_layout(height=height)
    # update_xaxes / update_yaxes merge rather than replace, preserving tickformat etc.
    fig.update_xaxes(showgrid=True, gridcolor=_GRID, zeroline=False, color=_DIM)
    fig.update_yaxes(showgrid=True, gridcolor=_GRID, zeroline=False, color=_DIM)
    if x_title:
        fig.update_xaxes(title=x_title)
    if y_title:
        fig.update_yaxes(title=y_title)
    return fig


def macro_heatmap_fig(gm_data: Dict[str, Any]) -> go.Figure:
    """Divergent RdYlGn heatmap of macro Z-composite scores.

    Rows = zones (US / EU / Asia), columns = indicators.
    Colour scale centred at 0: green = expanding, red = contracting.
    """
    zones = [z for z in ("US", "EU", "Asia") if z in gm_data]
    all_inds: List[str] = []
    for z in zones:
        for ind in gm_data.get(z, {}):
            if ind not in all_inds:
                all_inds.append(ind)

    z_matrix: List[List[float]] = []
    text_matrix: List[List[str]] = []

    for z in zones:
        row_z, row_t = [], []
        for ind in all_inds:
            d = gm_data.get(z, {}).get(ind)
            if d and d.get("latest") is not None:
                zc  = float(d.get("z_composite", 0.0))
                lv  = d.get("latest", 0.0)
                tr  = d.get("trend", "neutral")
                arr = "▲" if tr == "expanding" else "▼" if tr == "contracting" else "→"
                row_z.append(zc)
                row_t.append(f"{ind}<br>{arr} {lv:.1f}<br>Z={zc:+.2f}")
            else:
                row_z.append(0.0)
                row_t.append(f"{ind}<br>n/a")
        z_matrix.append(row_z)
        text_matrix.append(row_t)

    fig = go.Figure(go.Heatmap(
        z=z_matrix,
        x=all_inds,
        y=zones,
        text=text_matrix,
        hovertemplate="%{text}<extra></extra>",
        colorscale="RdYlGn",
        zmid=0,
        zmin=-2,
        zmax=2,
        xgap=4,
        ygap=4,
        showscale=True,
        colorbar=dict(
            title=dict(text="Z", font=dict(color=_DIM, size=8)),
            tickfont=dict(color=_DIM, size=8),
            len=0.85,
            thickness=10,
            bgcolor="rgba(0,0,0,0)",
            outlinewidth=0,
        ),
    ))
    apply_pro_theme(fig, title="Global Macro Z-Score Heatmap", height=180, hovermode="closest")
    fig.update_xaxes(tickfont=dict(size=8, color=_DIM), side="bottom", showgrid=False)
    fig.update_yaxes(tickfont=dict(size=9, color=_TEXT), showgrid=False)
    return fig


def dcf_waterfall_fig(results: List[Dict[str, Any]]) -> go.Figure:
    """Visualise how Ridge macro features shift intrinsic value.

    Single ticker  → go.Waterfall (Classic -> Macro Adj -> ML-DCF).
    Multiple tickers → grouped delta bar chart per ticker.
    """
    valid = [r for r in results if not r.get("error") and r.get("classic_fv", 0) > 0]
    if not valid:
        return go.Figure()

    if len(valid) == 1:
        r       = valid[0]
        classic = r["classic_fv"]
        delta   = r["ml_fv"] - classic
        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "total"],
            x=["Classic DCF", "Macro Adj (Ridge)", "ML-DCF"],
            y=[classic, delta, 0],
            text=[
                f"${classic:.1f}",
                f"{'+'if delta >= 0 else ''}{delta:.1f}",
                f"${r['ml_fv']:.1f}",
            ],
            textposition="outside",
            textfont=dict(size=9, color=_TEXT),
            connector=dict(line=dict(color=_GRID, width=1, dash="dot")),
            increasing_marker_color=_GREEN,
            decreasing_marker_color=_RED,
            totals_marker_color=_BLUE,
        ))
        title = f"{r['ticker']} — Classic DCF vs ML-DCF"
    else:
        tickers = [r["ticker"] for r in valid]
        deltas  = [r["ml_fv"] - r["classic_fv"] for r in valid]
        clrs    = [_GREEN if d >= 0 else _RED for d in deltas]
        txts    = [f"{'+'if d >= 0 else ''}{d:.1f}" for d in deltas]
        fig = go.Figure(go.Bar(
            x=tickers,
            y=deltas,
            marker_color=clrs,
            text=txts,
            textposition="outside",
            textfont=dict(size=9, color=_TEXT),
        ))
        fig.add_hline(y=0, line=dict(color=_GRID, width=1))
        title = "ML-DCF Macro Adjustment vs Classic DCF ($)"

    return apply_pro_theme(fig, title=title, height=260, y_title="Δ Fair Value ($)")
