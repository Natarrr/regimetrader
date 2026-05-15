"""regime_trader/ui/streamlit_app.py
Streamlit orchestrator for the Regime Trader dashboard.

Pages (sidebar navigation):
  Dashboard       — regime / portfolio + VIX sparkline + market/macro intel
  Monetary Pulse  — Friedman/Kuznets/Prescott yield-curve & M2 velocity
  Volatility Brain— Engle GJR-GARCH + Merton Distance-to-Default
  Valuation Radar — Shiller CAPE + Excess CAPE Yield
  Contagion Web   — Leontief I-O shock propagation
  Regime Prediction — Lucas/Sargent HMM composite regime + Minsky alert

Dashboard tabs:
  Live Monitor    — regime / portfolio + VIX sparkline
  Market Intel    — Smart Money discovery picks + explainability panel
  Macro Intel     — Commodity conviction + macro shocks + partial-data badge
  Trade Log       — stub
  Regime History  — stub
  Portfolio Sync  — stub

Run:
  streamlit run regime_trader/ui/streamlit_app.py
"""
from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv

from regime_trader.utils.logging_cfg import configure_logging

# ── Bootstrap ─────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_ROOT / ".env", override=False)
configure_logging()
log = logging.getLogger(__name__)

# Pages directory (project root / pages/)
_PAGES_DIR = _ROOT / "pages"

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Regime Trader",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

_FMP_KEY = os.getenv("FMP_API_KEY", "")
_HAS_FMP = bool(_FMP_KEY)

_ALPACA_KEY    = os.getenv("ALPACA_API_KEY", "")
_ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY", "")
_ALPACA_PAPER  = "paper" in os.getenv("ALPACA_BASE_URL", "paper").lower()
_HAS_ALPACA    = bool(_ALPACA_KEY and _ALPACA_SECRET)

_EDGAR_USER_AGENT = os.getenv("EDGAR_USER_AGENT", "regime-trader n.tardy@hotmail.fr")

# ── Optional import: generate_macro_synthesis (graceful fallback) ─────────────
try:
    from regime_trader.scanners.market_intel_macro import generate_macro_synthesis as _generate_macro_synthesis
except ImportError:
    def _generate_macro_synthesis(  # type: ignore[misc]
        prices: Dict, convictions: Dict, indicators: Dict
    ) -> List[str]:
        del prices, convictions, indicators
        return ["Macro synthesis unavailable — market_intel_macro not installed."]


# ── Page module loader ────────────────────────────────────────────────────────

_PAGE_MODULE_MTIMES: dict[str, float] = {}


def _load_page_module(name: str, filename: str):
    """Load a page module; re-execute if the source file has changed since last load.

    Caches the module in sys.modules so @st.cache_data survives across Streamlit
    re-runs, but invalidates the cache automatically on file modification so that
    code changes take effect without a full process restart.

    Returns the module or None if the file is missing / imports fail.
    """
    key  = f"_rt_page_{name}"
    path = _PAGES_DIR / filename
    if not path.exists():
        log.warning("Page file not found: %s", path)
        return None

    current_mtime = path.stat().st_mtime
    if key in sys.modules and _PAGE_MODULE_MTIMES.get(key) == current_mtime:
        return sys.modules[key]

    if key in sys.modules:
        log.info("Page %s modified — reloading", filename)

    try:
        spec = importlib.util.spec_from_file_location(key, path)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
        _PAGE_MODULE_MTIMES[key] = current_mtime
        return mod
    except Exception as exc:
        log.warning("Failed to load page %s: %s", filename, exc)
        sys.modules.pop(key, None)
        _PAGE_MODULE_MTIMES.pop(key, None)
        return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_insider_pct(val: Any) -> str:
    """Format insider_value_pct_mcap robustly for display.

    The field may arrive as a percentage (e.g. 0.05 = 0.05 % of mktcap) from
    the current scanner, or as a raw fraction (e.g. 0.0005) from older cached
    payloads. Heuristic: values ≤ 1.0 and > 0 are treated as fractions and
    multiplied by 100; values > 1.0 are already percentages.

    Args:
        val: Raw value from a ScanResult dict.

    Returns:
        Human-readable percentage string, e.g. "0.0500%" or "—".
    """
    if val is None:
        return "—"
    try:
        fval = float(val)
    except (TypeError, ValueError):
        return "—"
    pct = fval * 100.0 if 0 < abs(fval) <= 1.0 else fval
    return f"{pct:.4f}%"


def _safe_payload() -> Dict[str, Any]:
    """Return a consistent empty payload shape used on load errors."""
    return {"results": [], "cached": False, "computed_at": "error"}


# ── Cached data loaders ───────────────────────────────────────────────────────

_STATE_FILE = _ROOT / "data" / "market_state.json"


@st.cache_data(ttl=60, show_spinner=False)
def _load_market_state() -> Optional[Dict[str, Any]]:
    """Read engine-produced market_state.json with a 60 s TTL.

    Returns the parsed dict (keys: last_updated, macro_status, alpha_picks)
    or None when the file is absent or corrupted — caller shows Engine Offline.
    """
    try:
        return json.loads(_STATE_FILE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception as exc:
        log.warning("market_state.json read failed: %s", exc)
        return None


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def _load_discovery(limit: int = 5) -> Dict[str, Any]:
    """Return discovery picks; uses module-level TTL cache internally.

    Returns _safe_payload() on any exception so the UI never crashes on
    a scanner failure.

    Args:
        limit: Number of top picks to request.

    Returns:
        Discovery payload dict with at minimum a 'results' list.
    """
    try:
        from regime_trader.scanners.discovery_scanner import get_top_alpha_picks_sync
        return get_top_alpha_picks_sync(limit=limit)
    except Exception as exc:
        log.warning("discovery load failed: %s", exc)
        return _safe_payload()


@st.cache_data(ttl=3600, show_spinner=False)
def _load_commodity_prices() -> Dict[str, Optional[Dict]]:
    """Fetch all commodity prices in a bounded thread-pool (30 s wall timeout).

    Returns:
        {ticker: price_dict | None} for each commodity in COMMODITY_UNIVERSE.
    """
    from regime_trader.scanners.market_intel_macro import COMMODITY_UNIVERSE, fetch_commodity_prices

    prices: Dict[str, Optional[Dict]] = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {
            pool.submit(fetch_commodity_prices, c): c["ticker"]
            for c in COMMODITY_UNIVERSE
        }
        try:
            for fut in as_completed(futures, timeout=30):
                ticker = futures[fut]
                try:
                    prices[ticker] = fut.result()
                except Exception as exc:
                    log.warning("commodity fetch failed %s: %s", ticker, exc)
                    prices[ticker] = None
        except FutureTimeoutError:
            log.warning("commodity fetch timed out — marking remaining tickers as unavailable")
            for ticker in futures.values():
                if ticker not in prices:
                    prices[ticker] = None
    return prices


@st.cache_data(ttl=3600, show_spinner=False)
def _load_macro_indicators() -> Dict[str, Optional[Dict]]:
    """Fetch macro indicators in a bounded thread-pool (20 s wall timeout).

    Returns:
        {ticker: indicator_dict | None} for each indicator in MACRO_INDICATORS.
    """
    from regime_trader.scanners.market_intel_macro import MACRO_INDICATORS, fetch_macro_indicator

    indicators: Dict[str, Optional[Dict]] = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(fetch_macro_indicator, ind["ticker"]): ind["ticker"]
            for ind in MACRO_INDICATORS
        }
        try:
            for fut in as_completed(futures, timeout=20):
                ticker = futures[fut]
                try:
                    indicators[ticker] = fut.result()
                except Exception as exc:
                    log.warning("macro indicator fetch failed %s: %s", ticker, exc)
                    indicators[ticker] = None
        except FutureTimeoutError:
            log.warning("macro indicator fetch timed out")
            for ticker in futures.values():
                if ticker not in indicators:
                    indicators[ticker] = None
    return indicators


@st.cache_data(ttl=60, show_spinner=False)
def _load_alpaca_account() -> Dict[str, Any]:
    """Fetch Alpaca account + positions. TTL 60 s for near-real-time data."""
    try:
        from alpaca.trading.client import TradingClient
        client = TradingClient(_ALPACA_KEY, _ALPACA_SECRET, paper=_ALPACA_PAPER)
        acct = client.get_account()
        positions = client.get_all_positions()

        equity      = float(acct.equity)
        last_equity = float(acct.last_equity)
        daily_pnl   = equity - last_equity
        daily_pnl_pct = (daily_pnl / last_equity * 100) if last_equity else 0.0

        pos_rows = []
        amount_invested = 0.0
        for p in sorted(positions, key=lambda x: float(x.market_value or 0), reverse=True):
            mkt_val = float(p.market_value or 0)
            amount_invested += mkt_val
            pos_rows.append({
                "Symbol":    p.symbol,
                "Side":      p.side.value.capitalize(),
                "Qty":       float(p.qty),
                "Entry":     float(p.avg_entry_price),
                "Price":     float(p.current_price),
                "Mkt Value": mkt_val,
                "Unreal. P&L": float(p.unrealized_pl),
                "Unreal. %": float(p.unrealized_plpc) * 100,
                "Day P&L":   float(p.unrealized_intraday_pl),
                "Day %":     float(p.unrealized_intraday_plpc) * 100,
            })

        return {
            "equity":           equity,
            "buying_power":     float(acct.buying_power),
            "cash":             float(acct.cash),
            "amount_invested":  amount_invested,
            "portfolio_value":  float(acct.portfolio_value),
            "daily_pnl":        daily_pnl,
            "daily_pnl_pct":    daily_pnl_pct,
            "positions":        pos_rows,
            "status":           acct.status.value,
            "paper":            _ALPACA_PAPER,
        }
    except Exception as exc:
        log.warning("Alpaca account load failed: %s", exc)
        return {"error": str(exc)}


@st.cache_data(ttl=300, show_spinner=False)
def _load_regime() -> Dict[str, Any]:
    """Fetch latest VIX and derive market regime label. TTL 5 min."""
    try:
        import yfinance as yf
        from regime_trader.models.regime_detector import vix_rule
        df = yf.download("^VIX", period="2d", interval="1d",
                         progress=False, auto_adjust=True)
        if df.empty:
            return {"regime": "Unknown", "vix": None}
        vix = float(df["Close"].squeeze().dropna().iloc[-1])
        return {"regime": vix_rule(vix), "vix": vix}
    except Exception as exc:
        log.warning("Regime load failed: %s", exc)
        return {"regime": "Unknown", "vix": None}


@st.cache_data(ttl=300, show_spinner=False)
def _load_vix_history() -> Optional[List[float]]:
    """Fetch 10-day VIX history for the sparkline. Returns None on failure."""
    try:
        import yfinance as yf
        df = yf.download("^VIX", period="10d", interval="1d",
                         progress=False, auto_adjust=True)
        if df.empty:
            return None
        series = df["Close"].squeeze().dropna()
        return [float(v) for v in series.tolist()]
    except Exception as exc:
        log.debug("VIX history load failed: %s", exc)
        return None


_PERIOD_MAP = {"1W": ("1W", "1H"), "1M": ("1M", "1D"), "3M": ("3M", "1D"), "1Y": ("1A", "1D")}


@st.cache_data(ttl=300, show_spinner=False)
def _load_portfolio_history(period: str = "1M") -> Optional[Dict[str, Any]]:
    """Fetch Alpaca portfolio equity history for the given period.

    Args:
        period: One of '1W', '1M', '3M', '1Y'.

    Returns:
        Dict with keys timestamps, equity, profit_loss_pct, base_value — or None.
    """
    if not _HAS_ALPACA:
        return None
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetPortfolioHistoryRequest

        alpaca_period, timeframe = _PERIOD_MAP.get(period, ("1M", "1D"))
        client = TradingClient(_ALPACA_KEY, _ALPACA_SECRET, paper=_ALPACA_PAPER)
        req = GetPortfolioHistoryRequest(period=alpaca_period, timeframe=timeframe)
        h = client.get_portfolio_history(filter=req)

        # Filter out None equity entries (market closures)
        pairs = [
            (datetime.fromtimestamp(t, tz=timezone.utc), e)
            for t, e in zip(h.timestamp, h.equity)
            if e is not None
        ]
        if not pairs:
            return None

        timestamps, equity = zip(*pairs)
        return {
            "timestamps":      list(timestamps),
            "equity":          list(equity),
            "profit_loss_pct": [p for p, e in zip(h.profit_loss_pct or [], h.equity or []) if e is not None],
            "base_value":      h.base_value,
        }
    except Exception as exc:
        log.warning("Portfolio history load failed: %s", exc)
        return None


def _render_portfolio_chart(history: Dict[str, Any]) -> None:
    """Render a Revolut-style portfolio performance line chart using Plotly.

    Design principles (Revolut aesthetic):
      - Gradient area fill below the equity curve (green/red based on P&L direction)
      - Smooth spline line with no markers
      - Transparent plot background blending into Streamlit dark theme
      - Minimal axes: no x-gridlines, subtle y-gridlines on the right
      - Clean hover tooltip showing dollar value and date
    """
    import plotly.graph_objects as go

    timestamps = history["timestamps"]
    equity     = history["equity"]

    if len(equity) < 2:
        st.caption("Not enough data to render chart.")
        return

    base     = equity[0]
    current  = equity[-1]
    positive = current >= base

    line_color = "#00d26a" if positive else "#ff4d6d"
    fill_rgba  = "rgba(0,210,106,0.12)" if positive else "rgba(255,77,109,0.12)"

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=timestamps,
        y=equity,
        fill="tozeroy",
        fillcolor=fill_rgba,
        line=dict(color=line_color, width=2.5, shape="spline", smoothing=0.7),
        mode="lines",
        hovertemplate="<b>$%{y:,.2f}</b><br>%{x|%b %d, %Y}<extra></extra>",
    ))

    # Baseline reference — subtle dashed line at opening value
    fig.add_hline(
        y=base,
        line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dot"),
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=8, b=0),
        xaxis=dict(
            showgrid=False,
            showline=False,
            zeroline=False,
            tickfont=dict(color="#8b8b9e", size=11),
            tickformat="%b %d",
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(255,255,255,0.05)",
            gridwidth=1,
            showline=False,
            zeroline=False,
            tickfont=dict(color="#8b8b9e", size=11),
            tickformat="$,.0f",
            side="right",
        ),
        hovermode="x unified",
        showlegend=False,
        height=240,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


# ── Sidebar: navigation + settings ───────────────────────────────────────────

_NAV_PAGES = [
    ("📊 Dashboard",          None,                    None),
    ("💰 Monetary Pulse",     "1_Monetary_Pulse.py",   "monetary_pulse"),
    ("📈 Volatility Brain",   "2_Volatility_Brain.py", "volatility_brain"),
    ("🔭 Valuation Radar",    "3_Valuation_Radar.py",  "valuation_radar"),
    ("🕸️ Contagion Web",      "4_Contagion_Web.py",    "contagion_web"),
    ("🎯 Regime Prediction",  "5_Regime_Prediction.py","regime_prediction"),
]


def _render_sidebar() -> str:
    """Render sidebar navigation + settings. Returns selected page label."""
    with st.sidebar:
        st.markdown("## 🧭 Navigate")
        labels = [label for label, _, _ in _NAV_PAGES]
        selected = st.radio(
            "page",
            labels,
            label_visibility="collapsed",
            key="_sidebar_nav",
        )

        st.divider()
        st.markdown("## ⚙️ Settings")

        with st.expander("Cache controls", expanded=False):
            if st.button("Clear engine state cache", key="clear_disc"):
                _load_market_state.clear()
                _load_discovery.clear()
                st.success("Engine state + discovery cache cleared.")
            if st.button("Clear commodity cache", key="clear_comm"):
                _load_commodity_prices.clear()
                _load_macro_indicators.clear()
                st.success("Commodity / macro cache cleared.")
            if st.button("Clear account cache", key="clear_acct"):
                _load_alpaca_account.clear()
                _load_regime.clear()
                _load_vix_history.clear()
                _load_portfolio_history.clear()
                st.success("Account / regime cache cleared.")

        with st.expander("Environment", expanded=False):
            st.caption(f"FMP key: **{'set' if _HAS_FMP else 'missing'}**")
            st.caption(f"Alpaca key: **{'set' if _HAS_ALPACA else 'missing'}**")
            st.caption(f"Paper trading: **{_ALPACA_PAPER}**")
            st.caption(f"EDGAR User-Agent: `{_EDGAR_USER_AGENT}`")

    return selected


# ── Helper: API key warning banner ────────────────────────────────────────────

def _require_fmp() -> bool:
    """Show a warning banner when FMP_API_KEY is absent.

    Returns:
        True if the key is set, False otherwise.
    """
    if not _HAS_FMP:
        st.warning(
            "**FMP_API_KEY not set.** "
            "Add it to your `.env` file or environment to enable live data. "
            "Showing cached or demo data where available.",
            icon="⚠️",
        )
    return _HAS_FMP


# ── Dashboard tab renderers ───────────────────────────────────────────────────

_CSS = """
<style>
/* ── Metric cards — Revolut-style glass cards ── */
div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 18px 22px 14px;
}
/* Metric value */
div[data-testid="metric-container"] > label + div {
    font-size: 1.55rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em;
}
/* Metric label */
div[data-testid="metric-container"] > label {
    font-size: 0.72rem !important;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    opacity: 0.55;
}
/* Dataframe rounded corners */
div[data-testid="stDataFrame"] > div {
    border-radius: 12px;
    overflow: hidden;
}
/* Period selector pills — compact */
div[data-testid="stHorizontalBlock"] .stRadio > div {
    flex-direction: row;
    gap: 6px;
}
</style>
"""


def _render_live_monitor() -> None:
    """Render the Live Monitor tab — Revolut-style layout.

    Layout:
      Row 1  : Regime  |  Portfolio Value  |  Daily P&L   (primary KPIs)
      Section: Portfolio performance chart  (Revolut-style, period selectable)
      Row 2  : Invested  |  Cash  |  Open Positions        (balance breakdown)
      Section: VIX 10-day sparkline
      Section: Positions table
    """
    st.markdown(_CSS, unsafe_allow_html=True)

    hdr, btn = st.columns([8, 1])
    hdr.header("Live Monitor")
    if btn.button("↻", key="live_refresh", help="Refresh account data"):
        _load_alpaca_account.clear()
        _load_regime.clear()
        _load_vix_history.clear()
        _load_portfolio_history.clear()

    regime_data = _load_regime()
    regime      = regime_data.get("regime", "Unknown")
    vix         = regime_data.get("vix")

    _REGIME_COLOR = {
        "Crash": "🔴", "Panic": "🟠", "Bear": "🟡",
        "Neutral": "⚪", "Bull": "🟢", "Euphoria": "💹",
    }
    regime_icon = _REGIME_COLOR.get(regime, "❓")

    # ── No Alpaca credentials — show regime only ──────────────────────────────
    if not _HAS_ALPACA:
        st.warning(
            "**ALPACA_API_KEY / ALPACA_SECRET_KEY not set.** "
            "Add them to `.env` and restart the app.",
            icon="⚠️",
        )
        col1, col2, col3 = st.columns(3)
        col1.metric("Regime",          f"{regime_icon} {regime}",
                    f"VIX {vix:.1f}" if vix else None)
        col2.metric("Portfolio Value",  "—")
        col3.metric("Daily P&L",        "—")
        _render_vix_sparkline()
        return

    with st.spinner("Loading account…"):
        acct = _load_alpaca_account()

    if "error" in acct:
        st.error(f"Alpaca connection failed: {acct['error']}")
        return

    paper_tag = " · Paper" if acct["paper"] else ""
    pnl       = acct["daily_pnl"]
    pnl_pct   = acct["daily_pnl_pct"]

    # ── Row 1: Primary KPIs ───────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Regime",
        f"{regime_icon} {regime}",
        f"VIX {vix:.1f}" if vix else None,
    )
    col2.metric(
        "Portfolio Value",
        f"${acct['portfolio_value']:,.2f}",
        f"{pnl_pct:+.2f}% today",
    )
    col3.metric(
        "Daily P&L",
        f"${pnl:+,.2f}",
        f"{pnl_pct:+.2f}%",
    )

    st.caption(f"Alpaca · Status: **{acct['status']}**{paper_tag}")

    # ── Portfolio performance chart ───────────────────────────────────────────
    period_col, spacer = st.columns([3, 7])
    period = period_col.radio(
        "period",
        ["1W", "1M", "3M", "1Y"],
        index=1,
        horizontal=True,
        label_visibility="collapsed",
        key="portfolio_period",
    )

    with st.spinner("Loading chart…"):
        port_history = _load_portfolio_history(period)

    if port_history:
        _render_portfolio_chart(port_history)
    else:
        _render_vix_sparkline()

    # ── Row 2: Balance breakdown ──────────────────────────────────────────────
    col4, col5, col6 = st.columns(3)
    col4.metric("Invested",       f"${acct['amount_invested']:,.2f}")
    col5.metric("Cash",           f"${acct['cash']:,.2f}")
    col6.metric("Open Positions", len(acct["positions"]))

    # ── VIX sparkline (always shown below the portfolio chart) ────────────────
    _render_vix_sparkline()

    # ── Positions table ───────────────────────────────────────────────────────
    st.subheader(f"Positions ({len(acct['positions'])})")
    positions = acct["positions"]
    if not positions:
        st.info("No open positions.")
        return

    import pandas as pd

    df = pd.DataFrame(positions)
    df["Entry"]       = df["Entry"].map("${:,.2f}".format)
    df["Price"]       = df["Price"].map("${:,.2f}".format)
    df["Mkt Value"]   = df["Mkt Value"].map("${:,.2f}".format)
    df["Unreal. P&L"] = df["Unreal. P&L"].map("${:+,.2f}".format)
    df["Unreal. %"]   = df["Unreal. %"].map("{:+.2f}%".format)
    df["Day P&L"]     = df["Day P&L"].map("${:+,.2f}".format)
    df["Day %"]       = df["Day %"].map("{:+.2f}%".format)

    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_vix_sparkline() -> None:
    """Render a 10-day VIX sparkline using a Revolut-style Plotly chart. No-op on failure."""
    try:
        import plotly.graph_objects as go

        history = _load_vix_history()
        if not history or len(history) < 2:
            return

        # VIX above 20 = elevated risk; colour shifts amber → red
        latest_vix = history[-1]
        line_color = "#ff4d6d" if latest_vix >= 25 else "#f5a623" if latest_vix >= 18 else "#00d26a"
        fill_rgba  = f"rgba(245,166,35,0.10)" if latest_vix >= 18 else "rgba(0,210,106,0.10)"

        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=history,
            fill="tozeroy",
            fillcolor=fill_rgba,
            line=dict(color=line_color, width=2, shape="spline", smoothing=0.6),
            mode="lines",
            hovertemplate="VIX <b>%{y:.1f}</b><extra></extra>",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=4, b=0),
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(
                showgrid=True,
                gridcolor="rgba(255,255,255,0.04)",
                showticklabels=True,
                tickfont=dict(color="#8b8b9e", size=10),
                zeroline=False,
                side="right",
            ),
            showlegend=False,
            height=90,
        )
        st.caption(f"VIX — 10-day history  ·  latest **{latest_vix:.1f}**")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    except Exception as exc:
        log.debug("VIX sparkline render failed: %s", exc)


def _render_system_health() -> None:
    """Render the System Health expander with the last 10 lines of main.log."""
    log_file = _ROOT / "logs" / "main.log"
    with st.expander("🩺 System Health", expanded=False):
        st.caption(f"Log: `{log_file}`")
        try:
            lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
            tail = "\n".join(lines[-10:]) if lines else "(empty)"
            st.code(tail, language="text")
        except FileNotFoundError:
            st.info("Log file not found — engine has not run yet.")
        except Exception as exc:
            st.warning(f"Could not read log: {exc}")


def _render_market_intel() -> None:
    """Render the Market Intel tab — dumb UI reading from engine-produced JSON.

    This function makes zero API calls. All computation lives in
    backend/engine_worker.py which writes data/market_state.json.
    TTL is 60 s so new engine runs appear within one minute.
    """
    import pandas as pd

    st.header("Market Intel — Smart Money Discovery")

    # ── Refresh control ───────────────────────────────────────────────────────
    col_ref, col_ts = st.columns([1, 5])
    if col_ref.button("🔄 Refresh cache", key="mi_refresh"):
        _load_market_state.clear()
        st.rerun()

    # ── Load state ────────────────────────────────────────────────────────────
    state = _load_market_state()

    if state is None:
        st.error(
            "**⚠️ Engine Offline** — `data/market_state.json` not found.\n\n"
            "Run the engine worker to generate fresh data:\n"
            "```\npython -m backend.engine_worker\n```"
        )
        _render_system_health()
        return

    last_updated = state.get("last_updated", "—")
    col_ts.caption(f"Engine snapshot · {last_updated}")

    # ── Macro status banner ───────────────────────────────────────────────────
    macro = state.get("macro_status", {})
    regime          = macro.get("regime", "Unknown")
    conviction      = macro.get("conviction", 0.0)
    kill_switch     = macro.get("kill_switch_active", False)
    vix_latest      = macro.get("vix_latest", 0.0)

    _REGIME_COLOR = {
        "Bull":    "#00FFA3",
        "Neutral": "#60A5FA",
        "Bear":    "#FFB347",
        "Panic":   "#FF6B6B",
        "Crash":   "#FF2222",
    }
    rc = _REGIME_COLOR.get(regime, "#9E9E9E")

    if kill_switch:
        st.markdown(
            f"""<div style="background:#3A0000;border:2px solid #FF2222;
            border-radius:8px;padding:14px 20px;margin:8px 0;">
            <span style="font-size:1.4em;color:#FF2222;">⛔ MACRO KILL SWITCH ACTIVE</span>
            <span style="color:#FF8080;margin-left:16px;">
            Regime: <strong>{regime}</strong> · VIX {vix_latest:.1f} ·
            All picks are <strong>RISK BLOCKED</strong></span></div>""",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""<div style="background:{rc}18;border:1px solid {rc};
            border-radius:8px;padding:10px 20px;margin:8px 0;">
            <span style="font-size:1.1em;color:{rc};">✅ Regime: <strong>{regime}</strong></span>
            <span style="color:#B0B0B0;margin-left:16px;">
            Conviction {conviction:.0%} · VIX {vix_latest:.1f}</span></div>""",
            unsafe_allow_html=True,
        )

    m1, m2, m3 = st.columns(3)
    m1.metric("Regime", regime)
    m2.metric("Conviction", f"{conviction:.0%}")
    m3.metric("VIX", f"{vix_latest:.1f}")

    st.divider()

    # ── Alpha picks table ─────────────────────────────────────────────────────
    alpha_picks: List[Dict] = state.get("alpha_picks", [])

    if not alpha_picks:
        st.info("No alpha picks in current state. Run `python -m backend.engine_worker` to populate.")
        _render_system_health()
        return

    rows = []
    for r in alpha_picks:
        rows.append({
            "Symbol":        r.get("symbol", ""),
            "Smart Money":   float(r.get("smart_money_score", 0.0)),
            "Insider (45%)": float(r.get("insider_score", 0.0)),
            "Inst. (35%)":   float(r.get("institutional_score", 0.0)),
            "Momentum (20%)":float(r.get("momentum_score", 0.0)),
            "Insider $":     float(r.get("insider_value_usd", 0.0)),
            "Insider % Cap": _fmt_insider_pct(r.get("insider_value_pct_mcap")),
            "Vol Spike":     f"{r.get('volume_spike', 0.0):.2f}x",
            "Price Δ":       f"{r.get('price_change_pct', 0.0):+.2f}%",
            "Risk Block":    "⛔" if r.get("risk_block") else "✅",
            "Sources":       ", ".join(r.get("source_flags", [])),
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Smart Money":    st.column_config.ProgressColumn(
                "Smart Money", help="0.45·Insider + 0.35·Inst + 0.20·Momentum",
                min_value=0.0, max_value=1.0, format="%.3f",
            ),
            "Insider (45%)":  st.column_config.ProgressColumn(
                "Insider (45%)", min_value=0.0, max_value=1.0, format="%.3f",
            ),
            "Inst. (35%)":    st.column_config.ProgressColumn(
                "Inst. (35%)", min_value=0.0, max_value=1.0, format="%.3f",
            ),
            "Momentum (20%)": st.column_config.ProgressColumn(
                "Momentum (20%)", min_value=0.0, max_value=1.0, format="%.3f",
            ),
            "Insider $": st.column_config.NumberColumn(
                "Insider $", format="$%,.0f",
            ),
        },
    )

    _render_explainability(alpha_picks)

    with st.expander("Raw JSON state"):
        st.code(json.dumps(state, indent=2, default=str), language="json")

    _render_system_health()


def _render_explainability(results: List[Dict]) -> None:
    """Render per-ticker explainability expanders showing component scores and evidence.

    Reads only from the payload already returned by get_top_alpha_picks_sync() —
    no additional network calls are made.
    """
    _WEIGHTS = {
        "insider_score":       ("Insider",       0.25),
        "institutional_score": ("Institutional", 0.20),
        "momentum_score":      ("Momentum",      0.20),
        "smart_money_score":   ("Smart Money",   0.35),
    }

    for r in results:
        symbol = r.get("symbol", "?")
        score  = r.get("smart_money_score", 0.0)
        with st.expander(f"📊 {symbol} — score {score:.3f}  (click to expand)"):
            st.markdown("**Component scores**")
            comp_rows = []
            for field, (label, weight) in _WEIGHTS.items():
                raw = r.get(field, 0.0)
                try:
                    raw = float(raw)
                except (TypeError, ValueError):
                    raw = 0.0
                comp_rows.append({
                    "Component": label,
                    "Weight":    f"{weight:.0%}",
                    "Score":     f"{raw:.3f}",
                    "Contribution": f"{raw * weight:.4f}",
                })

            import pandas as pd
            st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)

            evidence: List[Dict] = r.get("evidence", []) or []
            if evidence:
                st.markdown("**Evidence**")
                for ev in evidence:
                    ev_type    = ev.get("type", "—")
                    ev_id      = ev.get("id", "—")
                    ev_summary = ev.get("summary", "—")
                    st.markdown(f"- `[{ev_type}]` **{ev_id}** — {ev_summary}")
            else:
                st.caption("No detailed evidence available for this ticker.")


def _render_macro_intel() -> None:
    """Render the Macro Intel tab — commodity conviction + macro shocks."""
    from regime_trader.scanners.market_intel_macro import (
        COMMODITY_UNIVERSE,
        MACRO_INDICATORS,
        calc_macro_conviction,
        check_macro_shocks,
    )

    st.header("Macro Intel — Commodity Conviction")
    _require_fmp()

    col_refresh, _ = st.columns([1, 4])
    if col_refresh.button("Refresh macro data", key="macro_refresh"):
        _load_commodity_prices.clear()
        _load_macro_indicators.clear()

    with st.spinner("Fetching commodity prices…"):
        prices     = _load_commodity_prices()
        indicators = _load_macro_indicators()

    n_missing_comm = sum(1 for v in prices.values() if v is None)
    n_missing_ind  = sum(1 for v in indicators.values() if v is None)
    if n_missing_comm or n_missing_ind:
        st.warning(
            f"⚠️ Partial data: {n_missing_comm} commodity feed(s) and "
            f"{n_missing_ind} macro indicator(s) unavailable. "
            "Rows show '—' where data is missing.",
            icon="⚠️",
        )

    alerts = check_macro_shocks(prices)
    for alert in alerts:
        fn = st.error if alert["level"] == "error" else st.warning
        fn(alert["message"])

    sentiment_map: Dict[str, float] = {}
    convictions: Dict[str, Dict] = {}

    rows = []
    for c in COMMODITY_UNIVERSE:
        ticker = c["ticker"]
        data   = prices.get(ticker)
        if data is None:
            rows.append({
                "Name": c["name"], "Ticker": ticker, "Price": "—",
                "1d": "—", "5d": "—", "Conviction": "No data",
            })
            continue
        cv = calc_macro_conviction(data, sentiment_map)
        convictions[ticker] = cv
        rows.append({
            "Name": c["name"],
            "Ticker": ticker,
            "Price": f"{data['price']:.2f} {c['unit']}",
            "1d": f"{data['ret_1d']:+.2%}",
            "5d": f"{data['ret_5d']:+.2%}",
            "RSI": f"{data.get('rsi14', 0):.0f}",
            "Conviction": cv["conviction_label"],
            "Score": f"{cv['composite']:.3f}",
        })

    import pandas as pd
    st.subheader("Commodity Scorecard")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Macro Narrative")
    for para in _generate_macro_synthesis(prices, convictions, indicators):
        st.markdown(f"> {para}")

    st.subheader("Macro Indicators")
    ind_rows = []
    for ind in MACRO_INDICATORS:
        data = indicators.get(ind["ticker"])
        ind_rows.append({
            "Name": ind["name"],
            "Ticker": ind["ticker"],
            "Value": f"{data['price']:.2f} {ind['unit']}" if data else "—",
            "1d": f"{data['ret_1d']:+.2%}" if data else "—",
            "5d": f"{data['ret_5d']:+.2%}" if data else "—",
        })
    st.dataframe(pd.DataFrame(ind_rows), use_container_width=True, hide_index=True)


def _render_trade_log() -> None:
    """Render the Trade Log tab (stub — requires broker NDJSON output)."""
    st.header("Trade Log")
    st.info("Parses trades.log (NDJSON) — connect your broker output to enable this tab.")


def _render_regime_history() -> None:
    """Render the Regime History tab (stub — requires HMM engine output)."""
    st.header("Regime History")
    st.info("Requires HMM engine output logs. Run the regime pipeline to populate.")


def _render_portfolio_sync() -> None:
    """Render the Portfolio Sync tab — upload a brokerage CSV for reconciliation."""
    st.header("Portfolio Sync")
    st.info("Upload a brokerage CSV to preview and execute position sync.")
    uploaded = st.file_uploader("Upload brokerage CSV", type=["csv"])
    if uploaded:
        import pandas as pd
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(20), use_container_width=True)


def _render_dashboard() -> None:
    """Render the main dashboard with all six tabs."""
    st.title("Regime Trader Dashboard")
    tabs = st.tabs([
        "📊 Live Monitor",
        "🧠 Market Intel",
        "🌍 Macro Intel",
        "📋 Trade Log",
        "📈 Regime History",
        "🔄 Portfolio Sync",
    ])
    with tabs[0]:
        _render_live_monitor()
    with tabs[1]:
        _render_market_intel()
    with tabs[2]:
        _render_macro_intel()
    with tabs[3]:
        _render_trade_log()
    with tabs[4]:
        _render_regime_history()
    with tabs[5]:
        _render_portfolio_sync()


# ── Main layout ───────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point: render sidebar navigation then dispatch to selected page.

    Called by Streamlit on every re-run. Quant pages are loaded lazily via
    importlib so a missing backend dependency only breaks that page, not the
    whole dashboard.
    """
    selected = _render_sidebar()

    # Build lookup from label → (filename, mod_name)
    _page_map = {label: (fn, mn) for label, fn, mn in _NAV_PAGES}

    if selected == "📊 Dashboard" or selected not in _page_map:
        _render_dashboard()
        return

    filename, mod_name = _page_map[selected]
    if filename is None:
        _render_dashboard()
        return

    mod = _load_page_module(mod_name, filename)
    if mod is None:
        st.error(
            f"**{selected}** could not be loaded. "
            "Check that all backend dependencies are installed and try again."
        )
        with st.expander("Troubleshooting"):
            st.markdown(
                "This page requires packages from `backend/`. "
                "Ensure `backend/data/fred_service.py`, `backend/quant_models/`, "
                "and related modules are present in the project root."
            )
        return

    if not hasattr(mod, "render"):
        st.error(f"Page module `{filename}` has no `render()` function.")
        return

    try:
        mod.render()
    except Exception as exc:
        log.exception("Page render failed: %s — %s", selected, exc)
        st.error(f"**{selected}** encountered an error: {exc}")


if __name__ == "__main__":
    main()
