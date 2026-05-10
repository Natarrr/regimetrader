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
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError, as_completed
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

def _load_page_module(name: str, filename: str):
    """Load a page module once; cache it in sys.modules so @st.cache_data persists.

    Returns the module or None if the file is missing / imports fail.
    """
    key = f"_rt_page_{name}"
    if key in sys.modules:
        return sys.modules[key]
    path = _PAGES_DIR / filename
    if not path.exists():
        log.warning("Page file not found: %s", path)
        return None
    try:
        spec = importlib.util.spec_from_file_location(key, path)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod
        spec.loader.exec_module(mod)
        return mod
    except Exception as exc:
        log.warning("Failed to load page %s: %s", filename, exc)
        # Remove partial entry so next run retries
        sys.modules.pop(key, None)
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
        for p in sorted(positions, key=lambda x: float(x.market_value or 0), reverse=True):
            pos_rows.append({
                "Symbol":    p.symbol,
                "Side":      p.side.value.capitalize(),
                "Qty":       float(p.qty),
                "Entry":     float(p.avg_entry_price),
                "Price":     float(p.current_price),
                "Mkt Value": float(p.market_value),
                "Unreal. P&L": float(p.unrealized_pl),
                "Unreal. %": float(p.unrealized_plpc) * 100,
                "Day P&L":   float(p.unrealized_intraday_pl),
                "Day %":     float(p.unrealized_intraday_plpc) * 100,
            })

        return {
            "equity":          equity,
            "buying_power":    float(acct.buying_power),
            "portfolio_value": float(acct.portfolio_value),
            "daily_pnl":       daily_pnl,
            "daily_pnl_pct":   daily_pnl_pct,
            "positions":       pos_rows,
            "status":          acct.status.value,
            "paper":           _ALPACA_PAPER,
        }
    except Exception as exc:
        log.warning("Alpaca account load failed: %s", exc)
        return {"error": str(exc)}


@st.cache_data(ttl=300, show_spinner=False)
def _load_regime() -> Dict[str, Any]:
    """Fetch latest VIX and derive market regime label. TTL 5 min."""
    try:
        import yfinance as yf
        from regime.regime_detector import vix_rule
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
            if st.button("Clear discovery cache", key="clear_disc"):
                _load_discovery.clear()
                st.success("Discovery cache cleared.")
            if st.button("Clear commodity cache", key="clear_comm"):
                _load_commodity_prices.clear()
                _load_macro_indicators.clear()
                st.success("Commodity / macro cache cleared.")
            if st.button("Clear account cache", key="clear_acct"):
                _load_alpaca_account.clear()
                _load_regime.clear()
                _load_vix_history.clear()
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

def _render_live_monitor() -> None:
    """Render the Live Monitor tab — live regime + Alpaca account + VIX sparkline."""
    hdr, btn = st.columns([8, 1])
    hdr.header("Live Monitor")
    if btn.button("↻", key="live_refresh", help="Refresh account data"):
        _load_alpaca_account.clear()
        _load_regime.clear()
        _load_vix_history.clear()

    regime_data = _load_regime()
    regime      = regime_data.get("regime", "Unknown")
    vix         = regime_data.get("vix")

    _REGIME_COLOR = {
        "Crash": "🔴", "Panic": "🟠", "Bear": "🟡",
        "Neutral": "⚪", "Bull": "🟢", "Euphoria": "💹",
    }
    regime_icon = _REGIME_COLOR.get(regime, "❓")

    if not _HAS_ALPACA:
        st.warning(
            "**ALPACA_API_KEY / ALPACA_SECRET_KEY not set.** "
            "Add them to `.env` and restart the app.",
            icon="⚠️",
        )
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Regime", f"{regime_icon} {regime}",
                    f"VIX {vix:.1f}" if vix else None)
        col2.metric("Portfolio Value", "—")
        col3.metric("Open Positions", "—")
        col4.metric("Daily P&L", "—")
        _render_vix_sparkline()
        return

    with st.spinner("Loading account…"):
        acct = _load_alpaca_account()

    if "error" in acct:
        st.error(f"Alpaca connection failed: {acct['error']}")
        return

    paper_tag = " · Paper trading" if acct["paper"] else ""
    pnl       = acct["daily_pnl"]
    pnl_pct   = acct["daily_pnl_pct"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Regime", f"{regime_icon} {regime}",
                f"VIX {vix:.1f}" if vix else None)
    col2.metric("Portfolio Value", f"${acct['portfolio_value']:,.2f}")
    col3.metric("Open Positions",  len(acct["positions"]))
    col4.metric("Daily P&L",       f"${pnl:+,.2f}", f"{pnl_pct:+.2f}%")

    st.caption(
        f"Alpaca · Status: **{acct['status']}**{paper_tag} · "
        f"Buying power: **${acct['buying_power']:,.2f}**"
    )

    _render_vix_sparkline()

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
    """Render a 10-day VIX sparkline below the regime metrics. No-op on failure."""
    try:
        history = _load_vix_history()
        if not history or len(history) < 2:
            return
        import pandas as pd
        df = pd.DataFrame({"VIX": history})
        st.caption("VIX — 10-day history")
        st.line_chart(df, height=100, use_container_width=True)
    except Exception as exc:
        log.debug("VIX sparkline render failed: %s", exc)


def _render_market_intel() -> None:
    """Render the Market Intel tab — Smart Money discovery picks + explainability."""
    st.header("Market Intel — Smart Money Discovery")
    _require_fmp()

    limit = st.slider("Top picks", min_value=3, max_value=20, value=5, step=1)

    col_refresh, col_status = st.columns([1, 4])
    force = col_refresh.button("Force refresh", key="disc_refresh")

    payload: Dict[str, Any]
    if force:
        _load_discovery.clear()
        try:
            from regime_trader.scanners.discovery_scanner import force_refresh_sync
            with st.spinner("Running fresh scan…"):
                payload = force_refresh_sync(limit=limit)
            st.success("Scan complete.")
        except Exception as exc:
            log.warning("force refresh failed: %s", exc)
            payload = _safe_payload()
            st.warning("Scan failed — check application logs.")
    else:
        try:
            with st.spinner("Loading discovery data…"):
                payload = _load_discovery(limit=limit)
        except Exception as exc:
            log.warning("discovery load failed: %s", exc)
            payload = _safe_payload()

    cached_flag = payload.get("cached", False)
    computed_at = payload.get("computed_at", "—")
    col_status.caption(
        f"{'Cached' if cached_flag else 'Fresh'} · computed {computed_at}"
    )

    results: List[Dict] = payload.get("results", [])

    if not results:
        st.info("No discovery results. Check FMP_API_KEY and try Force refresh.")
        return

    import pandas as pd

    _INSIDER_PCT_HELP = (
        "Insider % of Market Cap — values ≤ 1.0 from the scanner are raw fractions "
        "and are multiplied by 100 for display; values > 1.0 are already percentages."
    )

    rows = []
    for r in results:
        rows.append({
            "Symbol": r.get("symbol", ""),
            "Smart Money Score": f"{r.get('smart_money_score', 0):.3f}",
            "Insider Score": f"{r.get('insider_score', 0):.3f}",
            "Inst. Score": f"{r.get('institutional_score', 0):.3f}",
            "Momentum Score": f"{r.get('momentum_score', 0):.3f}",
            "Insider $": f"${r.get('insider_value_usd', 0):,.0f}",
            "Insider % MktCap": _fmt_insider_pct(r.get("insider_value_pct_mcap")),
            "Vol Spike": f"{r.get('volume_spike', 0):.2f}x",
            "Price Chg": f"{r.get('price_change_pct', 0):+.2f}%",
            "Sources": ", ".join(r.get("source_flags", [])),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(f"ℹ️ Insider % MktCap: {_INSIDER_PCT_HELP}")

    _render_explainability(results)

    with st.expander("Raw payload"):
        import json
        st.code(json.dumps(
            {k: v for k, v in payload.items() if not k.startswith("_")},
            indent=2, default=str,
        ), language="json")


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
