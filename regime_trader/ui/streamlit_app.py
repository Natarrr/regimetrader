"""regime_trader/ui/streamlit_app.py
Streamlit orchestrator for the Regime Trader dashboard.

Tabs:
  Live Monitor    — regime / portfolio stubs
  Market Intel    — Smart Money discovery picks
  Macro Intel     — Commodity conviction + macro shocks
  Trade Log       — stub
  Regime History  — stub
  Portfolio Sync  — stub

All heavy computation is delegated to:
  regime_trader.discovery_scanner   — get_top_alpha_picks_sync()
  regime_trader.market_intel_macro  — fetch_commodity_prices(), calc_macro_conviction()

Run:
  streamlit run regime_trader/ui/streamlit_app.py
"""
from __future__ import annotations

import logging
import os
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

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Regime Trader",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

_FMP_KEY = os.getenv("FMP_API_KEY", "")
_HAS_FMP = bool(_FMP_KEY)

_ALPACA_KEY    = os.getenv("ALPACA_API_KEY", "")
_ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY", "")
_ALPACA_PAPER  = "paper" in os.getenv("ALPACA_BASE_URL", "paper").lower()
_HAS_ALPACA    = bool(_ALPACA_KEY and _ALPACA_SECRET)

# ── Optional import: generate_macro_synthesis (graceful fallback) ─────────────
try:
    from regime_trader.market_intel_macro import generate_macro_synthesis as _generate_macro_synthesis
except ImportError:
    def _generate_macro_synthesis(  # type: ignore[misc]
        prices: Dict, convictions: Dict, indicators: Dict
    ) -> List[str]:
        del prices, convictions, indicators  # fallback stub — params unused by design
        return ["Macro synthesis unavailable — market_intel_macro not installed."]


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
        from regime_trader.discovery_scanner import get_top_alpha_picks_sync
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
    from regime_trader.market_intel_macro import COMMODITY_UNIVERSE, fetch_commodity_prices

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
    from regime_trader.market_intel_macro import MACRO_INDICATORS, fetch_macro_indicator

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


# ── Tab renderers ─────────────────────────────────────────────────────────────

def _render_live_monitor() -> None:
    """Render the Live Monitor tab — live regime + Alpaca account."""
    hdr, btn = st.columns([8, 1])
    hdr.header("Live Monitor")
    if btn.button("↻", key="live_refresh", help="Refresh account data"):
        _load_alpaca_account.clear()
        _load_regime.clear()

    # ── Regime ────────────────────────────────────────────────────────────────
    regime_data = _load_regime()
    regime      = regime_data.get("regime", "Unknown")
    vix         = regime_data.get("vix")

    _REGIME_COLOR = {
        "Crash": "🔴", "Panic": "🟠", "Bear": "🟡",
        "Neutral": "⚪", "Bull": "🟢", "Euphoria": "💹",
    }
    regime_icon = _REGIME_COLOR.get(regime, "❓")

    # ── Account ───────────────────────────────────────────────────────────────
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
        return

    with st.spinner("Loading account…"):
        acct = _load_alpaca_account()

    if "error" in acct:
        st.error(f"Alpaca connection failed: {acct['error']}")
        return

    paper_tag = " · Paper trading" if acct["paper"] else ""
    pnl     = acct["daily_pnl"]
    pnl_pct = acct["daily_pnl_pct"]

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


def _render_market_intel() -> None:
    """Render the Market Intel tab — Smart Money discovery picks."""
    st.header("Market Intel — Smart Money Discovery")
    _require_fmp()

    limit = st.slider("Top picks", min_value=3, max_value=20, value=5, step=1)

    col_refresh, col_status = st.columns([1, 4])
    force = col_refresh.button("Force refresh", key="disc_refresh")

    payload: Dict[str, Any]
    if force:
        _load_discovery.clear()
        try:
            from regime_trader.discovery_scanner import force_refresh_sync
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

    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    with st.expander("Raw payload"):
        import json
        st.code(json.dumps(
            {k: v for k, v in payload.items() if not k.startswith("_")},
            indent=2, default=str,
        ), language="json")


def _render_macro_intel() -> None:
    """Render the Macro Intel tab — commodity conviction + macro shocks."""
    from regime_trader.market_intel_macro import (
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
        prices = _load_commodity_prices()
        indicators = _load_macro_indicators()

    # Macro shock alerts
    alerts = check_macro_shocks(prices)
    for alert in alerts:
        fn = st.error if alert["level"] == "error" else st.warning
        fn(alert["message"])

    sentiment_map: Dict[str, float] = {}
    convictions: Dict[str, Dict] = {}

    rows = []
    for c in COMMODITY_UNIVERSE:
        ticker = c["ticker"]
        data = prices.get(ticker)
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
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    # Macro synthesis — guarded at module level; fallback returns a safe string
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
    st.dataframe(pd.DataFrame(ind_rows), width="stretch", hide_index=True)


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
        st.dataframe(df.head(20), width="stretch")


# ── Main layout ───────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point for the Streamlit dashboard.

    Renders all tabs. Called by the root-level streamlit_app.py shim on
    every Streamlit re-run (including interactions).
    """
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


if __name__ == "__main__":
    main()
