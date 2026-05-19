"""pages/6_Stock_Picker.py
Monthly stock pick leaderboard — sector picks + cap-tier picks.
Reads logs/top_lists.json (produced by edgar_3x pipeline). Zero API calls.

Run via sidebar: "📅 Stock Picker"
"""
from __future__ import annotations

import io
import json
import logging
import os
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from regime_trader.utils.formatting import score_bar as _score_bar_util

# Re-read .env on every page load so tokens added after server start are picked up
_ROOT_ENV = Path(__file__).parent.parent / ".env"
load_dotenv(_ROOT_ENV, override=True)

log = logging.getLogger(__name__)

_GH_OWNER    = "Natarrr"
_GH_REPO     = "regimetrader"
_GH_WORKFLOW = "edgar_3x.yml"


def _trigger_pipeline(pat: str, force: bool = False) -> tuple[bool, str]:
    """POST workflow_dispatch to GitHub Actions. Returns (success, message)."""
    try:
        import requests
        url = (
            f"https://api.github.com/repos/{_GH_OWNER}/{_GH_REPO}"
            f"/actions/workflows/{_GH_WORKFLOW}/dispatches"
        )
        resp = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {pat}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            json={"ref": "main", "inputs": {"force_regen": "true" if force else "false"}},
            timeout=15,
        )
        if resp.status_code == 204:
            return True, "Pipeline triggered — GitHub Actions is starting the run."
        if resp.status_code == 401:
            return False, "GitHub token invalid or expired (401). Check GH_PAT in your .env."
        if resp.status_code == 404:
            return False, "Workflow not found (404). Check repo name and workflow file."
        return False, f"GitHub API returned {resp.status_code}: {resp.text[:200]}"
    except Exception as exc:
        return False, f"Request failed: {exc}"


def _sync_from_github(pat: str) -> tuple[bool, str]:
    """Download latest top-lists artifact from GitHub and write logs/top_lists.json.
    Returns (success, message). 'already_up_to_date' message means no write occurred."""
    try:
        import requests
        # 1. Find the latest top-lists artifact
        url = (
            f"https://api.github.com/repos/{_GH_OWNER}/{_GH_REPO}"
            "/actions/artifacts?name=top-lists&per_page=5"
        )
        resp = requests.get(
            url,
            headers={
                "Authorization": f"Bearer {pat}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=15,
        )
        if resp.status_code == 401:
            return False, "GitHub token invalid or expired (401)."
        if resp.status_code != 200:
            return False, f"GitHub API returned {resp.status_code}: {resp.text[:200]}"
        artifacts = resp.json().get("artifacts", [])
        active = [a for a in artifacts if not a.get("expired", True)]
        if not active:
            return False, "No active top-lists artifact found (may be expired)."
        artifact_id = active[0]["id"]

        # 2. Download the artifact zip
        zip_url = (
            f"https://api.github.com/repos/{_GH_OWNER}/{_GH_REPO}"
            f"/actions/artifacts/{artifact_id}/zip"
        )
        zip_resp = requests.get(
            zip_url,
            headers={
                "Authorization": f"Bearer {pat}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=30,
            allow_redirects=True,
        )
        if zip_resp.status_code != 200:
            return False, f"Artifact download returned {zip_resp.status_code}."

        # 3. Extract top_lists.json from the zip
        with zipfile.ZipFile(io.BytesIO(zip_resp.content)) as zf:
            if "top_lists.json" not in zf.namelist():
                return False, "top_lists.json not found in artifact zip."
            remote_data = json.loads(zf.read("top_lists.json").decode("utf-8"))

        # 4. Compare timestamps
        remote_ts = remote_data.get("generated_at", "")
        local_ts = ""
        if _TOP_LISTS.exists():
            try:
                local_ts = json.loads(_TOP_LISTS.read_text(encoding="utf-8")).get("generated_at", "")
            except Exception:
                pass

        if local_ts and remote_ts:
            try:
                from datetime import datetime as _dt
                def _parse_ts(s: str) -> _dt:
                    return _dt.fromisoformat(s.replace("Z", "+00:00"))
                if _parse_ts(local_ts) >= _parse_ts(remote_ts):
                    return True, "already_up_to_date"
            except Exception:
                if local_ts >= remote_ts:
                    return True, "already_up_to_date"

        # 5. Write atomically
        tmp = _TOP_LISTS.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(remote_data, indent=2), encoding="utf-8")
            tmp.replace(_TOP_LISTS)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        return True, f"Synced — data from {remote_ts[:16] or 'unknown'}"

    except Exception as exc:
        return False, f"Sync failed: {exc}"


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_last_run_status(pat: str) -> Dict[str, Any]:
    """Fetch the most recent edgar_3x run status from GitHub API. TTL 60s."""
    try:
        import requests
        url = (
            f"https://api.github.com/repos/{_GH_OWNER}/{_GH_REPO}"
            f"/actions/workflows/{_GH_WORKFLOW}/runs?per_page=1"
        )
        resp = requests.get(
            url,
            headers={
                "Authorization": f"Bearer {pat}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            timeout=10,
        )
        if resp.status_code != 200:
            return {}
        runs = resp.json().get("workflow_runs", [])
        if not runs:
            return {}
        r = runs[0]
        return {
            "status":     r.get("status"),       # queued / in_progress / completed
            "conclusion": r.get("conclusion"),    # success / failure / cancelled / None
            "created_at": r.get("created_at"),
            "html_url":   r.get("html_url"),
        }
    except Exception:
        return {}

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
    return _score_bar_util(score, width)


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
    _gh_pat_early = os.getenv("GH_PAT", "")

    col_ref, col_sync, col_ts = st.columns([1, 1, 5])

    if col_ref.button("↻ Refresh", key="sp_refresh"):
        _load_top_lists.clear()
        st.session_state["sp_just_refreshed"] = True
        st.rerun()

    if _gh_pat_early and col_sync.button("⬇ Sync", key="sp_sync", help="Download latest artifact from GitHub Actions"):
        with st.spinner("Syncing from GitHub…"):
            ok, msg = _sync_from_github(_gh_pat_early)
        if ok and msg == "already_up_to_date":
            st.toast("Already up to date", icon="✅")
        elif ok:
            _load_top_lists.clear()
            st.toast(msg, icon="✅")
            st.rerun()
        else:
            st.error(f"Sync failed: {msg}", icon="❌")

    data = _load_top_lists()

    if st.session_state.pop("sp_just_refreshed", False):
        st.toast("Cache cleared — showing latest data from disk.", icon="✅")

    if data is None:
        st.error(
            "**⚠️ No data** — `logs/top_lists.json` not found.\n\n"
            "Run the edgar_3x pipeline to generate picks:\n"
            "```\npython -m backend.market_intel.generate_top_lists --force\n```"
        )
        return

    generated_at = data.get("generated_at", "—")
    ticker_count = data.get("ticker_count", 0)

    try:
        from datetime import datetime, timezone as _tz
        _ts = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
        _age_h = (datetime.now(_tz.utc) - _ts).total_seconds() / 3600
        _date_str = _ts.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        _age_h = 0.0
        _date_str = generated_at[:16] or "—"

    col_ts.caption(f"Pipeline ran: **{_date_str}** · {ticker_count} tickers scored")

    # ── Pipeline trigger ───────────────────────────────────────────────────────
    _gh_pat = os.getenv("GH_PAT", "")
    _stale  = _age_h > 25

    if _stale or _gh_pat:
        with st.expander(
            "⚠️ Data is stale — run the pipeline" if _stale else "⚙️ Pipeline controls",
            expanded=_stale,
        ):
            if not _gh_pat:
                st.info(
                    "Add `GH_PAT=<your_token>` to your `.env` to enable one-click pipeline runs.\n\n"
                    "The token needs **`Actions: write`** scope on the `Natarrr/regimetrader` repo.",
                    icon="ℹ️",
                )
            else:
                # ── Last run status ────────────────────────────────────────────
                run_info = _fetch_last_run_status(_gh_pat)
                if run_info:
                    _status     = run_info.get("status", "")
                    _conclusion = run_info.get("conclusion")
                    _run_url    = run_info.get("html_url", "")
                    _run_ts     = run_info.get("created_at", "")[:16].replace("T", " ")
                    _in_prog    = _status in ("queued", "in_progress")

                    if _in_prog:
                        st.info(f"⏳ Pipeline is **{_status}** (started {_run_ts} UTC)", icon="⏳")
                    elif _conclusion == "success":
                        st.success(f"✅ Last run **succeeded** ({_run_ts} UTC) — click Refresh above to reload data.", icon="✅")
                    elif _conclusion in ("failure", "timed_out"):
                        st.error(f"❌ Last run **{_conclusion}** ({_run_ts} UTC) · [View logs]({_run_url})", icon="❌")
                    else:
                        st.caption(f"Last run: **{_conclusion or _status}** · {_run_ts} UTC")

                # ── Trigger button ─────────────────────────────────────────────
                btn_col, info_col = st.columns([2, 5])
                _force = btn_col.checkbox("Force regeneration", value=True, key="sp_force_regen",
                                          help="Pass force_regen=true to rebuild even if top_lists.json is fresh")
                if btn_col.button("▶ Run edgar_3x now", key="sp_run_pipeline", type="primary"):
                    with st.spinner("Dispatching workflow…"):
                        ok, msg = _trigger_pipeline(_gh_pat, force=_force)
                    if ok:
                        _fetch_last_run_status.clear()
                        st.toast(msg, icon="✅")
                        time.sleep(2)   # give GitHub a moment before next status poll
                        st.rerun()
                    else:
                        st.error(msg, icon="❌")

                info_col.caption(
                    f"Triggers the `edgar_3x` workflow on **{_GH_OWNER}/{_GH_REPO}** · "
                    "takes ~5 min · data auto-refreshes when you click ↻ Refresh above"
                )

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
