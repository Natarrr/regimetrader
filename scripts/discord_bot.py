# Path: scripts/discord_bot.py
"""ChatOps listener: `?score TICKER` → GitHub Actions workflow_dispatch.

Long-running daemon (NOT part of the CI pipeline). On `?score TSLA` it
sanitizes the ticker, optionally pre-validates dotted tickers against
config/ticker_registry.json, then POSTs to the GitHub REST endpoint

    https://api.github.com/repos/{owner}/{repo}/actions/workflows/
        ondemand_score.yml/dispatches

and acknowledges in-channel. The scored factor audit is posted back to the
channel by the workflow itself (send_discord.py via DISCORD_WEBHOOK_URL).

Environment:
    DISCORD_BOT_TOKEN   required — Discord bot token (message-content intent
                        must be enabled in the developer portal)
    GITHUB_TOKEN        required — PAT with repo+workflow scope (or
                        fine-grained Actions: read/write); the Actions-injected
                        token does not exist outside CI
    GITHUB_REPOSITORY   required — "owner/repo" slug
    GITHUB_REF          optional — branch to dispatch on (default: main)

Run:
    pip install -r scripts/requirements-bot.txt
    python scripts/discord_bot.py

`import discord` is deliberately LAZY (inside build_bot/main) so the test
suite can exercise the dispatch logic without discord.py installed.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path

import requests

# MUST mirror src/delivery/audit_payload._TICKER_RE — a ticker the bot accepts
# must never be rejected by the safety gate for format (asserted in
# tests/test_discord_bot.py::test_regex_mirrors_audit_gate).
_TICKER_RE = re.compile(r"^([A-Z]{1,5}|[A-Z0-9]{1,6}\.[A-Z]{1,2})$")

_GITHUB_API = "https://api.github.com"
_WORKFLOW_FILE = "ondemand_score.yml"
_DEFAULT_REGISTRY = Path(__file__).resolve().parents[1] / "config" / "ticker_registry.json"
_REGISTRY_SECTIONS = ("europe", "europe_mid", "asia", "asia_mid")


class DispatchError(RuntimeError):
    """GitHub rejected the workflow_dispatch request."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message


def sanitize_ticker(raw: str) -> str:
    """Uppercase/strip a user-supplied ticker; raise ValueError on bad format."""
    ticker = (raw or "").strip().upper()
    if not ticker:
        raise ValueError("Ticker is empty — usage: `?score TSLA`")
    if not _TICKER_RE.match(ticker):
        raise ValueError(
            f"`{ticker}` is not a valid ticker (expected e.g. `TSLA` or "
            f"`SAP.DE`)"
        )
    return ticker


def registry_check(ticker: str, registry_path: Path) -> str | None:
    """Best-effort pre-flight for dotted (international) tickers.

    Returns an error string when the ticker carries an international suffix
    but is not in ticker_registry.json — failing fast in-channel instead of
    burning a workflow run. Returns None for bare US tickers, registered
    tickers, or when the registry cannot be read (the workflow re-validates
    authoritatively, so an unreadable local registry must not block dispatch).
    """
    if "." not in ticker:
        return None
    try:
        registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[BOT] WARN: registry unreadable ({exc}) — deferring "
              f"validation to the workflow", file=sys.stderr)
        return None
    known = {
        entry.get("ticker")
        for section in _REGISTRY_SECTIONS
        for entry in registry.get(section, [])
    }
    if ticker not in known:
        return (f"`{ticker}` has an international suffix but is not in "
                f"ticker_registry.json — on-demand INTL scoring only covers "
                f"registered tickers")
    return None


def dispatch_workflow(
    ticker: str,
    *,
    repo: str,
    token: str,
    ref: str = "main",
    workflow_file: str = _WORKFLOW_FILE,
    timeout: float = 10.0,
) -> None:
    """POST a workflow_dispatch for ticker; raise DispatchError on rejection."""
    url = f"{_GITHUB_API}/repos/{repo}/actions/workflows/{workflow_file}/dispatches"
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        json={"ref": ref, "inputs": {"ticker": ticker}},
        timeout=timeout,
    )
    if resp.status_code != 204:
        try:
            message = resp.json().get("message", resp.text)
        except ValueError:
            message = resp.text
        raise DispatchError(resp.status_code, message)


def build_bot(*, repo: str, token: str, ref: str,
              registry_path: Path = _DEFAULT_REGISTRY):
    """Construct the discord.py bot (lazy import — see module docstring)."""
    import discord
    from discord.ext import commands

    intents = discord.Intents.default()
    intents.message_content = True  # privileged — enable in the dev portal
    bot = commands.Bot(command_prefix="?", intents=intents)

    @bot.event
    async def on_ready() -> None:
        print(f"[BOT] Logged in as {bot.user} — listening for ?score "
              f"(dispatching to {repo}@{ref})")
        print(f"[BOT] message_content intent active: {bot.intents.message_content}")

    @bot.event
    async def on_message(message) -> None:
        if message.author == bot.user:
            return
        print(f"[BOT] MSG from {message.author}: {repr(message.content)}")
        await bot.process_commands(message)

    @bot.command(name="score")
    async def score(ctx, raw_ticker: str | None = None) -> None:
        if not raw_ticker:
            await ctx.reply(
                "Usage: `?score TICKER` — e.g. `?score TSLA` (US) or "
                "`?score SAP.DE` (registered intl)")
            return
        try:
            ticker = sanitize_ticker(raw_ticker)
        except ValueError as exc:
            await ctx.reply(f"❌ {exc}")
            return
        registry_err = registry_check(ticker, registry_path)
        if registry_err:
            await ctx.reply(f"❌ {registry_err}")
            return
        try:
            # requests is synchronous — keep the event loop responsive.
            await asyncio.to_thread(
                dispatch_workflow, ticker, repo=repo, token=token, ref=ref)
        except DispatchError as exc:
            await ctx.reply(
                f"❌ GitHub dispatch failed (HTTP {exc.status}): {exc.message}")
            return
        except requests.RequestException as exc:
            await ctx.reply(f"❌ GitHub dispatch failed: network error — {exc}")
            return
        await ctx.reply(
            f"📨 Dispatched on-demand scoring for **{ticker}** — the factor "
            f"audit posts here in ~3-8 min "
            f"(https://github.com/{repo}/actions/workflows/{_WORKFLOW_FILE})")

    return bot


def main() -> int:
    # Load .env from repo root if present (local dev convenience)
    try:
        from dotenv import load_dotenv
        _env = Path(__file__).resolve().parents[1] / ".env"
        if _env.exists():
            load_dotenv(_env)
            print(f"[BOT] Loaded env from {_env}")
    except ImportError:
        pass

    bot_token = os.environ.get("DISCORD_BOT_TOKEN", "")
    gh_token = os.environ.get("GITHUB_TOKEN", "")
    missing = [name for name, val in (
        ("DISCORD_BOT_TOKEN", bot_token),
        ("GITHUB_TOKEN", gh_token),
    ) if not val]
    if missing:
        print(f"[BOT] ERROR: missing required env var(s): "
              f"{', '.join(missing)}", file=sys.stderr)
        return 2

    repo = os.environ.get("GITHUB_REPOSITORY", "")
    if "/" not in repo:
        print("[BOT] ERROR: GITHUB_REPOSITORY must be set to 'owner/repo'",
              file=sys.stderr)
        return 2
    ref = os.environ.get("GITHUB_REF", "main")

    bot = build_bot(repo=repo, token=gh_token, ref=ref)
    bot.run(bot_token)
    return 0


if __name__ == "__main__":
    sys.exit(main())
