# Path: tests/test_discord_bot.py
"""Unit tests for scripts/discord_bot.py — ChatOps ?score dispatch logic.

Only the pure functions are tested (sanitize_ticker, registry_check,
dispatch_workflow, DispatchError). `import discord` must stay LAZY inside
build_bot()/main() so CI never needs discord.py installed.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import requests_mock as requests_mock_lib  # noqa: F401 (plugin provides fixture)


def _load_bot():
    spec = importlib.util.spec_from_file_location(
        "discord_bot",
        Path(__file__).parents[1] / "scripts" / "discord_bot.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bot_mod():
    return _load_bot()


# ── Lazy-import contract ──────────────────────────────────────────────────────

def test_module_imports_without_discord_py(bot_mod):
    """Importing the module must not import discord.py (CI has no bot deps)."""
    assert bot_mod is not None
    # If discord.py is absent locally, module exec above already proves this;
    # belt-and-braces: the module namespace must not hold a discord binding.
    assert not hasattr(bot_mod, "discord")


# ── sanitize_ticker ───────────────────────────────────────────────────────────

class TestSanitizeTicker:
    def test_uppercases_and_strips(self, bot_mod):
        assert bot_mod.sanitize_ticker("  tsla ") == "TSLA"

    def test_intl_suffix_ok(self, bot_mod):
        assert bot_mod.sanitize_ticker("sap.de") == "SAP.DE"
        assert bot_mod.sanitize_ticker("700.hk") == "700.HK"

    def test_injection_rejected(self, bot_mod):
        with pytest.raises(ValueError):
            bot_mod.sanitize_ticker("TSLA;rm -rf /")

    def test_empty_rejected(self, bot_mod):
        with pytest.raises(ValueError):
            bot_mod.sanitize_ticker("   ")

    def test_too_long_rejected(self, bot_mod):
        with pytest.raises(ValueError):
            bot_mod.sanitize_ticker("toolong123")

    def test_regex_mirrors_audit_gate(self, bot_mod):
        """The bot regex must equal audit_payload._TICKER_RE — a ticker the
        bot accepts must never be rejected by the safety gate for format."""
        from src.delivery.audit_payload import _TICKER_RE
        assert bot_mod._TICKER_RE.pattern == _TICKER_RE.pattern


# ── registry_check ────────────────────────────────────────────────────────────

class TestRegistryCheck:
    @pytest.fixture
    def registry(self, tmp_path):
        p = tmp_path / "ticker_registry.json"
        p.write_text(json.dumps({
            "europe": [{"ticker": "SAP.DE"}],
            "asia":   [{"ticker": "700.HK"}],
        }), encoding="utf-8")
        return p

    def test_registered_intl_ok(self, bot_mod, registry):
        assert bot_mod.registry_check("SAP.DE", registry) is None

    def test_bare_us_ticker_ok(self, bot_mod, registry):
        assert bot_mod.registry_check("TSLA", registry) is None

    def test_unregistered_intl_returns_error(self, bot_mod, registry):
        err = bot_mod.registry_check("BAD.XX", registry)
        assert err is not None and "BAD.XX" in err

    def test_unreadable_registry_defers_to_workflow(self, bot_mod, tmp_path):
        assert bot_mod.registry_check("SAP.DE", tmp_path / "nope.json") is None


# ── dispatch_workflow ─────────────────────────────────────────────────────────

_URL = ("https://api.github.com/repos/owner/repo/actions/workflows/"
        "ondemand_score.yml/dispatches")


class TestDispatchWorkflow:
    def test_posts_workflow_dispatch(self, bot_mod, requests_mock):
        requests_mock.post(_URL, status_code=204)
        bot_mod.dispatch_workflow("TSLA", repo="owner/repo", token="ghp_x")
        req = requests_mock.last_request
        assert req.json() == {"ref": "main", "inputs": {"ticker": "TSLA"}}
        assert req.headers["Authorization"] == "Bearer ghp_x"
        assert req.headers["Accept"] == "application/vnd.github+json"
        assert req.headers["X-GitHub-Api-Version"] == "2022-11-28"

    def test_custom_ref(self, bot_mod, requests_mock):
        requests_mock.post(_URL, status_code=204)
        bot_mod.dispatch_workflow("TSLA", repo="owner/repo", token="t",
                                  ref="develop")
        assert requests_mock.last_request.json()["ref"] == "develop"

    def test_401_raises_with_github_message(self, bot_mod, requests_mock):
        requests_mock.post(_URL, status_code=401,
                           json={"message": "Bad credentials"})
        with pytest.raises(bot_mod.DispatchError) as exc_info:
            bot_mod.dispatch_workflow("TSLA", repo="owner/repo", token="bad")
        assert exc_info.value.status == 401
        assert "Bad credentials" in str(exc_info.value)

    def test_422_raises(self, bot_mod, requests_mock):
        requests_mock.post(_URL, status_code=422,
                           json={"message": "Workflow does not have "
                                            "workflow_dispatch trigger"})
        with pytest.raises(bot_mod.DispatchError) as exc_info:
            bot_mod.dispatch_workflow("TSLA", repo="owner/repo", token="t")
        assert exc_info.value.status == 422


# ── main() env contract ───────────────────────────────────────────────────────

def test_main_missing_env_exits_2(bot_mod, monkeypatch, capsys):
    for var in ("DISCORD_BOT_TOKEN", "GITHUB_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    assert bot_mod.main() == 2
    assert "DISCORD_BOT_TOKEN" in capsys.readouterr().err
