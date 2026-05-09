"""tests/test_check_secrets.py
Unit tests for scripts/check_secrets.py CI-friendly exit-code logic.

Stiglitz (2001 Nobel) — information asymmetry: the checker must reveal
presence/absence without revealing values; the exit code must match the
protection context, not just the presence state.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parent.parent / "scripts" / "check_secrets.py"


def _load() -> object:
    spec = importlib.util.spec_from_file_location("check_secrets", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


@pytest.fixture()
def cs():
    return _load()


# ── _is_protected_run ─────────────────────────────────────────────────────────

class TestIsProtectedRun:
    def test_require_secrets_env_true(self, cs, monkeypatch):
        monkeypatch.setenv("REQUIRE_SECRETS", "true")
        monkeypatch.delenv("GITHUB_REF", raising=False)
        assert cs._is_protected_run() is True

    def test_require_secrets_env_1(self, cs, monkeypatch):
        monkeypatch.setenv("REQUIRE_SECRETS", "1")
        monkeypatch.delenv("GITHUB_REF", raising=False)
        assert cs._is_protected_run() is True

    def test_main_branch_is_protected(self, cs, monkeypatch):
        monkeypatch.delenv("REQUIRE_SECRETS", raising=False)
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
        monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
        assert cs._is_protected_run() is True

    def test_master_branch_is_protected(self, cs, monkeypatch):
        monkeypatch.delenv("REQUIRE_SECRETS", raising=False)
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        monkeypatch.setenv("GITHUB_REF", "refs/heads/master")
        monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
        assert cs._is_protected_run() is True

    def test_pull_request_is_not_protected(self, cs, monkeypatch):
        monkeypatch.delenv("REQUIRE_SECRETS", raising=False)
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        monkeypatch.setenv("GITHUB_REF", "refs/pull/42/merge")
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
        assert cs._is_protected_run() is False

    def test_feature_branch_is_not_protected(self, cs, monkeypatch):
        monkeypatch.delenv("REQUIRE_SECRETS", raising=False)
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        monkeypatch.setenv("GITHUB_REF", "refs/heads/feature/my-branch")
        monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
        assert cs._is_protected_run() is False

    def test_local_dev_not_in_gh_actions(self, cs, monkeypatch):
        monkeypatch.delenv("REQUIRE_SECRETS", raising=False)
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
        monkeypatch.delenv("GITHUB_REF", raising=False)
        assert cs._is_protected_run() is False


# ── check_secrets exit codes ──────────────────────────────────────────────────

class TestCheckSecretsExitCodes:
    def test_all_present_exits_0_always(self, cs, monkeypatch):
        for k in cs.REQUIRED:
            monkeypatch.setenv(k, "dummy_value")
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
        _, code = cs.check_secrets(required=cs.REQUIRED, optional=())
        assert code == 0

    def test_missing_non_protected_exits_0(self, cs, monkeypatch):
        for k in cs.REQUIRED:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.delenv("REQUIRE_SECRETS", raising=False)
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
        monkeypatch.delenv("GITHUB_REF", raising=False)
        _, code = cs.check_secrets(required=cs.REQUIRED, optional=())
        assert code == 0

    def test_missing_on_main_exits_1(self, cs, monkeypatch):
        for k in cs.REQUIRED:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.delenv("REQUIRE_SECRETS", raising=False)
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
        monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
        _, code = cs.check_secrets(required=cs.REQUIRED, optional=())
        assert code == 1

    def test_missing_with_require_secrets_exits_1(self, cs, monkeypatch):
        for k in cs.REQUIRED:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("REQUIRE_SECRETS", "true")
        monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
        monkeypatch.delenv("GITHUB_REF", raising=False)
        _, code = cs.check_secrets(required=cs.REQUIRED, optional=())
        assert code == 1

    def test_pr_missing_exits_0(self, cs, monkeypatch):
        for k in cs.REQUIRED:
            monkeypatch.delenv(k, raising=False)
        monkeypatch.delenv("REQUIRE_SECRETS", raising=False)
        monkeypatch.setenv("GITHUB_ACTIONS", "true")
        monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
        monkeypatch.setenv("GITHUB_REF", "refs/pull/7/merge")
        _, code = cs.check_secrets(required=cs.REQUIRED, optional=())
        assert code == 0


# ── No values in output ───────────────────────────────────────────────────────

class TestNoSecretsLeaked:
    def test_values_not_in_output(self, cs, monkeypatch, capsys):
        secret_val = "ultra_secret_canary_xyz_9182"
        monkeypatch.setenv("FMP_API_KEY", secret_val)
        cs.check_secrets(required=("FMP_API_KEY",), optional=())
        out = capsys.readouterr().out
        assert secret_val not in out

    def test_optional_values_not_in_output(self, cs, monkeypatch, capsys):
        secret_val = "optional_canary_abc_4567"
        monkeypatch.setenv("ANTHROPIC_API_KEY", secret_val)
        cs.check_secrets(required=(), optional=("ANTHROPIC_API_KEY",))
        out = capsys.readouterr().out
        assert secret_val not in out
