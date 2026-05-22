"""tests/test_logging_cfg.py
Unit tests for SecretMaskFilter and mask_secret() in logging_cfg.

Stiglitz (2001 Nobel) — information asymmetry: verifies that the logging
layer never leaks live API-key values regardless of how they enter a record.
"""
from __future__ import annotations

import io
import logging
import sys
from pathlib import Path


ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from regime_trader.utils.logging_cfg import SecretMaskFilter, configure_logging, mask_secret


# ── Helpers ───────────────────────────────────────────────────────────────────

def _capture(monkeypatch, key: str, secret: str, message: str, *args: object) -> str:
    """Configure logging to a StringIO buffer and emit one record; return output."""
    monkeypatch.setenv(key, secret)
    buf = io.StringIO()
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    root.handlers = []
    try:
        configure_logging(level=logging.DEBUG, stream=buf, mask_env=True)
        logging.getLogger("test.mask").info(message, *args)
        return buf.getvalue()
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)


# ── mask_secret() ─────────────────────────────────────────────────────────────

class TestMaskSecret:
    def test_replaces_fmp_key(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "my_fmp_secret")
        result = mask_secret("url?apikey=my_fmp_secret&limit=10")
        assert "my_fmp_secret" not in result
        assert "<REDACTED>" in result

    def test_passthrough_when_no_match(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "actual_secret")
        result = mask_secret("no sensitive data here")
        assert result == "no sensitive data here"

    def test_empty_env_value_not_replaced(self, monkeypatch):
        monkeypatch.delenv("FMP_API_KEY", raising=False)
        result = mask_secret("some string")
        assert result == "some string"

    def test_masks_multiple_keys(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "fmp_token")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "anth_token")
        result = mask_secret("fmp=fmp_token anth=anth_token")
        assert "fmp_token" not in result
        assert "anth_token" not in result
        assert result.count("<REDACTED>") == 2

    def test_replaces_all_occurrences(self, monkeypatch):
        monkeypatch.setenv("FMP_API_KEY", "repeat_secret")
        result = mask_secret("a=repeat_secret b=repeat_secret")
        assert "repeat_secret" not in result
        assert result.count("<REDACTED>") == 2


# ── SecretMaskFilter via configure_logging() ──────────────────────────────────

class TestSecretMaskFilter:
    def test_masks_fmp_key_in_message(self, monkeypatch):
        out = _capture(monkeypatch, "FMP_API_KEY", "super_secret_123", "key=super_secret_123")
        assert "super_secret_123" not in out
        assert "<REDACTED>" in out

    def test_masks_alpaca_secret(self, monkeypatch):
        out = _capture(monkeypatch, "ALPACA_SECRET", "alpaca_xyz", "secret=%s", "alpaca_xyz")
        assert "alpaca_xyz" not in out

    def test_masks_anthropic_key_in_args(self, monkeypatch):
        out = _capture(monkeypatch, "ANTHROPIC_API_KEY", "claude_key", "calling %s", "claude_key")
        assert "claude_key" not in out
        assert "<REDACTED>" in out

    def test_plain_message_passes_through(self, monkeypatch):
        monkeypatch.delenv("FMP_API_KEY", raising=False)
        out = _capture(monkeypatch, "FMP_API_KEY", "", "hello world")
        assert "hello world" in out

    def test_mask_disabled_leaks(self, monkeypatch):
        """Documents the opt-out risk: mask_env=False allows secrets through."""
        secret = "plaintext_risk_demo_value"
        monkeypatch.setenv("FMP_API_KEY", secret)
        buf = io.StringIO()
        root = logging.getLogger()
        saved_handlers, saved_level = root.handlers[:], root.level
        root.handlers = []
        try:
            configure_logging(level=logging.DEBUG, stream=buf, mask_env=False)
            logging.getLogger("test.nomask").info("key=%s", secret)
            assert secret in buf.getvalue()
        finally:
            root.handlers = saved_handlers
            root.setLevel(saved_level)


# ── configure_logging() invariants ────────────────────────────────────────────

class TestConfigureLogging:
    def test_does_not_stack_handlers_on_repeated_calls(self):
        """Repeated configure_logging() calls replace the first handler."""
        root = logging.getLogger()
        n_before = len(root.handlers)
        configure_logging()
        configure_logging()
        configure_logging()
        assert len(root.handlers) <= n_before + 1

    def test_returns_none(self):
        assert configure_logging() is None

    def test_installs_secret_mask_filter_by_default(self):
        root = logging.getLogger()
        configure_logging()
        handler = root.handlers[0]
        filter_types = [type(f) for f in handler.filters]
        assert SecretMaskFilter in filter_types

    def test_no_filter_when_mask_env_false(self):
        root = logging.getLogger()
        configure_logging(mask_env=False)
        handler = root.handlers[0]
        filter_types = [type(f) for f in handler.filters]
        assert SecretMaskFilter not in filter_types


# ── get_logger() ──────────────────────────────────────────────────────────────

def test_get_logger_returns_named_logger():
    from regime_trader.utils.logging_cfg import get_logger

    a = get_logger("regime_trader.tests.a")
    b = get_logger("regime_trader.tests.b")
    assert a.name == "regime_trader.tests.a"
    assert b.name == "regime_trader.tests.b"
    assert a is not b
