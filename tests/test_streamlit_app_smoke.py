"""tests/test_streamlit_app_smoke.py
Smoke test for the Streamlit orchestrator.

Goals:
  1. The module imports cleanly (no syntax / dependency regressions).
  2. The data path used by the Market Intel tab — get_top_alpha_picks_sync()
     — returns a dict with the expected keys when wired against a stubbed
     discovery_scanner. We never hit the network.
  3. configure_logging() never logs environment variables or secrets.

Stiglitz (2001 Nobel) — Information asymmetry: secrets in logs are a one-way
information leak from the application to its operators / attackers. Test the
boundary, don't trust it.
"""
from __future__ import annotations

import io
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch


ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── 1. Import smoke ───────────────────────────────────────────────────────────


def test_streamlit_app_imports():
    """The orchestrator imports without side-effects beyond Streamlit init."""
    import regime_trader.ui.streamlit_app as app  # noqa: F401

    assert app is not None


def test_discovery_scanner_imports():
    import regime_trader.scanners.discovery_scanner as ds

    for fn_name in (
        "get_top_alpha_picks_sync",
        "force_refresh_sync",
        "fmp_screener",
        "fmp_insider_buys",
        "select_candidates",
        "enrich_with_momentum",
        "save_disc_cache",
        "load_disc_cache",
    ):
        assert callable(getattr(ds, fn_name)), f"{fn_name} not exposed"


def test_market_intel_macro_imports():
    import regime_trader.scanners.market_intel_macro as mim

    assert mim is not None


# ── 2. Data path: get_top_alpha_picks_sync with stubbed network ───────────────


_STUB_RESULTS: List[Dict[str, Any]] = [
    {
        "symbol": "ACME",
        "smart_money_score": 0.91,
        "insider_score": 1.0,
        "institutional_score": 0.6,
        "momentum_score": 0.5,
        "rationale": "stub",
    }
]


def test_get_top_alpha_picks_returns_expected_shape(tmp_path, monkeypatch):
    """The Market Intel tab calls get_top_alpha_picks_sync(); shape must be stable."""
    import regime_trader.scanners.discovery_scanner as ds

    # Force cache miss by pointing to a temp directory that doesn't have a cache file.
    monkeypatch.setattr(ds, "_DISC_CACHE_FILE", tmp_path / "disc_cache.json", raising=False)

    fake_payload = {
        "results": _STUB_RESULTS,
        "ts": 0,
        "regime": "Neutral",
    }
    with patch.object(ds, "run_scan", return_value=[]):
        # When run_scan returns [], the function should still return a dict
        # with results / ts / regime keys (graceful degradation path).
        with patch.object(ds, "save_disc_cache"):
            result = ds.get_top_alpha_picks_sync(limit=5)

    assert isinstance(result, dict)
    # Public contract: at minimum a 'results' list must be present.
    assert "results" in result
    assert isinstance(result["results"], list)


# ── 3. No-secrets-in-logs ─────────────────────────────────────────────────────


_SECRET_NEEDLE = "DECOY_SECRET_VALUE_DO_NOT_LOG_4f1a2b"


def test_configure_logging_does_not_emit_environment(monkeypatch):
    """
    configure_logging() must not iterate / dump os.environ — a regression that
    would expose API keys in the application log stream.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", _SECRET_NEEDLE)
    monkeypatch.setenv("FMP_API_KEY", _SECRET_NEEDLE)

    buffer = io.StringIO()

    # Reset root handlers so configure_logging starts fresh.
    root = logging.getLogger()
    saved_handlers = root.handlers[:]
    saved_level = root.level
    root.handlers = []

    try:
        from regime_trader.utils.logging_cfg import configure_logging

        configure_logging(level=logging.DEBUG, stream=buffer)

        # configure_logging itself emits nothing — it just configures a handler.
        # Now log a normal message and confirm nothing leaks.
        log = logging.getLogger("regime_trader.tests.smoke")
        log.info("regime check complete")

        captured = buffer.getvalue()
        assert _SECRET_NEEDLE not in captured, (
            "configure_logging or its initial messages leaked an env-var value"
        )
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)


def test_default_log_format_does_not_include_env(monkeypatch):
    """
    The default formatter must not interpolate environment variables.
    Sanity: emit a dummy log and verify the formatter doesn't expand $VAR.
    """
    monkeypatch.setenv("FMP_API_KEY", _SECRET_NEEDLE)

    buffer = io.StringIO()
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    root.handlers = []

    try:
        from regime_trader.utils.logging_cfg import configure_logging

        configure_logging(level=logging.INFO, stream=buffer)
        log = logging.getLogger("regime_trader.tests.fmt")
        log.info("checking format string $FMP_API_KEY")  # literal, must not expand

        out = buffer.getvalue()
        assert _SECRET_NEEDLE not in out
        # The literal token should pass through untouched
        assert "$FMP_API_KEY" in out
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)


def test_get_logger_returns_distinct_named_logger():
    from regime_trader.utils.logging_cfg import get_logger

    a = get_logger("regime_trader.tests.a")
    b = get_logger("regime_trader.tests.b")
    assert a.name == "regime_trader.tests.a"
    assert b.name == "regime_trader.tests.b"
    assert a is not b


# ── 4. Atomic cache round-trip (defence in depth — already covered elsewhere) ──


def test_save_load_disc_cache_round_trip(tmp_path, monkeypatch):
    """save_disc_cache must write atomically; load_disc_cache must read it back
    when the payload is well-formed (TTL not expired)."""
    import time

    import regime_trader.scanners.discovery_scanner as ds

    cache_file = tmp_path / "cache.json"
    monkeypatch.setattr(ds, "_DISC_CACHE_FILE", cache_file, raising=False)

    # Realistic payload: must include _expires_at in the future or load rejects it.
    payload = {
        "results": [{"symbol": "TEST"}],
        "regime": "Bull",
        "_expires_at": time.time() + 600,
    }
    ds.save_disc_cache(payload)

    assert cache_file.exists(), "atomic save did not produce the target file"
    loaded = ds.load_disc_cache()
    assert loaded is not None, "load_disc_cache returned None for a fresh entry"
    assert loaded["results"] == payload["results"]
    assert loaded["regime"] == "Bull"
    # Internal _expires_at should be stripped from the public-facing dict
    assert not any(k.startswith("_") for k in loaded.keys())


def test_save_disc_cache_is_atomic(tmp_path, monkeypatch):
    """Atomic write: no partial / temp files left behind on success."""
    import time

    import regime_trader.scanners.discovery_scanner as ds

    cache_file = tmp_path / "cache.json"
    monkeypatch.setattr(ds, "_DISC_CACHE_FILE", cache_file, raising=False)

    payload = {"results": [], "_expires_at": time.time() + 60}
    ds.save_disc_cache(payload)

    leftovers = [p for p in tmp_path.iterdir() if p.name != "cache.json"]
    assert leftovers == [], f"atomic write left temp files behind: {leftovers}"
