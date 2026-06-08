"""tests/test_metrics_and_alerts.py — exercise metrics_exporter + discord_notifier.

Markowitz frame: the canary's coverage ratio is the share of the target asset
universe that EDGAR returns — analogous to the realised diversification of a
target portfolio. Metric integrity is therefore the precondition for any
risk-aware decision downstream.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from monitoring import metrics_exporter as me
from monitoring.slack_notifier import send_discord_alert as send_slack_alert


# ── metrics_exporter.export_metrics ──────────────────────────────────────────

def _write_status(log_dir: Path, meta: dict) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    p = log_dir / "intel_source_status.json"
    p.write_text(json.dumps({"_edgar_meta": meta}), encoding="utf-8")
    return p


def test_export_metrics_writes_six_canonical_keys(tmp_path: Path) -> None:
    """Markowitz: the canary contract is a six-field signature; any extra or
    missing key invalidates downstream gates."""
    _write_status(tmp_path, {
        "last_run":             "2026-05-06T00:00:00+00:00",
        "run_duration_seconds": 12.34,
        "ticker_count":         10,
        "edgar_count":          7,
        "fmp_count":            3,
        "error_count":          0,
    })
    metrics = me.export_metrics(tmp_path)
    assert set(metrics) == {
        "last_run", "run_duration_seconds",
        "ticker_count", "edgar_count", "fmp_count", "error_count",
    }
    assert metrics["ticker_count"] == 10
    assert metrics["edgar_count"]  == 7
    assert metrics["fmp_count"]    == 3
    assert metrics["run_duration_seconds"] == 12.34
    assert (tmp_path / "metrics.json").exists()


def test_export_metrics_duration_override_wins(tmp_path: Path) -> None:
    """Kahneman: when CI measures wall-clock externally, prefer the observed
    value over the self-reported one — observation dominates self-report."""
    _write_status(tmp_path, {"run_duration_seconds": 1.0, "ticker_count": 5})
    metrics = me.export_metrics(tmp_path, duration_override_s=42.5)
    assert metrics["run_duration_seconds"] == 42.5


def test_export_metrics_missing_status_writes_tombstone(tmp_path: Path) -> None:
    """When intel_source_status.json is absent (pipeline aborted), export_metrics writes
    a tombstone metrics.json with pipeline_failed=True so the if:always() CI step
    doesn't cascade-fail on top of the real pipeline failure."""
    result = me.export_metrics(tmp_path)
    assert result["pipeline_failed"] is True
    assert result["ticker_count"] == 0
    assert (tmp_path / "metrics.json").exists()


def test_export_metrics_missing_meta_yields_zeros(tmp_path: Path) -> None:
    """Fama: when `_edgar_meta` is absent, the contract still holds — counts
    default to zero so the threshold gate naturally fails (low coverage)."""
    (tmp_path / "intel_source_status.json").write_text("{}", encoding="utf-8")
    metrics = me.export_metrics(tmp_path)
    assert metrics["ticker_count"] == 0
    assert metrics["edgar_count"]  == 0
    assert metrics["error_count"]  == 0


# ── slack_notifier.send_discord_alert ────────────────────────────────────────

class _FakeResp:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


def test_send_slack_empty_webhook_is_noop_returns_false() -> None:
    """Kahneman: a missing webhook is a configuration error, not a runtime
    crash — the pipeline must keep flowing (return False, never raise)."""
    assert send_slack_alert("", "title", "body") is False
    assert send_slack_alert(None, "title", "body") is False


def test_send_slack_2xx_returns_true(monkeypatch: pytest.MonkeyPatch) -> None:
    """Markowitz: a single 2xx is sufficient evidence of delivery — no need
    to retry once the channel has acknowledged."""
    calls = {"n": 0}

    def fake_post(url, json=None, timeout=None):  # noqa: A002
        calls["n"] += 1
        return _FakeResp(200)

    monkeypatch.setattr("monitoring.slack_notifier.requests.post", fake_post)
    assert send_slack_alert("https://hooks.slack/x", "t", "b") is True
    assert calls["n"] == 1


def test_send_slack_non_2xx_retries_then_returns_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kahneman: persistent failure must surface as False after the retry
    budget is exhausted — silent suppression would hide real outages."""
    calls = {"n": 0}

    def fake_post(url, json=None, timeout=None):  # noqa: A002
        calls["n"] += 1
        return _FakeResp(500, "boom")

    monkeypatch.setattr("monitoring.slack_notifier.requests.post", fake_post)
    monkeypatch.setattr("monitoring.slack_notifier.time.sleep", lambda *_: None)
    assert send_slack_alert("https://hooks.slack/x", "t", "b", max_retries=3) is False
    assert calls["n"] == 3


def test_send_slack_swallows_network_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kahneman: a thrown ConnectionError is the same loss event as a 5xx —
    treat it identically (False), never propagate."""
    def boom(url, json=None, timeout=None):  # noqa: A002
        raise RuntimeError("network down")

    monkeypatch.setattr("monitoring.slack_notifier.requests.post", boom)
    monkeypatch.setattr("monitoring.slack_notifier.time.sleep", lambda *_: None)
    assert send_slack_alert("https://hooks.slack/x", "t", "b", max_retries=2) is False
