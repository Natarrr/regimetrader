"""tests/test_alert_state.py — Kahneman: escalation after N consecutive losses.

Prospect-theory frame: a single loss is tolerable; a sustained streak shifts
the decision boundary. The canary mirrors this — escalation flips True only
after `ALERT_ESCALATE_AFTER` consecutive failures, and a single success
wipes the streak (canonical reset, not exponential decay).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from monitoring import alert_state as as_mod


@pytest.fixture
def state_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Kahneman: per-test isolation — no streak leaks between tests."""
    p = tmp_path / "alert_state.json"
    monkeypatch.setattr(as_mod, "_STATE_PATH", p)
    return p


def test_initial_call_with_ok_keeps_zero(state_path: Path) -> None:
    """Kahneman: first run, no losses observed → no escalation."""
    decision = as_mod.update_after_evaluation(ok=True)
    assert decision.consecutive_failures == 0
    assert decision.escalate is False


def test_single_failure_does_not_escalate(
    state_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kahneman: one loss is below the default 3-strike escalation threshold."""
    monkeypatch.setenv("ALERT_ESCALATE_AFTER", "3")
    decision = as_mod.update_after_evaluation(ok=False)
    assert decision.consecutive_failures == 1
    assert decision.escalate is False


def test_escalates_at_threshold(
    state_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kahneman: sustained losses (3 consecutive) cross the escalation boundary."""
    monkeypatch.setenv("ALERT_ESCALATE_AFTER", "3")
    d1 = as_mod.update_after_evaluation(ok=False)
    d2 = as_mod.update_after_evaluation(ok=False)
    d3 = as_mod.update_after_evaluation(ok=False)
    assert (d1.escalate, d2.escalate, d3.escalate) == (False, False, True)
    assert d3.consecutive_failures == 3


def test_success_resets_streak(
    state_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kahneman: a single gain wipes the loss memory — fresh start, no escalation."""
    monkeypatch.setenv("ALERT_ESCALATE_AFTER", "3")
    as_mod.update_after_evaluation(ok=False)
    as_mod.update_after_evaluation(ok=False)
    decision = as_mod.update_after_evaluation(ok=True)
    assert decision.consecutive_failures == 0
    assert decision.escalate is False
    state = as_mod.load_state()
    assert state["last_status"] == "ok"


def test_state_persists_across_calls(
    state_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kahneman: state file is the source of truth — survives process restart."""
    monkeypatch.setenv("ALERT_ESCALATE_AFTER", "10")
    as_mod.update_after_evaluation(ok=False)
    as_mod.update_after_evaluation(ok=False)
    state = as_mod.load_state()
    assert state["consecutive_failures"] == 2
    assert state["last_status"] == "fail"


def test_check_metrics_passes_escalate_true_at_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kahneman: at the streak boundary, check_metrics flips Slack escalate=True."""
    from monitoring import check_metrics as cm

    monkeypatch.setattr(as_mod, "_STATE_PATH", tmp_path / "alert_state.json")
    monkeypatch.setenv("ALERT_ESCALATE_AFTER", "2")
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack/x")

    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "metrics.json").write_text(json.dumps({
        "last_run":             "2026-05-06T00:00:00+00:00",
        "run_duration_seconds": 1.0,
        "ticker_count":         10,
        "edgar_count":          3,   # 30% coverage → fails default 60% gate
        "fmp_count":            0,
        "error_count":          0,
    }), encoding="utf-8")

    captured: list[bool] = []

    def fake_slack(webhook, title, body, *, escalate=False, **_kwargs):
        captured.append(escalate)
        return True

    monkeypatch.setattr(cm, "send_slack_alert", fake_slack)

    rc1 = cm.main(["--log-dir", str(log_dir)])
    rc2 = cm.main(["--log-dir", str(log_dir)])

    assert rc1 == 2 and rc2 == 2
    assert captured == [False, True]


def test_check_metrics_resets_after_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Kahneman: a passing run between failures must wipe the streak."""
    from monitoring import check_metrics as cm

    monkeypatch.setattr(as_mod, "_STATE_PATH", tmp_path / "alert_state.json")
    monkeypatch.setenv("ALERT_ESCALATE_AFTER", "2")
    monkeypatch.setenv("SLACK_WEBHOOK_URL", "https://hooks.slack/x")

    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    metrics_p = log_dir / "metrics.json"

    captured: list[bool] = []

    def fake_slack(webhook, title, body, *, escalate=False, **_kwargs):
        captured.append(escalate)
        return True

    monkeypatch.setattr(cm, "send_slack_alert", fake_slack)

    fail = json.dumps({
        "last_run": "x", "run_duration_seconds": 1.0,
        "ticker_count": 10, "edgar_count": 3, "fmp_count": 0, "error_count": 0,
    })
    ok = json.dumps({
        "last_run": "x", "run_duration_seconds": 1.0,
        "ticker_count": 10, "edgar_count": 9, "fmp_count": 1, "error_count": 0,
    })

    metrics_p.write_text(fail, encoding="utf-8"); cm.main(["--log-dir", str(log_dir)])
    metrics_p.write_text(ok,   encoding="utf-8"); cm.main(["--log-dir", str(log_dir)])
    metrics_p.write_text(fail, encoding="utf-8"); cm.main(["--log-dir", str(log_dir)])

    # 1st failure → False; success in between → no slack call (rc=0);
    # post-success failure should be back to False (streak reset).
    assert captured == [False, False]
    assert as_mod.load_state()["consecutive_failures"] == 1
