"""monitoring/alert_state.py — Kahneman: track consecutive canary failures.

Kahneman (2002 Nobel) prospect theory: agents weight repeated losses
super-linearly. The canary mirrors this — a single transient failure emits a
normal Slack alert, but ALERT_ESCALATE_AFTER consecutive failures escalate
to a louder one (paged operator, separate channel, etc.).

State file: .monitoring/alert_state.json
    {
        "consecutive_failures": int,
        "last_status":          "ok" | "fail",
        "last_run_ts":          float (epoch seconds)
    }

Pure-ish module — load/save are I/O, but `update_after_evaluation` is the
only public mutator and returns a NamedTuple so callers can assert on the
decision in tests without re-reading the file.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, NamedTuple

from src.utils.io import atomic_write_json

_STATE_PATH = Path(".monitoring/alert_state.json")


class AlertDecision(NamedTuple):
    consecutive_failures: int
    escalate: bool


def _default_state() -> Dict[str, Any]:
    return {"consecutive_failures": 0, "last_status": "ok", "last_run_ts": 0.0}


def load_state() -> Dict[str, Any]:
    """Kahneman: read persisted state; default to a clean slate on any error."""
    if not _STATE_PATH.exists():
        return _default_state()
    try:
        return json.loads(_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return _default_state()


def save_state(d: Dict[str, Any]) -> None:
    """Kahneman: persist state atomically — concurrent canary jobs must not race."""
    atomic_write_json(_STATE_PATH, d)


def update_after_evaluation(ok: bool) -> AlertDecision:
    """Kahneman: increment on loss, reset on gain — return the escalation decision.

    `escalate` is True iff (ok is False) AND consecutive_failures has reached
    the ALERT_ESCALATE_AFTER threshold (default 3). The caller is responsible
    for actually sending Slack — this function is pure decision + state.
    """
    state = load_state()
    threshold = int(os.getenv("ALERT_ESCALATE_AFTER", "3"))

    if ok:
        save_state({
            "consecutive_failures": 0,
            "last_status":          "ok",
            "last_run_ts":          time.time(),
        })
        return AlertDecision(0, False)

    fc = int(state.get("consecutive_failures", 0)) + 1
    save_state({
        "consecutive_failures": fc,
        "last_status":          "fail",
        "last_run_ts":          time.time(),
    })
    return AlertDecision(fc, fc >= threshold)
