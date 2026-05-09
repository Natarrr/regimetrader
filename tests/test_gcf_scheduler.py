"""tests/test_gcf_scheduler.py
Unit tests for cloud/scheduler/gcf_scheduler.py.

Covers:
  - _now_london(): returns Europe/London-aware datetime
  - _dst_info():   correct BST vs GMT detection
  - _within_window(): tolerance guard logic
  - _mask_token(): safe logging of credentials
  - _dispatch_workflow(): retry logic, 204/401/422/timeout handling
  - trigger_daily_discord(): HTTP handler — env validation, dispatch wiring, response codes
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ── Ensure project root on path ───────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ── Stub functions_framework before importing gcf_scheduler ───────────────────
_ff = types.ModuleType("functions_framework")
_ff.http = lambda fn: fn  # pass-through decorator
sys.modules.setdefault("functions_framework", _ff)

from cloud.scheduler.gcf_scheduler import (  # noqa: E402
    _dispatch_workflow,
    _dst_info,
    _mask_token,
    _now_london,
    _within_window,
    trigger_daily_discord,
)
from zoneinfo import ZoneInfo

_LONDON_TZ = ZoneInfo("Europe/London")


# ── _now_london ───────────────────────────────────────────────────────────────

class TestNowLondon:
    def test_returns_london_aware(self):
        dt = _now_london()
        assert dt.tzinfo is not None
        # ZoneInfo("Europe/London") key
        assert "London" in str(dt.tzinfo)

    def test_offset_is_0_or_1(self):
        dt = _now_london()
        hours = dt.utcoffset().total_seconds() / 3600
        assert hours in (0, 1), f"Unexpected UTC offset: {hours}"


# ── _dst_info ─────────────────────────────────────────────────────────────────

class TestDstInfo:
    def _make_dt(self, iso: str) -> datetime:
        return datetime.fromisoformat(iso).astimezone(_LONDON_TZ)

    def test_bst_july(self):
        dt = self._make_dt("2026-07-01T13:00:00+00:00")
        is_dst, hours = _dst_info(dt)
        assert is_dst is True
        assert hours == 1

    def test_gmt_january(self):
        dt = self._make_dt("2026-01-15T14:00:00+00:00")
        is_dst, hours = _dst_info(dt)
        assert is_dst is False
        assert hours == 0


# ── _within_window ────────────────────────────────────────────────────────────

class TestWithinWindow:
    def _dt(self, hour: int, minute: int) -> datetime:
        return datetime(2026, 7, 1, hour, minute, 0, tzinfo=_LONDON_TZ)

    def test_exactly_14_00(self):
        assert _within_window(self._dt(14, 0), 30) is True

    def test_13_31_inside_30min(self):
        assert _within_window(self._dt(13, 31), 30) is True

    def test_13_29_outside_30min(self):
        assert _within_window(self._dt(13, 29), 30) is False

    def test_14_29_inside_30min(self):
        assert _within_window(self._dt(14, 29), 30) is True

    def test_14_31_outside_30min(self):
        assert _within_window(self._dt(14, 31), 30) is False

    def test_zero_tolerance_always_true(self):
        assert _within_window(self._dt(9, 0), 0) is True
        assert _within_window(self._dt(23, 59), 0) is True


# ── _mask_token ───────────────────────────────────────────────────────────────

class TestMaskToken:
    def test_shows_first_four(self):
        result = _mask_token("ghp_ABCD1234")
        assert result.startswith("ghp_")
        assert "1234" not in result

    def test_short_token_masked(self):
        assert _mask_token("abc") == "***"

    def test_empty_masked(self):
        assert _mask_token("") == "***"


# ── _dispatch_workflow ────────────────────────────────────────────────────────

class TestDispatchWorkflow:
    def _mock_resp(self, status: int, body: str = "") -> MagicMock:
        r = MagicMock()
        r.status_code = status
        r.text = body
        r.headers = {}
        return r

    def test_204_success(self):
        with patch("cloud.scheduler.gcf_scheduler.requests.post") as mock_post:
            mock_post.return_value = self._mock_resp(204)
            ok, code, msg = _dispatch_workflow("owner/repo", "token")
        assert ok is True
        assert code == 204

    def test_401_no_retry(self):
        with patch("cloud.scheduler.gcf_scheduler.requests.post") as mock_post:
            mock_post.return_value = self._mock_resp(401)
            ok, code, msg = _dispatch_workflow("owner/repo", "token")
        assert ok is False
        assert code == 401
        assert mock_post.call_count == 1  # no retry on auth failure

    def test_422_no_retry(self):
        with patch("cloud.scheduler.gcf_scheduler.requests.post") as mock_post:
            mock_post.return_value = self._mock_resp(422, '{"message":"Bad ref"}')
            ok, code, msg = _dispatch_workflow("owner/repo", "token")
        assert ok is False
        assert code == 422
        assert mock_post.call_count == 1

    def test_500_retries_three_times(self):
        with patch("cloud.scheduler.gcf_scheduler.requests.post") as mock_post, \
             patch("cloud.scheduler.gcf_scheduler.time.sleep"):
            mock_post.return_value = self._mock_resp(500, "server error")
            ok, code, msg = _dispatch_workflow("owner/repo", "token")
        assert ok is False
        assert mock_post.call_count == 3

    def test_timeout_retries(self):
        import requests as req_lib
        with patch("cloud.scheduler.gcf_scheduler.requests.post",
                   side_effect=req_lib.exceptions.Timeout), \
             patch("cloud.scheduler.gcf_scheduler.time.sleep"):
            ok, code, msg = _dispatch_workflow("owner/repo", "token")
        assert ok is False
        assert "retries" in msg

    def test_success_on_second_attempt(self):
        with patch("cloud.scheduler.gcf_scheduler.requests.post") as mock_post, \
             patch("cloud.scheduler.gcf_scheduler.time.sleep"):
            mock_post.side_effect = [
                self._mock_resp(500, "error"),
                self._mock_resp(204),
            ]
            ok, code, _ = _dispatch_workflow("owner/repo", "token")
        assert ok is True
        assert mock_post.call_count == 2


# ── trigger_daily_discord (HTTP handler) ──────────────────────────────────────

class TestTriggerDailyDiscord:
    """Tests for the Cloud Function HTTP entry point."""

    def _call(self, env: dict, dispatch_result=(True, 204, "dispatched")):
        """Call trigger_daily_discord with a mock request and env."""
        mock_request = MagicMock()
        with patch.dict(os.environ, env, clear=False), \
             patch("cloud.scheduler.gcf_scheduler._dispatch_workflow",
                   return_value=dispatch_result):
            result = trigger_daily_discord(mock_request)
        return result

    def test_missing_token_returns_500(self):
        env = {"GITHUB_REPO": "owner/repo", "GITHUB_TOKEN_SCHEDULER": ""}
        body, code, _ = self._call(env)
        assert code == 500
        data = json.loads(body)
        assert data["status"] == "error"
        assert "GITHUB_TOKEN_SCHEDULER" in data["detail"]

    def test_missing_repo_returns_500(self):
        env = {"GITHUB_TOKEN_SCHEDULER": "ghp_abc", "GITHUB_REPO": ""}
        body, code, _ = self._call(env)
        assert code == 500
        data = json.loads(body)
        assert "GITHUB_REPO" in data["detail"]

    def test_successful_dispatch_returns_200(self):
        env = {"GITHUB_TOKEN_SCHEDULER": "ghp_abc", "GITHUB_REPO": "owner/repo"}
        body, code, _ = self._call(env)
        assert code == 200
        data = json.loads(body)
        assert data["status"] == "ok"
        assert "london_time" in data
        assert "dst" in data

    def test_dispatch_failure_returns_500(self):
        env = {"GITHUB_TOKEN_SCHEDULER": "ghp_abc", "GITHUB_REPO": "owner/repo"}
        body, code, _ = self._call(env, dispatch_result=(False, 500, "retries exhausted"))
        assert code == 500
        data = json.loads(body)
        assert data["status"] == "error"

    def test_dispatch_401_returns_400(self):
        env = {"GITHUB_TOKEN_SCHEDULER": "ghp_abc", "GITHUB_REPO": "owner/repo"}
        body, code, _ = self._call(env, dispatch_result=(False, 401, "unauthorized"))
        assert code == 400

    def test_response_includes_dst_info(self):
        env = {"GITHUB_TOKEN_SCHEDULER": "ghp_abc", "GITHUB_REPO": "owner/repo"}
        body, code, _ = self._call(env)
        data = json.loads(body)
        assert isinstance(data["dst"], bool)
        assert data["utc_offset_h"] in (0, 1)
