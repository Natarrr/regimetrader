"""cloud/scheduler/gcf_scheduler.py
DST-aware Google Cloud Function that dispatches the daily_toplists_discord
GitHub Actions workflow at exactly 14:00 Europe/London, year-round.

Deployment: see infra/gcf_deploy.sh
Entry point: trigger_daily_discord

Environment variables (injected via --set-secrets in deploy script):
  GITHUB_TOKEN_SCHEDULER   — GitHub PAT with `workflow` scope (from Secret Manager)
  GITHUB_REPO              — owner/repo, e.g. "Natarrr/regimetrader"
  TIME_TOLERANCE_MINUTES   — how many minutes either side of 14:00 are accepted
                             (default: 30; set to 0 to disable guard entirely)

Request contract:
  Cloud Scheduler POSTs with OIDC authentication.
  Body: ignored (idempotency guaranteed by Cloud Scheduler's at-least-once delivery).
  Response: JSON  {"status": "ok"|"skipped"|"error", "detail": "..."}

Retry policy:
  3 attempts; 5 s, 10 s linear backoff.
  Cloud Scheduler itself retries on HTTP 5xx or timeout — overlap is fine
  because GitHub Actions `workflow_dispatch` is idempotent within a run window.
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo  # type: ignore[no-redef]

import functions_framework
import requests

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("gcf_scheduler")

# ── Constants ─────────────────────────────────────────────────────────────────
_LONDON_TZ      = ZoneInfo("Europe/London")
_TARGET_HOUR    = 14
_TARGET_MINUTE  = 0
_WORKFLOW_FILE  = "daily_toplists_discord.yml"
_API_VERSION    = "2022-11-28"
_MAX_RETRIES    = 3
_BACKOFF_BASE_S = 5.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_london() -> datetime:
    """Return current UTC moment converted to Europe/London (BST or GMT)."""
    return datetime.now(timezone.utc).astimezone(_LONDON_TZ)


def _dst_info(dt: datetime) -> Tuple[bool, int]:
    """Return (is_dst, utc_offset_hours) for a London datetime."""
    utc_offset = dt.utcoffset()
    hours = int(utc_offset.total_seconds() / 3600) if utc_offset else 0
    is_dst = hours == 1
    return is_dst, hours


def _within_window(dt: datetime, tolerance_min: int) -> bool:
    """True if *dt* is within ±tolerance_min of 14:00 London."""
    if tolerance_min <= 0:
        return True
    target = dt.replace(hour=_TARGET_HOUR, minute=_TARGET_MINUTE, second=0, microsecond=0)
    delta  = abs((dt - target).total_seconds())
    return delta <= tolerance_min * 60


def _mask_token(token: str) -> str:
    """Return first 4 + masked remainder for safe logging."""
    if not token or len(token) < 8:
        return "***"
    return token[:4] + "*" * (len(token) - 4)


# ── GitHub dispatch ───────────────────────────────────────────────────────────

def _dispatch_workflow(
    repo: str,
    token: str,
    ref: str = "main",
    inputs: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, int, str]:
    """POST workflow_dispatch to GitHub API.

    Returns (success, http_status, message).
    Never raises — catches all exceptions.
    """
    url = f"https://api.github.com/repos/{repo}/actions/workflows/{_WORKFLOW_FILE}/dispatches"
    headers = {
        "Accept":               "application/vnd.github+json",
        "Authorization":        f"Bearer {token}",
        "X-GitHub-Api-Version": _API_VERSION,
    }
    body: Dict[str, Any] = {"ref": ref}
    if inputs:
        body["inputs"] = inputs

    for attempt in range(_MAX_RETRIES):
        if attempt > 0:
            wait = _BACKOFF_BASE_S * attempt
            log.warning("GitHub dispatch retry %d/%d in %.0fs", attempt + 1, _MAX_RETRIES, wait)
            time.sleep(wait)
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=15.0)
            if resp.status_code == 204:
                log.info(
                    "Dispatched %s on %s/%s (attempt=%d)",
                    _WORKFLOW_FILE, repo, ref, attempt + 1,
                )
                return True, 204, "dispatched"
            if resp.status_code == 422:
                msg = f"422 Unprocessable — ref '{ref}' may not exist or workflow not found"
                log.error(msg)
                return False, 422, msg
            if resp.status_code == 401:
                log.error("401 Unauthorized — check GITHUB_TOKEN_SCHEDULER scope (needs 'workflow')")
                return False, 401, "unauthorized"
            log.warning(
                "Unexpected status=%d body=%r (attempt=%d)",
                resp.status_code, resp.text[:200], attempt + 1,
            )
        except requests.exceptions.Timeout:
            log.warning("GitHub API timeout (attempt=%d)", attempt + 1)
        except Exception as exc:  # noqa: BLE001
            log.warning("GitHub API error (attempt=%d): %s", attempt + 1, str(exc)[:200])

    return False, 0, "all retries exhausted"


# ── Cloud Function entry point ────────────────────────────────────────────────

@functions_framework.http
def trigger_daily_discord(request):
    """HTTP Cloud Function — dispatches daily_toplists_discord at 14:00 London.

    Cloud Scheduler fires this every day at 14:00 Europe/London.
    A secondary DST guard logs a warning if it fires outside the ±30 min window
    (catches misconfigured schedulers but does NOT block — Cloud Scheduler's
    timezone support is the primary mechanism).
    """
    now_london = _now_london()
    is_dst, utc_offset_h = _dst_info(now_london)
    tolerance_min = int(os.environ.get("TIME_TOLERANCE_MINUTES", "30"))

    log.info(
        "Triggered at London=%s  DST=%s  UTC+%d",
        now_london.strftime("%Y-%m-%dT%H:%M:%S%z"),
        is_dst,
        utc_offset_h,
    )

    # Secondary time guard — warn only, never block
    if not _within_window(now_london, tolerance_min):
        log.warning(
            "Fired outside ±%d min window of 14:00 London (now=%02d:%02d). "
            "Check Cloud Scheduler timezone setting.",
            tolerance_min,
            now_london.hour,
            now_london.minute,
        )

    # Read secrets from env (mounted via --set-secrets in deploy script)
    token = os.environ.get("GITHUB_TOKEN_SCHEDULER", "")
    repo  = os.environ.get("GITHUB_REPO", "")
    ref   = os.environ.get("GITHUB_REF", "main")

    if not token:
        log.error("GITHUB_TOKEN_SCHEDULER not set")
        return (
            json.dumps({"status": "error", "detail": "GITHUB_TOKEN_SCHEDULER missing"}),
            500,
            {"Content-Type": "application/json"},
        )
    if not repo:
        log.error("GITHUB_REPO not set")
        return (
            json.dumps({"status": "error", "detail": "GITHUB_REPO missing"}),
            500,
            {"Content-Type": "application/json"},
        )

    log.info("Dispatching workflow for repo=%s ref=%s token=%s", repo, ref, _mask_token(token))

    # Build dispatch inputs
    dispatch_inputs: Dict[str, Any] = {
        "source_run_id": f"gcf-{now_london.strftime('%Y%m%dT%H%M')}",
        "dry_run":       "false",
    }

    ok, status_code, detail = _dispatch_workflow(repo, token, ref, dispatch_inputs)

    if ok:
        body = json.dumps({
            "status":        "ok",
            "detail":        detail,
            "london_time":   now_london.isoformat(),
            "dst":           is_dst,
            "utc_offset_h":  utc_offset_h,
        })
        return body, 200, {"Content-Type": "application/json"}

    body = json.dumps({
        "status":     "error",
        "detail":     detail,
        "http_status": status_code,
    })
    http_code = 500 if status_code not in (401, 422) else 400
    return body, http_code, {"Content-Type": "application/json"}
