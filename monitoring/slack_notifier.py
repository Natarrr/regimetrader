"""monitoring/slack_notifier.py — Discord webhook sender with retry/backoff.

Stdlib + requests only. Returns True on 2xx, False otherwise. Never raises.

Usage:
    from monitoring.slack_notifier import send_discord_alert
    ok = send_discord_alert(
        webhook=os.environ["DISCORD_WEBHOOK_URL"],
        title="EDGAR canary alert",
        body="Coverage ratio 0.42 < 0.60 threshold",
    )
"""
from __future__ import annotations

import logging
import time
from typing import Optional

try:
    import requests  # type: ignore
    _HAS_REQUESTS = True
except ImportError:
    requests = None  # type: ignore
    _HAS_REQUESTS = False

log = logging.getLogger("monitoring.discord")

# Discord embed colors
_COLOR_ALERT = 0xE74C3C      # red
_COLOR_ESCALATE = 0xFF0000   # bright red


def send_discord_alert(
    webhook: Optional[str],
    title: str,
    body: str,
    *,
    escalate: bool = False,
    max_retries: int = 3,
    timeout_s: float = 10.0,
    backoff_base_s: float = 0.5,
) -> bool:
    """Post a Discord embed message with exponential-backoff retry.

    Args:
        webhook:        Discord incoming webhook URL. Empty / None → no-op False.
        title:          Embed title line.
        body:           Embed description (Discord markdown supported).
        escalate:       When True, uses bright-red color and [ESCALATE] prefix.
        max_retries:    Total attempts including the first (default 3).
        timeout_s:      Per-request timeout.
        backoff_base_s: First retry waits this; subsequent waits double.

    Returns:
        True on Discord 2xx, False on every other outcome (never raises).
    """
    if not webhook:
        log.warning("send_discord_alert: webhook is empty — skipping (no-op)")
        return False
    if not _HAS_REQUESTS:
        log.error("send_discord_alert: 'requests' not installed — cannot post")
        return False

    display_title = f"🚨 [ESCALATE] {title}" if escalate else title
    color = _COLOR_ESCALATE if escalate else _COLOR_ALERT
    payload = {
        "embeds": [{
            "title": display_title,
            "description": body,
            "color": color,
        }]
    }

    for attempt in range(max_retries):
        if attempt > 0:
            wait = backoff_base_s * (2 ** (attempt - 1))   # 0.5, 1.0, 2.0 …
            time.sleep(wait)
        try:
            resp = requests.post(webhook, json=payload, timeout=timeout_s)
            if 200 <= resp.status_code < 300:
                return True
            log.warning("discord non-2xx status=%d body=%r attempt=%d/%d",
                        resp.status_code, resp.text[:200], attempt + 1, max_retries)
        except Exception as exc:
            log.warning("discord post failed attempt=%d/%d err=%s",
                        attempt + 1, max_retries, str(exc)[:200])
    return False


# Backward-compat alias — existing monkeypatches targeting send_slack_alert still work
send_slack_alert = send_discord_alert
