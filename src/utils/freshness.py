# Path: src/utils/freshness.py
"""FMP data freshness helpers — wall-clock validation of returned timestamps.

The FMP client is a thin wrapper (CLAUDE.md §2: zero trading math inside it), so
freshness *policy* lives here and is applied by consumers (src/ingestion). These
functions answer one question each and take an explicit ``now`` so the
wall-clock comparison is deterministic and testable.

Backs audit findings:
  F1 — quote `timestamp` was never compared to the clock (market.py).
  F2 — historical-series recency (`dates[-1]`) was never checked (fmp_fetcher.py).
"""
from __future__ import annotations

import logging
from datetime import datetime, time, timezone
from typing import Any, Optional

log = logging.getLogger(__name__)

_RTH_OPEN = time(9, 30)
_RTH_CLOSE = time(16, 0)


def _ny_now(now: Optional[datetime] = None) -> Optional[datetime]:
    """Return *now* in America/New_York, or None if the tz database is absent.

    Windows ships no IANA database; the runtime relies on the stdlib `tzdata`
    fallback (present in CI and confirmed locally). If the zone cannot be
    resolved we return None and callers treat the session as indeterminate.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    try:
        from zoneinfo import ZoneInfo  # noqa: PLC0415 — optional tzdata dependency
        return now.astimezone(ZoneInfo("America/New_York"))
    except Exception as exc:  # ZoneInfoNotFoundError or import failure
        log.warning("freshness: America/New_York unavailable (%s)", exc)
        return None


def is_us_rth(now: Optional[datetime] = None) -> bool:
    """True when *now* falls in US regular trading hours (Mon–Fri 09:30–16:00 ET).

    Half-open interval [09:30, 16:00). No US market-holiday calendar — a holiday
    reads as "open", which at worst flags a (legitimately stale) quote rather
    than wrongly rejecting one, since holidays produce no fresh quotes anyway.
    Returns False if the Eastern timezone cannot be resolved (fail-safe: never
    hard-reject on a clock we cannot localize).
    """
    ny = _ny_now(now)
    if ny is None:
        return False
    if ny.weekday() >= 5:  # Sat/Sun
        return False
    return _RTH_OPEN <= ny.time() < _RTH_CLOSE


def quote_age_seconds(
    quote_row: dict[str, Any], now: Optional[datetime] = None
) -> Optional[float]:
    """Age in seconds of an FMP quote, from its epoch `timestamp` field.

    None when the field is absent or unparseable — absence of a timestamp is not
    evidence of staleness (CLAUDE.md §2), so the caller decides how to treat it.
    """
    ts = quote_row.get("timestamp")
    if ts is None:
        return None
    try:
        ts_f = float(ts)
    except (TypeError, ValueError):
        return None
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.timestamp() - ts_f


def series_age_days(
    latest_date_str: str, now: Optional[datetime] = None
) -> Optional[int]:
    """Calendar-day age of the newest EOD bar (`dates[-1]`), or None if unparseable.

    Accepts "YYYY-MM-DD" or any ISO string whose first 10 chars are the date.
    """
    if not latest_date_str:
        return None
    try:
        bar_date = datetime.fromisoformat(str(latest_date_str)[:10]).date()
    except ValueError:
        return None
    if now is None:
        now = datetime.now(timezone.utc)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return (now.date() - bar_date).days


def is_us_listing(ticker: str) -> bool:
    """True for US listings (no exchange suffix). EU/Asia carry a dotted suffix
    (ASML.AS, 7203.T) and are governed by their own session, so the US-RTH
    staleness reject must not apply to them (see fmp_fetcher F1 gate)."""
    return bool(ticker) and "." not in ticker
