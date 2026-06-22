# Path: tests/test_freshness.py
"""Unit tests for src/utils/freshness.py — FMP data freshness helpers.

These back the freshness gates that stop present-but-stale FMP data from
reaching the scoring engine (audit findings F1/F2). The functions take an
explicit ``now`` so the wall-clock comparison is deterministic in tests.
"""
from __future__ import annotations

from datetime import datetime, timezone

from src.utils.freshness import (
    is_us_rth,
    quote_age_seconds,
    series_age_days,
    is_us_listing,
)

# 2026-06-22 is a Monday; 2026-06-20 is a Saturday. June ⇒ US Eastern is EDT (UTC-4).
_UTC = timezone.utc


class TestIsUsRth:
    def test_midday_weekday_is_open(self):
        # 14:00 UTC = 10:00 EDT, a Monday → open
        assert is_us_rth(datetime(2026, 6, 22, 14, 0, tzinfo=_UTC)) is True

    def test_one_minute_before_open_is_closed(self):
        # 13:29 UTC = 09:29 EDT → closed
        assert is_us_rth(datetime(2026, 6, 22, 13, 29, tzinfo=_UTC)) is False

    def test_exact_open_is_open(self):
        # 13:30 UTC = 09:30 EDT → open
        assert is_us_rth(datetime(2026, 6, 22, 13, 30, tzinfo=_UTC)) is True

    def test_one_minute_before_close_is_open(self):
        # 19:59 UTC = 15:59 EDT → open
        assert is_us_rth(datetime(2026, 6, 22, 19, 59, tzinfo=_UTC)) is True

    def test_exact_close_is_closed(self):
        # 20:00 UTC = 16:00 EDT → closed (half-open interval)
        assert is_us_rth(datetime(2026, 6, 22, 20, 0, tzinfo=_UTC)) is False

    def test_weekend_is_closed(self):
        # Saturday midday → closed
        assert is_us_rth(datetime(2026, 6, 20, 14, 0, tzinfo=_UTC)) is False


class TestQuoteAgeSeconds:
    def test_recent_timestamp(self):
        now = datetime(2026, 6, 22, 14, 0, tzinfo=_UTC)
        ts = now.timestamp() - 600  # 10 minutes ago
        age = quote_age_seconds({"timestamp": ts}, now)
        assert age is not None
        assert abs(age - 600) < 1.0

    def test_integer_epoch_seconds(self):
        now = datetime(2026, 6, 22, 14, 0, tzinfo=_UTC)
        ts = int(now.timestamp()) - 60
        age = quote_age_seconds({"timestamp": ts}, now)
        assert age is not None
        assert abs(age - 60) < 2.0

    def test_missing_timestamp_is_none(self):
        now = datetime(2026, 6, 22, 14, 0, tzinfo=_UTC)
        assert quote_age_seconds({"price": 123.0}, now) is None

    def test_garbage_timestamp_is_none(self):
        now = datetime(2026, 6, 22, 14, 0, tzinfo=_UTC)
        assert quote_age_seconds({"timestamp": "not-a-number"}, now) is None

    def test_empty_dict_is_none(self):
        now = datetime(2026, 6, 22, 14, 0, tzinfo=_UTC)
        assert quote_age_seconds({}, now) is None


class TestSeriesAgeDays:
    def test_friday_close_read_on_monday(self):
        now = datetime(2026, 6, 22, 14, 0, tzinfo=_UTC)  # Monday
        assert series_age_days("2026-06-19", now) == 3   # prior Friday

    def test_same_day_is_zero(self):
        now = datetime(2026, 6, 22, 14, 0, tzinfo=_UTC)
        assert series_age_days("2026-06-22", now) == 0

    def test_datetime_string_truncated_to_date(self):
        now = datetime(2026, 6, 22, 14, 0, tzinfo=_UTC)
        assert series_age_days("2026-06-19 00:00:00", now) == 3

    def test_empty_is_none(self):
        now = datetime(2026, 6, 22, 14, 0, tzinfo=_UTC)
        assert series_age_days("", now) is None

    def test_garbage_is_none(self):
        now = datetime(2026, 6, 22, 14, 0, tzinfo=_UTC)
        assert series_age_days("nope", now) is None


class TestIsUsListing:
    def test_plain_symbol_is_us(self):
        assert is_us_listing("AAPL") is True

    def test_suffixed_symbol_is_not_us(self):
        assert is_us_listing("ASML.AS") is False
        assert is_us_listing("7203.T") is False

    def test_empty_is_not_us(self):
        assert is_us_listing("") is False
