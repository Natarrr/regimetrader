# Path: research/tests/test_backfill_factors.py
"""Local-only tests for backfill_factors helpers (no FMP calls)."""
import pytest
from datetime import date, timedelta

# Import the pure helpers (no network calls)
from research.scripts.backfill_factors import (
    compute_momentum_at,
    compute_forward_return,
    compute_congress_score,
    _fridays,
)

FACTOR_KEYS = [
    "ticker", "snapshot_date", "insider_conviction", "insider_breadth",
    "congress", "news_sentiment", "news_buzz", "momentum_long",
    "volume_attention", "analyst_consensus", "quality_piotroski",
    "forward_return_21d", "spy_return_21d",
]


def _make_prices(n: int, base_close: float = 100.0) -> list[dict]:
    """Build n daily price records newest-first."""
    today = date(2026, 1, 1)
    records = []
    for i in range(n):
        d = today - timedelta(days=i)
        records.append({
            "date": str(d),
            "close": base_close * (1 + i * 0.001),
            "volume": 1_000_000 + i * 100,
        })
    return records


def test_fridays_count_and_weekday():
    fridays = _fridays(52)
    assert len(fridays) == 52
    for f in fridays:
        assert f.weekday() == 4  # Friday


def test_fridays_oldest_first():
    fridays = _fridays(10)
    for i in range(len(fridays) - 1):
        assert fridays[i] < fridays[i + 1]


def test_compute_momentum_at_insufficient_history():
    prices = _make_prices(100)  # need 252
    result, vol = compute_momentum_at(prices, date(2025, 12, 1))
    assert result is None
    assert vol is None


def test_compute_momentum_at_sufficient_history():
    prices = _make_prices(300)
    snap = date(2025, 12, 1)
    result, vol = compute_momentum_at(prices, snap)
    assert result is not None
    assert isinstance(result, float)


def test_compute_forward_return_within_range():
    prices = _make_prices(300, base_close=100.0)
    snap = date(2025, 12, 1)
    fwd = compute_forward_return(prices, snap, horizon=21)
    # Should return a float (positive or negative)
    assert fwd is not None
    assert isinstance(fwd, float)


def test_compute_forward_return_no_future_data():
    prices = _make_prices(50)  # all in the past
    snap = date(2024, 1, 1)
    fwd = compute_forward_return(prices, snap, horizon=21)
    # No prices before snap date → None
    assert fwd is None


def test_congress_score_empty():
    score = compute_congress_score("AAPL", [], date(2025, 6, 1))
    assert score == 0.0


def test_congress_score_capped_at_one():
    trades = [
        {"ticker": "AAPL", "transaction_date": "2025-05-15", "type": "Purchase"}
        for _ in range(10)
    ]
    score = compute_congress_score("AAPL", trades, date(2025, 6, 1))
    assert score == 1.0


def test_congress_score_out_of_window():
    trades = [{"ticker": "AAPL", "transaction_date": "2024-01-01", "type": "Purchase"}]
    score = compute_congress_score("AAPL", trades, date(2025, 6, 1))
    assert score == 0.0  # outside 90-day window


def test_no_lookahead_bias_in_fridays():
    """Fridays list must exclude the last 21 days (need fwd return)."""
    from datetime import date as dt
    fridays = _fridays(52)
    cutoff = dt.today() - timedelta(days=21)
    for f in fridays:
        assert f <= cutoff, f"Future snapshot {f} would create lookahead bias"
