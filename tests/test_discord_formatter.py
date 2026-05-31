"""tests/test_discord_formatter.py
Unit tests for Discord formatter helpers — 7-factor schema (P0 rewrite).
"""
from __future__ import annotations

import json


def _make_status(tickers=None, generated_at="2026-05-17T12:00:00+00:00"):
    """Minimal intel_source_status.json fixture for build_payload tests."""
    if tickers is None:
        tickers = [
            {"ticker": "AAPL", "final_score": 0.70, "badge": "WATCHLIST",
             "sector": "Technology", "cap_tier": "large", "market": "US",
             "insider_conviction_score_neutral": 0.80,
             "insider_breadth_score_neutral": 0.70,
             "congress_score_neutral": 0.0,
             "news_sentiment_score_neutral": 0.60,
             "news_buzz_score_neutral": 0.50,
             "momentum_long_score_neutral": 0.40,
             "volume_attention_score_neutral": 0.0,
             "ceo_conviction_tier": "CEO BUY"},
        ]
    return {
        "generated_at": generated_at,
        "source_run_id": "test-run",
        "ticker_count": len(tickers),
        "top_by_market": {"US": tickers, "EUROPE": [], "ASIA": []},
        "results": tickers,
    }


class TestTickerDetailField:
    """_ticker_detail_field renders the 3-line card format with 7 factors."""

    def _entry(self, **overrides):
        base = {
            "ticker": "AAPL",
            "final_score": 0.82,
            "badge": "HIGH BUY",
            "sector": "Technology",
            "cap_tier": "large",
            "market": "USA",
            "ceo_conviction_tier": "BUY",
            "factors": {
                "insider_conviction": 0.80,
                "insider_breadth": 0.70,
                "congress": 0.0,
                "news_sentiment": 0.60,
                "news_buzz": 0.50,
                "momentum_long": 0.40,
                "volume_attention": 0.0,
            },
        }
        base.update(overrides)
        return base

    def _all_scores(self):
        return [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.82, 0.9]

    def test_ticker_in_field(self):
        from scripts.send_toplists_discord import _ticker_detail_field
        f = _ticker_detail_field(
            1, self._entry(), all_scores=self._all_scores())
        assert "AAPL" in f["name"]

    def test_score_in_field(self):
        from scripts.send_toplists_discord import _ticker_detail_field
        f = _ticker_detail_field(
            1, self._entry(), all_scores=self._all_scores())
        assert "0.8200" in f["name"]

    def test_7factor_matrix_rendered(self):
        from scripts.send_toplists_discord import _ticker_detail_field
        f = _ticker_detail_field(
            1, self._entry(), all_scores=self._all_scores())
        val = f["value"]
        assert "IC:" in val
        assert "IB:" in val
        assert "CG:" in val
        assert "NS:" in val
        assert "NB:" in val
        assert "MO:" in val
        assert "VA:" in val

    def test_zero_factors_rendered_as_dash(self):
        from scripts.send_toplists_discord import _ticker_detail_field
        f = _ticker_detail_field(
            1, self._entry(), all_scores=self._all_scores())
        val = f["value"]
        # congress=0.0 and volume_attention=0.0 → should render as "—"
        assert "CG:—" in val
        assert "VA:—" in val

    def test_ceo_tier_shown(self):
        from scripts.send_toplists_discord import _ticker_detail_field
        entry = self._entry(insider_usd=22000, ceo_conviction_tier="CEO BUY")
        f = _ticker_detail_field(1, entry, all_scores=self._all_scores())
        assert "Insider" in f["value"]
        assert "CEO" in f["value"]

    def test_ceo_tier_absent_when_none(self):
        from scripts.send_toplists_discord import _ticker_detail_field
        entry = self._entry(ceo_conviction_tier=None)
        f = _ticker_detail_field(1, entry, all_scores=self._all_scores())
        assert "CEO" not in f["value"]

    def test_percentile_in_field(self):
        from scripts.send_toplists_discord import _ticker_detail_field
        # all_scores has 9 values, 0.82 is 8th → p88
        f = _ticker_detail_field(
            1, self._entry(), all_scores=self._all_scores())
        assert "p8" in f["name"]  # p80–p89 range

    def test_catalyst_line_present(self):
        from scripts.send_toplists_discord import _ticker_detail_field
        f = _ticker_detail_field(
            1, self._entry(), all_scores=self._all_scores())
        assert any(kw in f["value"] for kw in ["Insider",
                   "EPS", "congress", "vs SPY", "no primary"])


class TestComputePercentile:
    """_compute_percentile returns correct rank within population."""

    def test_top_score_is_p100(self):
        from scripts.send_toplists_discord import _compute_percentile
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        assert _compute_percentile(0.5, scores) == 100

    def test_bottom_score_is_low(self):
        from scripts.send_toplists_discord import _compute_percentile
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        pct = _compute_percentile(0.1, scores)
        assert pct <= 20

    def test_empty_population_returns_zero(self):
        from scripts.send_toplists_discord import _compute_percentile
        assert _compute_percentile(0.5, []) == 0


class TestComputeCatalyst:
    """_compute_catalyst returns top active factors in descending order."""

    def test_returns_top_two_drivers(self):
        from scripts.send_toplists_discord import _compute_catalyst
        entry = {
            "insider_usd": 12500,
            "ceo_conviction_tier": "CEO BUY",
            "earnings_surprise_pct": 0.12,
            "earnings_surprise_days": 8,
        }
        cat = _compute_catalyst(entry)
        assert "Insider" in cat
        assert "EPS" in cat

    def test_no_active_factors_returns_no_catalyst(self):
        from scripts.send_toplists_discord import _compute_catalyst
        entry = {"factors": {"insider_conviction": 0.0, "congress": 0.0}}
        cat = _compute_catalyst(entry)
        assert "no primary catalyst" in cat

    def test_zeros_excluded_from_catalyst(self):
        from scripts.send_toplists_discord import _compute_catalyst
        entry = {"factors": {"insider_conviction": 0.9, "congress": 0.0,
                             "news_sentiment": 0.0}}
        cat = _compute_catalyst(entry)
        assert "CG" not in cat


class TestBuildPayloadSchema:
    """build_payload handles both intel_source_status.json and legacy top_lists.json."""

    def test_status_schema_produces_embed(self):
        from scripts.send_toplists_discord import build_payload
        payload = build_payload(_make_status())
        assert "embeds" in payload
        assert len(payload["embeds"]) >= 1

    def test_status_schema_has_usa_section(self):
        from scripts.send_toplists_discord import build_payload
        payload = build_payload(_make_status())
        fields = payload["embeds"][0]["fields"]
        names = [f["name"] for f in fields]
        assert any("USA" in n for n in names)

    def test_legacy_schema_still_works(self):
        from scripts.send_toplists_discord import build_payload
        legacy = {
            "generated_at": "2026-05-17T12:00:00+00:00",
            "source_run_id": "test-run",
            "ticker_count": 1,
            "weights": {"edgar": 0.28, "insider": 0.23, "congress": 0.22,
                        "news": 0.15, "macro": 0.12},
            "top_buys": [{"ticker": "AAPL", "final_score": 0.70,
                          "badge": "WATCHLIST",
                          "factors": {"edgar": 0.7, "insider": 0.6,
                                      "congress": 0.5, "news": 0.6,
                                      "macro": 0.5},
                          "ceo_buy": False}],
            "mid_caps": [],
            "small_caps": [],
        }
        payload = build_payload(legacy)
        assert "embeds" in payload

    def test_stale_data_shows_warning(self):
        from scripts.send_toplists_discord import build_payload
        # generated_at > 25h ago → DATA IS Xh OLD stale alert in description
        status = _make_status(generated_at="2026-05-17T12:00:00+00:00")
        payload = build_payload(status)
        desc = payload["embeds"][0]["description"]
        assert "OLD" in desc

    def test_status_schema_preserves_esg_metadata(self):
        from scripts.send_toplists_discord import build_payload
        status = _make_status()
        status["top_by_market"]["US"][0]["esg_score"] = 22.0
        status["top_by_market"]["US"][0]["esg_flag"] = True
        payload = build_payload(status)
        names = [f["name"] for f in payload["embeds"][0]["fields"]]
        assert any("ESG!" in n for n in names)

    def test_percentile_includes_zero_values(self):
        from scripts.send_toplists_discord import _compute_percentile
        scores = [0.0, 0.2, 0.4, 0.6]
        assert _compute_percentile(0.0, scores) == 25
