"""tests/test_discord_formatter.py
Unit tests for Discord formatter helpers.
"""
from __future__ import annotations


class TestFactorGroup:
    """Tests for _factor_group — grouped factor rendering used in conviction field."""

    def test_renders_edgar_and_insider(self):
        from scripts.send_toplists_discord import _factor_group, _FUNDAMENTAL
        factors = {"edgar": 0.80, "insider": 0.70}
        out = _factor_group(factors, _FUNDAMENTAL)
        assert "0.80" in out
        assert "0.70" in out
        assert "EDGAR" in out
        assert "Insider" in out

    def test_renders_congress_and_news(self):
        from scripts.send_toplists_discord import _factor_group, _SENTIMENT
        factors = {"congress": 0.60, "news": 0.55}
        out = _factor_group(factors, _SENTIMENT)
        assert "0.60" in out
        assert "0.55" in out

    def test_renders_macro(self):
        from scripts.send_toplists_discord import _factor_group, _TECHNICAL
        factors = {"macro": 0.99}
        out = _factor_group(factors, _TECHNICAL)
        assert "0.99" in out

    def test_missing_key_omitted_not_defaulted(self):
        from scripts.send_toplists_discord import _factor_group, _FUNDAMENTAL
        # missing insider → only edgar rendered, no 0.00 filler
        out = _factor_group({"edgar": 0.72}, _FUNDAMENTAL)
        assert "0.72" in out
        assert "Insider" not in out

    def test_empty_factors_returns_dash(self):
        from scripts.send_toplists_discord import _factor_group, _FUNDAMENTAL
        assert _factor_group({}, _FUNDAMENTAL) == "—"


class TestConvictionField:
    """Conviction field must expose key signals for the #1 pick."""

    def _entry(self, **overrides):
        base = {
            "ticker": "AAPL", "final_score": 0.82, "badge": "HIGH BUY",
            "sector": "Technology", "market_cap": 3e12, "ceo_buy": False,
            "factors": {"edgar": 0.80, "insider": 0.90, "congress": 0.60,
                        "news": 0.70, "macro": 0.50},
        }
        base.update(overrides)
        return base

    def test_ticker_in_field(self):
        from scripts.send_toplists_discord import _conviction_field
        f = _conviction_field(self._entry())
        assert "AAPL" in f["value"]

    def test_score_in_field(self):
        from scripts.send_toplists_discord import _conviction_field
        f = _conviction_field(self._entry())
        assert "0.8200" in f["value"]

    def test_ceo_buy_tag_shown(self):
        from scripts.send_toplists_discord import _conviction_field
        f = _conviction_field(self._entry(ceo_buy=True))
        assert "CEO BUY" in f["value"]

    def test_ceo_buy_tag_absent_when_false(self):
        from scripts.send_toplists_discord import _conviction_field
        f = _conviction_field(self._entry(ceo_buy=False))
        assert "CEO BUY" not in f["value"]

    def test_factor_groups_all_present(self):
        from scripts.send_toplists_discord import _conviction_field
        f = _conviction_field(self._entry())
        assert "Fundamental" in f["value"]
        assert "Sentiment" in f["value"]
        assert "Technical" in f["value"]

    def test_missing_factor_omitted_not_zero(self):
        from scripts.send_toplists_discord import _conviction_field
        entry = self._entry()
        del entry["factors"]["insider"]
        f = _conviction_field(entry)
        assert "Insider" not in f["value"]


class TestBuildPayloadWeights:
    def _make_top_lists(self, weights):
        return {
            "generated_at":  "2026-05-17T12:00:00+00:00",
            "source_run_id": "test-run",
            "ticker_count":  10,
            "weights":       weights,
            "kill_switch":   False,
            "top_buys":      [{"ticker": "AAPL", "final_score": 0.70, "badge": "TACTICAL BUY",
                               "factors": {"edgar": 0.7, "insider": 0.6, "congress": 0.5,
                                           "news": 0.6, "macro": 0.5}, "ceo_buy": False}],
            "mid_caps":      [],
            "small_caps":    [],
        }

    def test_nominal_weights_no_redistribution_label(self):
        from scripts.send_toplists_discord import build_payload
        weights = {"edgar": 0.28, "insider": 0.23, "congress": 0.22, "news": 0.15, "macro": 0.12}
        payload = build_payload(self._make_top_lists(weights))
        desc = payload["embeds"][0]["description"]
        assert "feed down" not in desc, "nominal weights must not trigger redistribution warning"
        assert "redistributed" not in desc

    def test_redistributed_weights_shows_warning(self):
        from scripts.send_toplists_discord import build_payload
        # Simulate insider feed dead — weight redistributed to other factors
        weights = {"edgar": 0.359, "congress": 0.282, "news": 0.192, "macro": 0.154}
        payload = build_payload(self._make_top_lists(weights))
        desc = payload["embeds"][0]["description"]
        assert "feed down" in desc or "redistributed" in desc, (
            "redistributed weights must show a warning in description"
        )
