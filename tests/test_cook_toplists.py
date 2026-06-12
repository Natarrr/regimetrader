# Path: tests/test_cook_toplists.py
import json
import sys
import importlib.util
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Helpers to import the module under test (not yet implemented)
# ---------------------------------------------------------------------------


def _load_cook():
    """Dynamically import cook_toplists from the scripts/ directory."""
    spec = importlib.util.spec_from_file_location(
        "cook_toplists",
        Path(__file__).parents[1] / "src" / "delivery" / "cook_toplists.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def cook_mod():
    return _load_cook()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry(tmp_path):
    data = {
        "europe": [
            {"ticker": "SAP.DE", "name": "SAP", "sector": "Technology"},
            {"ticker": "ASML.AS", "name": "ASML", "sector": "Technology"},
        ],
        "asia": [
            {"ticker": "7203.T", "name": "Toyota", "sector": "Consumer Discretionary"},
            {"ticker": "9984.T", "name": "SoftBank", "sector": "Communication Services"},
        ],
    }
    p = tmp_path / "ticker_registry.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture
def us_payload(tmp_path):
    data = {
        "top_buys": [
            {
                "ticker": "MSFT",
                "final_score": 0.87,
                "badge": "HIGH BUY",
                "market": "USA",
                "factors": {
                    "insider_conviction": 0.80,
                    "insider_breadth": 0.75,
                    "congress": 0.60,
                    "news_sentiment": 0.70,
                    "news_buzz": 0.50,
                    "momentum_long": 0.90,
                    "volume_attention": 0.80,
                    "analyst_consensus": 0.70,
                    "quality_piotroski": 0.80,
                },
            },
            {
                "ticker": "AAPL",
                "final_score": 0.72,
                "badge": "TACTICAL BUY",
                "market": "USA",
                "factors": {
                    "insider_conviction": 0.60,
                    "insider_breadth": 0.55,
                    "congress": 0.40,
                    "news_sentiment": 0.65,
                    "news_buzz": 0.45,
                    "momentum_long": 0.80,
                    "volume_attention": 0.70,
                    "analyst_consensus": 0.65,
                    "quality_piotroski": 0.75,
                },
            },
        ],
        "vix": 17.3,
        "vix_regime": "Normal",
        "kill_switch": False,
        "ticker_count": 160,
        "weights": {
            "insider_conviction": 0.10,
            "insider_breadth": 0.10,
            "congress": 0.05,
            "news_sentiment": 0.15,
            "news_buzz": 0.10,
            "momentum_long": 0.15,
            "volume_attention": 0.10,
            "analyst_consensus": 0.10,
            "quality_piotroski": 0.15,
        },
    }
    p = tmp_path / "top_lists.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture
def intl_payload(tmp_path):
    data = [
        {
            "ticker": "SAP.DE",
            "composite_score": 0.81,
            "region_applied": "INTL",
            "factor_snapshots": {
                "news_sentiment": 0.85,
                "momentum_long": 0.80,
                "quality_piotroski": 0.78,
                "volume_attention": 0.65,
                "analyst_consensus": 0.72,
                "news_buzz": 0.60,
            },
        },
        {
            "ticker": "7203.T",
            "composite_score": 0.63,
            "region_applied": "INTL",
            "factor_snapshots": {
                "news_sentiment": 0.60,
                "momentum_long": 0.70,
                "quality_piotroski": 0.55,
                "volume_attention": 0.45,
                "analyst_consensus": 0.58,
                "news_buzz": 0.40,
            },
        },
        {
            "ticker": "ASML.AS",
            "composite_score": 0.55,
            "region_applied": "INTL",
            "factor_snapshots": {
                "news_sentiment": 0.50,
                "momentum_long": 0.60,
                "quality_piotroski": 0.50,
                "volume_attention": 0.40,
                "analyst_consensus": 0.55,
                "news_buzz": 0.35,
            },
        },
    ]
    p = tmp_path / "top_lists_intl.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# _badge tests
# ---------------------------------------------------------------------------


def test_badge_high_buy(cook_mod):
    assert cook_mod._badge(0.80) == "HIGH BUY"
    assert cook_mod._badge(0.95) == "HIGH BUY"


def test_badge_tactical_buy(cook_mod):
    assert cook_mod._badge(0.60) == "TACTICAL BUY"
    assert cook_mod._badge(0.79) == "TACTICAL BUY"


def test_badge_watchlist(cook_mod):
    assert cook_mod._badge(0.0) == "WATCHLIST"
    assert cook_mod._badge(0.59) == "WATCHLIST"


# ---------------------------------------------------------------------------
# _build_registry_map tests
# ---------------------------------------------------------------------------


def test_registry_map_europe(cook_mod, registry):
    m = cook_mod._build_registry_map(registry)
    assert m["SAP.DE"]["market"] == "EUROPE"
    assert m["ASML.AS"]["market"] == "EUROPE"


def test_registry_map_asia(cook_mod, registry):
    m = cook_mod._build_registry_map(registry)
    assert m["7203.T"]["market"] == "ASIA"
    assert m["9984.T"]["market"] == "ASIA"


def test_registry_map_missing_file(cook_mod, tmp_path):
    m = cook_mod._build_registry_map(tmp_path / "nonexistent.json")
    assert m == {}


# ---------------------------------------------------------------------------
# _normalize_intl_entry tests
# ---------------------------------------------------------------------------


def test_normalize_sets_final_score(cook_mod, registry):
    ticker_map = cook_mod._build_registry_map(registry)
    raw = {
        "ticker": "SAP.DE",
        "composite_score": 0.81,
        "region_applied": "INTL",
        "factor_snapshots": {"news_sentiment": 0.85, "momentum_long": 0.80},
    }
    result = cook_mod._normalize_intl_entry(raw, ticker_map, vix=15.0)
    # vix=15 → multiplier 1.00, so final_score == composite_score
    assert result["final_score"] == 0.81


def test_normalize_maps_market_from_registry(cook_mod, registry):
    ticker_map = cook_mod._build_registry_map(registry)
    eu_raw = {"ticker": "SAP.DE", "composite_score": 0.70, "region_applied": "INTL", "factor_snapshots": {}}
    asia_raw = {"ticker": "7203.T", "composite_score": 0.65, "region_applied": "INTL", "factor_snapshots": {}}
    assert cook_mod._normalize_intl_entry(eu_raw, ticker_map, vix=15.0)["market"] == "EUROPE"
    assert cook_mod._normalize_intl_entry(asia_raw, ticker_map, vix=15.0)["market"] == "ASIA"


def test_normalize_adds_congress_zero(cook_mod, registry):
    ticker_map = cook_mod._build_registry_map(registry)
    raw = {
        "ticker": "SAP.DE",
        "composite_score": 0.72,
        "region_applied": "INTL",
        "factor_snapshots": {"news_sentiment": 0.7},
    }
    result = cook_mod._normalize_intl_entry(raw, ticker_map, vix=15.0)
    assert result["factors"].get("congress") == 0.0


def test_normalize_congress_cannot_be_overridden(cook_mod, registry):
    """Even if factor_snapshots mistakenly carries a congress value, cook zeroes it."""
    ticker_map = cook_mod._build_registry_map(registry)
    raw = {
        "ticker": "SAP.DE",
        "composite_score": 0.72,
        "region_applied": "INTL",
        "factor_snapshots": {"congress": 0.99, "news_sentiment": 0.7},
    }
    result = cook_mod._normalize_intl_entry(raw, ticker_map, vix=15.0)
    assert result["factors"]["congress"] == 0.0


def test_normalize_computes_correct_badge(cook_mod, registry):
    ticker_map = cook_mod._build_registry_map(registry)
    # Use vix=15 (multiplier=1.00) so raw scores map directly to badge thresholds
    high = {"ticker": "SAP.DE", "composite_score": 0.82, "region_applied": "INTL", "factor_snapshots": {}}
    tact = {"ticker": "SAP.DE", "composite_score": 0.65, "region_applied": "INTL", "factor_snapshots": {}}
    watch = {"ticker": "SAP.DE", "composite_score": 0.40, "region_applied": "INTL", "factor_snapshots": {}}
    assert cook_mod._normalize_intl_entry(high, ticker_map, vix=15.0)["badge"] == "HIGH BUY"
    assert cook_mod._normalize_intl_entry(tact, ticker_map, vix=15.0)["badge"] == "TACTICAL BUY"
    assert cook_mod._normalize_intl_entry(watch, ticker_map, vix=15.0)["badge"] == "WATCHLIST"


# ---------------------------------------------------------------------------
# cook() integration tests
# ---------------------------------------------------------------------------


def test_cook_produces_regional_keys(cook_mod, tmp_path, us_payload, intl_payload, registry):
    out = tmp_path / "combined.json"
    cook_mod.cook(us_payload, intl_payload, registry, out)
    result = json.loads(out.read_text())
    assert "top_buys_usa" in result
    assert "top_buys_europe" in result
    assert "top_buys_asia" in result


def test_cook_preserves_vix_from_us(cook_mod, tmp_path, us_payload, intl_payload, registry):
    out = tmp_path / "combined.json"
    cook_mod.cook(us_payload, intl_payload, registry, out)
    result = json.loads(out.read_text())
    assert result["vix"] == 17.3
    assert result["vix_regime"] == "NORMAL"
    assert result["kill_switch"] is False


def test_cook_splits_intl_by_region(cook_mod, tmp_path, us_payload, intl_payload, registry):
    out = tmp_path / "combined.json"
    cook_mod.cook(us_payload, intl_payload, registry, out)
    result = json.loads(out.read_text())
    eu_tickers = [e["ticker"] for e in result["top_buys_europe"]]
    asia_tickers = [e["ticker"] for e in result["top_buys_asia"]]
    assert "SAP.DE" in eu_tickers
    assert "ASML.AS" in eu_tickers
    assert "7203.T" in asia_tickers
    assert "SAP.DE" not in asia_tickers
    assert "7203.T" not in eu_tickers


def test_cook_no_cross_contamination(cook_mod, tmp_path, us_payload, intl_payload, registry):
    out = tmp_path / "combined.json"
    cook_mod.cook(us_payload, intl_payload, registry, out)
    result = json.loads(out.read_text())
    for entry in result["top_buys_europe"] + result["top_buys_asia"]:
        assert entry["factors"].get("congress", 0.0) == 0.0, (
            f"{entry['ticker']}: congress must be 0.0 for non-US entries"
        )


def test_cook_ticker_count_is_combined(cook_mod, tmp_path, us_payload, intl_payload, registry):
    out = tmp_path / "combined.json"
    cook_mod.cook(us_payload, intl_payload, registry, out)
    result = json.loads(out.read_text())
    # 160 US + 3 INTL
    assert result["ticker_count"] == 163


def test_cook_us_entries_preserved_unchanged(cook_mod, tmp_path, us_payload, intl_payload, registry):
    out = tmp_path / "combined.json"
    cook_mod.cook(us_payload, intl_payload, registry, out)
    result = json.loads(out.read_text())
    usa = result["top_buys_usa"]
    assert usa[0]["ticker"] == "MSFT"
    assert usa[0]["final_score"] == 0.87
    assert usa[0]["factors"]["congress"] == 0.60  # US congress factor preserved


def test_cook_handles_top_buys_usa_key(cook_mod, tmp_path, intl_payload, registry):
    """US payload that already uses top_buys_usa (not top_buys) is handled correctly."""
    us_data = {
        "top_buys_usa": [
            {
                "ticker": "NVDA",
                "final_score": 0.91,
                "badge": "HIGH BUY",
                "market": "USA",
                "factors": {"congress": 0.5, "news_sentiment": 0.8},
            },
        ],
        "vix": 14.0,
        "vix_regime": "Normal",
        "kill_switch": False,
        "ticker_count": 50,
    }
    us_file = tmp_path / "us.json"
    us_file.write_text(json.dumps(us_data))
    out = tmp_path / "combined.json"
    cook_mod.cook(us_file, intl_payload, registry, out)
    result = json.loads(out.read_text())
    assert result["top_buys_usa"][0]["ticker"] == "NVDA"


def test_cook_missing_us_file_raises(cook_mod, tmp_path, intl_payload, registry):
    """main() should return exit code 1 when US input is missing."""
    sys.argv = [
        "cook_toplists.py",
        "--us-input", str(tmp_path / "nonexistent.json"),
        "--intl-input", str(intl_payload),
        "--registry", str(registry),
        "--output", str(tmp_path / "out.json"),
    ]
    rc = cook_mod.main()
    assert rc == 1


def test_cook_generated_at_is_present(cook_mod, tmp_path, us_payload, intl_payload, registry):
    out = tmp_path / "combined.json"
    cook_mod.cook(us_payload, intl_payload, registry, out)
    result = json.loads(out.read_text())
    assert "generated_at" in result
    assert result["generated_at"]  # non-empty string


# ---------------------------------------------------------------------------
# Bug-fix regression tests
# ---------------------------------------------------------------------------


def test_ticker_count_uses_actual_regional_entries(cook_mod, registry, tmp_path):
    """ticker_count must equal us_count + len(europe) + len(asia).

    Verifies the formula by asserting on the regional sub-list lengths independently,
    confirming the sum matches what cook() reports.
    """
    us_payload_path = tmp_path / "top_lists_us.json"
    us_payload_path.write_text(json.dumps({
        "top_buys": [{"ticker": "AAPL", "final_score": 0.85, "market": "USA", "factors": {}}],
        "vix": 18.0, "vix_regime": "Normal", "kill_switch": False, "ticker_count": 10,
    }), encoding="utf-8")
    intl_path = tmp_path / "top_lists_intl.json"
    # 2 EU + 2 Asia from registry; ticker_count = 10 + 2 + 2 = 14
    intl_path.write_text(json.dumps([
        {"ticker": "SAP.DE",  "composite_score": 0.75, "region_applied": "INTL", "factor_snapshots": {}, "pipeline": "INTL"},
        {"ticker": "ASML.AS", "composite_score": 0.70, "region_applied": "INTL", "factor_snapshots": {}, "pipeline": "INTL"},
        {"ticker": "7203.T",  "composite_score": 0.65, "region_applied": "INTL", "factor_snapshots": {}, "pipeline": "INTL"},
        {"ticker": "9984.T",  "composite_score": 0.60, "region_applied": "INTL", "factor_snapshots": {}, "pipeline": "INTL"},
    ]), encoding="utf-8")
    out = tmp_path / "out.json"
    cook_mod.cook(us_payload_path, intl_path, registry, out)
    result = json.loads(out.read_text())
    n_eu = len(result["top_buys_europe"])
    n_asia = len(result["top_buys_asia"])
    assert n_eu == 2, f"Expected 2 EU entries, got {n_eu}"
    assert n_asia == 2, f"Expected 2 Asia entries, got {n_asia}"
    expected = 10 + n_eu + n_asia
    assert result["ticker_count"] == expected, f"Expected {expected}, got {result['ticker_count']}"


def test_missing_vix_exits_with_code_1(cook_mod, registry, tmp_path):
    """cook() must exit(1) when vix is missing, not just print a warning."""
    us_path = tmp_path / "top_lists_us.json"
    us_path.write_text(json.dumps({
        "top_buys": [], "vix_regime": "Normal", "kill_switch": False,
    }), encoding="utf-8")
    intl_path = tmp_path / "top_lists_intl.json"
    intl_path.write_text(json.dumps([]), encoding="utf-8")
    out = tmp_path / "out.json"
    with pytest.raises(SystemExit) as exc_info:
        cook_mod.cook(us_path, intl_path, registry, out)
    assert exc_info.value.code != 0


def test_vix_dampening_applied_to_intl_entries(cook_mod, registry, tmp_path):
    """In CAPITULATION (VIX=35): high-quality survivor entries get 0.50x score dampening."""
    us_path = tmp_path / "top_lists_us.json"
    us_path.write_text(json.dumps({
        "top_buys": [], "vix": 35.0, "vix_regime": "CAPITULATION", "kill_switch": False, "ticker_count": 0,
    }), encoding="utf-8")
    intl_path = tmp_path / "top_lists_intl.json"
    # Entry must pass capitulation filter: beta ≤ 1.2 AND (piotroski ≥ 0.70 OR de_ratio ≤ 0.30)
    intl_path.write_text(json.dumps([
        {
            "ticker": "SAP.DE", "composite_score": 1.0, "region_applied": "INTL",
            "factor_snapshots": {"beta": 0.6, "quality_piotroski": 0.85, "debt_to_equity": 0.2},
            "pipeline": "INTL",
        },
    ]), encoding="utf-8")
    out = tmp_path / "out.json"
    cook_mod.cook(us_path, intl_path, registry, out)
    result = json.loads(out.read_text())
    # H-1 fix: under CAPITULATION, survivors are moved to watchlist; top_buys_* are emptied
    assert result["top_buys_europe"] == [], "top_buys_europe must be empty under CAPITULATION"
    watchlist = result.get("watchlist", [])
    assert len(watchlist) == 1, f"Expected 1 watchlist entry, got {len(watchlist)}"
    eu_entry = watchlist[0]
    # CAPITULATION multiplier = 0.50× (quality anchors survive, score dampened)
    assert abs(eu_entry["final_score"] - 0.50) < 1e-4, f"Expected 0.50, got {eu_entry['final_score']}"
    assert eu_entry.get("_capitulation_survivor") is True
    assert eu_entry["badge"] == "WATCHLIST"


def test_capitulation_dampening_symmetric_us_vs_intl(cook_mod, registry, tmp_path):
    """US final_score arrives pre-dampened from generate_top_lists
    (_apply_vix_overlay ×0.50 at VIX 35); the cook must NOT multiply it
    again, while applying exactly one ×0.50 to the raw INTL composite.
    Identical raw strength (1.0) must yield identical watchlist scores."""
    us_path = tmp_path / "top_lists_us.json"
    us_path.write_text(json.dumps({
        "top_buys_usa": [{
            "ticker": "JNJ", "final_score": 0.50, "badge": "WATCHLIST",
            "market": "USA", "pipeline": "US",
            "factors": {"quality_piotroski": 0.9},
        }],
        "vix": 35.0, "vix_regime": "CAPITULATION", "kill_switch": True, "ticker_count": 1,
    }), encoding="utf-8")
    intl_path = tmp_path / "top_lists_intl.json"
    intl_path.write_text(json.dumps([{
        "ticker": "SAP.DE", "composite_score": 1.0, "region_applied": "INTL",
        "factor_snapshots": {"quality_piotroski": 0.9},
        "pipeline": "INTL",
    }]), encoding="utf-8")
    out = tmp_path / "out.json"
    cook_mod.cook(us_path, intl_path, registry, out)
    watchlist = json.loads(out.read_text())["watchlist"]
    scores = {e["ticker"]: e["final_score"] for e in watchlist}
    assert abs(scores["JNJ"] - 0.50) < 1e-4, (
        f"US score re-dampened: expected 0.50, got {scores['JNJ']}")
    assert abs(scores["SAP.DE"] - 0.50) < 1e-4
    assert scores["JNJ"] == scores["SAP.DE"]


class TestSectorCountCap:
    def _make_entries(self, sectors):
        return [
            {
                "ticker": f"T{i}",
                "final_score": round(0.9 - 0.01 * i, 2),
                "factors": {"sector": s, "beta": 0.5, "quality_piotroski": 0.9},
                "market": "US",
                "badge": "HIGH BUY",
            }
            for i, s in enumerate(sectors)
        ]

    def test_max_two_per_sector(self, cook_mod):
        entries = self._make_entries(["Tech", "Tech", "Tech", "Health"])
        primary, overflow = cook_mod._apply_sector_count_cap(entries, max_per_sector=2)
        tech_in_primary = sum(1 for e in primary if e["factors"]["sector"] == "Tech")
        assert tech_in_primary == 2
        assert len(overflow) == 1

    def test_no_entries_lost(self, cook_mod):
        entries = self._make_entries(["A", "A", "A", "B"])
        primary, overflow = cook_mod._apply_sector_count_cap(entries, max_per_sector=2)
        assert len(primary) + len(overflow) == 4

    def test_descending_score_order(self, cook_mod):
        entries = self._make_entries(["Tech", "Health", "Tech"])
        primary, _ = cook_mod._apply_sector_count_cap(entries, max_per_sector=2)
        scores = [e["final_score"] for e in primary]
        assert scores == sorted(scores, reverse=True)

    def test_output_keys_in_combined(self, cook_mod, registry, tmp_path):
        """cook() output must include usa_overflow and eu_overflow keys."""
        us_path = tmp_path / "us.json"
        us_path.write_text(json.dumps({
            "top_buys": [], "vix": 18.0, "vix_regime": "NORMAL", "kill_switch": False, "ticker_count": 0,
        }), encoding="utf-8")
        intl_path = tmp_path / "intl.json"
        intl_path.write_text(json.dumps([]), encoding="utf-8")
        out = tmp_path / "out.json"
        cook_mod.cook(us_path, intl_path, registry, out)
        result = json.loads(out.read_text())
        assert "usa_overflow" in result
        assert "eu_overflow" in result
        assert "asia_overflow" in result


class TestSegmentByMarketCap:
    def _make_cap_entry(self, ticker, cap):
        return {"ticker": ticker, "market_cap": cap, "final_score": 0.85,
                "factors": {}, "badge": "HIGH BUY"}

    def test_three_tiers(self, cook_mod):
        entries = [
            self._make_cap_entry("LRG", 50_000_000_000),
            self._make_cap_entry("MID", 5_000_000_000),
            self._make_cap_entry("SML", 800_000_000),
            self._make_cap_entry("TIN", 50_000_000),   # excluded
        ]
        large, mid, small = cook_mod._segment_by_market_cap(entries)
        assert [e["ticker"] for e in large] == ["LRG"]
        assert [e["ticker"] for e in mid]   == ["MID"]
        assert [e["ticker"] for e in small] == ["SML"]
        assert not any(e["ticker"] == "TIN" for e in large + mid + small)

    def test_empty_lists_when_no_entries(self, cook_mod):
        large, mid, small = cook_mod._segment_by_market_cap([])
        assert large == mid == small == []

    def test_intl_entries_excluded_regardless_of_cap(self, cook_mod):
        """INTL market_cap is listing-currency (no FX normalization upstream) —
        a raw ¥/€ value treated as USD would skew MVO bracketing, so INTL
        entries are excluded from all pools until FX normalization exists."""
        intl = self._make_cap_entry("7203.T", 45_000_000_000_000)  # ¥45T raw
        intl["pipeline"] = "INTL"
        eu = self._make_cap_entry("ASML.AS", 5_000_000_000)
        eu["pipeline"] = "INTL"
        us = self._make_cap_entry("MSFT", 3_000_000_000_000)
        large, mid, small = cook_mod._segment_by_market_cap([intl, eu, us])
        assert [e["ticker"] for e in large] == ["MSFT"]
        assert mid == [] and small == []

    def test_mvo_pools_key_in_output(self, cook_mod, registry, tmp_path):
        """cook() output must include mvo_pools key."""
        us_path = tmp_path / "us.json"
        us_path.write_text(json.dumps({
            "top_buys": [], "vix": 18.0, "vix_regime": "NORMAL", "kill_switch": False, "ticker_count": 0,
        }), encoding="utf-8")
        intl_path = tmp_path / "intl.json"
        intl_path.write_text(json.dumps([]), encoding="utf-8")
        out = tmp_path / "out.json"
        cook_mod.cook(us_path, intl_path, registry, out)
        result = json.loads(out.read_text())
        assert "mvo_pools" in result


def test_intl_insider_meta_flows_engine_to_discord(cook_mod, registry, tmp_path):
    """End-to-end (plan verification 1b): FMPFetcher display meta survives
    StrategyEngine → _normalize_intl_entry → DiscordPayloadBuilder, so the
    EU desk line shows real insider $; the INTL MVO guard still excludes the
    entry from cap brackets despite a plausible-looking market_cap."""
    import json as _json
    import tempfile
    from datetime import datetime, timezone

    from src.delivery.send_discord import DiscordPayloadBuilder
    from src.engine.engine import StrategyEngine

    profile = {
        "region": "INTL",
        "active_factors": {"momentum_long": 0.60, "news_sentiment": 0.40},
        "output_filename": "test_out.json",
    }
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as f:
        _json.dump(profile, f)
        profile_path = f.name

    try:
        engine = StrategyEngine(profile_path)
        raw = [{
            "ticker": "ASML.AS",
            "metrics": {
                "momentum_long_score":  0.80,
                "news_sentiment_score": 0.90,
                "insider_usd":          150_000.0,
                "market_cap":           5e9,
                "return_12_1m":         0.18,
            },
        }]
        engine_entry = engine.score_ticker_pool(raw)[0]
    finally:
        import os
        os.unlink(profile_path)

    reg_map = cook_mod._build_registry_map(registry)
    normalized = cook_mod._normalize_intl_entry(engine_entry, reg_map, vix=17.0)
    assert normalized["insider_usd"] == 150_000.0
    assert normalized["market_cap"] == 5e9
    # INTL has no SPY-relative momentum: the absolute return is forwarded
    # under its own name, never as momentum_spy_relative.
    assert normalized["return_12_1m"] == 0.18
    assert "momentum_spy_relative" not in normalized
    assert normalized["market"] == "EUROPE"

    # Currency guard: $5B-looking cap must NOT bracket an INTL entry.
    large, mid, small = cook_mod._segment_by_market_cap([normalized])
    assert large == mid == small == []

    payload = DiscordPayloadBuilder({
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "vix":             17.0,
        "vix_regime":      "NORMAL",
        "kill_switch":     False,
        "ticker_count":    1,
        "top_buys_usa":    [],
        "top_buys_europe": [normalized],
        "top_buys_asia":   [],
        "watchlist":       [],
        "mvo_pools":       {},
    }).build()
    eu_field = next(f for f in payload["embeds"][0]["fields"]
                    if "EUROPE" in f["name"])
    assert "Insider $150k" in eu_field["value"]
    assert "+18.0% 12-1m abs" in eu_field["value"]
    assert "vs SPY" not in eu_field["value"]


def test_unregistered_intl_ticker_is_dropped(cook_mod, registry, tmp_path, capsys):
    """Registry is authoritative: an unregistered ticker must be dropped with a
    warning — never defaulted to EUROPE (which mislabeled it and could fail
    audit check D, blocking the whole send)."""
    us_path = tmp_path / "top_lists_us.json"
    us_path.write_text(json.dumps({
        "top_buys": [], "vix": 15.0, "vix_regime": "Normal", "kill_switch": False, "ticker_count": 0,
    }), encoding="utf-8")
    intl_path = tmp_path / "top_lists_intl.json"
    intl_path.write_text(json.dumps([
        {"ticker": "GHOST", "composite_score": 0.90, "region_applied": "INTL", "factor_snapshots": {}, "pipeline": "INTL"},
        {"ticker": "SAP.DE", "composite_score": 0.75, "region_applied": "INTL", "factor_snapshots": {}, "pipeline": "INTL"},
    ]), encoding="utf-8")
    out = tmp_path / "out.json"
    cook_mod.cook(us_path, intl_path, registry, out)
    result = json.loads(out.read_text())
    all_intl = (result["top_buys_europe"] + result["top_buys_asia"]
                + result["eu_mid_small"] + result["asia_mid_small"])
    tickers = {e["ticker"] for e in all_intl}
    assert "GHOST" not in tickers
    assert "SAP.DE" in tickers
    assert "not in ticker_registry.json" in capsys.readouterr().out


def test_pipeline_key_preserved_in_intl_entries(cook_mod, registry, tmp_path):
    """Normalized INTL entries must carry 'pipeline': 'INTL'."""
    us_path = tmp_path / "top_lists_us.json"
    us_path.write_text(json.dumps({
        "top_buys": [], "vix": 15.0, "vix_regime": "Normal", "kill_switch": False, "ticker_count": 0,
    }), encoding="utf-8")
    intl_path = tmp_path / "top_lists_intl.json"
    intl_path.write_text(json.dumps([
        {"ticker": "SAP.DE", "composite_score": 0.75, "region_applied": "INTL", "factor_snapshots": {}, "pipeline": "INTL"},
    ]), encoding="utf-8")
    out = tmp_path / "out.json"
    cook_mod.cook(us_path, intl_path, registry, out)
    result = json.loads(out.read_text())
    assert result["top_buys_europe"][0].get("pipeline") == "INTL"


# ---------------------------------------------------------------------------
# SMID leverage pool tests
# ---------------------------------------------------------------------------


class TestSmidLeveragePool:
    """_build_smid_leverage_pool: US small/mid-cap leverage sleeve.

    leverage_score = (0.50*final_score + 0.30*momentum_long
                      + 0.20*quality_piotroski) * pead_boost
    PEAD boost [Bernard & Thomas, 1989]: 1.10 iff earnings_surprise_pct > 0
    and 0 < earnings_surprise_days <= 60; absence is NEVER bearish.
    """

    def _us(self, ticker, score=0.72, cap=5e9, mom=0.70, qp=0.60, **kw):
        badge = ("HIGH BUY" if score >= 0.80
                 else "TACTICAL BUY" if score >= 0.60 else "WATCHLIST")
        return {"ticker": ticker, "final_score": score, "badge": badge,
                "market": "USA", "pipeline": "US", "market_cap": cap,
                "factors": {"momentum_long": mom, "quality_piotroski": qp,
                            "sector": kw.pop("sector", "Technology")}, **kw}

    # ── unit: composite + selection ──────────────────────────────────────

    def test_weights_sum_to_one(self, cook_mod):
        assert abs(sum(cook_mod._SMID_LEVERAGE_WEIGHTS.values()) - 1.0) < 1e-6

    def test_cap_bounds_inclusive(self, cook_mod):
        entries = [
            self._us("FLOOR",  cap=300_000_000),
            self._us("CEIL",   cap=10_000_000_000),
            self._us("MICRO",  cap=299_999_999),
            self._us("MEGA",   cap=10_000_000_001),
            self._us("NOCAP",  cap=0),
        ]
        pool = cook_mod._build_smid_leverage_pool(entries, top_n=10)
        tickers = {e["ticker"] for e in pool}
        assert tickers == {"FLOOR", "CEIL"}

    def test_top3_cap_and_descending(self, cook_mod):
        entries = [self._us(f"T{i}", score=0.60 + 0.05 * i) for i in range(5)]
        pool = cook_mod._build_smid_leverage_pool(entries)
        assert len(pool) == 3
        levs = [e["leverage_score"] for e in pool]
        assert levs == sorted(levs, reverse=True)
        assert [e["ticker"] for e in pool] == ["T4", "T3", "T2"]

    def test_intl_pipeline_excluded(self, cook_mod):
        """Same FX-normalization guard as _segment_by_market_cap: INTL
        market_cap is listing-currency, never bracketed as USD."""
        intl = self._us("ASML.AS", cap=5e9)
        intl["pipeline"] = "INTL"
        pool = cook_mod._build_smid_leverage_pool([intl, self._us("AAOI")])
        assert [e["ticker"] for e in pool] == ["AAOI"]

    def test_pead_boost_changes_ranking(self, cook_mod):
        a = self._us("AAA", score=0.70, mom=0.70, qp=0.70)
        b = self._us("BBB", score=0.68, mom=0.68, qp=0.68,
                     earnings_surprise_pct=5.0, earnings_surprise_days=30)
        pool = cook_mod._build_smid_leverage_pool([a, b])
        assert [e["ticker"] for e in pool] == ["BBB", "AAA"]
        assert pool[0]["leverage_score"] == round(0.68 * 1.10, 4)  # 0.748
        assert pool[1]["leverage_score"] == 0.70

    def test_pead_window_and_sign_boundaries(self, cook_mod):
        def lev(**kw):
            entry = self._us("X", score=0.70, mom=0.70, qp=0.70, **kw)
            return cook_mod._build_smid_leverage_pool([entry])[0]["leverage_score"]

        base = lev()  # no earnings meta at all
        assert base == 0.70
        assert lev(earnings_surprise_pct=5.0, earnings_surprise_days=60) == 0.77
        assert lev(earnings_surprise_pct=5.0, earnings_surprise_days=61) == base
        assert lev(earnings_surprise_pct=5.0, earnings_surprise_days=0) == base
        assert lev(earnings_surprise_pct=0.0, earnings_surprise_days=30) == base
        assert lev(earnings_surprise_pct=-3.0, earnings_surprise_days=30) == base
        # Absence is NEVER bearish: None meta scores identically to no meta.
        assert lev(earnings_surprise_pct=None, earnings_surprise_days=None) == base

    def test_source_entries_not_mutated(self, cook_mod):
        import copy
        entries = [self._us("AAA"), self._us("BBB", score=0.81)]
        snapshot = copy.deepcopy(entries)
        cook_mod._build_smid_leverage_pool(entries)
        assert entries == snapshot
        for original in entries:
            assert "leverage_score" not in original

    def test_leverage_score_may_exceed_one_final_score_untouched(self, cook_mod):
        entry = self._us("MAX", score=1.0, mom=1.0, qp=1.0,
                         earnings_surprise_pct=10.0, earnings_surprise_days=10)
        pool = cook_mod._build_smid_leverage_pool([entry])
        assert pool[0]["leverage_score"] == 1.1
        assert pool[0]["final_score"] == 1.0  # audit check A gates this key only

    # ── integration: cook() plumbing ─────────────────────────────────────

    def _cook(self, cook_mod, registry, tmp_path, us_data):
        us_path = tmp_path / "us.json"
        us_path.write_text(json.dumps(us_data), encoding="utf-8")
        intl_path = tmp_path / "intl.json"
        intl_path.write_text(json.dumps([]), encoding="utf-8")
        out = tmp_path / "out.json"
        cook_mod.cook(us_path, intl_path, registry, out)
        return json.loads(out.read_text())

    def test_cook_always_emits_top_buys_smid_key(self, cook_mod, registry, tmp_path):
        result = self._cook(cook_mod, registry, tmp_path, {
            "top_buys": [], "vix": 18.0, "vix_regime": "NORMAL",
            "kill_switch": False, "ticker_count": 0,
        })
        assert "top_buys_smid" in result
        assert result["top_buys_smid"] == []

    def test_cook_smid_includes_sector_overflow(self, cook_mod, registry, tmp_path):
        """The leverage sleeve is NOT sector-constrained: names pushed to
        usa_overflow by the sector count cap remain SMID candidates."""
        entries = [self._us(f"T{i}", score=0.80 - 0.01 * i, sector="Tech")
                   for i in range(3)]
        result = self._cook(cook_mod, registry, tmp_path, {
            "top_buys_usa": entries, "vix": 17.0, "vix_regime": "NORMAL",
            "kill_switch": False, "ticker_count": 3,
        })
        assert len(result["top_buys_usa"]) == 2  # sector cap = 2
        assert {e["ticker"] for e in result["top_buys_smid"]} == {"T0", "T1", "T2"}

    def test_cook_capitulation_empties_smid(self, cook_mod, registry, tmp_path):
        entry = self._us("JNJ", score=0.50, qp=0.90)
        entry["factors"]["beta"] = 0.5
        result = self._cook(cook_mod, registry, tmp_path, {
            "top_buys_usa": [entry], "vix": 35.0, "vix_regime": "CAPITULATION",
            "kill_switch": True, "ticker_count": 1,
        })
        assert result["top_buys_smid"] == []

    def test_cook_smid_excludes_large_caps(self, cook_mod, registry, tmp_path):
        result = self._cook(cook_mod, registry, tmp_path, {
            "top_buys_usa": [
                self._us("MSFT", score=0.90, cap=3e12, sector="Tech"),
                self._us("AAOI", score=0.70, cap=1.4e9, sector="Optics"),
            ],
            "vix": 17.0, "vix_regime": "NORMAL",
            "kill_switch": False, "ticker_count": 2,
        })
        assert result["top_buys_usa"][0]["ticker"] == "MSFT"
        assert [e["ticker"] for e in result["top_buys_smid"]] == ["AAOI"]
