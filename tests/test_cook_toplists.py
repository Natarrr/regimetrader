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
        Path(__file__).parents[1] / "scripts" / "cook_toplists.py",
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
    assert m["SAP.DE"] == "EUROPE"
    assert m["ASML.AS"] == "EUROPE"


def test_registry_map_asia(cook_mod, registry):
    m = cook_mod._build_registry_map(registry)
    assert m["7203.T"] == "ASIA"
    assert m["9984.T"] == "ASIA"


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
    result = cook_mod._normalize_intl_entry(raw, ticker_map)
    assert result["final_score"] == 0.81


def test_normalize_maps_market_from_registry(cook_mod, registry):
    ticker_map = cook_mod._build_registry_map(registry)
    eu_raw = {"ticker": "SAP.DE", "composite_score": 0.70, "region_applied": "INTL", "factor_snapshots": {}}
    asia_raw = {"ticker": "7203.T", "composite_score": 0.65, "region_applied": "INTL", "factor_snapshots": {}}
    assert cook_mod._normalize_intl_entry(eu_raw, ticker_map)["market"] == "EUROPE"
    assert cook_mod._normalize_intl_entry(asia_raw, ticker_map)["market"] == "ASIA"


def test_normalize_adds_congress_zero(cook_mod, registry):
    ticker_map = cook_mod._build_registry_map(registry)
    raw = {
        "ticker": "SAP.DE",
        "composite_score": 0.72,
        "region_applied": "INTL",
        "factor_snapshots": {"news_sentiment": 0.7},
    }
    result = cook_mod._normalize_intl_entry(raw, ticker_map)
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
    result = cook_mod._normalize_intl_entry(raw, ticker_map)
    assert result["factors"]["congress"] == 0.0


def test_normalize_computes_correct_badge(cook_mod, registry):
    ticker_map = cook_mod._build_registry_map(registry)
    high = {"ticker": "SAP.DE", "composite_score": 0.82, "region_applied": "INTL", "factor_snapshots": {}}
    tact = {"ticker": "SAP.DE", "composite_score": 0.65, "region_applied": "INTL", "factor_snapshots": {}}
    watch = {"ticker": "SAP.DE", "composite_score": 0.40, "region_applied": "INTL", "factor_snapshots": {}}
    assert cook_mod._normalize_intl_entry(high, ticker_map)["badge"] == "HIGH BUY"
    assert cook_mod._normalize_intl_entry(tact, ticker_map)["badge"] == "TACTICAL BUY"
    assert cook_mod._normalize_intl_entry(watch, ticker_map)["badge"] == "WATCHLIST"


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
    assert result["vix_regime"] == "Normal"
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
