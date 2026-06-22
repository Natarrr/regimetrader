import sys
import os
import json
import tempfile
from datetime import date
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.backtest_signals import (
    SignalRecord,
    HorizonStats,
    _parse_snapshot,
    _market_flag,
    _failure_label,
    _alpha_decay_tag,
    _truncate_field,
    build_backtest_discord_payload,
)


def _snapshot_data_with_market():
    """Minimal snapshot data with market and company_name fields."""
    return {
        "generated_at": "2026-01-10T12:00:00Z",
        "weights": {"edgar": 0.28, "insider": 0.23, "congress": 0.22, "news": 0.15, "momentum": 0.12},
        "top_buys": [
            {
                "ticker": "6758.T",
                "final_score": 0.85,  # Must be >= 0.80 for HIGH BUY
                "badge": "HIGH BUY",
                "market": "ASIA",
                "company_name": "Sony",
                # quality_piotroski present → no retroactive pre-EU-gate discount;
                # this test checks market/company_name passthrough, not the gate.
                "factors": {"edgar": 0.8, "quality_piotroski": 0.6}
            }
        ]
    }


def test_signal_record_has_market_field():
    rec = SignalRecord(
        ticker="6758.T",
        signal_date=date(2026, 1, 10),
        badge="HIGH BUY",
        cap_tier="large",
        final_score=0.72,
        factors={"edgar": 0.8},
        weights={"edgar": 0.28},
        strategy_era="v2",
        source_file="test.json",
        entry_next_day=True,
        market="ASIA",
        company_name="Sony",
    )
    assert rec.market == "ASIA"
    assert rec.company_name == "Sony"


def test_parse_snapshot_captures_market_and_company_name():
    snap_data = _snapshot_data_with_market()

    # Write to a temporary file so _parse_snapshot can read it
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(snap_data, f)
        temp_path = f.name

    try:
        records = _parse_snapshot(Path(temp_path))
        assert len(records) == 1, f"Expected 1 record, got {len(records)}"
        r = records[0]
        assert r.market == "ASIA", f"Expected market='ASIA', got '{r.market}'"
        assert r.company_name == "Sony", f"Expected company_name='Sony', got '{r.company_name}'"
    finally:
        Path(temp_path).unlink()


def test_market_flag_usa():
    assert _market_flag("USA") == "🇺🇸"


def test_market_flag_europe():
    assert _market_flag("EUROPE") == "🇪🇺"


def test_market_flag_asia():
    assert _market_flag("ASIA") == "🇯🇵"


def test_market_flag_unknown():
    assert _market_flag("OTHER") == "🌐"


def test_failure_label_stopped_out():
    # Returns STOPPED OUT when T+5 return <= -0.05
    assert _failure_label(-0.06) == "STOPPED OUT"


def test_failure_label_mean_reverted():
    # Returns MEAN REVERTED when return is between -0.05 and -0.015
    assert _failure_label(-0.03) == "MEAN REVERTED"


def test_failure_label_noise():
    # Returns NOISE when return is between -0.015 and 0
    assert _failure_label(-0.01) == "NOISE"


def test_failure_label_winner():
    # Returns empty string for positive returns (not a failure)
    assert _failure_label(0.02) == ""


def test_alpha_decay_tag_decaying():
    # Alpha is decaying when T+20 < T+10 * 0.5
    assert _alpha_decay_tag(0.04, 0.01) == "⚠ ALPHA DECAY"


def test_alpha_decay_tag_negative():
    # Alpha is decaying when T+20 goes negative
    assert _alpha_decay_tag(0.04, -0.01) == "⚠ ALPHA DECAY"


def test_alpha_decay_tag_healthy():
    # No decay when T+20 >= T+10 * 0.5 and T+20 >= 0
    assert _alpha_decay_tag(0.04, 0.03) == ""


def test_truncate_field_within_limit():
    assert _truncate_field("hello", 1024) == "hello"


def test_truncate_field_over_limit():
    long_str = "x" * 1100
    result = _truncate_field(long_str, 1024)
    assert len(result) <= 1024
    assert result.endswith("…")


# ── Payload test helpers ──────────────────────────────────────────────────────

def _make_horizon_stats(horizon, count, win_rate, avg_return, avg_alpha, profit_factor=1.5):
    return HorizonStats(
        horizon=horizon, count=count, win_rate=win_rate,
        avg_return=avg_return, median_return=avg_return,
        max_drawdown=-0.05, profit_factor=profit_factor,
        avg_alpha=avg_alpha,
    )


def _make_signal_record(ticker, badge, score, cap_tier="large", market="USA", r10=0.03, alpha10=0.02):
    rec = SignalRecord(
        ticker=ticker, signal_date=date(2026, 5, 1), badge=badge,
        cap_tier=cap_tier, final_score=score,
        factors={"edgar": 0.8, "insider": 0.6, "congress": 0.5, "news": 0.5, "momentum": 0.4},
        weights={"edgar": 0.28, "insider": 0.23, "congress": 0.22, "news": 0.15, "momentum": 0.12},
        strategy_era="v2", source_file="test.json",
        entry_next_day=False, market=market,
    )
    rec.entry_price = 100.0
    rec.returns = {5: 0.01, 10: r10, 20: 0.02}
    rec.alpha = {5: 0.005, 10: alpha10, 20: 0.015}
    rec.spy_returns = {5: 0.005, 10: 0.01, 20: 0.005}
    return rec


def _make_full_payload_inputs():
    badge_stats = {
        "HIGH BUY": [
            _make_horizon_stats(5,  6, 0.667, 0.0145, 0.010),
            _make_horizon_stats(10, 6, 0.667, 0.0320, 0.0185),
            _make_horizon_stats(20, 6, 0.500, 0.0210, 0.012),
        ],
        "TACTICAL BUY": [
            _make_horizon_stats(5,  6, 0.500, 0.0080, 0.003),
            _make_horizon_stats(10, 6, 0.500, 0.0110, 0.002),
            _make_horizon_stats(20, 6, 0.333, -0.0045, -0.001),
        ],
    }
    cap_stats = {
        "large": _make_horizon_stats(10, 4, 0.750, 0.0245, 0.015),
        "mid":   _make_horizon_stats(10, 2, 0.500, 0.0110, 0.005),
        "small": _make_horizon_stats(10, 3, 0.333, -0.0095, -0.004),
    }
    records = [
        _make_signal_record("PLTR",   "HIGH BUY",    0.92, market="USA",    r10=0.054, alpha10=0.041),
        _make_signal_record("SAP.DE", "HIGH BUY",    0.78, market="EUROPE", r10=0.012, alpha10=0.009),
        _make_signal_record("AMD",    "TACTICAL BUY",0.65, market="USA",    r10=-0.021, alpha10=-0.030),
    ]
    worst = [records[2]]
    return badge_stats, cap_stats, records, worst


# ── Payload tests ─────────────────────────────────────────────────────────────

def test_payload_structure_has_embeds():
    badge_stats, cap_stats, records, worst = _make_full_payload_inputs()
    payload = build_backtest_discord_payload(
        report_date=date(2026, 5, 22),
        badge_stats=badge_stats,
        cap_stats=cap_stats,
        worst=worst,
        total_signals=12,
        records=records,
    )
    assert "embeds" in payload
    assert isinstance(payload["embeds"], list)
    assert len(payload["embeds"]) == 1


def test_payload_title_contains_performance_log():
    badge_stats, cap_stats, records, worst = _make_full_payload_inputs()
    payload = build_backtest_discord_payload(
        report_date=date(2026, 5, 22),
        badge_stats=badge_stats,
        cap_stats=cap_stats,
        worst=worst,
        total_signals=12,
        records=records,
    )
    title = payload["embeds"][0]["title"]
    assert "STRATEGY PERFORMANCE LOG" in title


def test_payload_description_contains_n():
    badge_stats, cap_stats, records, worst = _make_full_payload_inputs()
    payload = build_backtest_discord_payload(
        report_date=date(2026, 5, 22),
        badge_stats=badge_stats,
        cap_stats=cap_stats,
        worst=worst,
        total_signals=12,
        records=records,
    )
    description = payload["embeds"][0]["description"]
    assert "N=12" in description


def test_payload_colour_green_when_wr_high():
    badge_stats, cap_stats, records, worst = _make_full_payload_inputs()
    # HIGH BUY T+10 win_rate is 0.667 >= 0.55 → green
    payload = build_backtest_discord_payload(
        report_date=date(2026, 5, 22),
        badge_stats=badge_stats,
        cap_stats=cap_stats,
        worst=worst,
        total_signals=12,
        records=records,
    )
    assert payload["embeds"][0]["color"] == 0x00FF00


def test_payload_colour_red_when_wr_low():
    badge_stats, cap_stats, records, worst = _make_full_payload_inputs()
    # Override HIGH BUY T+10 with low win_rate < 0.45
    badge_stats["HIGH BUY"][1] = _make_horizon_stats(10, 6, 0.333, -0.01, -0.005)
    payload = build_backtest_discord_payload(
        report_date=date(2026, 5, 22),
        badge_stats=badge_stats,
        cap_stats=cap_stats,
        worst=worst,
        total_signals=12,
        records=records,
    )
    assert payload["embeds"][0]["color"] == 0xFF0000


def test_payload_fields_have_high_buy_block():
    badge_stats, cap_stats, records, worst = _make_full_payload_inputs()
    payload = build_backtest_discord_payload(
        report_date=date(2026, 5, 22),
        badge_stats=badge_stats,
        cap_stats=cap_stats,
        worst=worst,
        total_signals=12,
        records=records,
    )
    fields = payload["embeds"][0]["fields"]
    names = [f["name"] for f in fields]
    assert any("HIGH BUY" in n for n in names)
    assert any("TACTICAL BUY" in n for n in names)


def test_payload_field_contains_profit_factor():
    badge_stats, cap_stats, records, worst = _make_full_payload_inputs()
    payload = build_backtest_discord_payload(
        report_date=date(2026, 5, 22),
        badge_stats=badge_stats,
        cap_stats=cap_stats,
        worst=worst,
        total_signals=12,
        records=records,
    )
    fields = payload["embeds"][0]["fields"]
    hb_field = next((f for f in fields if "HIGH BUY" in f["name"]), None)
    assert hb_field is not None
    assert "PF" in hb_field["value"]


def test_payload_field_shows_alpha_decay():
    badge_stats, cap_stats, records, worst = _make_full_payload_inputs()
    # TACTICAL BUY T+10=0.0110, T+20=-0.0045 (negative) → decay
    payload = build_backtest_discord_payload(
        report_date=date(2026, 5, 22),
        badge_stats=badge_stats,
        cap_stats=cap_stats,
        worst=worst,
        total_signals=12,
        records=records,
    )
    fields = payload["embeds"][0]["fields"]
    tb_field = next((f for f in fields if "TACTICAL BUY" in f["name"]), None)
    assert tb_field is not None
    assert "Decay" in tb_field["value"] or "DECAY" in tb_field["value"]


def test_payload_cap_tier_block_present():
    badge_stats, cap_stats, records, worst = _make_full_payload_inputs()
    payload = build_backtest_discord_payload(
        report_date=date(2026, 5, 22),
        badge_stats=badge_stats,
        cap_stats=cap_stats,
        worst=worst,
        total_signals=12,
        records=records,
    )
    fields = payload["embeds"][0]["fields"]
    names = [f["name"] for f in fields]
    assert any("Capacity" in n or "Tier" in n for n in names)


def test_payload_edge_trajectory_shows_market_flag():
    badge_stats, cap_stats, records, worst = _make_full_payload_inputs()
    payload = build_backtest_discord_payload(
        report_date=date(2026, 5, 22),
        badge_stats=badge_stats,
        cap_stats=cap_stats,
        worst=worst,
        total_signals=12,
        records=records,
    )
    fields = payload["embeds"][0]["fields"]
    edge_field = next((f for f in fields if "Edge" in f["name"] or "Trajectory" in f["name"]), None)
    assert edge_field is not None
    assert "🇺🇸" in edge_field["value"] or "🇪🇺" in edge_field["value"]


def test_payload_detractors_shows_failure_label():
    badge_stats, cap_stats, records, worst = _make_full_payload_inputs()
    # worst[0] is AMD with r10=-0.021 → "MEAN REVERTED" or "STOPPED OUT"
    payload = build_backtest_discord_payload(
        report_date=date(2026, 5, 22),
        badge_stats=badge_stats,
        cap_stats=cap_stats,
        worst=worst,
        total_signals=12,
        records=records,
    )
    fields = payload["embeds"][0]["fields"]
    risk_field = next((f for f in fields if "Risk" in f["name"] or "Detractor" in f["name"]), None)
    assert risk_field is not None
    value = risk_field["value"]
    assert "STOPPED OUT" in value or "MEAN REVERTED" in value or "NOISE" in value


def test_payload_all_field_values_within_1024_chars():
    badge_stats, cap_stats, records, worst = _make_full_payload_inputs()
    payload = build_backtest_discord_payload(
        report_date=date(2026, 5, 22),
        badge_stats=badge_stats,
        cap_stats=cap_stats,
        worst=worst,
        total_signals=12,
        records=records,
    )
    fields = payload["embeds"][0]["fields"]
    for f in fields:
        assert len(f["value"]) <= 1024, f"Field '{f['name']}' value exceeds 1024 chars: {len(f['value'])}"
