"""tests/test_backtest_retro_discount.py — retroactive pre-EU-gate score discount.

load_archive_snapshot deflates EU/Asia entries that predate the Piotroski gate
(missing quality_piotroski) by 0.6x; _parse_snapshot must CONSUME that adjusted
score for both the badge threshold and the recorded final_score. The bug under
test: the adjusted score was computed but never read, and the discount loop only
scanned `top_buys` (missing the combined-cooked `top_buys_europe/_asia` lists).
"""
from __future__ import annotations

import json

from scripts.backtest_signals import _parse_snapshot


def _write(tmp_path, payload):
    p = tmp_path / "2026-05-01_top_lists.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _eu_entry(score, factors=None):
    return {"ticker": "ASML.AS", "market": "EUROPE", "badge": "TACTICAL BUY",
            "final_score": score, "cap_tier": "large", "factors": factors or {}}


def test_pre_gate_eu_discount_filters_membership(tmp_path):
    # 0.95 * 0.6 = 0.57 < 0.60 TACTICAL threshold → dropped once discounted.
    # Entry lives in the combined-cooked top_buys_europe list (not top_buys).
    payload = {"generated_at": "2026-05-01T10:00:00+00:00",
               "top_buys_europe": [_eu_entry(0.95)]}
    assert _parse_snapshot(_write(tmp_path, payload)) == []


def test_gate_active_disables_discount(tmp_path):
    # With the gate already active the discount must NOT apply, so the same
    # 0.95 EU entry survives at its raw score (control for the filter above).
    payload = {"generated_at": "2026-05-01T10:00:00+00:00",
               "piotroski_eu_gate_active": True,
               "top_buys_europe": [_eu_entry(0.95)]}
    recs = _parse_snapshot(_write(tmp_path, payload))
    assert len(recs) == 1
    assert recs[0].final_score == 0.95


def test_us_entry_untouched_by_discount(tmp_path):
    payload = {"generated_at": "2026-05-01T10:00:00+00:00",
               "top_buys_usa": [{"ticker": "AAPL", "market": "USA",
                                 "badge": "TACTICAL BUY", "final_score": 0.70,
                                 "cap_tier": "large", "factors": {}}]}
    recs = _parse_snapshot(_write(tmp_path, payload))
    assert len(recs) == 1 and recs[0].final_score == 0.70


def test_eu_entry_with_piotroski_present_not_discounted(tmp_path):
    payload = {"generated_at": "2026-05-01T10:00:00+00:00",
               "top_buys_europe": [_eu_entry(0.70, {"quality_piotroski": 0.5})]}
    recs = _parse_snapshot(_write(tmp_path, payload))
    assert len(recs) == 1 and recs[0].final_score == 0.70
