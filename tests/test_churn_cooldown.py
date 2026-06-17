# Path: tests/test_churn_cooldown.py
"""Selection churn — tenure tracking + cooldown rotation.

A name that dominates the top-N for too many consecutive runs is forced to sit
out one run so capital rotates ("exclusion temporaire après signal"). Tenure is
tracked across runs in logs/universe_state.json; removals/additions are logged
to logs/universe_churn.ndjson with reasons (the transparency the brief asks for).
"""
from __future__ import annotations

from src.scoring.churn import (
    cooled_top_n,
    tickers_on_cooldown,
    update_tenure,
)


def _entry(ticker, score):
    return {"ticker": ticker, "final_score": score}


class TestTickersOnCooldown:
    def test_flags_names_at_or_above_max_tenure(self):
        state = {"A": 3, "B": 1, "C": 4}
        assert tickers_on_cooldown(state, max_tenure=3) == {"A", "C"}


class TestCooledTopN:
    def test_overtenured_leader_is_rotated_out(self):
        entries = [_entry("A", 0.9), _entry("B", 0.8),
                   _entry("C", 0.7), _entry("D", 0.6)]
        state = {"A": 3}                      # A has led 3 runs running
        selected, events = cooled_top_n(entries, state, max_tenure=3, n=2)

        assert [e["ticker"] for e in selected] == ["B", "C"]   # A rotated out
        assert any(ev["ticker"] == "A" and ev["action"] == "cooldown"
                   for ev in events)
        # the promoted name is logged as added with its reason
        assert any(ev["ticker"] == "B" and ev["action"] == "added"
                   for ev in events)

    def test_under_tenure_leader_is_kept(self):
        entries = [_entry("A", 0.9), _entry("B", 0.8), _entry("C", 0.7)]
        selected, events = cooled_top_n(entries, {"A": 1}, max_tenure=3, n=2)
        assert [e["ticker"] for e in selected] == ["A", "B"]
        assert not any(ev["action"] == "cooldown" for ev in events)


class TestUpdateTenure:
    def test_increments_selected_resets_cooled_drops_rest(self):
        prev = {"A": 2, "B": 0, "X": 1}
        new = update_tenure(prev, selected=["A", "C"], cooled_out=["B"])
        assert new == {"A": 3, "C": 1, "B": 0}   # X dropped (not selected)
