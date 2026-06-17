# Path: research/tests/test_ic_engine.py
"""IC engine — backfill NDJSON → snapshots → research/ic_report.json."""
from __future__ import annotations

import json
from datetime import date

from research.scripts.ic_engine import records_to_snapshots, run
from src.research.ic_metrics import snapshot_ic


class TestRecordsToSnapshots:
    def test_groups_by_date_and_derives_spy_relative_excess(self):
        records = [
            {"ticker": "A", "snapshot_date": "2025-01-31",
             "momentum_long": 0.1, "forward_return_21d": 0.05,
             "spy_return_21d": 0.02},
            {"ticker": "B", "snapshot_date": "2025-01-31",
             "momentum_long": 0.2, "forward_return_21d": 0.06,
             "spy_return_21d": 0.02},
            {"ticker": "A", "snapshot_date": "2025-02-28",
             "momentum_long": 0.3, "forward_return_21d": 0.01,
             "spy_return_21d": 0.03},
        ]
        snaps = records_to_snapshots(records)

        assert [s["date"] for s in snaps] == [date(2025, 1, 31), date(2025, 2, 28)]
        s1 = snaps[0]
        excess = {r["ticker"]: r["excess_return_21d"] for r in s1["rows"]}
        assert abs(excess["A"] - 0.03) < 1e-9   # 0.05 - 0.02
        assert abs(excess["B"] - 0.04) < 1e-9   # 0.06 - 0.02
        # IC of the factor vs the derived excess label is computable
        assert snapshot_ic(s1, "momentum_long",
                           return_key="excess_return_21d") == 1.0


class TestRun:
    def test_writes_report_with_full_schema(self, tmp_path):
        records = [
            {"ticker": t, "snapshot_date": d, "momentum_long": s,
             "forward_return_21d": r, "spy_return_21d": 0.0}
            for d, rows in {
                # snap1: factor ranks exactly with returns        → IC = 1.0
                "2025-01-31": [("A", 0.1, 0.01), ("B", 0.2, 0.02),
                               ("C", 0.3, 0.03)],
                # snap2: one inversion (B/C returns swapped)       → IC = 0.5
                "2025-03-31": [("A", 0.1, 0.01), ("B", 0.2, 0.03),
                               ("C", 0.3, 0.02)],
            }.items()
            for t, s, r in rows
        ]
        backfill = tmp_path / "factor_scores.ndjson"
        backfill.write_text(
            "\n".join(json.dumps(r) for r in records), encoding="utf-8")
        out = tmp_path / "ic_report.json"

        payload = run(backfill, out, ["momentum_long"])

        assert out.exists()
        on_disk = json.loads(out.read_text(encoding="utf-8"))
        assert on_disk == payload
        assert payload["_meta"]["horizon_days"] == 21
        assert payload["_meta"]["n_snapshots"] == 2
        # factor dicts live at the TOP LEVEL (consumer-compatible schema)
        mom = payload["momentum_long"]
        assert mom["mean_ic"] == 0.75         # mean of IC 1.0 and 0.5
        assert mom["ic_positive_rate"] == 1.0
        assert mom["weight_recommendation"] == "increase"
        assert "ic_t_stat" in mom and "n_effective" in mom

    def test_report_is_consumable_by_portfolio_optimizer(self, tmp_path, monkeypatch):
        """The whole point of WS1: unblock portfolio_optimizer._ic_estimate()."""
        import backend.market_intel.portfolio_optimizer as po

        records = [
            {"ticker": t, "snapshot_date": d, "momentum_long": s,
             "forward_return_21d": r, "spy_return_21d": 0.0}
            for d, rows in {
                "2025-01-31": [("A", 0.1, 0.01), ("B", 0.2, 0.02),
                               ("C", 0.3, 0.03)],
                "2025-03-31": [("A", 0.1, 0.01), ("B", 0.2, 0.03),
                               ("C", 0.3, 0.02)],
            }.items()
            for t, s, r in rows
        ]
        backfill = tmp_path / "factor_scores.ndjson"
        backfill.write_text(
            "\n".join(json.dumps(r) for r in records), encoding="utf-8")
        out = tmp_path / "ic_report.json"
        run(backfill, out, ["momentum_long"])

        monkeypatch.setattr(po, "_IC_REPORT_PATH", out)
        # mean_ic 0.75 → estimate is the measured IC, not the 0.03 fallback
        assert po._ic_estimate() == 0.75
