"""tests/test_universe_files.py — EU/APAC universe file integrity.

Region purity at the door (plan: suffix misclassification caught on load),
schema, dedupe, and liquidity-floor sanity for the committed CSVs built by
tools/build_universes.py.
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.config.weights import get_region

_SCHEMA = ["ticker", "name", "sector", "cap_tier", "exchange", "adv_usd"]
_CASES = [
    ("config/universe_eu.csv", "EU", 5e6),
    ("config/universe_apac.csv", "ASIA", 3e6),
]


def _load(path):
    with Path(path).open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return reader.fieldnames, list(reader)


@pytest.mark.parametrize("path,region,min_adv", _CASES,
                         ids=["eu", "apac"])
class TestUniverseFiles:
    def test_schema(self, path, region, min_adv):
        fieldnames, _ = _load(path)
        assert fieldnames == _SCHEMA

    def test_region_purity(self, path, region, min_adv):
        _, rows = _load(path)
        misplaced = [r["ticker"] for r in rows
                     if get_region(r["ticker"]) != region]
        assert misplaced == [], f"non-{region} tickers: {misplaced[:10]}"

    def test_no_duplicates(self, path, region, min_adv):
        _, rows = _load(path)
        tickers = [r["ticker"] for r in rows]
        assert len(tickers) == len(set(tickers))

    def test_minimum_breadth(self, path, region, min_adv):
        # Target is >=80 (sector x cap buckets clear MIN_BUCKET_SIZE=5);
        # hard floor 30 catches catastrophic regressions without making CI
        # hostage to screener coverage on any given rebuild day.
        _, rows = _load(path)
        assert len(rows) >= 30, f"only {len(rows)} names in {path}"

    def test_liquidity_floor_and_tiers(self, path, region, min_adv):
        _, rows = _load(path)
        for r in rows:
            assert int(r["adv_usd"]) >= min_adv, r["ticker"]
            assert r["cap_tier"] in ("large", "mid", "small"), r["ticker"]

    def test_exchange_breadth(self, path, region, min_adv):
        # Regression for the alphabetical-truncation bias: digit-first
        # tickers (KR/HK) monopolized the candidate cap and Japan/India
        # vanished entirely. A regional universe must span >= 3 suffixes.
        _, rows = _load(path)
        suffixes = {r["ticker"].rsplit(".", 1)[-1] for r in rows}
        assert len(suffixes) >= 3, f"{path} spans only {suffixes}"


class TestStratifiedTruncation:
    def test_cap_keeps_suffix_breadth(self):
        from tools.build_universes import _stratified
        candidates = (
            [f"00{i}.KS" for i in range(50)]      # digits sort first
            + [f"7{i:03d}.T" for i in range(50)]
            + [f"INFY{i}.NS" for i in range(50)]  # letters sort last
        )
        capped = _stratified({t: {} for t in candidates}, cap=30)
        suffixes = {t.rsplit(".", 1)[-1] for t in capped}
        assert suffixes == {"KS", "T", "NS"}
        assert len(capped) == 30

    def test_cap_above_population_returns_all(self):
        from tools.build_universes import _stratified
        capped = _stratified({"A.PA": {}, "B.DE": {}}, cap=10)
        assert sorted(capped) == ["A.PA", "B.DE"]
