"""tests/scoring/test_contamination_guard.py — v3.0 cross-contamination layers.

Layer 2 (defensive runtime): out-of-mask factors with non-None values are
forced to None in prod (with `_contamination_masked` evidence) and hard-fail
under STRICT_REGION_GUARD=1 (CI / staging drill). Market mismatches always
raise — wrong-pool rows are never silently rescored.
"""
from __future__ import annotations

import copy

import pytest

from src.config.factor_matrix import FACTOR_MATRIX_V3
from src.scoring.engine_v3 import assert_region_isolation, score_universe_v3


def _universe(region, market, prefix, n=6):
    names = list(FACTOR_MATRIX_V3[region].keys())
    rows = []
    for i in range(n):
        row = {"ticker": f"{prefix}{i}", "sector": "Technology",
               "cap_tier": "large", "market": market,
               "quality_piotroski_raw": 8}
        for name in names:
            row[f"{name}_score"] = 0.7
        rows.append(row)
    return rows


class TestInjectionMasking:
    def test_eu_congress_injection_masked_and_score_unchanged(self):
        clean = _universe("EU", "EUROPE", "E")
        dirty = copy.deepcopy(clean)
        dirty[0]["congress_score"] = 0.9
        res_clean = score_universe_v3(clean, "EU")
        res_dirty = score_universe_v3(dirty, "EU")
        assert (res_dirty[0]["final_score_v3"]
                == res_clean[0]["final_score_v3"])
        assert res_dirty[0]["_contamination_masked"] == ["congress"]

    def test_asia_13f_injection_masked(self):
        rows = _universe("ASIA", "ASIA", "A")
        rows[2]["inst_flow_13f_score"] = 0.95
        result = score_universe_v3(rows, "ASIA")
        assert result[2]["_contamination_masked"] == ["inst_flow_13f"]
        assert result[2]["final_score_v3"] == result[0]["final_score_v3"]

    def test_us_dividend_sustain_injection_masked(self):
        rows = _universe("US", "USA", "T")
        rows[1]["dividend_sustain_score"] = 0.9
        result = score_universe_v3(rows, "US")
        assert result[1]["_contamination_masked"] == ["dividend_sustain"]
        assert result[1]["final_score_v3"] == result[0]["final_score_v3"]

    def test_input_rows_not_mutated(self):
        rows = _universe("EU", "EUROPE", "E")
        rows[0]["congress_score"] = 0.9
        score_universe_v3(rows, "EU")
        assert rows[0]["congress_score"] == 0.9  # caller's data untouched


class TestStrictMode:
    def test_strict_env_raises(self, monkeypatch):
        monkeypatch.setenv("STRICT_REGION_GUARD", "1")
        rows = _universe("EU", "EUROPE", "E")
        rows[0]["congress_score"] = 0.9
        with pytest.raises(ValueError, match="congress"):
            score_universe_v3(rows, "EU")

    def test_strict_param_overrides_env(self, monkeypatch):
        monkeypatch.delenv("STRICT_REGION_GUARD", raising=False)
        rows = _universe("EU", "EUROPE", "E")
        rows[0]["congress_score"] = 0.9
        with pytest.raises(ValueError, match="congress"):
            assert_region_isolation(rows, "EU", strict=True)


class TestMarketMismatch:
    def test_wrong_pool_always_raises(self):
        rows = _universe("EU", "EUROPE", "E")
        rows.append(_universe("US", "USA", "T", n=1)[0])
        with pytest.raises(ValueError, match="isolation"):
            score_universe_v3(rows, "EU")

    def test_unknown_region_rejected(self):
        with pytest.raises(ValueError):
            score_universe_v3(_universe("US", "USA", "T"), "LATAM")
