"""tests/test_golden_record.py
Golden Record regression test — pins scoring output as immutable truth.

Loads the 2026-05-16 intel_source_status fixture and verifies that
generate() produces the exact same final_scores (within float epsilon),
ranking order, and badges as the stored top_lists golden record.

This test fails immediately if a code change alters the scoring formula
or normalization logic — an intentional tripwire.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN_INPUT  = FIXTURES / "golden_intel_source_2026_05_16.json"
GOLDEN_OUTPUT = FIXTURES / "golden_top_lists_2026_05_16.json"

# VIX value captured at the time the golden record was produced
_GOLDEN_VIX = 18.43


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def golden_input() -> Dict[str, Any]:
    return _load(GOLDEN_INPUT)


@pytest.fixture(scope="module")
def golden_output() -> Dict[str, Any]:
    return _load(GOLDEN_OUTPUT)


@pytest.fixture(scope="module")
def generated(tmp_path_factory, golden_input) -> Dict[str, Any]:
    from backend.market_intel.generate_top_lists import generate

    log_dir = tmp_path_factory.mktemp("golden_logs")
    with patch(
        "backend.market_intel.generate_top_lists._read_vix",
        return_value=_GOLDEN_VIX,
    ):
        return generate(golden_input, run_id="golden-regression", log_dir=log_dir)


class TestGoldenRecord:
    # ── top_buys ────────────────────────────────────────────────────────────────

    def test_top_buys_count(self, generated, golden_output):
        assert len(generated["top_buys"]) == len(golden_output["top_buys"])

    def test_top_buys_ranking_order(self, generated, golden_output):
        gen_tickers  = [e["ticker"] for e in generated["top_buys"]]
        gold_tickers = [e["ticker"] for e in golden_output["top_buys"]]
        assert gen_tickers == gold_tickers, (
            f"Top-buy order changed.\n  got:      {gen_tickers}\n  expected: {gold_tickers}"
        )

    @pytest.mark.parametrize("rank", [0, 1, 2, 3, 4])
    def test_top_buys_final_score(self, rank, generated, golden_output):
        gen_score  = generated["top_buys"][rank]["final_score"]
        gold_score = golden_output["top_buys"][rank]["final_score"]
        ticker     = golden_output["top_buys"][rank]["ticker"]
        assert abs(gen_score - gold_score) < 1e-4, (
            f"{ticker} rank {rank+1}: score changed from {gold_score} to {gen_score}"
        )

    @pytest.mark.parametrize("rank", [0, 1, 2, 3, 4])
    def test_top_buys_badge(self, rank, generated, golden_output):
        gen_badge  = generated["top_buys"][rank]["badge"]
        gold_badge = golden_output["top_buys"][rank]["badge"]
        ticker     = golden_output["top_buys"][rank]["ticker"]
        assert gen_badge == gold_badge, (
            f"{ticker} rank {rank+1}: badge changed from {gold_badge!r} to {gen_badge!r}"
        )

    def test_top_buys_cap_tiers(self, generated, golden_output):
        gen_tiers  = {e["ticker"]: e["cap_tier"] for e in generated["top_buys"]}
        gold_tiers = {e["ticker"]: e["cap_tier"] for e in golden_output["top_buys"]}
        assert gen_tiers == gold_tiers

    # ── mid_caps ────────────────────────────────────────────────────────────────

    def test_mid_caps_order(self, generated, golden_output):
        gen  = [e["ticker"] for e in generated["mid_caps"]]
        gold = [e["ticker"] for e in golden_output["mid_caps"]]
        assert gen == gold

    def test_mid_caps_scores(self, generated, golden_output):
        for gen_e, gold_e in zip(generated["mid_caps"], golden_output["mid_caps"]):
            assert abs(gen_e["final_score"] - gold_e["final_score"]) < 1e-4, (
                f'{gold_e["ticker"]} mid_cap score changed'
            )

    # ── small_caps ──────────────────────────────────────────────────────────────

    def test_small_caps_order(self, generated, golden_output):
        gen  = [e["ticker"] for e in generated["small_caps"]]
        gold = [e["ticker"] for e in golden_output["small_caps"]]
        assert gen == gold

    def test_small_caps_scores(self, generated, golden_output):
        for gen_e, gold_e in zip(generated["small_caps"], golden_output["small_caps"]):
            assert abs(gen_e["final_score"] - gold_e["final_score"]) < 1e-4, (
                f'{gold_e["ticker"]} small_cap score changed'
            )

    # ── kill_switch ──────────────────────────────────────────────────────────────

    def test_kill_switch_off(self, generated, golden_output):
        # VIX 18.43 < 30 → no kill switch
        assert generated["kill_switch"] is False
        assert golden_output["kill_switch"] is False

    def test_vix_multiplier_is_one(self, generated):
        assert generated["vix_multiplier"] == 1.0

    # ── ticker_count ─────────────────────────────────────────────────────────────

    def test_ticker_count(self, generated, golden_output):
        assert generated["ticker_count"] == golden_output["ticker_count"]

    # ── Piotroski canary (Bug 3) ──────────────────────────────────────────────────

    def test_piotroski_score_distribution(self, golden_input):
        """Canary: at least 30% of US tickers should have quality_piotroski_score
        != round(3/9, 4) sentinel. If this fails, ratios-ttm endpoint is broken."""
        sentinel = round(3 / 9, 4)  # = 0.3333
        us_results = [
            r for r in golden_input.get("results", [])
            if r.get("market", "USA") == "USA"
        ]
        if not us_results:
            pytest.skip("No US results in golden fixture")

        flat = [r for r in us_results if r.get("quality_piotroski_score") == sentinel]
        assert len(flat) < len(us_results) * 0.70, (
            f"PIOTROSKI FLAT: {len(flat)}/{len(us_results)} US tickers at sentinel "
            f"{sentinel} — >70% at missing-score value; ratios-ttm may be broken"
        )
