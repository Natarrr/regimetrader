"""tests/test_earnings_analyzer.py
Unit tests for analysis/earnings_analyzer.py.

Covers:
  - build_shortlist: quintile floor, watchlist merging, dedup, cap, empty input
  - build_prompt: output format, insider block, institutional block, risk block
  - cross_check_citations: verified citations, hallucinated accessions, non-EDGAR sources
  - should_auto_execute: all gate combinations, threshold overrides

Granger (2003 Nobel) — causal signals must be validated in isolation before
combining them; these tests verify each gate independently.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.earnings_analyzer import (
    PROMPT_VERSION,
    build_prompt,
    build_shortlist,
    cross_check_citations,
    should_auto_execute,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _candidate(symbol: str, smart_money_score: float = 0.85) -> Dict[str, Any]:
    """Return a minimal ScanResult dict."""
    return {
        "symbol": symbol,
        "smart_money_score": smart_money_score,
        "insider_score": 0.6,
        "institutional_score": 0.7,
        "momentum_score": 0.5,
    }


def _valid_analysis(**overrides) -> Dict[str, Any]:
    base = {
        "score": 80,
        "confidence": 0.85,
        "reasons": ["Strong insider buying", "Institutional accumulation"],
        "citations": [{"source": "EDGAR Form-4", "loc": "0001234567-24-000001"}],
        "recommended_action": "BUY",
    }
    base.update(overrides)
    return base


def _form4_event(accession: str, role: str = "CEO",
                 shares: int = 10_000, price: float = 50.0) -> Dict[str, Any]:
    return {
        "transaction_code": "P",
        "transaction_date": "2024-03-15",
        "reporting_role": role,
        "shares": shares,
        "price": price,
        "filing_accession": accession,
        "type": "form4",
    }


def _13f_event(accession: str, holder: str = "Vanguard",
               shares: int = 500_000) -> Dict[str, Any]:
    return {
        "type": "13f",
        "reporting_person": holder,
        "shares": shares,
        "filing_accession": accession,
    }


# ── build_shortlist ───────────────────────────────────────────────────────────

class TestBuildShortlist:
    """Verify shortlist selection logic.

    Granger (2003 Nobel) — causality filtering: only symbols above the
    information threshold (quintile_floor) should proceed to Claude analysis.
    """

    def test_basic_quintile_filter(self):
        candidates = [
            _candidate("AAPL", 0.90),
            _candidate("MSFT", 0.85),
            _candidate("IBM",  0.70),   # below 80/100 floor
        ]
        result = build_shortlist(candidates, quintile_floor=80)
        assert "AAPL" in result
        assert "MSFT" in result
        assert "IBM" not in result

    def test_returns_sorted_by_score(self):
        candidates = [
            _candidate("MSFT", 0.82),
            _candidate("AAPL", 0.95),
            _candidate("TSLA", 0.88),
        ]
        result = build_shortlist(candidates, quintile_floor=80)
        assert result[0] == "AAPL"
        assert result[1] == "TSLA"
        assert result[2] == "MSFT"

    def test_watchlist_symbols_included_regardless_of_score(self):
        candidates = [_candidate("AAPL", 0.90)]
        result = build_shortlist(candidates, watchlist=["NVDA"], quintile_floor=80)
        assert "NVDA" in result

    def test_watchlist_deduplication(self):
        candidates = [_candidate("AAPL", 0.90)]
        result = build_shortlist(candidates, watchlist=["AAPL"], quintile_floor=80)
        assert result.count("AAPL") == 1

    def test_max_symbols_cap(self):
        candidates = [_candidate(f"S{i:03d}", 0.90) for i in range(30)]
        result = build_shortlist(candidates, quintile_floor=80, max_symbols=10)
        assert len(result) == 10

    def test_empty_candidates_empty_result(self):
        result = build_shortlist([], quintile_floor=80)
        assert result == []

    def test_empty_candidates_watchlist_only(self):
        result = build_shortlist([], watchlist=["AMZN"], quintile_floor=80)
        assert result == ["AMZN"]

    def test_symbol_normalised_to_uppercase(self):
        candidates = [{"symbol": "aapl", "smart_money_score": 0.90}]
        result = build_shortlist(candidates, quintile_floor=80)
        assert "AAPL" in result

    def test_missing_symbol_skipped(self):
        candidates = [{"smart_money_score": 0.90}]
        result = build_shortlist(candidates, quintile_floor=80)
        assert result == []

    def test_all_below_floor_returns_empty(self):
        candidates = [_candidate("AAPL", 0.50), _candidate("MSFT", 0.60)]
        result = build_shortlist(candidates, quintile_floor=80)
        assert result == []

    def test_floor_boundary_inclusive(self):
        """Score exactly at quintile_floor should be included."""
        candidates = [_candidate("AAPL", 0.80)]  # 0.80 * 100 == 80.0
        result = build_shortlist(candidates, quintile_floor=80.0)
        assert "AAPL" in result

    def test_duplicate_candidates_deduplicated(self):
        candidates = [_candidate("AAPL", 0.90), _candidate("AAPL", 0.92)]
        result = build_shortlist(candidates, quintile_floor=80)
        assert result.count("AAPL") == 1


# ── build_prompt ──────────────────────────────────────────────────────────────

class TestBuildPrompt:
    """Verify prompt template renders correctly with various inputs.

    Arrow (1972 Nobel) — information quality entering the model determines
    the quality of the analysis; these tests validate the information channel.
    """

    def _base_quant(self, **overrides):
        data = {
            "smart_money_score": 0.85,
            "insider_score": 0.70,
            "institutional_score": 0.65,
            "momentum_score": 0.55,
        }
        data.update(overrides)
        return data

    def test_symbol_present_in_prompt(self):
        prompt = build_prompt("AAPL", self._base_quant(), [], "Bull")
        assert "AAPL" in prompt

    def test_regime_present_in_prompt(self):
        for regime in ("Bull", "Bear", "Panic", "Crash", "Euphoria"):
            prompt = build_prompt("MSFT", self._base_quant(), [], regime)
            assert regime in prompt

    def test_quant_score_in_prompt(self):
        prompt = build_prompt("AAPL", self._base_quant(smart_money_score=0.90), [], "Neutral")
        assert "90.0" in prompt

    def test_insider_block_with_events(self):
        events = [_form4_event("0001234567-24-000001", role="CEO")]
        prompt = build_prompt("AAPL", self._base_quant(), events, "Bull")
        assert "CEO" in prompt
        assert "0001234567-24-000001" in prompt

    def test_insider_block_capped_at_10(self):
        events = [_form4_event(f"0001234567-24-{i:06d}") for i in range(15)]
        prompt = build_prompt("AAPL", self._base_quant(), events, "Bull")
        # Count [EDGAR entries — should not exceed 10
        assert prompt.count("[EDGAR") <= 10

    def test_no_insider_events_shows_fallback(self):
        prompt = build_prompt("AAPL", self._base_quant(), [], "Bull")
        assert "No key-insider" in prompt

    def test_institutional_block_with_events(self):
        events = [_13f_event("0009876543-24-000099", holder="BlackRock")]
        prompt = build_prompt("AAPL", self._base_quant(), events, "Bull")
        assert "BlackRock" in prompt

    def test_institutional_block_capped_at_5(self):
        events = [_13f_event(f"0009876543-24-{i:06d}", holder=f"Fund{i}")
                  for i in range(8)]
        prompt = build_prompt("AAPL", self._base_quant(), events, "Bull")
        count = sum(1 for line in prompt.splitlines() if "Fund" in line)
        assert count <= 5

    def test_no_institutional_events_shows_fallback(self):
        prompt = build_prompt("AAPL", self._base_quant(), [], "Bull")
        assert "No 13F data" in prompt

    def test_risk_block_bear_regime(self):
        prompt = build_prompt("AAPL", self._base_quant(), [], "Bear")
        assert "Bear" in prompt
        assert "systemic risk" in prompt.lower() or "Adverse" in prompt

    def test_risk_block_weak_momentum(self):
        prompt = build_prompt("AAPL", self._base_quant(momentum_score=0.1), [], "Neutral")
        assert "momentum" in prompt.lower()

    def test_risk_block_no_insider_buying(self):
        prompt = build_prompt("AAPL", self._base_quant(insider_score=0.0), [], "Bull")
        assert "insider" in prompt.lower()

    def test_no_risk_flags_default_message(self):
        quant = self._base_quant(momentum_score=0.8, insider_score=0.9)
        prompt = build_prompt("AAPL", quant, [], "Neutral")
        assert "No specific risk flags" in prompt

    def test_prompt_version_constant(self):
        """Verify PROMPT_VERSION is v1.3 — cache-busting relies on this."""
        assert PROMPT_VERSION == "v1.3"

    def test_returns_string(self):
        prompt = build_prompt("AAPL", self._base_quant(), [], "Bull")
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_transcript_section_present_when_provided(self):
        transcript = "CEO: We are raising guidance for Q3. Revenue will be..."
        prompt = build_prompt("AAPL", self._base_quant(), [], "Bull",
                              transcript=transcript)
        assert "Recent Earnings Call" in prompt
        assert "CEO: We are raising guidance" in prompt

    def test_transcript_truncated_to_transcript_max_chars(self):
        transcript = "X" * 5000
        prompt = build_prompt("AAPL", self._base_quant(), [], "Bull",
                              transcript=transcript, transcript_max_chars=2000)
        assert prompt.count("X") == 2000

    def test_transcript_none_shows_fallback(self):
        prompt = build_prompt("AAPL", self._base_quant(), [], "Bull",
                              transcript=None)
        assert "No transcript available" in prompt

    def test_transcript_empty_string_shows_fallback(self):
        prompt = build_prompt("AAPL", self._base_quant(), [], "Bull",
                              transcript="")
        assert "No transcript available" in prompt

    def test_transcript_instructions_present_when_transcript_provided(self):
        prompt = build_prompt("AAPL", self._base_quant(), [], "Bull",
                              transcript="Some transcript text.")
        assert "Forward guidance tone" in prompt
        assert "Management confidence" in prompt
        assert "buybacks" in prompt.lower() or "buyback" in prompt.lower()

    def test_transcript_instructions_present_without_transcript(self):
        """Instructions block always included so Claude knows what to look for."""
        prompt = build_prompt("AAPL", self._base_quant(), [], "Bull",
                              transcript=None)
        assert "Forward guidance tone" in prompt

    def test_no_transcript_arg_backward_compat(self):
        """Callers that don't pass transcript still get a valid prompt."""
        prompt = build_prompt("AAPL", self._base_quant(), [], "Bull")
        assert "AAPL" in prompt
        assert "No transcript available" in prompt

    def test_prompt_version_bumped_to_v1_3(self):
        """PROMPT_VERSION must be v1.3 after this change — cache-bust."""
        assert PROMPT_VERSION == "v1.3"


# ── cross_check_citations ─────────────────────────────────────────────────────

class TestCrossCheckCitations:
    """Test the EDGAR citation verification gate.

    Akerlof (2001 Nobel) — information asymmetry between Claude's output and
    the verified filing dataset must be resolved before auto-execution.
    """

    def test_verified_citation_no_violations(self):
        analysis = _valid_analysis(citations=[
            {"source": "EDGAR Form-4", "loc": "0001234567-24-000001"}
        ])
        events = [_form4_event("0001234567-24-000001")]
        violations = cross_check_citations(analysis, events)
        assert violations == []

    def test_hallucinated_accession_creates_violation(self):
        analysis = _valid_analysis(citations=[
            {"source": "EDGAR Form-4", "loc": "9999999999-99-999999"}
        ])
        events = [_form4_event("0001234567-24-000001")]
        violations = cross_check_citations(analysis, events)
        assert len(violations) == 1
        assert "9999999999-99-999999" in violations[0]

    def test_non_edgar_source_skipped(self):
        """Citations from FMP/yfinance should not be cross-checked."""
        analysis = _valid_analysis(citations=[
            {"source": "FMP API", "loc": "some-fmp-ref"}
        ])
        violations = cross_check_citations(analysis, [])
        assert violations == []

    def test_empty_loc_on_edgar_source_is_violation(self):
        analysis = _valid_analysis(citations=[
            {"source": "EDGAR Form-4", "loc": ""}
        ])
        violations = cross_check_citations(analysis, [])
        assert len(violations) == 1
        assert "empty" in violations[0].lower()

    def test_multiple_citations_partial_violation(self):
        analysis = _valid_analysis(citations=[
            {"source": "EDGAR Form-4", "loc": "0001234567-24-000001"},  # valid
            {"source": "EDGAR Form-4", "loc": "0000000000-00-000000"},  # hallucinated
        ])
        events = [_form4_event("0001234567-24-000001")]
        violations = cross_check_citations(analysis, events)
        assert len(violations) == 1

    def test_no_citations_no_violations(self):
        analysis = _valid_analysis(citations=[])
        violations = cross_check_citations(analysis, [])
        assert violations == []

    def test_sec_in_source_also_triggers_check(self):
        """'SEC' in source is equivalent to 'EDGAR' for cross-check."""
        analysis = _valid_analysis(citations=[
            {"source": "SEC filing", "loc": "9999999999-99-999999"}
        ])
        events = []
        violations = cross_check_citations(analysis, events)
        assert len(violations) == 1

    def test_multiple_events_all_verified(self):
        events = [
            _form4_event("0001234567-24-000001"),
            _form4_event("0001234567-24-000002"),
        ]
        analysis = _valid_analysis(citations=[
            {"source": "EDGAR Form-4", "loc": "0001234567-24-000001"},
            {"source": "EDGAR Form-4", "loc": "0001234567-24-000002"},
        ])
        violations = cross_check_citations(analysis, events)
        assert violations == []

    def test_events_without_accession_ignored_in_known_set(self):
        """Events missing filing_accession do not pollute the known set."""
        events = [{"type": "form4", "shares": 1000}]  # no filing_accession
        analysis = _valid_analysis(citations=[
            {"source": "EDGAR Form-4", "loc": "0001234567-24-000001"}
        ])
        violations = cross_check_citations(analysis, events)
        assert len(violations) == 1


# ── should_auto_execute ───────────────────────────────────────────────────────

class TestShouldAutoExecute:
    """Verify all gate combinations for auto-execution.

    Black-Scholes (1997 Nobel) — execution decisions must be bounded by
    rigorous multi-factor threshold logic to avoid catastrophic positions.
    """

    def test_all_gates_pass_returns_true(self):
        analysis = _valid_analysis(score=80, confidence=0.85, recommended_action="BUY")
        assert should_auto_execute(85.0, analysis, []) is True

    def test_citation_violations_blocks(self):
        analysis = _valid_analysis(score=85, confidence=0.90, recommended_action="BUY")
        assert should_auto_execute(90.0, analysis, ["hallucination!"]) is False

    def test_quant_score_below_threshold_blocks(self):
        analysis = _valid_analysis(score=85, confidence=0.90, recommended_action="BUY")
        assert should_auto_execute(79.9, analysis, []) is False

    def test_quant_score_at_boundary_passes(self):
        analysis = _valid_analysis(score=85, confidence=0.90, recommended_action="BUY")
        assert should_auto_execute(80.0, analysis, []) is True

    def test_confidence_below_threshold_blocks(self):
        analysis = _valid_analysis(score=85, confidence=0.79, recommended_action="BUY")
        assert should_auto_execute(85.0, analysis, []) is False

    def test_confidence_at_boundary_passes(self):
        analysis = _valid_analysis(score=85, confidence=0.80, recommended_action="BUY")
        assert should_auto_execute(85.0, analysis, []) is True

    def test_claude_score_below_threshold_blocks(self):
        analysis = _valid_analysis(score=69, confidence=0.90, recommended_action="BUY")
        assert should_auto_execute(85.0, analysis, []) is False

    def test_claude_score_at_boundary_passes(self):
        analysis = _valid_analysis(score=70, confidence=0.90, recommended_action="BUY")
        assert should_auto_execute(85.0, analysis, []) is True

    def test_sell_action_blocks(self):
        analysis = _valid_analysis(score=85, confidence=0.90, recommended_action="SELL")
        assert should_auto_execute(85.0, analysis, []) is False

    def test_hold_action_blocks(self):
        analysis = _valid_analysis(score=85, confidence=0.90, recommended_action="HOLD")
        assert should_auto_execute(85.0, analysis, []) is False

    def test_watch_action_blocks(self):
        analysis = _valid_analysis(score=85, confidence=0.90, recommended_action="WATCH")
        assert should_auto_execute(85.0, analysis, []) is False

    def test_reduce_action_blocks(self):
        analysis = _valid_analysis(score=85, confidence=0.90, recommended_action="REDUCE")
        assert should_auto_execute(85.0, analysis, []) is False

    def test_none_violations_treated_as_empty(self):
        """None citation_violations defaults to no violations."""
        analysis = _valid_analysis(score=85, confidence=0.90, recommended_action="BUY")
        assert should_auto_execute(85.0, analysis, None) is True

    def test_custom_thresholds(self):
        """Custom threshold overrides should override env defaults."""
        analysis = _valid_analysis(score=60, confidence=0.70, recommended_action="BUY")
        # With lenient thresholds this should pass
        result = should_auto_execute(
            70.0, analysis, [],
            quant_threshold=65.0,
            claude_confidence_min=0.65,
            claude_score_min=55,
        )
        assert result is True

    def test_custom_thresholds_still_block_on_action(self):
        """Lenient thresholds cannot override action gate."""
        analysis = _valid_analysis(score=60, confidence=0.70, recommended_action="HOLD")
        result = should_auto_execute(
            70.0, analysis, [],
            quant_threshold=65.0,
            claude_confidence_min=0.65,
            claude_score_min=55,
        )
        assert result is False
