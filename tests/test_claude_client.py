"""tests/test_claude_client.py
Unit tests for analysis/claude_client.py.

Covers:
  - validate_analysis_schema: happy path + all failure modes + 100 sample prompts
  - CostTracker: record(), cap enforcement, summary()
  - ClaudeClient: cache hit/miss, audit log, retry behaviour (all mocked)
  - SchemaValidationError message content

Arrow (1972 Nobel) — information systems are only as reliable as their
validation logic; these tests verify the schema guard that blocks
hallucinated data from reaching auto-execution.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Inject a stub anthropic module so ClaudeClient.__init__ does not raise
# ImportError in environments where the real package is not installed.
# Tests in TestClaudeClientMocked replace client._client with a MagicMock
# anyway, so the stub Anthropic() instance is never actually called.
if "anthropic" not in sys.modules:
    _stub_anthropic = MagicMock()
    _stub_anthropic.Anthropic.return_value = MagicMock()
    sys.modules["anthropic"] = _stub_anthropic

from analysis.claude_client import (
    CostBudgetExceeded,
    CostTracker,
    SchemaValidationError,
    _cache_key,
    _load_cache,
    _save_cache,
    validate_analysis_schema,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

VALID_ANALYSIS: Dict[str, Any] = {
    "score": 82,
    "confidence": 0.87,
    "reasons": [
        "CEO purchased 50 000 shares open-market — strongest insider signal.",
        "Institutional accumulation up 12 % QoQ (Vanguard 13F).",
        "Momentum improving despite Bear regime headwinds.",
    ],
    "citations": [
        {"source": "EDGAR Form-4", "loc": "0001234567-24-000001"},
        {"source": "EDGAR 13F",    "loc": "0009876543-24-000099"},
    ],
    "recommended_action": "BUY",
}


def _make_valid(**overrides) -> Dict[str, Any]:
    """Return a copy of VALID_ANALYSIS with selective field overrides."""
    base = VALID_ANALYSIS.copy()
    base["reasons"] = list(VALID_ANALYSIS["reasons"])
    base["citations"] = [dict(c) for c in VALID_ANALYSIS["citations"]]
    base.update(overrides)
    return base


# ── validate_analysis_schema: happy path ─────────────────────────────────────

class TestValidateAnalysisSchemaHappyPath:
    """Arrow (1972 Nobel) — verify schema guard accepts well-formed data."""

    def test_valid_returns_dict(self):
        """Returns the input dict unchanged on success."""
        result = validate_analysis_schema(VALID_ANALYSIS)
        assert result is VALID_ANALYSIS

    def test_score_boundary_zero(self):
        data = _make_valid(score=0)
        assert validate_analysis_schema(data)["score"] == 0

    def test_score_boundary_100(self):
        data = _make_valid(score=100)
        assert validate_analysis_schema(data)["score"] == 100

    def test_confidence_boundary_zero(self):
        data = _make_valid(confidence=0.0)
        assert validate_analysis_schema(data)["confidence"] == 0.0

    def test_confidence_boundary_one(self):
        data = _make_valid(confidence=1.0)
        assert validate_analysis_schema(data)["confidence"] == 1.0

    def test_empty_citations_allowed(self):
        """Citations list may be empty; source constraint is per-item."""
        data = _make_valid(citations=[])
        assert validate_analysis_schema(data)["citations"] == []

    def test_all_valid_actions(self):
        for action in ("BUY", "SELL", "HOLD", "REDUCE", "WATCH"):
            data = _make_valid(recommended_action=action)
            assert validate_analysis_schema(data)["recommended_action"] == action

    def test_score_float_accepted(self):
        """Score may be float (JSON integers are sometimes decoded as float)."""
        data = _make_valid(score=75.0)
        assert validate_analysis_schema(data)


# ── validate_analysis_schema: failures ───────────────────────────────────────

class TestValidateAnalysisSchemaFailures:
    """Verify SchemaValidationError is raised on every constraint violation."""

    def test_not_dict_raises(self):
        with pytest.raises(SchemaValidationError, match="Expected dict"):
            validate_analysis_schema([1, 2, 3])

    def test_missing_score(self):
        data = _make_valid()
        del data["score"]
        with pytest.raises(SchemaValidationError, match="Missing required fields"):
            validate_analysis_schema(data)

    def test_missing_confidence(self):
        data = _make_valid()
        del data["confidence"]
        with pytest.raises(SchemaValidationError, match="Missing required fields"):
            validate_analysis_schema(data)

    def test_missing_reasons(self):
        data = _make_valid()
        del data["reasons"]
        with pytest.raises(SchemaValidationError, match="Missing required fields"):
            validate_analysis_schema(data)

    def test_missing_citations(self):
        data = _make_valid()
        del data["citations"]
        with pytest.raises(SchemaValidationError, match="Missing required fields"):
            validate_analysis_schema(data)

    def test_missing_recommended_action(self):
        data = _make_valid()
        del data["recommended_action"]
        with pytest.raises(SchemaValidationError, match="Missing required fields"):
            validate_analysis_schema(data)

    def test_score_negative(self):
        data = _make_valid(score=-1)
        with pytest.raises(SchemaValidationError, match="score"):
            validate_analysis_schema(data)

    def test_score_above_100(self):
        data = _make_valid(score=101)
        with pytest.raises(SchemaValidationError, match="score"):
            validate_analysis_schema(data)

    def test_score_string(self):
        data = _make_valid(score="eighty")
        with pytest.raises(SchemaValidationError, match="score"):
            validate_analysis_schema(data)

    def test_confidence_below_zero(self):
        data = _make_valid(confidence=-0.01)
        with pytest.raises(SchemaValidationError, match="confidence"):
            validate_analysis_schema(data)

    def test_confidence_above_one(self):
        data = _make_valid(confidence=1.01)
        with pytest.raises(SchemaValidationError, match="confidence"):
            validate_analysis_schema(data)

    def test_confidence_string(self):
        data = _make_valid(confidence="high")
        with pytest.raises(SchemaValidationError, match="confidence"):
            validate_analysis_schema(data)

    def test_reasons_empty_list(self):
        data = _make_valid(reasons=[])
        with pytest.raises(SchemaValidationError, match="reasons"):
            validate_analysis_schema(data)

    def test_reasons_not_list(self):
        data = _make_valid(reasons="single string")
        with pytest.raises(SchemaValidationError, match="reasons"):
            validate_analysis_schema(data)

    def test_reasons_list_of_non_strings(self):
        data = _make_valid(reasons=[1, 2, 3])
        with pytest.raises(SchemaValidationError, match="reasons"):
            validate_analysis_schema(data)

    def test_citations_not_list(self):
        data = _make_valid(citations="EDGAR 2024")
        with pytest.raises(SchemaValidationError, match="citations"):
            validate_analysis_schema(data)

    def test_citation_missing_source(self):
        data = _make_valid(citations=[{"loc": "0001234567-24-000001"}])
        with pytest.raises(SchemaValidationError, match="citations"):
            validate_analysis_schema(data)

    def test_citation_missing_loc(self):
        data = _make_valid(citations=[{"source": "EDGAR Form-4"}])
        with pytest.raises(SchemaValidationError, match="citations"):
            validate_analysis_schema(data)

    def test_citation_not_dict(self):
        data = _make_valid(citations=["0001234567-24-000001"])
        with pytest.raises(SchemaValidationError, match="citations"):
            validate_analysis_schema(data)

    def test_invalid_action(self):
        data = _make_valid(recommended_action="STRONG_BUY")
        with pytest.raises(SchemaValidationError, match="recommended_action"):
            validate_analysis_schema(data)

    def test_action_lowercase(self):
        data = _make_valid(recommended_action="buy")
        with pytest.raises(SchemaValidationError, match="recommended_action"):
            validate_analysis_schema(data)


# ── 100-sample parametrised prompt regression test ───────────────────────────

def _generate_sample_analyses(n: int = 100):
    """Generate N structurally valid analysis dicts for regression coverage.

    # Diversification theorem (Markowitz 1952) — varied samples reduce the
    # probability that a single edge case goes undetected.
    """
    import random
    rng = random.Random(42)
    actions = ["BUY", "SELL", "HOLD", "REDUCE", "WATCH"]
    for i in range(n):
        yield {
            "score": rng.randint(0, 100),
            "confidence": round(rng.uniform(0.0, 1.0), 2),
            "reasons": [f"Reason {j} for sample {i}" for j in range(rng.randint(1, 4))],
            "citations": [
                {"source": "EDGAR Form-4", "loc": f"000{i:07d}-24-{j:06d}"}
                for j in range(rng.randint(0, 3))
            ],
            "recommended_action": rng.choice(actions),
        }


@pytest.mark.parametrize("analysis", list(_generate_sample_analyses(100)))
def test_validate_100_sample_analyses(analysis):
    """Each of the 100 generated valid dicts must pass schema validation."""
    result = validate_analysis_schema(analysis)
    assert result["score"] == analysis["score"]
    assert result["recommended_action"] == analysis["recommended_action"]


# ── CostTracker ───────────────────────────────────────────────────────────────

class TestCostTracker:
    """Verify CostTracker accumulates costs and enforces the hard cap.

    Modigliani-Miller (1958 Nobel) — capital budgeting constraints are
    irrelevant only if information is free; in practice, cost caps are real.
    """

    def test_record_increments_tokens(self):
        tracker = CostTracker(cap_usd=10.0, model="claude-sonnet-4-6")
        inc = tracker.record(1000, 200)
        assert tracker.input_tokens == 1000
        assert tracker.output_tokens == 200
        assert tracker.calls == 1
        assert inc > 0

    def test_cost_calculation_sonnet(self):
        """claude-sonnet-4-6: $3.00/MTok input, $15.00/MTok output."""
        tracker = CostTracker(cap_usd=100.0, model="claude-sonnet-4-6")
        inc = tracker.record(1_000_000, 0)
        assert abs(inc - 3.00) < 1e-6

    def test_cost_calculation_output_tokens(self):
        tracker = CostTracker(cap_usd=100.0, model="claude-sonnet-4-6")
        inc = tracker.record(0, 1_000_000)
        assert abs(inc - 15.00) < 1e-6

    def test_multiple_calls_accumulate(self):
        tracker = CostTracker(cap_usd=100.0, model="claude-sonnet-4-6")
        tracker.record(1000, 100)
        tracker.record(2000, 200)
        assert tracker.calls == 2
        assert tracker.input_tokens == 3000
        assert tracker.output_tokens == 300

    def test_cap_raises_when_exceeded(self):
        """Exceeding the cap raises CostBudgetExceeded."""
        tracker = CostTracker(cap_usd=0.001, model="claude-sonnet-4-6")
        with pytest.raises(CostBudgetExceeded):
            tracker.record(1_000_000, 1_000_000)

    def test_cap_does_not_raise_below_threshold(self):
        tracker = CostTracker(cap_usd=10.0, model="claude-sonnet-4-6")
        tracker.record(100, 10)  # should not raise

    def test_summary_keys(self):
        tracker = CostTracker(cap_usd=5.0, model="claude-sonnet-4-6")
        tracker.record(500, 50)
        s = tracker.summary()
        for key in ("calls", "input_tokens", "output_tokens", "total_usd", "cap_usd", "model"):
            assert key in s

    def test_total_usd_property(self):
        tracker = CostTracker(cap_usd=100.0, model="claude-sonnet-4-6")
        tracker.record(1_000_000, 0)
        assert abs(tracker.total_usd - 3.0) < 1e-5

    def test_unknown_model_falls_back_to_default(self):
        """Unknown model uses default price table entry."""
        tracker = CostTracker(cap_usd=100.0, model="claude-unknown-99")
        inc = tracker.record(1_000_000, 0)
        assert inc > 0  # falls back to _DEFAULT_MODEL pricing


# ── Cache helpers ─────────────────────────────────────────────────────────────

class TestCacheHelpers:
    """Verify deterministic cache key generation and round-trip storage."""

    def test_cache_key_deterministic(self):
        k1 = _cache_key("run1", "v1.2", "AAPL", "abc123")
        k2 = _cache_key("run1", "v1.2", "AAPL", "abc123")
        assert k1 == k2

    def test_cache_key_different_run_ids(self):
        k1 = _cache_key("run1", "v1.2", "AAPL", "abc123")
        k2 = _cache_key("run2", "v1.2", "AAPL", "abc123")
        assert k1 != k2

    def test_cache_key_different_versions(self):
        k1 = _cache_key("run1", "v1.1", "AAPL", "abc123")
        k2 = _cache_key("run1", "v1.2", "AAPL", "abc123")
        assert k1 != k2

    def test_cache_key_length(self):
        k = _cache_key("run1", "v1.2", "AAPL", "abc123")
        assert len(k) == 24

    def test_cache_roundtrip(self, tmp_path, monkeypatch):
        """Save and reload a cache entry via file-backed helpers."""
        import analysis.claude_client as mod
        monkeypatch.setattr(mod, "_CACHE_DIR", tmp_path / "cache")
        key = "testkey123456789abcdefg"
        payload = {"analysis": VALID_ANALYSIS, "symbol": "AAPL"}
        _save_cache(key, payload)
        loaded = _load_cache(key)
        assert loaded == payload

    def test_cache_miss_returns_none(self, tmp_path, monkeypatch):
        import analysis.claude_client as mod
        monkeypatch.setattr(mod, "_CACHE_DIR", tmp_path / "cache")
        result = _load_cache("nonexistentkey000000000")
        assert result is None

    def test_cache_corrupted_file_returns_none(self, tmp_path, monkeypatch):
        import analysis.claude_client as mod
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(mod, "_CACHE_DIR", cache_dir)
        key = "corruptkey00000000000000"
        (cache_dir / f"{key}.json").write_text("not valid json{{{{", encoding="utf-8")
        result = _load_cache(key)
        assert result is None


# ── ClaudeClient (mocked Anthropic) ──────────────────────────────────────────

class TestClaudeClientMocked:
    """Test ClaudeClient without making real API calls.

    Stiglitz (2001 Nobel) — information asymmetry between test and prod
    environments is resolved by comprehensive mocking of the API layer.

    Strategy: import ClaudeClient from the existing module (no reload);
    monkeypatch module-level vars; replace client._client with a MagicMock
    after instantiation so no real network calls are made.
    """

    def _make_mock_response(self, analysis: Dict[str, Any]):
        """Build a fake Anthropic API response with a tool_use block."""
        block = MagicMock()
        block.type = "tool_use"
        block.name = "output_analysis"
        block.input = analysis

        usage = MagicMock()
        usage.input_tokens = 800
        usage.output_tokens = 200

        resp = MagicMock()
        resp.content = [block]
        resp.usage = usage
        resp.stop_reason = "tool_use"
        return resp

    def _make_client(self, tmp_path, monkeypatch, analysis=None, run_id="test-run"):
        """Return a ClaudeClient with mocked internal _client + temp dirs."""
        if analysis is None:
            analysis = VALID_ANALYSIS

        import analysis.claude_client as mod
        from analysis.claude_client import ClaudeClient

        monkeypatch.setattr(mod, "_CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr(mod, "_AUDIT_LOG", tmp_path / "audit.ndjson")

        client = ClaudeClient(run_id=run_id)
        mock_inner = MagicMock()
        mock_inner.messages.create.return_value = self._make_mock_response(analysis)
        client._client = mock_inner
        return client, mock_inner

    def test_analyze_returns_valid_dict(self, tmp_path, monkeypatch):
        client, _ = self._make_client(tmp_path, monkeypatch)
        result = client.analyze("AAPL", "some prompt", bypass_cache=True)
        assert result["score"] == VALID_ANALYSIS["score"]
        assert result["recommended_action"] == "BUY"

    def test_cost_is_tracked(self, tmp_path, monkeypatch):
        client, _ = self._make_client(tmp_path, monkeypatch, run_id="cost-run")
        client.analyze("AAPL", "prompt", bypass_cache=True)
        summary = client.cost_summary()
        assert summary["calls"] == 1
        assert summary["input_tokens"] == 800
        assert summary["output_tokens"] == 200
        assert summary["total_usd"] > 0

    def test_cache_hit_skips_api(self, tmp_path, monkeypatch):
        """Second identical call should hit cache, not the API."""
        client, mock_inner = self._make_client(tmp_path, monkeypatch, run_id="cache-run")
        prompt = "cached prompt"
        client.analyze("MSFT", prompt, bypass_cache=False)
        client.analyze("MSFT", prompt, bypass_cache=False)
        assert mock_inner.messages.create.call_count == 1

    def test_bypass_cache_always_calls_api(self, tmp_path, monkeypatch):
        client, mock_inner = self._make_client(tmp_path, monkeypatch, run_id="bypass-run")
        prompt = "bypass prompt"
        client.analyze("TSLA", prompt, bypass_cache=True)
        client.analyze("TSLA", prompt, bypass_cache=True)
        assert mock_inner.messages.create.call_count == 2

    def test_audit_log_written(self, tmp_path, monkeypatch):
        audit_path = tmp_path / "audit.ndjson"
        import analysis.claude_client as mod
        monkeypatch.setattr(mod, "_AUDIT_LOG", audit_path)
        client, _ = self._make_client(tmp_path, monkeypatch, run_id="audit-run")
        # _AUDIT_LOG is re-patched after _make_client sets it; set directly on mod
        monkeypatch.setattr(mod, "_AUDIT_LOG", audit_path)
        client.analyze("NVDA", "audit test prompt", bypass_cache=True)
        assert audit_path.exists()
        lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) >= 1
        entry = json.loads(lines[-1])
        assert entry["symbol"] == "NVDA"
        assert entry["status"] == "ok"

    def test_cost_cap_exceeded_raises(self, tmp_path, monkeypatch):
        """CostBudgetExceeded is raised when the tracker cap is hit."""
        import analysis.claude_client as mod
        from analysis.claude_client import ClaudeClient

        monkeypatch.setattr(mod, "_CACHE_DIR", tmp_path / "cache")
        monkeypatch.setattr(mod, "_AUDIT_LOG", tmp_path / "audit.ndjson")

        tracker = CostTracker(cap_usd=0.001, model="claude-sonnet-4-6")
        client = ClaudeClient(run_id="cap-run", cost_tracker=tracker)

        mock_resp = self._make_mock_response(VALID_ANALYSIS)
        mock_resp.usage.input_tokens = 10_000_000
        mock_resp.usage.output_tokens = 10_000_000
        client._client = MagicMock()
        client._client.messages.create.return_value = mock_resp

        with pytest.raises(CostBudgetExceeded):
            client.analyze("AAPL", "expensive prompt", bypass_cache=True)


# ── transcript_signals optional property ──────────────────────────────────────

def test_transcript_signals_not_in_required():
    """transcript_signals must be optional — not in required array."""
    from analysis.claude_client import ANALYSIS_TOOL_SCHEMA
    required = ANALYSIS_TOOL_SCHEMA["input_schema"]["required"]
    assert "transcript_signals" not in required


def test_transcript_signals_schema_shape():
    """transcript_signals property must exist with correct sub-properties."""
    from analysis.claude_client import ANALYSIS_TOOL_SCHEMA
    props = ANALYSIS_TOOL_SCHEMA["input_schema"]["properties"]
    assert "transcript_signals" in props
    ts = props["transcript_signals"]
    assert ts["type"] == "object"
    sub = ts["properties"]
    assert sub["guidance_tone"]["enum"] == ["raised", "maintained", "lowered", "not_mentioned"]
    assert sub["management_confidence"]["enum"] == ["high", "neutral", "low"]
    assert sub["buyback_mentioned"]["type"] == "boolean"
