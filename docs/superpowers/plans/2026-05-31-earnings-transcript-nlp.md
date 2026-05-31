# Earnings Transcript NLP Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Feed FMP earnings call transcript excerpts into Claude's prompt so it can cross-reference management guidance tone and confidence against quantitative factor scores.

**Architecture:** `FMPClient` gains `get_earnings_transcript()` (new method, "transcript" cache bucket, soft-fail); `build_prompt()` gains an optional `transcript` param and injects a new section; `run_analysis()` gains an optional `fmp_client` param, fetches the transcript per symbol, and passes it through. `ANALYSIS_TOOL_SCHEMA` gains an optional `transcript_signals` output property. `ClaudeClient` is untouched.

**Tech Stack:** Python 3.11, `requests`, `anthropic>=0.28.0`, `pytest`, existing FMPClient cache infrastructure.

---

## File Map

| File | Change |
|---|---|
| `regime_trader/services/fmp_client.py` | Add `get_earnings_transcript()` method |
| `analysis/earnings_analyzer.py` | Add `transcript` param to `build_prompt()`; add `fmp_client` param to `run_analysis()`; bump `PROMPT_VERSION` |
| `analysis/claude_client.py` | Add optional `transcript_signals` to `ANALYSIS_TOOL_SCHEMA` |
| `.github/workflows/hybrid_pipeline.yml` | Update preflight cost gate token estimate |
| `tests/test_fmp_client.py` | Add `TestGetEarningsTranscript` class |
| `tests/test_earnings_analyzer.py` | Add transcript tests to `TestBuildPrompt`; add transcript tests to `run_analysis` |

---

## Task 1: `FMPClient.get_earnings_transcript` — tests first

**Files:**
- Test: `tests/test_fmp_client.py`

- [ ] **Step 1: Write the failing tests**

Append this class to `tests/test_fmp_client.py`:

```python
class TestGetEarningsTranscript:
    """get_earnings_transcript fetches stable/earning-call-transcript-latest.

    Cache bucket: "transcript" (24h TTL).
    Soft-fail: returns None on error, empty list, or FMPEndpointError.
    max_chars=3000 fetch ceiling is intentionally larger than build_prompt's
    2000-char injection limit — no second network call needed if budget changes.
    """

    def test_returns_content_truncated_to_max_chars(self, client):
        long_content = "A" * 5000
        payload = [{"symbol": "AAPL", "quarter": 1, "year": 2026,
                    "date": "2026-01-15", "content": long_content}]
        with patch.object(client._session, "get", return_value=_ok_resp(payload)):
            result = client.get_earnings_transcript("AAPL", max_chars=3000)
        assert result == "A" * 3000

    def test_returns_full_content_when_shorter_than_max_chars(self, client):
        content = "Short transcript text."
        payload = [{"symbol": "AAPL", "quarter": 1, "year": 2026,
                    "date": "2026-01-15", "content": content}]
        with patch.object(client._session, "get", return_value=_ok_resp(payload)):
            result = client.get_earnings_transcript("AAPL")
        assert result == content

    def test_returns_none_on_empty_list(self, client):
        with patch.object(client._session, "get", return_value=_empty_resp()):
            result = client.get_earnings_transcript("AAPL")
        assert result is None

    def test_returns_none_on_missing_content_key(self, client):
        payload = [{"symbol": "AAPL", "quarter": 1, "year": 2026, "date": "2026-01-15"}]
        with patch.object(client._session, "get", return_value=_ok_resp(payload)):
            result = client.get_earnings_transcript("AAPL")
        assert result is None

    def test_returns_none_when_no_api_key(self, tmp_path, monkeypatch):
        monkeypatch.delenv("FMP_API_KEY", raising=False)
        c = FMPClient(api_key="", cache_root=tmp_path / "fmp")
        result = c.get_earnings_transcript("AAPL")
        assert result is None

    def test_caches_result(self, client):
        content = "Transcript text."
        payload = [{"symbol": "AAPL", "quarter": 1, "year": 2026,
                    "date": "2026-01-15", "content": content}]
        with patch.object(client._session, "get", return_value=_ok_resp(payload)) as mock_get:
            client.get_earnings_transcript("AAPL")
            client.get_earnings_transcript("AAPL")
        assert mock_get.call_count == 1  # second call served from cache

    def test_returns_none_on_fmp_endpoint_error(self, client):
        from regime_trader.services.fmp_client import FMPEndpointError
        with patch.object(client, "_get", side_effect=FMPEndpointError("earning-call-transcript-latest", 404)):
            result = client.get_earnings_transcript("AAPL")
        assert result is None

    def test_returns_none_on_network_exception(self, client):
        with patch.object(client, "_get", side_effect=RuntimeError("timeout")):
            result = client.get_earnings_transcript("AAPL")
        assert result is None
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_fmp_client.py::TestGetEarningsTranscript -v
```

Expected: 7 failures with `AttributeError: 'FMPClient' object has no attribute 'get_earnings_transcript'`

---

## Task 2: Implement `FMPClient.get_earnings_transcript`

**Files:**
- Modify: `regime_trader/services/fmp_client.py`

- [ ] **Step 3: Add the method**

In [fmp_client.py](regime_trader/services/fmp_client.py), insert after the `get_cash_flow_statements` method (before `# ── Health report ──`):

```python
def get_earnings_transcript(self, ticker: str, max_chars: int = 3000) -> Optional[str]:
    """Executive remarks from the most recent earnings call.

    Fetches stable/earning-call-transcript-latest (limit=1).
    Returns content[:max_chars] on success; None on any failure.

    max_chars (default 3000) is intentionally larger than build_prompt's
    transcript_max_chars (default 2000) — the delta sits in memory and is
    discarded. This avoids a second network call if the prompt budget changes.

    Cache bucket: "transcript" (24h TTL — transcripts don't change after
    publication). Soft-fail: FMPEndpointError and network exceptions return
    None; the transcript is additive enrichment, not a scored factor.
    """
    if not self._api_key:
        return None
    cached = self._cache_read("transcript", ticker)
    if cached is not None:
        return cached
    try:
        data = self._get(
            "earning-call-transcript-latest",
            {"symbol": ticker, "limit": 1},
            bucket="transcript",
        ) or []
        if not isinstance(data, list) or not data:
            return None
        content = data[0].get("content")
        if not content:
            return None
        result = content[:max_chars]
        self._cache_write("transcript", ticker, result)
        return result
    except FMPEndpointError:
        return None
    except Exception as exc:
        log.debug("get_earnings_transcript %s failed: %s", ticker, exc)
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_fmp_client.py::TestGetEarningsTranscript -v
```

Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add regime_trader/services/fmp_client.py tests/test_fmp_client.py
git commit -m "feat(fmp): add get_earnings_transcript() — stable/earning-call-transcript-latest"
```

---

## Task 3: `build_prompt` transcript param — tests first

**Files:**
- Test: `tests/test_earnings_analyzer.py`

- [ ] **Step 6: Write the failing tests**

Append these test methods inside the existing `TestBuildPrompt` class in `tests/test_earnings_analyzer.py`:

```python
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
```

- [ ] **Step 7: Run tests to verify they fail**

```
pytest tests/test_earnings_analyzer.py::TestBuildPrompt -v
```

Expected: the 8 new tests fail (signature mismatch, version still "v1.2"), existing tests still pass.

---

## Task 4: Implement `build_prompt` transcript param + bump PROMPT_VERSION

**Files:**
- Modify: `analysis/earnings_analyzer.py`

- [ ] **Step 8: Bump PROMPT_VERSION to v1.3**

In [earnings_analyzer.py](analysis/earnings_analyzer.py), line 37:

```python
# Before
PROMPT_VERSION = "v1.2"

# After
PROMPT_VERSION = "v1.3"
```

- [ ] **Step 9: Update `_USER_PROMPT_TEMPLATE` to add transcript section and extended instructions**

Replace the existing `_USER_PROMPT_TEMPLATE` string (lines 105-135 of [earnings_analyzer.py](analysis/earnings_analyzer.py)):

```python
# v1.3 — added transcript section and qualitative cross-reference instructions
_USER_PROMPT_TEMPLATE = """\
## Equity Analysis Request — {symbol}

### Quant Signal Summary
- Composite quant score: {quant_score:.1f}/100
- Insider conviction: {insider_score:.2f}
- Institutional accumulation: {inst_score:.2f}
- Momentum: {momentum_score:.2f}
- Current market regime: {regime}

### Recent Insider Transactions (EDGAR Form-4, last 90 days)
{insider_block}

### Institutional Position Changes (EDGAR 13F, last quarter)
{inst_block}

### Key Risk Factors
{risk_block}

### Recent Earnings Call (last quarter — executive remarks excerpt)
{transcript_block}

### Analysis Instructions
Synthesize the quantitative signal and the SEC filing evidence above.
Focus on:
1. Are insider purchases consistent in size and role (CEO/CFO > Director)?
2. Are institutions accumulating or distributing?
3. What is the most likely 30-day price catalyst?
4. What are the top 2 risks that could invalidate the long thesis?

If a transcript is provided, identify:
1. Forward guidance tone (raised/maintained/lowered)
2. Management confidence signals (hedging language vs conviction)
3. Any mention of buybacks, M&A, or restructuring
Cross-reference these qualitative signals against the quantitative factors.

Return your analysis via the output_analysis tool with score, confidence,
reasons (≥3 points), citations (anchor each fact to a filing), and
recommended_action.
"""
```

- [ ] **Step 10: Update `build_prompt` signature and body**

Replace the existing `build_prompt` function signature and its return call in [earnings_analyzer.py](analysis/earnings_analyzer.py):

```python
def build_prompt(
    symbol: str,
    quant_data: Dict[str, Any],
    parsed_events: List[Dict[str, Any]],
    regime: str = "Unknown",
    transcript: Optional[str] = None,
    transcript_max_chars: int = 2000,
) -> str:
    """Build a compressed, token-efficient prompt for a single symbol.

    Context budget: ≤ 4000 tokens. Insider events are capped at 10 rows.
    Institutional changes are capped at 5 rows. Transcript injected up to
    transcript_max_chars (default 2000) — smaller than the FMPClient fetch
    ceiling of 3000 so the prompt budget can change without a new network call.

    Args:
        symbol:              Ticker.
        quant_data:          Dict from discovery_scanner ScanResult.
        parsed_events:       Form-4 events from edgar_parse.parse_form4_file().
        regime:              Current regime label (VIX/HMM output).
        transcript:          Raw transcript text from FMPClient.get_earnings_transcript().
                             None when unavailable — prompt uses a fallback message.
        transcript_max_chars: Max chars of transcript injected into prompt (default 2000).

    Returns:
        Formatted user-turn prompt string.
    """
```

Then, just before the final `return _USER_PROMPT_TEMPLATE.format(...)`, add the transcript block:

```python
    # ── Transcript block ───────────────────────────────────────────────────────
    if transcript:
        transcript_block = transcript[:transcript_max_chars]
    else:
        transcript_block = "No transcript available — analysis based on filing data only."
```

And add `transcript_block=transcript_block,` to the `.format()` call:

```python
    return _USER_PROMPT_TEMPLATE.format(
        symbol=symbol,
        quant_score=quant_data.get("smart_money_score", 0) * 100,
        insider_score=quant_data.get("insider_score", 0),
        inst_score=quant_data.get("institutional_score", 0),
        momentum_score=quant_data.get("momentum_score", 0),
        regime=regime,
        insider_block=insider_block,
        inst_block=inst_block,
        risk_block=risk_block,
        transcript_block=transcript_block,
    )
```

- [ ] **Step 11: Run all earnings_analyzer tests**

```
pytest tests/test_earnings_analyzer.py -v
```

Expected: all pass, including the 8 new transcript tests. The existing `test_prompt_version_constant` that asserted `"v1.2"` must now be updated to `"v1.3"` — find it and fix it:

```python
    def test_prompt_version_constant(self):
        """Verify PROMPT_VERSION is v1.3 — cache-busting relies on this."""
        assert PROMPT_VERSION == "v1.3"
```

Re-run until all green.

- [ ] **Step 12: Commit**

```bash
git add analysis/earnings_analyzer.py tests/test_earnings_analyzer.py
git commit -m "feat(analyzer): add transcript param to build_prompt, bump PROMPT_VERSION to v1.3"
```

---

## Task 5: `run_analysis` `fmp_client` param — tests first

**Files:**
- Test: `tests/test_earnings_analyzer.py`

- [ ] **Step 13: Write the failing tests**

Add this new class to `tests/test_earnings_analyzer.py`:

```python
# ── run_analysis transcript injection ─────────────────────────────────────────

class TestRunAnalysisTranscript:
    """Verify run_analysis fetches transcripts and passes them to build_prompt."""

    def _mock_claude_client(self):
        from unittest.mock import MagicMock
        client = MagicMock()
        client.analyze.return_value = _valid_analysis()
        return client

    def _mock_fmp_client(self, transcript: str | None = "Transcript text."):
        from unittest.mock import MagicMock
        fmp = MagicMock()
        fmp.get_earnings_transcript.return_value = transcript
        return fmp

    def test_fmp_client_injected_and_called_per_symbol(self):
        from analysis.earnings_analyzer import run_analysis
        fmp = self._mock_fmp_client()
        claude = self._mock_claude_client()

        run_analysis(
            shortlist=["AAPL", "MSFT"],
            quant_map={"AAPL": _candidate("AAPL"), "MSFT": _candidate("MSFT")},
            filings_map={},
            client=claude,
            fmp_client=fmp,
        )

        assert fmp.get_earnings_transcript.call_count == 2
        fmp.get_earnings_transcript.assert_any_call("AAPL")
        fmp.get_earnings_transcript.assert_any_call("MSFT")

    def test_transcript_none_does_not_abort_analysis(self):
        from analysis.earnings_analyzer import run_analysis
        fmp = self._mock_fmp_client(transcript=None)
        claude = self._mock_claude_client()

        results = run_analysis(
            shortlist=["AAPL"],
            quant_map={"AAPL": _candidate("AAPL")},
            filings_map={},
            client=claude,
            fmp_client=fmp,
        )

        assert len(results) == 1
        assert results[0].error is None

    def test_transcript_fetch_exception_does_not_abort_analysis(self):
        from analysis.earnings_analyzer import run_analysis
        from unittest.mock import MagicMock
        fmp = MagicMock()
        fmp.get_earnings_transcript.side_effect = RuntimeError("network error")
        claude = self._mock_claude_client()

        results = run_analysis(
            shortlist=["AAPL"],
            quant_map={"AAPL": _candidate("AAPL")},
            filings_map={},
            client=claude,
            fmp_client=fmp,
        )

        assert len(results) == 1
        assert results[0].error is None  # transcript failure ≠ analysis failure

    def test_fmp_client_instantiated_when_not_provided(self):
        """When fmp_client is None, run_analysis creates one internally."""
        from analysis.earnings_analyzer import run_analysis
        from unittest.mock import MagicMock, patch
        claude = self._mock_claude_client()

        with patch("analysis.earnings_analyzer.FMPClient") as MockFMP:
            mock_instance = MagicMock()
            mock_instance.get_earnings_transcript.return_value = None
            MockFMP.return_value = mock_instance

            run_analysis(
                shortlist=["AAPL"],
                quant_map={"AAPL": _candidate("AAPL")},
                filings_map={},
                client=claude,
                fmp_client=None,
            )

        MockFMP.assert_called_once()
        mock_instance.get_earnings_transcript.assert_called_once_with("AAPL")
```

- [ ] **Step 14: Run tests to verify they fail**

```
pytest tests/test_earnings_analyzer.py::TestRunAnalysisTranscript -v
```

Expected: 4 failures — `run_analysis` doesn't accept `fmp_client` yet.

---

## Task 6: Implement `run_analysis` `fmp_client` param

**Files:**
- Modify: `analysis/earnings_analyzer.py`

- [ ] **Step 15: Add FMPClient import**

At the top of [earnings_analyzer.py](analysis/earnings_analyzer.py), add the import after the existing imports:

```python
from regime_trader.services.fmp_client import FMPClient
```

- [ ] **Step 16: Update `run_analysis` signature and body**

Replace the `run_analysis` function signature:

```python
def run_analysis(
    shortlist: List[str],
    quant_map: Dict[str, Dict[str, Any]],
    filings_map: Dict[str, List[Dict[str, Any]]],
    regime: str = "Unknown",
    *,
    client: Optional[ClaudeClient] = None,
    fmp_client: Optional[FMPClient] = None,
    run_id: Optional[str] = None,
    bypass_cache: bool = False,
) -> List[AnalysisResult]:
```

Inside the function body, immediately after `if client is None: client = ClaudeClient(run_id=run_id)`, add:

```python
    if fmp_client is None:
        fmp_client = FMPClient()
```

Then, inside the `for symbol in shortlist:` loop, replace:

```python
        try:
            prompt = build_prompt(symbol, quant_data, parsed_events, regime)
```

with:

```python
        try:
            try:
                transcript = fmp_client.get_earnings_transcript(symbol)
            except Exception as exc:
                log.warning("[ANALYZER] transcript fetch failed for %s: %s", symbol, exc)
                transcript = None
            prompt = build_prompt(symbol, quant_data, parsed_events, regime,
                                  transcript=transcript)
```

- [ ] **Step 17: Run the full test suite**

```
pytest tests/test_earnings_analyzer.py -v
```

Expected: all pass.

- [ ] **Step 18: Commit**

```bash
git add analysis/earnings_analyzer.py tests/test_earnings_analyzer.py
git commit -m "feat(analyzer): inject fmp_client into run_analysis, fetch transcript per symbol"
```

---

## Task 7: `ANALYSIS_TOOL_SCHEMA` — add optional `transcript_signals`

**Files:**
- Modify: `analysis/claude_client.py`
- Test: `tests/test_claude_client.py`

- [ ] **Step 19: Write the failing test**

Read `tests/test_claude_client.py` briefly to find where schema tests live, then append:

```python
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
```

- [ ] **Step 20: Run tests to verify they fail**

```
pytest tests/test_claude_client.py::test_transcript_signals_not_in_required tests/test_claude_client.py::test_transcript_signals_schema_shape -v
```

Expected: 2 failures — key not present yet.

- [ ] **Step 21: Add `transcript_signals` to `ANALYSIS_TOOL_SCHEMA`**

In [claude_client.py](analysis/claude_client.py), inside `ANALYSIS_TOOL_SCHEMA["input_schema"]["properties"]`, add after the `"recommended_action"` block:

```python
            "transcript_signals": {
                "type": "object",
                "description": (
                    "Qualitative signals extracted from the earnings transcript. "
                    "Omit entirely when no transcript was provided."
                ),
                "properties": {
                    "guidance_tone": {
                        "type": "string",
                        "enum": ["raised", "maintained", "lowered", "not_mentioned"],
                        "description": "Whether management raised, maintained, or lowered forward guidance.",
                    },
                    "management_confidence": {
                        "type": "string",
                        "enum": ["high", "neutral", "low"],
                        "description": "Overall confidence tone (hedging language vs conviction).",
                    },
                    "buyback_mentioned": {
                        "type": "boolean",
                        "description": "True if buybacks, M&A, or restructuring were mentioned.",
                    },
                },
            },
```

**Do not add `"transcript_signals"` to the `"required"` array.**

- [ ] **Step 22: Run tests to verify they pass**

```
pytest tests/test_claude_client.py::test_transcript_signals_not_in_required tests/test_claude_client.py::test_transcript_signals_schema_shape -v
```

Expected: 2 passed.

- [ ] **Step 23: Commit**

```bash
git add analysis/claude_client.py tests/test_claude_client.py
git commit -m "feat(schema): add optional transcript_signals to ANALYSIS_TOOL_SCHEMA"
```

---

## Task 8: Update preflight cost gate token estimate in `hybrid_pipeline.yml`

**Files:**
- Modify: `.github/workflows/hybrid_pipeline.yml`

- [ ] **Step 24: Update the arithmetic and comment**

In [hybrid_pipeline.yml](.github/workflows/hybrid_pipeline.yml), find the preflight cost gate step (lines ~210-213). Replace:

```python
          # Conservative estimate: ~1k input + 300 output tokens per symbol
          # at claude-sonnet-4-6 pricing (\$3/Mtoken input, \$15/Mtoken output)
          input_cost  = len(shortlist) * 1000 * 3.00  / 1_000_000
          output_cost = len(shortlist) * 300  * 15.00 / 1_000_000
```

with:

```python
          # ~1500 input + 300 output tokens per symbol (includes 2000-char transcript block)
          # at claude-sonnet-4-6 pricing (\$3/Mtoken input, \$15/Mtoken output)
          input_cost  = len(shortlist) * 1500 * 3.00  / 1_000_000
          output_cost = len(shortlist) * 300  * 15.00 / 1_000_000
```

- [ ] **Step 25: Verify the YAML is valid**

```
python -c "import yaml; yaml.safe_load(open('.github/workflows/hybrid_pipeline.yml'))"
```

Expected: no output (no error).

- [ ] **Step 26: Commit**

```bash
git add .github/workflows/hybrid_pipeline.yml
git commit -m "chore(ci): update preflight cost gate token estimate to 1500 input (transcript block)"
```

---

## Task 9: Full suite verification

- [ ] **Step 27: Run the full test suite**

```
pytest tests/ -v --tb=short 2>&1 | tail -30
```

Expected: all existing tests pass, new tests pass, no regressions.

- [ ] **Step 28: Smoke-check the import chain**

```
python -c "
from regime_trader.services.fmp_client import FMPClient
from analysis.earnings_analyzer import build_prompt, run_analysis, PROMPT_VERSION
from analysis.claude_client import ANALYSIS_TOOL_SCHEMA
print('PROMPT_VERSION:', PROMPT_VERSION)
print('transcript_signals in schema:', 'transcript_signals' in ANALYSIS_TOOL_SCHEMA['input_schema']['properties'])
print('transcript_signals in required:', 'transcript_signals' in ANALYSIS_TOOL_SCHEMA['input_schema']['required'])
c = FMPClient(api_key='')
print('get_earnings_transcript callable:', callable(c.get_earnings_transcript))
"
```

Expected output:
```
PROMPT_VERSION: v1.3
transcript_signals in schema: True
transcript_signals in required: False
get_earnings_transcript callable: True
```

---

## Self-Review Checklist

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| `get_earnings_transcript(ticker, max_chars=3000)` in FMPClient | Tasks 1–2 |
| Cache bucket "transcript" 24h TTL | Task 2 Step 3 |
| Return `None` on error / empty / no key | Tasks 1–2 |
| `build_prompt` optional `transcript` + `transcript_max_chars` params | Tasks 3–4 |
| New `### Recent Earnings Call` section | Task 4 Step 9 |
| "No transcript available" fallback | Task 4 Steps 9–10 |
| Extended Analysis Instructions with qualitative cross-reference | Task 4 Step 9 |
| `PROMPT_VERSION` bumped to v1.3 | Task 4 Step 8 |
| `run_analysis` optional `fmp_client` param | Tasks 5–6 |
| `FMPClient()` auto-instantiated when `fmp_client=None` | Task 6 Step 16 |
| Per-symbol transcript fetch with soft-fail | Task 6 Step 16 |
| `transcript_signals` optional in `ANALYSIS_TOOL_SCHEMA` | Task 7 |
| Not in `required` array | Task 7 Steps 19–21 |
| `hybrid_pipeline.yml` cost gate estimate updated | Task 8 |
| `ClaudeClient` untouched | No task — nothing changes there |
| `validate_analysis_schema` untouched | No task — nothing changes there |

All requirements covered.
