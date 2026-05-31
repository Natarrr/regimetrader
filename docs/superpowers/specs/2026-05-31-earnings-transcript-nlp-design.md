# Design: Earnings Transcript NLP Integration

**Date:** 2026-05-31  
**Status:** Approved  
**Prompt version bump:** v1.2 → v1.3 (cache-busting — prompt intent changes)

---

## Problem

Claude currently receives only numerical factor scores per symbol. It has no access to the company's own words. The most information-dense qualitative signal — earnings call transcripts — is already available on FMP Ultimate (`stable/earning-call-transcript-latest`) but is not being fetched or injected.

Feeding executive remarks directly into the prompt enables Claude to cross-reference management guidance tone and confidence against the quant factors, producing a materially richer analysis.

---

## Scope

Four discrete file changes. No changes to `ClaudeClient`, `validate_analysis_schema`, or `hybrid_pipeline.yml` job structure.

---

## Changes

### 1. `FMPClient.get_earnings_transcript` — new method

**File:** `regime_trader/services/fmp_client.py`

```python
def get_earnings_transcript(self, ticker: str, max_chars: int = 3000) -> str | None:
```

- Fetches `stable/earning-call-transcript-latest?symbol={ticker}&limit=1`
- Response shape: `[{symbol, quarter, year, date, content}]`
- Returns `content[:max_chars]` on success; `None` on error, empty list, or missing `content`
- Cache bucket: `"transcript"` (already in `_TTL` at 24h — transcripts don't change after publication)
- `FMPEndpointError` is caught and returns `None` — a missing transcript is a **soft failure**, not a pipeline abort. The circuit-breaker pattern is reserved for factors that zero a score; transcripts are additive enrichment only.

**`max_chars` rationale:** The fetch ceiling (default 3000) is intentionally larger than the prompt injection limit (default 2000 in `build_prompt`). This gives `build_prompt` room to truncate to its own context budget without requiring a second network call if the budget changes. The delta (up to 1000 chars) sits in memory briefly and is discarded — document this explicitly so the asymmetry is not read as a bug.

---

### 2. `build_prompt` — optional `transcript` param

**File:** `analysis/earnings_analyzer.py`

```python
def build_prompt(
    symbol: str,
    quant_data: Dict[str, Any],
    parsed_events: List[Dict[str, Any]],
    regime: str = "Unknown",
    transcript: str | None = None,
    transcript_max_chars: int = 2000,
) -> str:
```

**New section injected between `### Key Risk Factors` and `### Analysis Instructions`:**

```
### Recent Earnings Call (last quarter — executive remarks excerpt)
{transcript_block}
```

Where:
- `transcript_block = transcript[:transcript_max_chars]` if transcript is not None/empty
- else `"No transcript available — analysis based on filing data only."`

**`Analysis Instructions` addition:**

```
If a transcript is provided, identify:
1. Forward guidance tone (raised/maintained/lowered)
2. Management confidence signals (hedging language vs conviction)
3. Any mention of buybacks, M&A, or restructuring
Cross-reference these qualitative signals against the quantitative factors.
```

**Backward compatibility:** All existing callers pass positional args through `regime`. New params `transcript` and `transcript_max_chars` are keyword-only with defaults — no existing call site breaks.

**`PROMPT_VERSION` bump:** `v1.2` → `v1.3`. This is a cache-bust, not a patch: the prompt structure changes and cached v1.2 responses should not be served for v1.3 prompts.

---

### 3. `run_analysis` — optional `fmp_client` param

**File:** `analysis/earnings_analyzer.py`

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

- `fmp_client = fmp_client or FMPClient()` — mirrors the existing `client` pattern exactly
- Instantiated once per `run_analysis` call, shared across all symbols in the shortlist
- Per symbol: `transcript = fmp_client.get_earnings_transcript(symbol)` wrapped in try/except; on any exception, logs a warning and continues with `transcript=None`
- `transcript` passed to `build_prompt()`

**Ownership:** `FMPClient` is instantiated in `earnings_analyzer`, not in `ClaudeClient`. `ClaudeClient` remains a pure Anthropic wrapper with no FMP dependency.

---

### 4. `ANALYSIS_TOOL_SCHEMA` — optional `transcript_signals` property

**File:** `analysis/claude_client.py`

New optional property added to `input_schema.properties`:

```json
"transcript_signals": {
  "type": "object",
  "properties": {
    "guidance_tone": {
      "type": "string",
      "enum": ["raised", "maintained", "lowered", "not_mentioned"]
    },
    "management_confidence": {
      "type": "string",
      "enum": ["high", "neutral", "low"]
    },
    "buyback_mentioned": {
      "type": "boolean"
    }
  }
}
```

**Not added to `required` array** — backward compatible. Claude may omit this field when no transcript is provided. `validate_analysis_schema` checks only `required` fields; no change needed there.

---

### 5. `hybrid_pipeline.yml` — cost gate estimate update

**File:** `.github/workflows/hybrid_pipeline.yml` (claude job, preflight cost gate step)

Update the per-symbol token estimate from `~1k input + 300 output` to `~1500 input + 300 output` to reflect the transcript block addition (~500 tokens at 4 chars/token).

```python
# Before
input_cost  = len(shortlist) * 1000 * 3.00  / 1_000_000
output_cost = len(shortlist) * 300  * 15.00 / 1_000_000

# After
input_cost  = len(shortlist) * 1500 * 3.00  / 1_000_000
output_cost = len(shortlist) * 300  * 15.00 / 1_000_000
```

Update the comment to: `# ~1500 input + 300 output tokens per symbol (includes 2000-char transcript block)`

---

## Data flow

```
run_analysis(shortlist, quant_map, filings_map, regime, fmp_client=None)
  │
  ├─ [once] fmp_client = fmp_client or FMPClient()
  │
  └─ [per symbol]
       transcript = fmp_client.get_earnings_transcript(symbol)   # None on any failure
       prompt = build_prompt(symbol, quant_data, events, regime, transcript)
       analysis = client.analyze(...)        # may include transcript_signals in output
```

---

## Error handling

| Failure point | Behaviour |
|---|---|
| `FMPEndpointError` on transcript fetch | Caught → `None` transcript → prompt uses "No transcript available" fallback |
| Empty list response (no transcript filed yet) | Returns `None` → same fallback |
| `transcript_signals` absent from Claude output | `validate_analysis_schema` unchanged — not in `required` — no error |
| Any other exception in transcript fetch | Logged as warning → `transcript=None` → pipeline continues |

---

## Token budget

| Component | Tokens (approx) |
|---|---|
| Existing prompt | ~1000 |
| Transcript block (2000 chars @ 4 chars/tok) | +500 |
| New instruction text | +30 |
| **New total per symbol** | **~1530** |

Cost delta per symbol at claude-sonnet-4-6 input pricing ($3/M): **+$0.0016**. Negligible.

---

## What does not change

- `ClaudeClient` — untouched
- `validate_analysis_schema` — untouched
- All existing callers of `build_prompt` and `run_analysis` — new params are optional with safe defaults
- `hybrid_pipeline.yml` job structure — only the token estimate arithmetic changes
