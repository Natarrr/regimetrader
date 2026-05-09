# Prompt Catalog — Hybrid Pipeline

Versioning convention: `v<major>.<minor>`
- **Major bump** → cache is invalidated; all symbols re-analysed next run
- **Minor bump** → wording change only; cache remains valid

---

## Current: v1.2

File: `analysis/earnings_analyzer.py`, constant `PROMPT_VERSION = "v1.2"`

### Change vs v1.1

Added explicit `### Current market regime: {regime}` line to the Quant Signal
Summary block so the model can modulate conviction in Bear/Panic/Crash regimes.

### System prompt

```
You are a senior equity analyst specializing in insider-filing forensics and
institutional positioning. You have access to compressed SEC filing data for
the ticker under analysis.

Rules:
1. Every factual claim MUST cite a specific filing: use citations[].
2. Do NOT invent accession numbers or filing dates.
3. If data is insufficient for a claim, express uncertainty in confidence.
4. recommended_action must be consistent with score (score ≥ 70 → BUY or HOLD).
5. Respond ONLY via the output_analysis tool — no free-form text.
```

### User prompt template

```
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

### Analysis Instructions
Synthesize the quantitative signal and the SEC filing evidence above.
Focus on:
1. Are insider purchases consistent in size and role (CEO/CFO > Director)?
2. Are institutions accumulating or distributing?
3. What is the most likely 30-day price catalyst?
4. What are the top 2 risks that could invalidate the long thesis?

Return your analysis via the output_analysis tool with score, confidence,
reasons (≥3 points), citations (anchor each fact to a filing), and
recommended_action.
```

### Context budget

| Block | Max rows | Rationale |
|-------|----------|-----------|
| Insider (Form-4) | 10 | ~200 tokens; focuses on highest-conviction buys |
| Institutional (13F) | 5 | ~100 tokens; top holders only |
| Risk factors | unbounded | Typically 1–3 items; low token cost |

Total estimate: ~800 input tokens per symbol.

### Output schema (`ANALYSIS_TOOL_SCHEMA`)

```json
{
  "score":              <int 0–100>,
  "confidence":         <float 0.0–1.0>,
  "reasons":            ["reason1", "reason2", ...],
  "citations":          [{"source": "EDGAR Form-4", "loc": "accession-number"}, ...],
  "recommended_action": "BUY" | "SELL" | "HOLD" | "REDUCE" | "WATCH"
}
```

### Citation validation

Citations with `source` containing "EDGAR" or "SEC" must provide a `loc` that
matches a known accession number in the parsed filings. Any mismatch is logged
as a violation and blocks auto-execution.

Non-EDGAR sources (e.g., "FMP API", "yfinance") are passed through without
accession checks.

---

## Changelog

### v1.2 (current)
- Added `Current market regime: {regime}` to Quant Signal Summary
- Added "Adverse macro regime" risk flag for Bear/Panic/Crash regimes
- Cache key updated; first run on v1.2 will re-query all symbols

### v1.1
- Added `### Key Risk Factors` block
- Weakened momentum and no-insider-buying risk flags added
- `recommended_action` consistency rule added to system prompt (score ≥ 70 → BUY or HOLD)

### v1.0
- Initial prompt: insider block + institutional block only
- No regime context
- No explicit risk block

---

## Regression Testing

The parametrised test `tests/test_claude_client.py::test_validate_100_sample_analyses`
verifies that `validate_analysis_schema()` accepts all 100 generated samples
covering the full valid range of each field.

To run schema regression tests:

```bash
pytest tests/test_claude_client.py -k "test_validate_100" -v
```

To test the full prompt builder:

```bash
pytest tests/test_earnings_analyzer.py -k "TestBuildPrompt" -v
```

---

## Adding a New Prompt Version

1. Increment `PROMPT_VERSION` in `analysis/earnings_analyzer.py`
2. Update `_USER_PROMPT_TEMPLATE` and/or `_SYSTEM_PROMPT`
3. Add a changelog entry above with the date and changes
4. Run `pytest tests/test_earnings_analyzer.py` to verify builder correctness
5. Do one dry-run (`DRY_RUN=true`) to inspect prompt previews before going live
6. For major version bumps, set `bypass_cache=True` for the first production run
