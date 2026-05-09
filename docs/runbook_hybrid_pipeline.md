# Runbook: Hybrid Quant + Claude Pipeline

## Overview

The hybrid pipeline combines quantitative EDGAR/FMP scoring with Claude LLM qualitative analysis.
It runs automatically at 08:30 ET on trading days via GitHub Actions.

```
quant job           →  claude job          →  gate job
─────────────────      ─────────────────      ─────────────────
EDGAR/FMP scoring      ClaudeClient.analyze   Schema validation
Regime detection       cross_check_citations  Auto-exec summary
Artifact upload        Cost accounting        GITHUB_STEP_SUMMARY
```

---

## Prerequisites

### API Keys (GitHub Secrets)

| Secret | Required | Purpose |
|--------|----------|---------|
| `ANTHROPIC_API_KEY` | Yes (claude job) | Claude API access |
| `FMP_API_KEY` | Yes | Financial Modeling Prep data |
| `SEC_USER_AGENT` | Yes | SEC EDGAR rate-limit compliance |
| `DISCORD_WEBHOOK_URL` | Optional | Run summary notification |

### Cost Controls (Env / Workflow Inputs)

| Variable                   | Default | Effect                                          |
| -------------------------- | ------- | ----------------------------------------------- |
| `CLAUDE_COST_CAP_USD`      | `2.00`  | Hard cap per run; raises `CostBudgetExceeded`   |
| `SHORTLIST_QUINTILE_FLOOR` | `80`    | Min quant score (0–100) to include in shortlist |

---

## Local Development

### 1. Install dependencies

```bash
pip install -r requirements-ci.txt
pip install hmmlearn>=0.3.0 scikit-learn>=1.4.0 anthropic>=0.28.0
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in: ANTHROPIC_API_KEY, FMP_API_KEY, SEC_USER_AGENT
```

### 3. Run quant scoring locally

```bash
python -m regime_trader.discovery_scanner --limit 5
```

### 4. Run regime detection

```python
import yfinance as yf
import pandas as pd
from regime.regime_detector import RegimeDetector

vix = yf.download("^VIX", period="2y", interval="1d")["Close"].squeeze()
spy = yf.download("SPY", period="2y", interval="1d")["Close"].squeeze()
returns = spy.pct_change().dropna()

det = RegimeDetector()
det.fit(vix, returns)
print(det.predict(vix, returns))
```

### 5. Run Claude analysis (dry run)

```bash
DRY_RUN=true python - <<'EOF'
import json
from analysis.claude_client import ClaudeClient
from analysis.earnings_analyzer import build_shortlist, build_prompt

shortlist = ["AAPL", "MSFT"]
client = ClaudeClient(run_id="local-test")
for sym in shortlist:
    prompt = build_prompt(sym, {"smart_money_score": 0.85}, [], "Neutral")
    print(f"[DRY RUN] Would analyse {sym}:")
    print(prompt[:300])
    print("---")
EOF
```

---

## Manual Trigger

Go to **Actions → hybrid_pipeline → Run workflow** and set:

| Input | Recommended value |
|-------|-----------------|
| `force_refresh` | `true` to bypass discovery cache |
| `cost_cap_usd` | `2.00` (default) |
| `shortlist_floor` | `80` (default) |
| `dry_run` | `true` to validate without Claude API calls |

---

## Cost Budget Guidance

### Per-symbol cost estimate (claude-sonnet-4-6)

- Average prompt: ~800 input tokens
- Average response: ~200 output tokens
- Cost per symbol: ~$0.0054

### Budget planning

| Shortlist size | Estimated cost |
|---------------|----------------|
| 5 symbols | ~$0.03 |
| 10 symbols | ~$0.05 |
| 20 symbols (max) | ~$0.11 |

The default `$2.00` cap covers up to 20 symbols with 10x safety margin.

### Changing the cap

```bash
# Via workflow_dispatch input:
cost_cap_usd: "1.00"

# Via environment variable (local):
export CLAUDE_COST_CAP_USD=1.00
```

---

## Schema Validation

Every Claude response is validated against `ANALYSIS_TOOL_SCHEMA` before use.

Required fields:

| Field | Type | Constraint |
|-------|------|-----------|
| `score` | int | [0, 100] |
| `confidence` | float | [0.0, 1.0] |
| `reasons` | list[str] | min 1 item |
| `citations` | list[{source, loc}] | may be empty |
| `recommended_action` | str | BUY \| SELL \| HOLD \| REDUCE \| WATCH |

### Citation cross-check

EDGAR citations (`source` contains "EDGAR" or "SEC") are validated against
parsed filings before auto-execution is permitted. Any unverifiable accession
number blocks the entire symbol from auto-execution.

---

## Auto-Execution Gate

All five conditions must pass simultaneously:

1. `quant_score >= 80` (top-quintile composite)
2. `claude_analysis["confidence"] >= 0.80`
3. `claude_analysis["score"] >= 70`
4. `recommended_action == "BUY"`
5. `citation_violations == []`

```python
from analysis.earnings_analyzer import should_auto_execute
result = should_auto_execute(quant_score, claude_analysis, citation_violations)
```

---

## Credit Stress Regime

The Credit Stress signal is an optional fourth signal in the ensemble, sourced
entirely from free market data (yfinance). **Zero LLM tokens.**

### Data sources

| Proxy            | Ticker   | Rationale                                                    |
| ---------------- | -------- | ------------------------------------------------------------ |
| High Yield       | `JNK`    | iShares HY Corp Bond ETF; price falls when HY spreads widen  |
| Investment Grade | `LQD`    | iShares IG Corp Bond ETF; price falls when IG spreads widen  |
| MOVE index       | `^MOVE`  | ICE BofA bond vol index; rises with rate/credit stress       |

All tickers are overridable via env: `CREDIT_HY_TICKER`, `CREDIT_IG_TICKER`, `CREDIT_MOVE_TICKER`.

### Credit Stress Score formula

```text
credit_raw   = 0.35*z_hy + 0.25*z_ig + 0.20*slope_hy_norm + 0.10*hy_ig_ratio + 0.10*z_move
credit_score = weighted mean of sigmoid(component / 2 * 3)  ∈  [0, 1]
```

Components are **positively signed for stress** (falling HY price = positive z\_hy).
Missing components are automatically excluded; weight is redistributed.

### Credit regime thresholds

| Score      | Regime  |
| ---------- | ------- |
| ≥ 0.75     | CRISIS  |
| 0.60–0.75  | STRESS  |
| 0.40–0.60  | CAUTION |
| < 0.40     | NORMAL  |

### Asymmetric persistence filter

| Target regime    | Days required               |
| ---------------- | --------------------------- |
| NORMAL / CAUTION | 1 (immediate de-escalation) |
| STRESS           | 2 consecutive               |
| CRISIS           | 3 consecutive               |

### Override rules (applied after ensemble vote, before persistence filter)

| Condition                              | Override                                         |
| -------------------------------------- | ------------------------------------------------ |
| `credit == CRISIS`                     | Ensemble output forced to ≥ Bear                 |
| `credit == STRESS` AND `VIX < 20`      | Ensemble output forced to ≥ Bear (early warning) |

### Using the credit signal locally

```python
from regime.credit_regime_detector import CreditRegimeDetector
from regime.regime_detector import RegimeDetector

# Standalone credit signal
det = CreditRegimeDetector()
regime, score, features = det.predict_latest(window=300)
print(f"Credit regime: {regime.value}  score={score:.3f}")

# Integrated 4-signal ensemble
import yfinance as yf, pandas as pd
vix = yf.download("^VIX", period="2y")["Close"].squeeze()
spy = yf.download("SPY",  period="2y")["Close"].squeeze()
returns = spy.pct_change().dropna()

# Pre-compute credit scores
hy  = yf.download("JNK", period="2y")["Close"].squeeze()
lqd = yf.download("LQD", period="2y")["Close"].squeeze()
credit_scores = det.compute_features_series(hy_prices=hy, ig_prices=lqd)

ensemble = RegimeDetector(w_vix=0.30, w_hmm=0.25, w_ml=0.25, w_credit=0.20)
ensemble.fit(vix, returns)
label = ensemble.predict(vix, returns, credit_scores=credit_scores)
print(f"Ensemble regime (4-signal): {label}")
```

### Env variables

| Variable | Default | Effect |
|----------|---------|--------|
| `CREDIT_HY_TICKER` | `JNK` | HY ETF proxy |
| `CREDIT_IG_TICKER` | `LQD` | IG ETF proxy |
| `CREDIT_MOVE_TICKER` | `^MOVE` | Bond vol proxy |

---

## Regime Detection

### VIX threshold rule (CI / fast path)

| VIX level | Regime |
|-----------|--------|
| ≥ 45 | Crash |
| 35–45 | Panic |
| 25–35 | Bear |
| 15–25 | Neutral |
| 12–15 | Bull |
| < 12 | Euphoria |

### Ensemble (production)

```python
det = RegimeDetector()
det.fit(vix_series, returns_series)
label = det.predict(vix_series, returns_series)
report = det.backtest_report(vix_series, returns_series)
```

Ensemble weights: HMM 0.45, ML 0.35, VIX rule 0.20.
Persistence filter: 2 consecutive signals required before regime switch.

---

## Audit Trail

Every Claude call is written to `logs/claude_audit.ndjson` (NDJSON, one entry per call):

```json
{"ts": "2026-05-08T12:31:00Z", "run_id": "gh-12345", "symbol": "AAPL",
 "prompt_version": "v1.2", "model": "claude-sonnet-4-6", "attempt": 1,
 "input_tokens": 812, "output_tokens": 198, "cost_usd": 0.005394,
 "elapsed_s": 2.1, "cache_key": "a3f9...", "status": "ok"}
```

Artifacts retained 30 days: `claude-audit-{run_id}` bundle includes
`claude_results.json`, `cost_summary.json`, `claude_audit.ndjson`.

---

## Troubleshooting

### `CostBudgetExceeded` raised mid-run

The run hit the cost cap. Either increase `CLAUDE_COST_CAP_USD` or reduce
`SHORTLIST_QUINTILE_FLOOR` to trim the shortlist.

### Citation violations blocking auto-execution

Claude cited an EDGAR accession number that is not in the parsed filings.
Review `claude_results.json` → `citation_violations` for the specific symbols.
This is by design: hallucinated accession numbers are blocked before execution.

### `ModuleNotFoundError: No module named 'hmmlearn'`

```bash
pip install hmmlearn>=0.3.0
```

### Empty shortlist

Check `SHORTLIST_QUINTILE_FLOOR` — if set above 90, very few symbols qualify.
Verify that `discovery_scanner` is returning results (`data/pipeline/shortlist.json`).

### Schema validation failure

Run locally with `DRY_RUN=false` on a single symbol, inspect `claude_results.json`.
If Claude is returning free-form text instead of tool_use output, check that
`tool_choice={"type":"tool","name":"output_analysis"}` is set in `ClaudeClient.analyze()`.

---

## Caching

### Discovery cache

Location: `data/cache/discovery_cache.json`
TTL: 6 hours
Bypass: `force_refresh=true` in workflow input, or `--force-refresh` CLI flag.

### Claude response cache

Location: `data/cache/claude/{key}.json`
Key: SHA256(run_id | prompt_version | symbol | prompt_hash)[:24]
Bypass: `bypass_cache=True` in `ClaudeClient.analyze()`.
Cache is ONLY invalidated by changing `PROMPT_VERSION` major component (e.g., v1.x → v2.0).
