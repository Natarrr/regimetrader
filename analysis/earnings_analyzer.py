"""analysis/earnings_analyzer.py
Earnings / insider-filing qualitative analysis layer.

Responsibilities:
  - Prompt templates with semantic versioning (PROMPT_VERSION)
  - Compressed context builder (fits ≤ 4k tokens per symbol)
  - EDGAR factual cross-check: validate Claude citations against parsed filings
  - Shortlist selection: top-quintile quant score + explicit watchlist
  - Auto-execution gate: quant_score ≥ threshold AND claude confidence ≥ threshold

Prompt versioning convention:
  v<major>.<minor>   — major bump invalidates cache; minor is backward-compatible
  Current: v1.2

Pipeline flow (sync, Streamlit-safe):
  build_shortlist(candidates, watchlist) → List[str]
  build_prompt(symbol, quant_data, filings) → str
  cross_check_citations(analysis, parsed_events) → List[str]   # violations
  should_auto_execute(quant_score, claude_analysis) → bool
  run_analysis(shortlist, quant_map, filings_map) → List[AnalysisResult]
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .claude_client import ClaudeClient, CostTracker, validate_analysis_schema
from regime_trader.services.fmp_client import FMPClient

log = logging.getLogger(__name__)

# ── Prompt versioning ─────────────────────────────────────────────────────────
# MAJOR.MINOR are bumped manually when prompt intent / wording changes.
# The hash suffix is auto-generated from the build_prompt source so any
# unintentional drift in the function body is captured in the audit log.
# PROMPT_VERSION is assigned after build_prompt is defined (bottom of module).
_PROMPT_VERSION_BASE = "v1.3"


def get_prompt_version() -> str:
    """Return prompt version string with auto-generated source hash.

    Format: "v1.3-{sha256[:4]}" — minor stays manual, hash detects silent drift.
    A changed hash in claude_audit.ndjson means build_prompt changed without
    a manual version bump — always investigate before deploying to production.

    Safe to call at any time after the module has finished loading.
    """
    import hashlib, inspect  # noqa: PLC0415
    try:
        src = inspect.getsource(build_prompt)
        h = hashlib.sha256(src.encode()).hexdigest()[:4]
        return f"{_PROMPT_VERSION_BASE}-{h}"
    except Exception:
        return _PROMPT_VERSION_BASE


# Populated at the bottom of this module after build_prompt is defined.
PROMPT_VERSION = _PROMPT_VERSION_BASE

# ── Auto-execution thresholds ─────────────────────────────────────────────────
_DEFAULT_QUANT_THRESHOLD = float(os.getenv("AUTO_EXEC_QUANT_MIN", "80"))
_DEFAULT_CLAUDE_CONFIDENCE = float(os.getenv("AUTO_EXEC_CLAUDE_CONF", "0.80"))
_DEFAULT_CLAUDE_SCORE_MIN = int(os.getenv("AUTO_EXEC_CLAUDE_SCORE_MIN", "70"))

# ── Shortlist: top quintile floor ─────────────────────────────────────────────
_QUINTILE_FLOOR = float(os.getenv("SHORTLIST_QUINTILE_FLOOR", "80"))  # top 20%


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class AnalysisResult:
    """Complete analysis output for one symbol.

    Akerlof (2001 Nobel) — information asymmetry reduced by pairing quant
    signals with qualitative grounding in verified SEC filings.
    """
    symbol: str
    quant_score: float
    claude_analysis: Optional[Dict[str, Any]]
    citation_violations: List[str]
    auto_execute: bool
    prompt_version: str = PROMPT_VERSION
    error: Optional[str] = None
    analyzed_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def is_valid(self) -> bool:
        return self.claude_analysis is not None and len(self.citation_violations) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "quant_score": self.quant_score,
            "claude_score": self.claude_analysis.get("score") if self.claude_analysis else None,
            "claude_confidence": self.claude_analysis.get("confidence") if self.claude_analysis else None,
            "recommended_action": (
                self.claude_analysis.get("recommended_action") if self.claude_analysis else None
            ),
            "citation_violations": self.citation_violations,
            "auto_execute": self.auto_execute,
            "prompt_version": self.prompt_version,
            "error": self.error,
            "analyzed_at": self.analyzed_at,
        }


# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a senior equity analyst specializing in insider-filing forensics and \
institutional positioning. You have access to compressed SEC filing data for \
the ticker under analysis.

Rules:
1. Every factual claim MUST cite a specific filing: use citations[].
2. Do NOT invent accession numbers or filing dates.
3. If data is insufficient for a claim, express uncertainty in confidence.
4. recommended_action must be consistent with score (score ≥ 70 → BUY or HOLD).
5. Respond ONLY via the output_analysis tool — no free-form text.
"""

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
    # ── Insider block (capped at 10 rows) ─────────────────────────────────────
    buy_events = [
        e for e in parsed_events
        if str(e.get("transaction_code", "")).upper() == "P"
    ][:10]

    if buy_events:
        rows = []
        for e in buy_events:
            date = e.get("transaction_date", "?")[:10]
            role = e.get("reporting_role", "Unknown")
            shares = e.get("shares", 0)
            price = e.get("price", 0)
            value = e.get("value") or (shares * price if price else 0)
            acc = e.get("filing_accession", "?")
            rows.append(
                f"  - {date}  {role}  {shares:,.0f} shares @ ${price:.2f}  "
                f"(${value:,.0f})  [EDGAR {acc}]"
            )
        insider_block = "\n".join(rows)
    else:
        insider_block = "  No key-insider open-market purchases in last 90 days."

    # ── Institutional block (capped at 5 rows) ─────────────────────────────────
    inst_events = [
        e for e in parsed_events
        if e.get("type") in ("13f", "institutional")
    ][:5]

    if inst_events:
        rows = []
        for e in inst_events:
            holder = e.get("reporting_person", "?")
            change = e.get("shares", 0)
            acc = e.get("filing_accession", "?")
            direction = "+" if change >= 0 else ""
            rows.append(f"  - {holder}: {direction}{change:,.0f} shares  [EDGAR {acc}]")
        inst_block = "\n".join(rows)
    else:
        inst_block = "  No 13F data available for this period."

    # ── Risk block ─────────────────────────────────────────────────────────────
    risk_items = []
    if quant_data.get("momentum_score", 0) < 0.3:
        risk_items.append("Weak price momentum may signal distribution phase.")
    if quant_data.get("insider_score", 0) == 0:
        risk_items.append("No insider buying — thesis relies entirely on institutional signal.")
    if regime in ("Bear", "Panic", "Crash"):
        risk_items.append(f"Adverse macro regime ({regime}) — systemic risk elevated.")
    if not risk_items:
        risk_items.append("No specific risk flags from quant model.")
    risk_block = "\n".join(f"  - {r}" for r in risk_items)

    # ── Transcript block ───────────────────────────────────────────────────────
    if transcript:
        transcript_block = transcript[:transcript_max_chars]
    else:
        transcript_block = "No transcript available — analysis based on filing data only."

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


# Assign after build_prompt is defined so inspect.getsource() succeeds.
PROMPT_VERSION = get_prompt_version()


# ── Shortlist builder ─────────────────────────────────────────────────────────

def build_shortlist(
    candidates: List[Dict[str, Any]],
    watchlist: Optional[List[str]] = None,
    quintile_floor: float = _QUINTILE_FLOOR,
    max_symbols: int = 20,
) -> List[str]:
    """Select symbols for Claude analysis: top quintile + explicit watchlist.

    Cost control: Claude is called ONLY on this shortlist.
    Granger (2003 Nobel) — causal analysis is most valuable at the margin
    where quant signals are already strong (top quintile).

    Args:
        candidates:     List of ScanResult dicts with 'symbol' and
                        'smart_money_score' (0–1).
        watchlist:      Additional symbols to include regardless of score.
        quintile_floor: Minimum quant score (0–100) to include.
        max_symbols:    Hard cap on shortlist size (cost control).

    Returns:
        Deduplicated list of ticker symbols.
    """
    selected: List[str] = []
    seen: set = set()

    # Top quintile by smart_money_score
    sorted_candidates = sorted(
        candidates,
        key=lambda c: c.get("smart_money_score", 0),
        reverse=True,
    )
    for c in sorted_candidates:
        sym = str(c.get("symbol", "")).upper().strip()
        score_100 = c.get("smart_money_score", 0) * 100
        if not sym or sym in seen:
            continue
        if score_100 >= quintile_floor:
            selected.append(sym)
            seen.add(sym)

    # Explicit watchlist
    for sym in (watchlist or []):
        sym = sym.upper().strip()
        if sym and sym not in seen:
            selected.append(sym)
            seen.add(sym)

    result = selected[:max_symbols]
    log.info(
        "[SHORTLIST] %d symbols selected (floor=%.0f, watchlist=%d, cap=%d)",
        len(result), quintile_floor, len(watchlist or []), max_symbols,
    )
    return result


# ── EDGAR factual cross-check ─────────────────────────────────────────────────

def cross_check_citations(
    analysis: Dict[str, Any],
    parsed_events: List[Dict[str, Any]],
) -> List[str]:
    """Verify each Claude citation against parsed EDGAR filings.

    Prevents hallucinated accession numbers from reaching auto-execution.
    Returns a list of violation strings (empty = all citations verified).

    Validation logic:
      - Each citation with source containing "EDGAR" must have a loc that
        matches a known accession number in parsed_events.
      - Citations without "EDGAR" in source are passed through (e.g., "FMP",
        "yfinance" — not verifiable against local filings).

    Args:
        analysis:       Validated analysis dict from Claude.
        parsed_events:  All parsed Form-4/13F events for the symbol.

    Returns:
        List of human-readable violation strings.
    """
    known_accessions = {
        str(e.get("filing_accession", "")).strip()
        for e in parsed_events
        if e.get("filing_accession")
    }

    violations: List[str] = []
    for cit in analysis.get("citations", []):
        source = str(cit.get("source", ""))
        loc = str(cit.get("loc", "")).strip()

        if "EDGAR" not in source.upper() and "SEC" not in source.upper():
            continue  # Non-EDGAR source; skip accession check

        if not loc:
            violations.append(
                f"Citation from '{source}' has empty 'loc' field — cannot verify."
            )
            continue

        if loc not in known_accessions:
            violations.append(
                f"Citation '{source}' references '{loc}' "
                f"which is NOT in parsed filings ({len(known_accessions)} known). "
                "Possible hallucination — blocking auto-execution."
            )

    if violations:
        log.warning(
            "[CROSSCHECK] %d citation violation(s): %s",
            len(violations), "; ".join(violations),
        )
    return violations


# ── Auto-execution gate ───────────────────────────────────────────────────────

def should_auto_execute(
    quant_score: float,
    claude_analysis: Dict[str, Any],
    citation_violations: Optional[List[str]] = None,
    *,
    quant_threshold: float = _DEFAULT_QUANT_THRESHOLD,
    claude_confidence_min: float = _DEFAULT_CLAUDE_CONFIDENCE,
    claude_score_min: int = _DEFAULT_CLAUDE_SCORE_MIN,
) -> bool:
    """Return True only when all hard gates pass.

    Gates (all must be satisfied):
      1. quant_score ≥ quant_threshold (default 80/100)
      2. claude_analysis["confidence"] ≥ claude_confidence_min (default 0.80)
      3. claude_analysis["score"] ≥ claude_score_min (default 70/100)
      4. recommended_action in BUY set
      5. No citation violations (hallucination guard)

    Args:
        quant_score:          Quant composite score 0–100.
        claude_analysis:      Validated analysis dict.
        citation_violations:  Output of cross_check_citations().
        quant_threshold:      Min quant score for auto-exec.
        claude_confidence_min: Min Claude confidence for auto-exec.
        claude_score_min:     Min Claude score for auto-exec.

    Returns:
        True if auto-execution is permitted.
    """
    if citation_violations:
        log.info("[GATE] auto-exec blocked: %d citation violations", len(citation_violations))
        return False

    if quant_score < quant_threshold:
        log.info("[GATE] auto-exec blocked: quant_score=%.1f < %.1f", quant_score, quant_threshold)
        return False

    conf = claude_analysis.get("confidence", 0.0)
    if conf < claude_confidence_min:
        log.info("[GATE] auto-exec blocked: confidence=%.2f < %.2f", conf, claude_confidence_min)
        return False

    score = claude_analysis.get("score", 0)
    if score < claude_score_min:
        log.info("[GATE] auto-exec blocked: claude_score=%d < %d", score, claude_score_min)
        return False

    action = claude_analysis.get("recommended_action", "")
    if action not in ("BUY",):
        log.info("[GATE] auto-exec blocked: action=%s (only BUY triggers auto-exec)", action)
        return False

    log.info("[GATE] auto-exec APPROVED: quant=%.1f conf=%.2f score=%d action=%s",
             quant_score, conf, score, action)
    return True


# ── Main analysis runner ──────────────────────────────────────────────────────

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
    """Run Claude analysis on the full shortlist.

    Args:
        shortlist:    Symbols to analyse (output of build_shortlist()).
        quant_map:    {symbol: ScanResult dict} from discovery_scanner.
        filings_map:  {symbol: List[parsed_event_dict]} from edgar_parse.
        regime:       Current regime label.
        client:       Optional pre-configured ClaudeClient.
        run_id:       Pipeline run identifier for cache namespacing.
        bypass_cache: Force fresh Claude calls.

    Returns:
        List[AnalysisResult] in shortlist order.
    """
    if client is None:
        client = ClaudeClient(run_id=run_id)

    if fmp_client is None:
        fmp_client = FMPClient()

    results: List[AnalysisResult] = []

    for symbol in shortlist:
        quant_data = quant_map.get(symbol, {})
        parsed_events = filings_map.get(symbol, [])
        quant_score = quant_data.get("smart_money_score", 0.0) * 100

        try:
            try:
                transcript = fmp_client.get_earnings_transcript(symbol)
            except Exception as exc:
                log.warning("[ANALYZER] transcript fetch failed for %s: %s", symbol, exc)
                transcript = None
            prompt = build_prompt(symbol, quant_data, parsed_events, regime,
                                  transcript=transcript)
            analysis = client.analyze(
                symbol=symbol,
                prompt=prompt,
                prompt_version=PROMPT_VERSION,
                system=_SYSTEM_PROMPT,
                bypass_cache=bypass_cache,
            )

            violations = cross_check_citations(analysis, parsed_events)
            auto_exec = should_auto_execute(quant_score, analysis, violations)

            results.append(AnalysisResult(
                symbol=symbol,
                quant_score=quant_score,
                claude_analysis=analysis,
                citation_violations=violations,
                auto_execute=auto_exec,
            ))

        except Exception as exc:
            log.error("[ANALYZER] Failed for %s: %s", symbol, exc)
            results.append(AnalysisResult(
                symbol=symbol,
                quant_score=quant_score,
                claude_analysis=None,
                citation_violations=[],
                auto_execute=False,
                error=str(exc),
            ))

    approved = sum(1 for r in results if r.auto_execute)
    log.info(
        "[ANALYZER] Done: %d/%d auto-execute approved", approved, len(results)
    )
    return results
