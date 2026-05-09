"""analysis/claude_client.py
Anthropic Claude wrapper for the hybrid quant pipeline.

Responsibilities:
  - Structured JSON output via tool_use (forced schema)
  - Exponential-backoff retries on transient errors
  - Per-run cost tracking with configurable hard cap
  - Prompt/response cache keyed by (run_id, prompt_version, symbol)
  - Audit log: every call is written to logs/claude_audit.ndjson

Usage:
    client = ClaudeClient()
    response = client.analyze(symbol="AAPL", prompt="...", prompt_version="v1.2")

Cost caps (env vars):
    CLAUDE_COST_CAP_USD   : abort run when cumulative cost exceeds this (default $2.00)
    CLAUDE_MODEL          : model ID (default claude-sonnet-4-6)
    ANTHROPIC_API_KEY     : required
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ── Pricing table (USD per million tokens, May 2026) ──────────────────────────
_PRICE_TABLE: Dict[str, Dict[str, float]] = {
    "claude-opus-4-7":               {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6":             {"input":  3.00, "output": 15.00},
    "claude-haiku-4-5-20251001":     {"input":  0.25, "output":  1.25},
}
_DEFAULT_MODEL = "claude-sonnet-4-6"

# ── Defaults ──────────────────────────────────────────────────────────────────
_DEFAULT_COST_CAP_USD = float(os.getenv("CLAUDE_COST_CAP_USD", "2.00"))
_CACHE_DIR = Path(os.getenv("CLAUDE_CACHE_DIR", "data/cache/claude"))
_AUDIT_LOG = Path(os.getenv("CLAUDE_AUDIT_LOG", "logs/claude_audit.ndjson"))

_MAX_RETRIES = 3
_BACKOFF_BASE = 1.5   # seconds; doubles each attempt

# ── Output JSON schema enforced via Anthropic tool_use ────────────────────────
ANALYSIS_TOOL_SCHEMA: Dict[str, Any] = {
    "name": "output_analysis",
    "description": (
        "Return a structured qualitative analysis of the equity. "
        "All fields are required. citations must reference parsed EDGAR filings only."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "score": {
                "type": "integer",
                "minimum": 0,
                "maximum": 100,
                "description": "Composite conviction score 0–100 (100 = strongest long).",
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence in the analysis given available data quality.",
            },
            "reasons": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "description": "Ordered list of reasoning steps (most important first).",
            },
            "citations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "loc": {"type": "string"},
                    },
                    "required": ["source", "loc"],
                },
                "description": "Factual claims anchored to specific filing references.",
            },
            "recommended_action": {
                "type": "string",
                "enum": ["BUY", "SELL", "HOLD", "REDUCE", "WATCH"],
                "description": "Recommended portfolio action.",
            },
        },
        "required": ["score", "confidence", "reasons", "citations", "recommended_action"],
    },
}


# ── Cost budget ───────────────────────────────────────────────────────────────

class CostBudgetExceeded(RuntimeError):
    """Raised when cumulative token cost exceeds the configured hard cap."""


@dataclass
class CostTracker:
    """Accumulates token usage across all calls in one pipeline run.

    Akerlof (2001 Nobel) — information has value only if its cost is bounded;
    this tracker enforces that bound operationally.
    """
    cap_usd: float = _DEFAULT_COST_CAP_USD
    model: str = _DEFAULT_MODEL
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    _cost_usd: float = field(default=0.0, init=False, repr=False)

    def record(self, input_tok: int, output_tok: int) -> float:
        """Record token usage and return incremental cost. Raises if cap exceeded."""
        prices = _PRICE_TABLE.get(self.model, _PRICE_TABLE[_DEFAULT_MODEL])
        inc = (input_tok * prices["input"] + output_tok * prices["output"]) / 1_000_000
        self._cost_usd += inc
        self.input_tokens += input_tok
        self.output_tokens += output_tok
        self.calls += 1
        log.info(
            "[COST] call=%d  +$%.4f  total=$%.4f / cap=$%.2f  (in=%d out=%d)",
            self.calls, inc, self._cost_usd, self.cap_usd, input_tok, output_tok,
        )
        if self._cost_usd > self.cap_usd:
            raise CostBudgetExceeded(
                f"Cost cap ${self.cap_usd:.2f} exceeded (actual ${self._cost_usd:.4f}). "
                "Increase CLAUDE_COST_CAP_USD or reduce shortlist size."
            )
        return inc

    @property
    def total_usd(self) -> float:
        return round(self._cost_usd, 6)

    def summary(self) -> Dict[str, Any]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_usd": self.total_usd,
            "cap_usd": self.cap_usd,
            "model": self.model,
        }


# ── Cache helpers ─────────────────────────────────────────────────────────────

def _cache_key(run_id: str, prompt_version: str, symbol: str, prompt_hash: str) -> str:
    raw = f"{run_id}|{prompt_version}|{symbol}|{prompt_hash}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _cache_path(key: str) -> Path:
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR / f"{key}.json"


def _load_cache(key: str) -> Optional[Dict[str, Any]]:
    p = _cache_path(key)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_cache(key: str, payload: Dict[str, Any]) -> None:
    try:
        p = _cache_path(key)
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        log.warning("cache write failed: %s", exc)


# ── Audit logger ──────────────────────────────────────────────────────────────

def _audit(entry: Dict[str, Any]) -> None:
    try:
        _AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _AUDIT_LOG.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:
        log.warning("audit write failed: %s", exc)


# ── Main client ───────────────────────────────────────────────────────────────

class ClaudeClient:
    """Anthropic Claude wrapper for shortlist qualitative analysis.

    Design principles:
      - tool_use forces structured JSON — never parse free-form text
      - Cache keyed by (run_id, prompt_version, symbol, prompt_hash)
      - Exponential backoff on rate-limit / server errors
      - Hard cost cap per run; raises CostBudgetExceeded if exceeded
    """

    def __init__(
        self,
        model: Optional[str] = None,
        cost_tracker: Optional[CostTracker] = None,
        run_id: Optional[str] = None,
    ) -> None:
        try:
            import anthropic
            self._anthropic = anthropic
        except ImportError as e:
            raise ImportError(
                "anthropic package required: pip install anthropic>=0.28.0"
            ) from e

        self.model = model or os.getenv("CLAUDE_MODEL", _DEFAULT_MODEL)
        self.tracker = cost_tracker or CostTracker(model=self.model)
        self.run_id = run_id or datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S")
        self._client = self._anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY", "")
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def analyze(
        self,
        symbol: str,
        prompt: str,
        *,
        prompt_version: str = "v1.0",
        system: Optional[str] = None,
        max_tokens: int = 1024,
        bypass_cache: bool = False,
    ) -> Dict[str, Any]:
        """Run a single-symbol analysis through Claude.

        Returns the validated JSON dict from ANALYSIS_TOOL_SCHEMA.
        Raises CostBudgetExceeded if the run-level cap is exceeded.

        Args:
            symbol:         Ticker being analysed.
            prompt:         User-turn prompt (compressed context injected here).
            prompt_version: Semver string for cache invalidation / regression.
            system:         Optional system prompt override.
            max_tokens:     Max output tokens (default 1024).
            bypass_cache:   If True, skip cache lookup and always call API.

        Returns:
            Validated analysis dict conforming to ANALYSIS_TOOL_SCHEMA.
        """
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        cache_key = _cache_key(self.run_id, prompt_version, symbol, prompt_hash)

        # ── Cache hit ──────────────────────────────────────────────────────────
        if not bypass_cache:
            cached = _load_cache(cache_key)
            if cached is not None:
                log.info("[CLAUDE] cache hit  symbol=%s  key=%s", symbol, cache_key)
                return cached["analysis"]

        # ── Build messages ─────────────────────────────────────────────────────
        sys_prompt = system or (
            "You are a senior equity analyst with deep expertise in SEC filings, "
            "insider trading patterns, and institutional positioning. "
            "Respond ONLY via the output_analysis tool. "
            "Ground every factual claim in a specific citation from the provided context."
        )
        messages = [{"role": "user", "content": prompt}]

        # ── Call with retries ──────────────────────────────────────────────────
        last_exc: Optional[Exception] = None
        for attempt in range(1, _MAX_RETRIES + 1):
            try:
                t0 = time.monotonic()
                resp = self._client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=sys_prompt,
                    tools=[ANALYSIS_TOOL_SCHEMA],
                    tool_choice={"type": "tool", "name": "output_analysis"},
                    messages=messages,
                )
                elapsed = time.monotonic() - t0

                # ── Extract tool_use block ────────────────────────────────────
                analysis = self._extract_tool_result(resp)
                validate_analysis_schema(analysis)

                # ── Cost tracking ─────────────────────────────────────────────
                usage = resp.usage
                cost = self.tracker.record(usage.input_tokens, usage.output_tokens)

                # ── Persist cache ─────────────────────────────────────────────
                cache_payload = {
                    "run_id": self.run_id,
                    "symbol": symbol,
                    "prompt_version": prompt_version,
                    "prompt_hash": prompt_hash,
                    "model": self.model,
                    "analysis": analysis,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "cost_usd": cost,
                    "elapsed_s": round(elapsed, 2),
                    "cached_at": datetime.now(timezone.utc).isoformat(),
                }
                _save_cache(cache_key, cache_payload)

                # ── Audit log ─────────────────────────────────────────────────
                _audit({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "run_id": self.run_id,
                    "symbol": symbol,
                    "prompt_version": prompt_version,
                    "model": self.model,
                    "attempt": attempt,
                    "input_tokens": usage.input_tokens,
                    "output_tokens": usage.output_tokens,
                    "cost_usd": cost,
                    "elapsed_s": round(elapsed, 2),
                    "cache_key": cache_key,
                    "status": "ok",
                })

                log.info(
                    "[CLAUDE] ok  symbol=%s  score=%s  confidence=%s  cost=$%.4f  %.1fs",
                    symbol, analysis.get("score"), analysis.get("confidence"), cost, elapsed,
                )
                return analysis

            except CostBudgetExceeded:
                raise
            except Exception as exc:
                last_exc = exc
                is_rate_limit = "rate" in str(exc).lower() or "529" in str(exc)
                wait = _BACKOFF_BASE * (2 ** (attempt - 1)) * (2 if is_rate_limit else 1)
                log.warning(
                    "[CLAUDE] attempt %d/%d failed  symbol=%s  err=%s  retry_in=%.1fs",
                    attempt, _MAX_RETRIES, symbol, exc, wait,
                )
                _audit({
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "run_id": self.run_id,
                    "symbol": symbol,
                    "prompt_version": prompt_version,
                    "attempt": attempt,
                    "status": "error",
                    "error": str(exc),
                })
                if attempt < _MAX_RETRIES:
                    time.sleep(wait)

        raise RuntimeError(
            f"Claude call failed after {_MAX_RETRIES} attempts for {symbol}: {last_exc}"
        )

    def analyze_batch(
        self,
        items: List[Dict[str, str]],
        *,
        prompt_version: str = "v1.0",
        system: Optional[str] = None,
        max_tokens: int = 1024,
        bypass_cache: bool = False,
    ) -> List[Dict[str, Any]]:
        """Analyse a shortlist sequentially, respecting cost cap.

        Args:
            items: List of {"symbol": str, "prompt": str} dicts.

        Returns:
            List of {"symbol": str, "analysis": dict, "error": str|None} results.
        """
        results = []
        for item in items:
            sym = item["symbol"]
            try:
                analysis = self.analyze(
                    symbol=sym,
                    prompt=item["prompt"],
                    prompt_version=prompt_version,
                    system=system,
                    max_tokens=max_tokens,
                    bypass_cache=bypass_cache,
                )
                results.append({"symbol": sym, "analysis": analysis, "error": None})
            except CostBudgetExceeded:
                log.error("[CLAUDE] Cost cap hit — aborting batch at %s", sym)
                results.append({"symbol": sym, "analysis": None,
                                "error": "cost_cap_exceeded"})
                break
            except Exception as exc:
                log.error("[CLAUDE] Failed for %s: %s", sym, exc)
                results.append({"symbol": sym, "analysis": None, "error": str(exc)})
        return results

    # ── Internals ──────────────────────────────────────────────────────────────

    def _extract_tool_result(self, resp: Any) -> Dict[str, Any]:
        """Pull the tool_use input dict from an Anthropic response."""
        for block in resp.content:
            if block.type == "tool_use" and block.name == "output_analysis":
                return block.input
        raise ValueError(
            f"No tool_use block in response. stop_reason={resp.stop_reason}. "
            f"content={resp.content}"
        )

    def cost_summary(self) -> Dict[str, Any]:
        return self.tracker.summary()


# ── Schema validation (pure Python, no external dep) ─────────────────────────

class SchemaValidationError(ValueError):
    pass


def validate_analysis_schema(data: Any) -> Dict[str, Any]:
    """Validate that *data* conforms to ANALYSIS_TOOL_SCHEMA input_schema.

    Raises SchemaValidationError with a descriptive message on violation.
    Returns the validated dict unchanged.

    Validates:
        - Required fields present
        - score: int [0, 100]
        - confidence: float [0.0, 1.0]
        - reasons: non-empty list of strings
        - citations: list of {source, loc} objects
        - recommended_action: one of BUY|SELL|HOLD|REDUCE|WATCH
    """
    if not isinstance(data, dict):
        raise SchemaValidationError(f"Expected dict, got {type(data).__name__}")

    required = {"score", "confidence", "reasons", "citations", "recommended_action"}
    missing = required - data.keys()
    if missing:
        raise SchemaValidationError(f"Missing required fields: {missing}")

    score = data["score"]
    if not isinstance(score, (int, float)) or not (0 <= score <= 100):
        raise SchemaValidationError(f"score must be int/float in [0,100], got {score!r}")

    conf = data["confidence"]
    if not isinstance(conf, (int, float)) or not (0.0 <= conf <= 1.0):
        raise SchemaValidationError(f"confidence must be float in [0,1], got {conf!r}")

    reasons = data["reasons"]
    if not isinstance(reasons, list) or len(reasons) == 0:
        raise SchemaValidationError("reasons must be non-empty list")
    if not all(isinstance(r, str) for r in reasons):
        raise SchemaValidationError("reasons must be list[str]")

    citations = data["citations"]
    if not isinstance(citations, list):
        raise SchemaValidationError("citations must be a list")
    for i, c in enumerate(citations):
        if not isinstance(c, dict) or "source" not in c or "loc" not in c:
            raise SchemaValidationError(
                f"citations[{i}] must have 'source' and 'loc' keys, got {c!r}"
            )

    action = data["recommended_action"]
    valid_actions = {"BUY", "SELL", "HOLD", "REDUCE", "WATCH"}
    if action not in valid_actions:
        raise SchemaValidationError(
            f"recommended_action must be one of {valid_actions}, got {action!r}"
        )

    return data
