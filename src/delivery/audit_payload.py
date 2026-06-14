#!/usr/bin/env python3
"""Pre-flight audit for Discord top_lists payload. Exit 0 = pass, 1 = fail."""
import sys
import json
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class PipelineAuditError(Exception):
    """Base class for all pipeline audit failures."""


class ScoreDivergenceError(PipelineAuditError):
    """final_score is outside the valid [0, 1] range."""


class BadgeMismatchError(PipelineAuditError):
    """Badge label does not match the final_score threshold."""


class SortingError(PipelineAuditError):
    """top_buys entries are not sorted descending by final_score."""


class CrossContaminationError(PipelineAuditError):
    """EU/Asia ticker carries a non-zero congress factor."""


class GeographicLeakageError(PipelineAuditError):
    """Ticker suffix/market tag mismatch."""


class StructuralIntegrityError(PipelineAuditError):
    """Field format or Discord limit violation."""


class VIXCoherenceError(PipelineAuditError):
    """VIX value is implausible (negative or absurdly large)."""


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Badge thresholds — the single source of truth for this audit. Must match
# generate_top_lists._BADGES and send_toplists_discord._badge_from_score.
_BADGE_THRESHOLDS = [(0.80, "HIGH BUY"), (0.60, "TACTICAL BUY"), (0.00, "WATCHLIST")]

# VIX coherence bound used by check F (plausible-range guard, not a regime label).
_VIX_MAX = 200.0

# Ticker regex: up to 5 uppercase letters (USA) or alphanumeric + dot + 1-2 uppercase (intl)
_TICKER_RE = re.compile(r"^([A-Z]{1,5}|[A-Z0-9]{1,6}\.[A-Z]{1,2})$")

# Non-USA markets
_FOREIGN_MARKETS = {"EUROPE", "ASIA"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _expected_badge(score: float) -> str:
    """Return the badge label that corresponds to score."""
    for threshold, label in _BADGE_THRESHOLDS:
        if score >= threshold:
            return label
    return "WATCHLIST"


def _iter_all_entries(data: dict):
    """Yield every entry across all score buckets (deduped by ticker)."""
    seen: set = set()
    for bucket in (
        "top_buys", "top_buys_usa", "top_buys_europe", "top_buys_asia",
        # SMID leverage sleeve: copies of US entries ranked by leverage_score —
        # covered by checks A/B/D/E; deliberately EXCLUDED from check C (the
        # list is not sorted by final_score) and leverage_score itself is a
        # ranking key that check A does not gate (it may exceed 1.0).
        "top_buys_smid",
        "usa_overflow", "eu_overflow", "asia_overflow",
        "mid_caps", "small_caps",
        # CAPITULATION moves every entry into watchlist (cook_toplists) — the
        # safety gate must keep auditing in exactly that regime. eu/asia
        # mid-small are cooked into the combined payload alongside it.
        "watchlist", "eu_mid_small", "asia_mid_small",
    ):
        for entry in data.get(bucket, []):
            key = (bucket, entry.get("ticker", ""))
            if key not in seen:
                seen.add(key)
                yield entry


_ON_DEMAND_PIPELINES = {"US", "INTL"}


def _audit_on_demand(data: dict) -> None:
    """Check H — on-demand single-ticker block (on_demand_ticker).

    The block lives outside the bulk buckets, so checks A-E/G never see it;
    this applies the same gate semantics to the embedded entry. Daily payloads
    never carry the key, so this is a no-op for the standard pipeline.
    """
    block = data["on_demand_ticker"]
    if not isinstance(block, dict):
        raise StructuralIntegrityError(
            f"on_demand_ticker must be a dict, got {type(block).__name__}"
        )

    ticker = block.get("ticker")
    if not isinstance(ticker, str) or not ticker:
        raise StructuralIntegrityError(
            f"on_demand_ticker.ticker={ticker!r} must be a non-empty string"
        )
    if not _TICKER_RE.match(ticker):
        raise StructuralIntegrityError(
            f"on_demand ticker {ticker!r} does not match allowed format "
            f"(e.g. 'MSFT', 'SAP.DE')"
        )

    pipeline = block.get("pipeline")
    if pipeline not in _ON_DEMAND_PIPELINES:
        raise StructuralIntegrityError(
            f"on_demand_ticker.pipeline={pipeline!r} must be one of "
            f"{sorted(_ON_DEMAND_PIPELINES)}"
        )

    if "scoring_mode" not in block:
        raise StructuralIntegrityError(
            "on_demand_ticker.scoring_mode is missing — single-ticker scores "
            "must disclose their normalization mode"
        )

    entry = block.get("entry")
    if not isinstance(entry, dict):
        raise StructuralIntegrityError(
            "on_demand_ticker.entry is missing or not a dict"
        )
    if entry.get("ticker") != ticker:
        raise StructuralIntegrityError(
            f"on_demand_ticker.ticker={ticker!r} != entry.ticker="
            f"{entry.get('ticker')!r}"
        )

    # A. Score range
    score = entry.get("final_score")
    if score is None or not (0.0 <= score <= 1.0):
        raise ScoreDivergenceError(
            f"On-demand ticker {ticker!r}: final_score={score!r} is outside [0, 1]"
        )

    # B. Badge consistency
    badge = entry.get("badge", "")
    expected = _expected_badge(score)
    if badge != expected:
        raise BadgeMismatchError(
            f"On-demand ticker {ticker!r}: score={score:.4f} expects "
            f"badge={expected!r}, got {badge!r}"
        )

    # D. Geographic leakage — suffix vs market
    market = entry.get("market", "USA")
    has_suffix = "." in ticker
    if has_suffix and market not in _FOREIGN_MARKETS:
        raise GeographicLeakageError(
            f"On-demand ticker {ticker!r} has an international suffix but "
            f"market={market!r}; expected EUROPE or ASIA"
        )
    if not has_suffix and market in _FOREIGN_MARKETS:
        raise GeographicLeakageError(
            f"On-demand ticker {ticker!r} has no suffix but market={market!r}; "
            f"expected USA"
        )

    # E. Cross-contamination — non-US entries must have congress == 0
    if market in _FOREIGN_MARKETS:
        congress_val = entry.get("factors", {}).get("congress", 0.0)
        if congress_val > 0.0:
            raise CrossContaminationError(
                f"On-demand ticker {ticker!r} ({market}): congress factor="
                f"{congress_val!r} > 0; non-US tickers must not carry a "
                f"congress signal"
            )

    # E2. Dynamic INTL ceiling — same semantics as the bulk check
    if market in _FOREIGN_MARKETS:
        try:
            from src.config.weights import get_weights as _get_weights
            _weights = _get_weights(ticker)
            _avail = sum(
                w for f, w in _weights.items()
                if f not in {"congress", "transcript_tone"}
            ) if _weights else 1.0
        except Exception:
            _avail = 1.0
        if float(score) > round(_avail, 6) + 1e-4:
            raise ScoreDivergenceError(
                f"On-demand ticker {ticker!r} ({market}): final_score="
                f"{score:.4f} exceeds dynamic available-factor ceiling "
                f"{_avail:.4f}. Possible US-factor injection."
            )


# ---------------------------------------------------------------------------
# Public audit function
# ---------------------------------------------------------------------------

def audit(top_lists_path="logs/top_lists.json") -> bool:
    """Run all pre-flight checks.

    Parameters
    ----------
    top_lists_path:
        Either a file-system path (str or Path) or a dict already parsed from
        top_lists.json.  Passing a dict allows unit-tests to avoid I/O.

    Returns
    -------
    True on success.  Raises a PipelineAuditError subclass on the first
    detected violation.
    """
    if isinstance(top_lists_path, dict):
        data = top_lists_path
    else:
        data = json.loads(Path(top_lists_path).read_text(encoding="utf-8"))

    # ------------------------------------------------------------------
    # H. On-demand single-ticker block — key-gated, daily payloads skip
    # ------------------------------------------------------------------
    if "on_demand_ticker" in data:
        _audit_on_demand(data)

    # ------------------------------------------------------------------
    # A. Score range — every entry across all buckets
    # ------------------------------------------------------------------
    for entry in _iter_all_entries(data):
        ticker = entry.get("ticker", "?")
        score = entry.get("final_score")
        if score is None or not (0.0 <= score <= 1.0):
            raise ScoreDivergenceError(
                f"Ticker {ticker!r}: final_score={score!r} is outside [0, 1]"
            )

    # ------------------------------------------------------------------
    # B. Badge consistency — every entry
    # ------------------------------------------------------------------
    for entry in _iter_all_entries(data):
        ticker = entry.get("ticker", "?")
        score = entry["final_score"]
        badge = entry.get("badge", "")
        expected = _expected_badge(score)
        if badge != expected:
            raise BadgeMismatchError(
                f"Ticker {ticker!r}: score={score:.4f} expects badge={expected!r}, "
                f"got {badge!r}"
            )

    # ------------------------------------------------------------------
    # C. Sort order — all top_buys lists must be descending by final_score
    # ------------------------------------------------------------------
    for list_key in ("top_buys", "top_buys_usa", "top_buys_europe", "top_buys_asia"):
        bucket = data.get(list_key, [])
        for i in range(len(bucket) - 1):
            a = bucket[i]["final_score"]
            b = bucket[i + 1]["final_score"]
            if a < b:
                raise SortingError(
                    f"{list_key}[{i}].final_score={a:.4f} < {list_key}[{i+1}].final_score={b:.4f} "
                    f"— list is not sorted descending"
                )

    # ------------------------------------------------------------------
    # D. Geographic leakage — ticker suffix vs market field
    # ------------------------------------------------------------------
    for entry in _iter_all_entries(data):
        ticker = entry.get("ticker", "")
        market = entry.get("market", "USA")
        has_suffix = "." in ticker
        if has_suffix and market not in _FOREIGN_MARKETS:
            raise GeographicLeakageError(
                f"Ticker {ticker!r} has an international suffix but market={market!r}; "
                f"expected EUROPE or ASIA"
            )
        if not has_suffix and market in _FOREIGN_MARKETS:
            raise GeographicLeakageError(
                f"Ticker {ticker!r} has no suffix but market={market!r}; "
                f"expected USA"
            )

    # ------------------------------------------------------------------
    # E. Cross-contamination — EU/Asia entries must have congress == 0
    # ------------------------------------------------------------------
    for entry in _iter_all_entries(data):
        ticker = entry.get("ticker", "?")
        market = entry.get("market", "USA")
        if market in _FOREIGN_MARKETS:
            factors = entry.get("factors", {})
            congress_val = factors.get("congress", 0.0)
            if congress_val > 0.0:
                raise CrossContaminationError(
                    f"Ticker {ticker!r} ({market}): congress factor={congress_val!r} > 0; "
                    f"non-US tickers must not carry a congress signal"
                )

    # ------------------------------------------------------------------
    # E2. Dynamic range validation — per-ticker international score ceiling
    #     Ceiling = sum of available factor weights for the ticker's region.
    #     WEIGHTS_EU / WEIGHTS_ASIA / WEIGHTS_GLOBAL all zero congress and
    #     transcript_tone, so available weight = 1.0 for all intl regions.
    #     Score > ceiling + tolerance signals US-factor injection or arithmetic
    #     error (also caught by check A above for > 1.0).
    # ------------------------------------------------------------------
    try:
        from src.config.weights import get_weights as _get_weights
        _weights_import_ok = True
    except ImportError:
        _weights_import_ok = False

    _CEILING_TOLERANCE = 1e-4

    for entry in _iter_all_entries(data):
        ticker = entry.get("ticker", "?")
        market = entry.get("market", "USA")
        if market not in _FOREIGN_MARKETS:
            continue

        score = float(entry.get("final_score", 0.0))
        factors = entry.get("factors", {})

        try:
            _weights = _get_weights(ticker) if _weights_import_ok else {}
            _intl_available_weight = (
                sum(w for f, w in _weights.items() if f not in {"congress", "transcript_tone"})
                if _weights else 1.0
            )
        except Exception:
            _intl_available_weight = 1.0

        _dynamic_ceiling = round(_intl_available_weight, 6)

        if score > _dynamic_ceiling + _CEILING_TOLERANCE:
            raise ScoreDivergenceError(
                f"Ticker {ticker!r} ({market}): final_score={score:.4f} exceeds "
                f"dynamic available-factor ceiling {_dynamic_ceiling:.4f}. "
                f"Available weights sum: {_intl_available_weight:.4f}. "
                f"Possible US-factor injection. factors={factors}"
            )

    # ------------------------------------------------------------------
    # F. VIX coherence — value must be a plausible positive float
    # ------------------------------------------------------------------
    vix = data.get("vix")
    if vix is None or not (0.0 <= float(vix) <= _VIX_MAX):
        raise VIXCoherenceError(
            f"VIX={vix!r} is outside the plausible range [0, {_VIX_MAX:.0f}]"
        )

    # ------------------------------------------------------------------
    # G. Structural Discord limits — ticker format validation
    # ------------------------------------------------------------------
    for list_key in ("top_buys", "top_buys_usa", "top_buys_europe", "top_buys_asia"):
        for entry in data.get(list_key, []):
            ticker = entry.get("ticker", "")
            if not _TICKER_RE.match(ticker):
                raise StructuralIntegrityError(
                    f"Ticker {ticker!r} does not match allowed format "
                    f"(e.g. 'MSFT', 'SAP.DE')"
                )

    return True


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Pre-flight audit for Discord top_lists payload."
    )
    parser.add_argument(
        "--input",
        default="logs/top_lists.json",
        help="Path to top_lists.json (default: logs/top_lists.json)",
    )
    args = parser.parse_args()

    try:
        audit(args.input)
        print("[AUDIT] PASSED — all checks green")
        return 0
    except PipelineAuditError as exc:
        print(
            f"[AUDIT] FAILED — {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    except FileNotFoundError as exc:
        print(f"[AUDIT] FAILED — file not found: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
