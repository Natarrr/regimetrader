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
    for bucket in ("top_buys", "top_buys_usa", "top_buys_europe", "top_buys_asia",
                   "mid_caps", "small_caps"):
        for entry in data.get(bucket, []):
            key = (bucket, entry.get("ticker", ""))
            if key not in seen:
                seen.add(key)
                yield entry


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
