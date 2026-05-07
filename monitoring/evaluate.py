"""monitoring/evaluate.py — pure threshold gate for the canary.

Markowitz frame: the canary acceptance test is a joint constraint on two
independent axes — data coverage (analogous to portfolio completeness) and
realised error count (analogous to drawdown tolerance). Both must clear
simultaneously, mirroring the joint feasibility region of mean-variance
optimisation.

Coverage gate:  $\\text{coverage} = \\frac{\\text{edgar\\_count}}{\\text{ticker\\_count}} \\geq c_{\\min}$
Error gate:     $\\text{error\\_count} \\leq e_{\\max}$

Pure function — no I/O, no side effects, no logging. Importable by
`monitoring.check_metrics` (CLI entry) and `tests/test_check_metrics.py`
(truth table) without coupling either to the other.
"""
from __future__ import annotations

from typing import List, Tuple


def evaluate(
    metrics: dict,
    min_coverage: float = 0.6,
    max_errors: int = 0,
) -> Tuple[bool, List[str]]:
    """Markowitz: return (ok, reasons) for the canary's joint-constraint gate.

    A degenerate ticker_count short-circuits to a single reason — coverage
    is undefined when the denominator is non-positive, so subsequent checks
    would be meaningless. Otherwise both gates are evaluated independently
    and every breach is reported, so operators see the full diagnosis.
    """
    ticker_count = metrics.get("ticker_count", 0)
    if not isinstance(ticker_count, (int, float)) or ticker_count <= 0:
        return False, ["ticker_count must be > 0"]

    edgar  = metrics.get("edgar_count", 0) or 0
    errors = metrics.get("error_count", 0) or 0
    coverage = edgar / ticker_count

    reasons: List[str] = []
    if coverage < min_coverage:
        reasons.append(
            f"EDGAR coverage {edgar}/{ticker_count} below threshold {min_coverage:.2f}"
        )
    if errors > max_errors:
        reasons.append(f"error_count {errors} exceeds max_errors {max_errors}")

    return (len(reasons) == 0), reasons
