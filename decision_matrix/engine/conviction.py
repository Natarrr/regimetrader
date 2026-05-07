"""decision_matrix/engine/conviction.py
Build the per-symbol conviction map from raw score rows.
"""
from __future__ import annotations

from typing import Dict, Iterable

from decision_matrix.engine.config import CONVICTION_GRADES
from decision_matrix.engine.models import ConvictionScore
from decision_matrix.engine.utils import clamp, pick_grade


def build_conviction_map(raw_rows: Iterable[Dict]) -> Dict[str, ConvictionScore]:
    """Convert iterable of score dicts to a symbol-keyed conviction map.

    Each row must have at least "symbol".  Optional "conviction" defaults to 0.5.

    Args:
        raw_rows: e.g. a list of dicts from conviction DataFrame.iterrows()

    Returns:
        Dict mapping symbol -> ConvictionScore
    """
    out: Dict[str, ConvictionScore] = {}
    for row in raw_rows or []:
        sym = row.get("symbol")
        if not sym:
            continue
        cv = clamp(float(row.get("conviction", 0.5)), 0.0, 1.0)
        grade, color = pick_grade(cv, CONVICTION_GRADES)
        out[sym] = ConvictionScore(
            symbol=sym,
            conviction=cv,
            grade=grade,
            grade_color=color,
        )
    return out


def crash_conviction_override(
    conviction_map: Dict[str, ConvictionScore],
    regime_label: str,
) -> Dict[str, ConvictionScore]:
    """In Crash regime: force all conviction scores to 0 and grade to C.

    Does not mutate input — returns a new dict.

    Why: the Brief says "Unified Conviction = 0 across all symbols" in Crash,
    but without this override the engine still uses real conviction data.
    """
    if regime_label != "Crash":
        return conviction_map
    return {
        sym: ConvictionScore(symbol=sym, conviction=0.0, grade="C", grade_color="#ff4444")
        for sym in conviction_map
    }
