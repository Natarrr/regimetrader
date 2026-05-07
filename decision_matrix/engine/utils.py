"""decision_matrix/engine/utils.py
Pure utility functions — no side effects, no imports from engine siblings.
"""
from __future__ import annotations

from typing import List, Tuple


def safe_div(n: float, d: float, default: float = 0.0) -> float:
    """Divide n by d; return default if d is zero."""
    if d == 0:
        return default
    return n / d


def clamp(x: float, lo: float, hi: float) -> float:
    """Clamp x to [lo, hi]."""
    return max(lo, min(hi, x))


def normalize_risk(raw: float, max_raw: float) -> float:
    """Map raw risk score to 0-100 scale."""
    return clamp(safe_div(raw, max_raw, 0.0) * 100.0, 0.0, 100.0)


def pick_grade(conviction: float, grade_table: List[Tuple]) -> Tuple[str, str]:
    """Return (grade_label, hex_color) for a given conviction score.

    grade_table: [(min_score, label, color), ...] sorted descending.
    """
    for threshold, grade, color in grade_table:
        if conviction >= threshold:
            return grade, color
    return "C", "#ff4444"
