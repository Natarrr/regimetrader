"""regime_trader/utils/formatting.py
Shared display-formatting helpers used by both the Streamlit UI and Discord sender.
"""
from __future__ import annotations


def score_bar(score: float, width: int = 10) -> str:
    """Compact ASCII progress bar: ████░░░░

    Args:
        score: Value in [0, 1].
        width: Total number of characters (default 10).

    Returns:
        String of filled (█) and empty (░) blocks.
    """
    filled = min(width, max(0, round(score * width)))
    return "█" * filled + "░" * (width - filled)
