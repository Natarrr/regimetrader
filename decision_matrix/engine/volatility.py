"""decision_matrix/engine/volatility.py
ATR volatility detection for the Decision Matrix engine.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from decision_matrix.engine.models import TechnicalSignal


def detect_volatility(
    signals: Dict[str, TechnicalSignal],
) -> Tuple[bool, List[str]]:
    """Return (any_alert, [symbols_with_atr_alert]).

    Args:
        signals: symbol -> TechnicalSignal map from _get_technical_signals.

    Returns:
        (any_alert, vol_symbols) where vol_symbols is the list of tickers
        whose ATR is > 20% above their 30-day baseline.
    """
    vol_syms = [s.symbol for s in signals.values() if s.atr_alert]
    return bool(vol_syms), vol_syms
