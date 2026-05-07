"""decision_matrix/engine/models.py
Typed dataclasses for the Decision Matrix compute engine.

All fields use plain Python types (float, str, bool, list) so the dataclasses
are JSON-serialisable and can be passed to FastAPI response models or logged
directly.  Use `dataclasses.asdict(state)` to convert to a plain dict.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TechnicalSignal:
    """Per-symbol technical indicators computed from OHLCV history."""
    symbol: str
    beta: Optional[float] = None
    rsi14: Optional[float] = None
    rsi_label: str = ""
    rsi_color: str = "#555555"
    trend_status: str = ""
    trend_color: str = "#555555"
    atr14: Optional[float] = None
    atr_pct: float = 0.0          # % above 30-day ATR baseline
    atr_alert: bool = False       # True when atr_pct > 20
    daily_action: str = ""
    action_color: str = "#555555"


@dataclass
class ConvictionScore:
    """Regime-adjusted unified conviction for a symbol."""
    symbol: str
    conviction: float = 0.5       # [0, 1]
    grade: str = "B"
    grade_color: str = "#ffbb33"


@dataclass
class Position:
    """Single open position snapshot."""
    symbol: str
    qty: float
    avg_cost: float
    price: float
    market_value: float
    unreal_pnl: float
    unreal_pct: float             # decimal e.g. -0.05 = -5%


@dataclass
class ActionRow:
    """Fully resolved action recommendation for one position."""
    symbol: str
    action: str                   # SELL | TRIM | HOLD | ADD | BUY MORE
    urgency: int                  # lower = more urgent (matches ACTION_URGENCY)
    reason: str
    intel: float                  # composite intel score [0, 1]
    unreal_pct: float
    qty: float
    cost: float
    price: float
    mv: float                     # market value
    pnl: float
    trend: str
    trend_color: str
    rsi: Optional[float]
    rsi_label: str
    rsi_color: str
    atr_stop: Optional[float]
    risk_usd: Optional[float]
    conviction: float
    grade: str
    grade_color: str
    atr_alert: bool
    atr_pct: float


@dataclass
class RegimeState:
    """Current HMM + composite regime state."""
    label: str                    # Crash | Panic | Bear | Neutral | Bull | Euphoria | Mania
    confidence: float             # HMM posterior [0, 1]
    probs: Dict[str, float] = field(default_factory=dict)
    is_uncertain: bool = False
    stability_bars: int = 0
    flicker_count: int = 0


@dataclass
class RiskState:
    """Portfolio risk score (0-100) and per-component breakdown."""
    total: float                  # 0-100 normalised
    breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass
class DecisionMatrixState:
    """Complete output of the Decision Matrix compute engine.

    Pass this to the Streamlit render layer; it must not perform any
    computation — only read from this state object.
    """
    regime: RegimeState
    risk: RiskState
    action_rows: List[ActionRow]
    brief_items: List[tuple]      # (badge, color, title, description)
    sector_warnings: List[str]
    volatility_symbols: List[str]
    portfolio_beta: Optional[float]
    total_mv: float
    total_unreal: float
    daily_pnl_pct: float
    alloc_frac: float
    minsky_trace: Dict            # raw trigger provenance for audit
