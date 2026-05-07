"""
core/models.py
──────────────
Shared domain models for the Regime-Intel morning-report pipeline.

All classes are frozen dataclasses where state must not change after
construction, and regular dataclasses where the orchestrator needs
to mutate (e.g., PortfolioState.positions updated from broker).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional


# ── Enumerations ──────────────────────────────────────────────────────────────


class Direction(str, Enum):
    LONG  = "long"
    SHORT = "short"
    FLAT  = "flat"


class ActionType(str, Enum):
    BUY    = "BUY"
    SELL   = "SELL"
    HOLD   = "HOLD"
    REDUCE = "REDUCE"


class RegimeLabel(str, Enum):
    CRASH    = "Crash"
    PANIC    = "Panic"
    BEAR     = "Bear"
    NEUTRAL  = "Neutral"
    BULL     = "Bull"
    EUPHORIA = "Euphoria"
    MANIA    = "Mania"
    UNKNOWN  = "Unknown"


# ── Core signal ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Signal:
    """
    A trading signal for one symbol.

    Attributes
    ----------
    symbol        : Ticker (e.g. "AAPL").
    direction     : "long" | "short" | "flat".
    target_weight : Target portfolio weight as a fraction (0.0 – 1.0).
    confidence    : Overall conviction score (0.0 – 1.0).
    justification : Human-readable explanation for the report.
    generated_at  : UTC timestamp when the signal was produced.
    """

    symbol:        str
    direction:     str         # Direction enum value
    target_weight: float       # 0.0 – 1.0
    confidence:    float       # 0.0 – 1.0
    justification: str
    generated_at:  datetime = field(default_factory=lambda: datetime.utcnow())

    def __post_init__(self) -> None:
        if not (0.0 <= self.target_weight <= 1.5):
            raise ValueError(f"target_weight {self.target_weight} out of range [0, 1.5]")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence {self.confidence} out of range [0, 1]")


# ── Portfolio state ───────────────────────────────────────────────────────────


@dataclass
class Position:
    """One open position in the portfolio."""

    symbol:        str
    qty:           float          # shares (positive = long, negative = short)
    avg_cost:      float          # average entry price
    current_price: float          # last known price
    sector:        str = "Unknown"

    @property
    def market_value(self) -> float:
        return self.qty * self.current_price

    @property
    def unrealised_pnl(self) -> float:
        return (self.current_price - self.avg_cost) * self.qty

    @property
    def unrealised_pnl_pct(self) -> float:
        if self.avg_cost == 0:
            return 0.0
        return (self.current_price / self.avg_cost - 1.0) * (1 if self.qty > 0 else -1)


@dataclass
class PortfolioState:
    """
    Snapshot of the portfolio at a given moment.

    Attributes
    ----------
    equity     : Total portfolio value (cash + market value of positions).
    cash       : Uninvested cash.
    positions  : Open positions keyed by symbol.
    drawdown   : Current drawdown from peak equity (negative number, e.g. -0.05).
    as_of      : UTC timestamp of the snapshot.
    """

    equity:    float
    cash:      float
    positions: Dict[str, Position] = field(default_factory=dict)
    drawdown:  float               = 0.0
    as_of:     datetime            = field(default_factory=lambda: datetime.utcnow())

    @property
    def invested(self) -> float:
        """Total market value of all open positions."""
        return sum(p.market_value for p in self.positions.values())

    @property
    def cash_fraction(self) -> float:
        if self.equity == 0:
            return 1.0
        return self.cash / self.equity

    def weight_of(self, symbol: str) -> float:
        """Current portfolio weight of a symbol (0.0 if not held)."""
        if self.equity == 0:
            return 0.0
        pos = self.positions.get(symbol)
        if pos is None:
            return 0.0
        return abs(pos.market_value) / self.equity

    @property
    def total_unrealised_pnl(self) -> float:
        return sum(p.unrealised_pnl for p in self.positions.values())


# ── Intelligence score ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class IntelligenceScore:
    """
    Multi-source conviction score for a single symbol.

    All sub-scores are normalised to [0, 1] before combining.
    Values closer to 1.0 are bullish; values closer to 0.0 are bearish.
    A neutral reading is 0.50.

    Attributes
    ----------
    symbol         : Ticker.
    flow_score     : Options-flow signal (UnusualWhales / Unusual activity).
    sentiment_score: Social-sentiment signal (Reddit / ApeWisdom).
    insider_score  : Insider-buying signal (OpenInsider).
    macro_score    : Macro / regime context score (HMM + VIX).
    final_conviction: Geometric weighted mean of the four sub-scores.
    raw_data       : Optional dict of raw API payloads for audit.
    scored_at      : UTC timestamp.
    """

    symbol:           str
    flow_score:       float   # institutional holders conviction [0, 1]  (legacy: was whale)
    sentiment_score:  float   # StockTwits sentiment             [0, 1]
    insider_score:    float   # FMP v4 insider trading           [0, 1]
    macro_score:      float   # Finnhub analyst consensus        [0, 1]
    final_conviction: float   # geometric weighted mean          [0, 1]
    congress_score:   float   = 0.50   # Finnhub news + VADER     [0, 1]  (legacy: was news)
    raw_data:         Dict    = field(default_factory=dict)
    scored_at:        datetime = field(default_factory=lambda: datetime.utcnow())
    # ── Dynamic weighting metadata (v3+) ──────────────────────────────────────
    confidence_level: float     = 1.0    # fraction of pillars with real data [0, 1]
    pillar_weights:   Dict      = field(default_factory=dict)   # {pillar: weight used}
    weight_triggers:  List[str] = field(default_factory=list)   # fired rule labels

    @property
    def label(self) -> str:
        """Human-readable conviction tier (aligned with _label_for_score in streamlit_app.py)."""
        c = self.final_conviction
        if c >= 0.78:
            return "🟢 Strong Bullish"
        if c >= 0.63:
            return "🟡 Moderate Bullish"
        if c >= 0.58:
            return "⚪ Mildly Bullish"    # direction="long" threshold
        if c > 0.42:
            return "⚪ Neutral"            # direction="neutral" band
        if c >= 0.37:
            return "🟠 Mildly Bearish"   # direction="short" threshold
        if c >= 0.22:
            return "🔴 Moderate Bearish"
        return "🔴 Strong Bearish"

    @property
    def color(self) -> str:
        """HTML color for dashboard display."""
        c = self.final_conviction
        if c >= 0.78:
            return "#00C851"   # strong green
        if c >= 0.63:
            return "#7CB342"   # light green
        if c >= 0.58:
            return "#9E9E9E"   # grey-green (mildly bullish)
        if c > 0.42:
            return "#9E9E9E"   # grey (neutral)
        if c >= 0.37:
            return "#FF8800"   # orange (mildly bearish)
        if c >= 0.22:
            return "#FF4444"   # red
        return "#CC0000"       # deep red


# ── Rebalancing recommendation ────────────────────────────────────────────────


@dataclass(frozen=True)
class RebalanceAction:
    """
    A portfolio action recommendation generated by the rebalancing engine.

    Attributes
    ----------
    symbol          : Ticker.
    action          : BUY | SELL | REDUCE | HOLD.
    current_weight  : Current portfolio weight (fraction).
    target_weight   : Target portfolio weight (fraction).
    delta_weight    : target - current (positive = need to buy more).
    estimated_value : Approximate dollar value of the transaction.
    signal          : The Signal that drove this recommendation.
    score           : The IntelligenceScore backing it.
    """

    symbol:          str
    action:          str              # ActionType value
    current_weight:  float
    target_weight:   float
    delta_weight:    float
    estimated_value: float
    signal:          Signal
    score:           IntelligenceScore

    @property
    def is_buy(self) -> bool:
        return self.action in (ActionType.BUY, ActionType.BUY.value)

    @property
    def urgency_label(self) -> str:
        abs_delta = abs(self.delta_weight)
        if abs_delta > 0.10:
            return "High"
        if abs_delta > 0.05:
            return "Medium"
        return "Low"


# ── Morning report data ───────────────────────────────────────────────────────


@dataclass
class MorningReportData:
    """
    Complete data payload for the morning HTML email report.

    Assembled by the orchestrator and handed to MorningReporter.send().
    """

    report_date:       datetime
    regime_label:      str
    regime_confidence: float
    regime_probs:      Dict[str, float]

    portfolio:         PortfolioState
    rebalance_actions: List[RebalanceAction]
    top_opportunities: List[IntelligenceScore]    # top-5 from universe scan

    vix_level:         Optional[float] = None
    spy_change_pct:    Optional[float] = None
    notes:             List[str]       = field(default_factory=list)
