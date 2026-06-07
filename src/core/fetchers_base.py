from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MarketEnum(str, Enum):
    USA = "USA"
    EUROPE = "EUROPE"
    ASIA = "ASIA"


@dataclass
class TickerEntry:
    ticker: str
    market: MarketEnum
    sector: str
    cap_tier: str
    source_reliability: float
    raw_factors: dict[str, Any] = field(default_factory=dict)


class BaseMarketFetcher(abc.ABC):

    @property
    @abc.abstractmethod
    def market(self) -> MarketEnum: ...

    @abc.abstractmethod
    def prepare(self, tickers: list[str]) -> list[TickerEntry]: ...

    @abc.abstractmethod
    def source_reliability(self, ticker: str) -> float: ...
