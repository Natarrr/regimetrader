# Path: src/services/fmp/client.py
"""Composed FMPClient — shared core + per-category endpoint mixins."""
from __future__ import annotations

from src.services.fmp.core import FMPCore
from src.services.fmp.market import MarketDataMixin
from src.services.fmp.ownership import OwnershipFlowMixin
from src.services.fmp.estimates import EstimatesSentimentMixin
from src.services.fmp.fundamentals import FundamentalsMixin


class FMPClient(
    FMPCore,
    MarketDataMixin,
    OwnershipFlowMixin,
    EstimatesSentimentMixin,
    FundamentalsMixin,
):
    """Unified FMP Ultimate client — stable/ routes, full premium surface.

    Args:
        api_key:    FMP API key. Defaults to FMP_API_KEY env var.
        cache_root: Directory for file-based TTL cache. Defaults to .cache/fmp/.
    """
