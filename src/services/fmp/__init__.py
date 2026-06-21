# Path: src/services/fmp/__init__.py
"""FMP Ultimate client package (modular per-category split)."""
from __future__ import annotations

from src.services.fmp.client import FMPClient
from src.services.fmp.core import (
    FMPEndpointError,
    fmp_prices_to_arrays,
    _DEAD_ENDPOINTS,
    _TTL,
    _STABLE,
)

__all__ = ["FMPClient", "FMPEndpointError", "fmp_prices_to_arrays",
           "_DEAD_ENDPOINTS", "_TTL", "_STABLE"]
