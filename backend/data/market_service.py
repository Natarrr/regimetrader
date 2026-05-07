"""backend/data/market_service.py
Thin re-export of the existing data/market_data.py MarketData class.

The backend imports from here so the router layer has a single stable
import path, and the underlying implementation can be swapped without
touching router code.
"""
from __future__ import annotations

import sys
import os

# Ensure project root is on sys.path so relative imports resolve
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data.market_data import MarketData  # noqa: F401  (re-exported)

__all__ = ["MarketData"]
