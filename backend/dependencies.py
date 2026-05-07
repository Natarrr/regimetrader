"""backend/dependencies.py
Shared FastAPI dependency injection — caching, sys.path wiring.

Provides a single location to configure how the existing core/ modules
(macro_global, hmm_engine, etc.) are made available to routers without
modifying the originals.
"""
from __future__ import annotations

import os
import sys

# Ensure the project root (regime_trader/) is importable so that
# core/, hmm_engine/, data/ etc. resolve without package install.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Expose project root path for other modules that need it
PROJECT_ROOT: str = _PROJECT_ROOT
