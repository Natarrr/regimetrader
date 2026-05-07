"""backend/market_intel/config.py — pipeline-wide constants.

Environment variables (read at import time):
    EDGAR_USER_AGENT — required by SEC fair-access policy. Format: "Name email@host"
                       (alias: SEC_USER_AGENT — kept for back-compat)
    FMP_API_KEY      — used by fmp_fallback (optional but recommended).
    MARKET_INTEL_DATA_DIR — overrides default data/raw/ location.
    MARKET_INTEL_LOG_LEVEL — DEBUG | INFO | WARNING (default INFO).
"""
from __future__ import annotations

import os
from pathlib import Path

# ── SEC EDGAR ─────────────────────────────────────────────────────────────────

SEC_BASE = "https://www.sec.gov"
SEC_DATA_BASE = "https://data.sec.gov"

# SEC requires a User-Agent identifying the requester (name + email).
# Spec env var is EDGAR_USER_AGENT; SEC_USER_AGENT kept as backward-compatible alias.
USER_AGENT: str = (
    os.getenv("EDGAR_USER_AGENT")
    or os.getenv("SEC_USER_AGENT")
    or "Nathan MarketIntel n.tardy@hotmail.fr"
)

# Fair access: max 10 req/s. Use 7 to leave headroom.
REQ_PER_SEC: float = 7.0
MIN_SPACING_S: float = 1.0 / REQ_PER_SEC

# ── Storage ───────────────────────────────────────────────────────────────────

_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent.parent

DATA_DIR: Path = Path(
    os.getenv("MARKET_INTEL_DATA_DIR", str(_PROJECT_ROOT / "data" / "raw" / "edgar"))
)
LOG_DIR: Path = _PROJECT_ROOT / "logs"
CACHE_DIR: Path = _PROJECT_ROOT / "data" / "cache"

# Ticker→CIK map TTL (SEC publishes daily but it changes rarely).
TICKER_MAP_TTL_DAYS: int = 7

# ── Rollback flag ─────────────────────────────────────────────────────────────

# When False, the adapter skips EDGAR and goes straight to FMP. Use this as a
# kill-switch during the canary if EDGAR latency / coverage is unacceptable.
# Toggle via scripts/rollback_toggle.sh or by editing .env directly.
EDGAR_FIRST: bool = os.getenv("EDGAR_FIRST", "true").strip().lower() in ("1", "true", "yes", "y")


# ── Filing filters ────────────────────────────────────────────────────────────

FORM_TYPES_INSIDER = ("4", "4/A")
FORM_TYPES_INSTITUTIONAL = ("13F-HR", "13F-HR/A")

# Default cap on filings fetched per ticker per run.
DEFAULT_LIMIT_FORMS: int = 5

# Window for scoring (Form-4 events older than this are dropped).
INSIDER_WINDOW_DAYS: int = 90

# ── HTTP retry policy ─────────────────────────────────────────────────────────

HTTP_RETRIES: int = 3
HTTP_BACKOFF_BASE_S: float = 0.5   # 0.5, 1.0, 2.0
HTTP_TIMEOUT_S: float = 20.0

# ── FMP ───────────────────────────────────────────────────────────────────────

FMP_API_KEY: str = os.getenv("FMP_API_KEY", "")
FMP_BASE: str = "https://financialmodelingprep.com/api"
