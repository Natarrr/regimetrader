"""regime_trader/services/universe.py
Ticker universe manager with stratified rotation and adaptive prioritisation.

Markowitz (1990 Nobel) — a diversified, stratified ticker universe ensures
broad market coverage without concentrating all EDGAR/FMP budget on a single
sector.

Design:
  - Maintains large_cap / mid_cap / small_cap lists with sector tags.
  - Rotation scheduler: each ticker processed ≥ once / week.
  - Stratification: sector quotas prevent technology domination.
  - Prioritisation: priority = α·news_vol + β·recent_insider + γ·vol_spike.
  - `get_tickers_for_day(date, budget)` allocates the daily fetch budget.
  - Rotation state persisted atomically under .cache/universe/state.json.

Public API:
    UniverseManager.get_tickers_for_day(date, budget) → List[str]
    UniverseManager.record_processed(tickers, date)
    UniverseManager.set_priority_scores(scores: Dict[str, float])
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Tuple

log = logging.getLogger(__name__)

# ── Default universes ──────────────────────────────────────────────────────────
# Each entry: (ticker, sector)
# Large-cap (S&P 500 core), mid-cap (S&P 400 sample), small-cap (Russell 2000 sample)

_LARGE_CAP: List[Tuple[str, str]] = [
    ("AAPL",  "Technology"), ("MSFT",  "Technology"), ("NVDA",  "Technology"),
    ("GOOGL", "Technology"), ("META",  "Technology"), ("AMZN",  "Consumer"),
    ("TSLA",  "Automotive"), ("AMD",   "Technology"), ("INTC",  "Technology"),
    ("QCOM",  "Technology"), ("AVGO",  "Technology"), ("CRM",   "Technology"),
    ("ADBE",  "Technology"), ("ORCL",  "Technology"), ("NOW",   "Technology"),
    ("JPM",   "Financials"), ("BAC",   "Financials"), ("WFC",   "Financials"),
    ("GS",    "Financials"), ("MS",    "Financials"), ("BLK",   "Financials"),
    ("V",     "Financials"), ("MA",    "Financials"), ("AXP",   "Financials"),
    ("JNJ",   "Healthcare"), ("LLY",   "Healthcare"), ("ABBV",  "Healthcare"),
    ("MRK",   "Healthcare"), ("UNH",   "Healthcare"), ("PFE",   "Healthcare"),
    ("ISRG",  "Healthcare"), ("VRTX",  "Healthcare"), ("REGN",  "Healthcare"),
    ("XOM",   "Energy"),     ("CVX",   "Energy"),     ("COP",   "Energy"),
    ("CAT",   "Industrial"), ("HON",   "Industrial"), ("GE",    "Industrial"),
    ("BA",    "Industrial"), ("LMT",   "Industrial"), ("RTX",   "Industrial"),
    ("HD",    "Retail"),     ("WMT",   "Retail"),     ("TGT",   "Retail"),
    ("MCD",   "Consumer"),   ("SBUX",  "Consumer"),   ("NKE",   "Consumer"),
    ("PG",    "Consumer"),   ("KO",    "Consumer"),   ("PEP",   "Consumer"),
    ("NEE",   "Utilities"),  ("DUK",   "Utilities"),  ("SO",    "Utilities"),
    ("PLD",   "REIT"),       ("AMT",   "REIT"),       ("EQIX",  "REIT"),
    ("BRK-B", "Financials"), ("SPGI",  "Financials"), ("CME",   "Financials"),
    ("TMO",   "Healthcare"), ("DHR",   "Healthcare"), ("SYK",   "Healthcare"),
    ("LOW",   "Retail"),     ("BKNG",  "Travel"),     ("ABNB",  "Travel"),
    ("AMAT",  "Technology"), ("LRCX",  "Technology"), ("TXN",   "Technology"),
    ("ETN",   "Industrial"), ("UPS",   "Industrial"), ("FDX",   "Industrial"),
    ("ABT",   "Healthcare"), ("MDT",   "Healthcare"), ("BMY",   "Healthcare"),
    ("C",     "Financials"), ("PGR",   "Financials"), ("CB",    "Financials"),
    ("SLB",   "Energy"),     ("OXY",   "Energy"),     ("HAL",   "Energy"),
    ("GM",    "Automotive"), ("F",     "Automotive"),
    ("NFLX",  "Technology"), ("DIS",   "Media"),      ("CMCSA", "Media"),
]

_MID_CAP: List[Tuple[str, str]] = [
    ("PODD",  "Healthcare"), ("HOLX",  "Healthcare"), ("TECH",  "Healthcare"),
    ("WDAY",  "Technology"), ("DDOG",  "Technology"), ("ZS",    "Technology"),
    ("CRWD",  "Technology"), ("OKTA",  "Technology"), ("SPLK",  "Technology"),
    ("FSLR",  "Energy"),     ("ENPH",  "Energy"),     ("SEDG",  "Energy"),
    ("ALLY",  "Financials"), ("SFM",   "Retail"),     ("WING",  "Consumer"),
    ("RCL",   "Travel"),     ("CCL",   "Travel"),     ("NCLH",  "Travel"),
    ("HLT",   "Travel"),     ("MAR",   "Travel"),
    ("NUE",   "Materials"),  ("FCX",   "Materials"),  ("CLF",   "Materials"),
    ("RPM",   "Materials"),  ("SON",   "Materials"),
    ("CABO",  "Telecom"),    ("LUMN",  "Telecom"),
    ("FR",    "REIT"),       ("REXR",  "REIT"),       ("EGP",   "REIT"),
    ("HALO",  "Healthcare"), ("IONS",  "Healthcare"), ("ACAD",  "Healthcare"),
    ("SMTC",  "Technology"), ("FORM",  "Technology"), ("SITM",  "Technology"),
]

_SMALL_CAP: List[Tuple[str, str]] = [
    ("FIZZ",  "Consumer"),   ("MGNI",  "Technology"), ("PRCT",  "Healthcare"),
    ("PAYO",  "Financials"), ("CSWI",  "Industrial"), ("VCEL",  "Healthcare"),
    ("TTGT",  "Technology"), ("PWSC",  "Technology"), ("IDYA",  "Healthcare"),
    ("XPOF",  "Consumer"),   ("CORT",  "Healthcare"), ("AMPH",  "Healthcare"),
    ("SWTX",  "Healthcare"), ("RCUS",  "Healthcare"), ("RVNC",  "Healthcare"),
    ("DVAX",  "Healthcare"), ("BLNK",  "Technology"), ("NTRA",  "Healthcare"),
    ("PRVA",  "Healthcare"), ("ARWR",  "Healthcare"), ("APLT",  "Healthcare"),
    ("ITRM",  "Healthcare"), ("FATE",  "Healthcare"), ("MGNX",  "Healthcare"),
    ("INVA",  "Healthcare"), ("TBPH",  "Healthcare"), ("MDXG",  "Healthcare"),
    ("GOSS",  "Technology"), ("RLAY",  "Healthcare"), ("PGEN",  "Healthcare"),
    ("TGTX",  "Healthcare"), ("KYMR",  "Healthcare"), ("KPTI",  "Healthcare"),
    ("BLUE",  "Healthcare"), ("PBYI",  "Healthcare"), ("HOOK",  "Technology"),
]

# Sector weight quotas: max fraction of daily budget per sector
_SECTOR_QUOTAS: Dict[str, float] = {
    "Technology": 0.30,
    "Healthcare": 0.25,
    "Financials": 0.15,
    "Consumer":   0.10,
    "Industrial": 0.08,
    "Energy":     0.05,
    "Retail":     0.03,
    "Materials":  0.02,
    "Utilities":  0.01,
    "REIT":       0.01,
    "_other":     0.10,
}

# Markowitz weights for priority score
_ALPHA = 0.4    # news volume weight
_BETA  = 0.4    # recent insider activity weight
_GAMMA = 0.2    # volatility spike weight

# Rotation window
_ROTATION_WINDOW_DAYS = 7

_STATE_PATH = Path(__file__).parent.parent.parent / ".cache" / "universe" / "state.json"

# Build combined universe
_ALL_TICKERS: List[Tuple[str, str, str]] = (
    [(t, s, "large") for t, s in _LARGE_CAP]
    + [(t, s, "mid")   for t, s in _MID_CAP]
    + [(t, s, "small") for t, s in _SMALL_CAP]
)


class UniverseManager:
    """Markowitz (1990 Nobel) — stratified, rotation-based ticker universe manager.

    Allocates a daily fetch budget across tickers ensuring:
    1. Every ticker is processed at least once per rotation window.
    2. Sector quotas prevent concentration in a single sector.
    3. Higher-priority tickers (news, insider activity, vol spike) get fetched first.

    Args:
        state_path: Path for persisting rotation state (default under .cache/).
    """

    def __init__(self, state_path: Optional[Path] = None) -> None:
        self._state_path      = Path(state_path) if state_path else _STATE_PATH
        self._priority_scores: Dict[str, float] = {}
        self._state           = self._load_state()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_tickers_for_day(
        self,
        target_date: "str | date",
        budget:      int,
    ) -> List[str]:
        """Markowitz (1990 Nobel) — Allocate daily budget across stratified tickers.

        Priority = α·news_vol + β·recent_insider + γ·vol_spike.
        Tickers not seen in the last _ROTATION_WINDOW_DAYS get a rotation boost.
        Sector quotas cap concentration.

        Args:
            target_date: The date for which to build the ticker list.
            budget:      Number of tickers to return.

        Returns:
            Ordered list of tickers (highest priority first), length ≤ budget.
        """
        date_str = _to_date_str(target_date)
        processed = self._state.get("processed", {})

        # Score each ticker
        scored: List[Tuple[float, str, str, str]] = []  # (score, ticker, sector, cap)
        for ticker, sector, cap in _ALL_TICKERS:
            last = processed.get(ticker, "1970-01-01")
            days_since = (_parse_date(date_str) - _parse_date(last)).days
            rotation_boost = 10.0 if days_since >= _ROTATION_WINDOW_DAYS else 0.0
            priority = (
                _ALPHA * self._priority_scores.get(ticker + "_news", 0.0)
                + _BETA  * self._priority_scores.get(ticker + "_insider", 0.0)
                + _GAMMA * self._priority_scores.get(ticker + "_vol", 0.0)
                + rotation_boost
            )
            scored.append((priority, ticker, sector, cap))

        # Sort descending by score
        scored.sort(key=lambda x: -x[0])

        # Apply sector quotas
        sector_budget: Dict[str, int] = {
            s: max(1, int(budget * q)) for s, q in _SECTOR_QUOTAS.items()
        }
        sector_counts: Dict[str, int] = defaultdict(int)
        selected: List[str] = []

        for _, ticker, sector, _ in scored:
            if len(selected) >= budget:
                break
            quota_key = sector if sector in _SECTOR_QUOTAS else "_other"
            if sector_counts[quota_key] < sector_budget.get(quota_key, budget):
                selected.append(ticker)
                sector_counts[quota_key] += 1

        log.info(
            "universe: %s → %d tickers (budget=%d, sectors=%s)",
            date_str,
            len(selected),
            budget,
            dict(sector_counts),
        )
        return selected

    def record_processed(
        self,
        tickers:     List[str],
        target_date: "str | date",
    ) -> None:
        """Mark tickers as processed for the given date (updates rotation state)."""
        date_str = _to_date_str(target_date)
        processed = self._state.setdefault("processed", {})
        for t in tickers:
            processed[t] = date_str
        self._save_state()

    def set_priority_scores(self, scores: Dict[str, float]) -> None:
        """Update priority scores.  Keys: '<ticker>_news', '<ticker>_insider', '<ticker>_vol'."""
        self._priority_scores.update(scores)

    def coverage_stats(self) -> Dict[str, int]:
        """Return how many tickers were seen in the last 7 days vs total."""
        processed = self._state.get("processed", {})
        cutoff = (datetime.now(timezone.utc).date() - timedelta(days=7)).isoformat()
        recent = sum(1 for v in processed.values() if v >= cutoff)
        return {
            "total_universe": len(_ALL_TICKERS),
            "processed_7d":   recent,
            "not_processed_7d": len(_ALL_TICKERS) - recent,
        }

    # ── State persistence ──────────────────────────────────────────────────────

    def _load_state(self) -> Dict:
        if not self._state_path.exists():
            return {}
        try:
            return json.loads(self._state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _save_state(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self._state, indent=2).encode("utf-8")
        fd, tmp = tempfile.mkstemp(
            prefix=".state.", suffix=".tmp", dir=str(self._state_path.parent)
        )
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(payload)
                fh.flush()
        finally:
            try:
                os.replace(tmp, self._state_path)
            except Exception:
                pass


# ── Utility ────────────────────────────────────────────────────────────────────

def _to_date_str(d: "str | date") -> str:
    return d if isinstance(d, str) else d.strftime("%Y-%m-%d")


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()
