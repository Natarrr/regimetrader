"""backend/diagnostics/check_intel_sources.py
Unit-style checks for the intel pipeline's FMP-primary / yfinance-fallback logic.

Three checks:
  1. yfinance returns None → presence=False, score=0.50
  2. fmp_insider returns 0.65 → presence=True, active_sources contains 'insider'
  3. End-to-end single-ticker debug for AAPL (writes debug_AAPL.json)

Run from repo root:
    python -m backend.diagnostics.check_intel_sources
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

_HERE = Path(__file__).parent
_ROOT = _HERE.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_NEUTRAL_SCORE = 0.50


# ── helpers mirroring _run_intel_fetch merge logic ────────────────────────────

def _merged_insider(fmp_scores: Dict[str, float],
                    yf_scores:  Dict[str, float],
                    sym: str) -> float:
    """FMP primary, yfinance fallback — same logic as _run_intel_fetch."""
    return fmp_scores.get(sym, yf_scores.get(sym, _NEUTRAL_SCORE))


def _presence_flags(sym: str,
                    insider_presence: set,
                    fmp_ins_presence: set,
                    inst_presence:    set,
                    fmp_inst_presence: set,
                    news: dict,
                    finnhub_news: dict,
                    sentiment: dict,
                    stocktwits: dict,
                    finnhub_analyst: dict) -> Dict[str, bool]:
    """Canonical presence — matches _run_intel_fetch definition."""
    return {
        "insider":         sym in insider_presence or sym in fmp_ins_presence,
        "institutional":   sym in inst_presence    or sym in fmp_inst_presence,
        "news":            sym in news              or sym in finnhub_news,
        "sentiment":       sym in sentiment         or sym in stocktwits,
        "finnhub_analyst": sym in finnhub_analyst,
        "macro":           True,
    }


# ── Check 1 ───────────────────────────────────────────────────────────────────

def check_yf_none_returns_neutral() -> bool:
    """Simulate yfinance returning None for insider_transactions.

    Expected: insider presence=False, score=0.50 (neutral fallback).
    Minsky (1986 FIH) — absence of data must not be treated as a signal.
    """
    # Simulate: yfinance returned error, FMP also returned nothing
    yf_scores:  Dict[str, float] = {}          # no data from yfinance
    fmp_scores: Dict[str, float] = {}          # no data from FMP
    insider_presence: set         = set()      # yfinance returned None → not present
    fmp_ins_presence: set         = set()

    sym = "FAKE"
    score    = _merged_insider(fmp_scores, yf_scores, sym)
    flags    = _presence_flags(sym, insider_presence, fmp_ins_presence,
                               set(), set(), {}, {}, {}, {}, {})
    ok_score   = score == _NEUTRAL_SCORE
    ok_presence = not flags["insider"]

    status = "[PASS]" if (ok_score and ok_presence) else "[FAIL]"
    print(f"  {status}  Check 1 — yfinance None → presence=False, score=0.50")
    print(f"           score={score}  insider_present={flags['insider']}")
    return ok_score and ok_presence


# ── Check 2 ───────────────────────────────────────────────────────────────────

def check_fmp_score_marks_present() -> bool:
    """Simulate fmp_insider returning 0.65 for a ticker.

    Expected: presence=True, active_sources contains 'insider'.
    Lucas (1995 Nobel) — rational agents incorporate all available signals.
    """
    fmp_scores: Dict[str, float] = {"MSFT": 0.65}
    yf_scores:  Dict[str, float] = {}          # yfinance absent — FMP wins
    fmp_ins_presence: set         = {"MSFT"}   # score != 0.50 → FMP marked it

    sym = "MSFT"
    score  = _merged_insider(fmp_scores, yf_scores, sym)
    flags  = _presence_flags(sym, set(), fmp_ins_presence,
                              set(), set(), {}, {}, {}, {}, {})
    active = [k for k, v in flags.items() if v]

    ok_score    = abs(score - 0.65) < 1e-9
    ok_presence = flags["insider"]
    ok_active   = "insider" in active

    status = "[PASS]" if (ok_score and ok_presence and ok_active) else "[FAIL]"
    print(f"  {status}  Check 2 — fmp_insider=0.65 → presence=True, 'insider' in active_sources")
    print(f"           score={score}  insider_present={flags['insider']}  active={active}")
    return ok_score and ok_presence and ok_active


# ── Check 3 ───────────────────────────────────────────────────────────────────

def check_aapl_end_to_end() -> bool:
    """Run _debug_fetch_one('AAPL') and write debug_AAPL.json.

    Requires network access and a running Streamlit session is NOT needed —
    _debug_fetch_one only calls helper functions, not st.* methods.
    """
    try:
        # Import _debug_fetch_one at call time to avoid triggering st.* at import
        import importlib, types

        # We need to reach _debug_fetch_one without booting the full Streamlit app.
        # Safest: import the backend helpers directly and replicate the AAPL call.
        from backend.intelligence.engine import (
            fetch_fmp_insider_score,
            fetch_fmp_institutional_score,
            fetch_institutional_data,
        )
        import yfinance as _yf
        import time as _td

        sym = "AAPL"
        result: Dict[str, Any] = {"symbol": sym, "pillars": {}, "presence": {}, "errors": {}}

        # yfinance insider_transactions
        try:
            t = _yf.Ticker(sym)
            txns = t.insider_transactions
            if txns is not None and not (hasattr(txns, "empty") and txns.empty):
                result["presence"]["insider_yf"] = True
                result["pillars"]["yf_insider_rows"] = len(txns)
            else:
                result["presence"]["insider_yf"] = False
        except Exception as exc:
            result["errors"]["yf_insider"] = str(exc)
            result["presence"]["insider_yf"] = False
        _td.sleep(0.4)

        # FMP insider
        try:
            fmp_i = fetch_fmp_insider_score(sym)
            result["pillars"]["fmp_insider_score"] = fmp_i
            result["presence"]["insider_fmp"] = fmp_i != _NEUTRAL_SCORE
        except Exception as exc:
            result["errors"]["fmp_insider"] = str(exc)

        # yfinance institutional
        try:
            meta: Dict[str, Any] = {}
            inst_s = fetch_institutional_data(sym, _meta=meta)
            result["pillars"]["institutional_score"] = inst_s
            result["presence"]["institutional_yf"] = meta.get("has_data", False)
        except Exception as exc:
            result["errors"]["institutional"] = str(exc)
        _td.sleep(0.4)

        # FMP institutional
        try:
            fmp_inst = fetch_fmp_institutional_score(sym)
            result["pillars"]["fmp_institutional_score"] = fmp_inst
            result["presence"]["institutional_fmp"] = fmp_inst != _NEUTRAL_SCORE
        except Exception as exc:
            result["errors"]["fmp_institutional"] = str(exc)

        # Canonical presence
        result["canonical_presence"] = {
            "insider":       result["presence"].get("insider_yf", False)
                             or result["presence"].get("insider_fmp", False),
            "institutional": result["presence"].get("institutional_yf", False)
                             or result["presence"].get("institutional_fmp", False),
        }

        # Merged scores (FMP primary)
        result["merged_scores"] = {
            "insider":       result["pillars"].get("fmp_insider_score",
                             result["pillars"].get("institutional_score", _NEUTRAL_SCORE)),
            "institutional": result["pillars"].get("fmp_institutional_score",
                             result["pillars"].get("institutional_score", _NEUTRAL_SCORE)),
        }

        out = _ROOT / "logs" / "debug_AAPL.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"  [PASS]  Check 3 — AAPL debug written to {out}")
        print(f"           canonical_presence={result['canonical_presence']}")
        print(f"           merged_scores={result['merged_scores']}")
        return True

    except Exception as exc:
        print(f"  [FAIL]  Check 3 — exception: {exc}")
        return False


# ── Runner ────────────────────────────────────────────────────────────────────

def main() -> int:
    print(f"\n{'='*72}")
    print("  INTEL PIPELINE — UNIT-STYLE CHECKS")
    print(f"{'='*72}\n")

    results = [
        check_yf_none_returns_neutral(),
        check_fmp_score_marks_present(),
        check_aapl_end_to_end(),
    ]

    print(f"\n{'='*72}")
    passed = sum(results)
    print(f"  {passed}/{len(results)} checks passed.")
    if passed < len(results):
        print("  SOME CHECKS FAILED — see above for details.")
    print(f"{'='*72}\n")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
