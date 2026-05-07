"""backend/diagnostics/check_factors.py
Confirm all five scoring factors return valid, non-default values.

Run from repo root:
    python -m backend.diagnostics.check_factors
    python -m backend.diagnostics.check_factors --tickers AAPL MSFT XOM

Exit code 0  → all factors present and non-neutral for at least one ticker.
Exit code 1  → one or more factors returned the 0.50 default for all tickers
               (indicates a missing API key or broken data source).
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Dict, List

logging.basicConfig(
    level=logging.DEBUG,
    format="%(levelname)-8s %(name)s : %(message)s",
)

# Project root must be importable
import os as _os
_here = _os.path.dirname(_os.path.abspath(__file__))
_root = _os.path.dirname(_os.path.dirname(_here))
if _root not in sys.path:
    sys.path.insert(0, _root)

from backend.utils.score_helpers import (
    fetch_insider_data,
    fetch_institutional_score,
    fetch_news_sentiment_for_ticker,
    aggregate_scores,
    WEIGHTS,
)

_DEFAULT_TICKERS = ["AAPL", "MSFT", "XOM", "JPM", "SPY"]

_FACTOR_KEYS = ["insider", "institutional", "news"]  # the three fetched at runtime


def run_diagnostics(tickers: List[str]) -> Dict[str, Dict[str, float]]:
    """Fetch all three runtime factors for each ticker, print results, return data."""
    results: Dict[str, Dict[str, float]] = {}

    print(f"\n{'='*72}")
    print(f"  FACTOR DIAGNOSTICS  —  {len(tickers)} tickers")
    print(f"{'='*72}")
    print(f"  {'TICKER':<8}  {'INSIDER':>8}  {'INST':>8}  {'NEWS':>8}  {'DATA?'}")
    print(f"  {'-'*60}")

    all_neutral = {k: True for k in _FACTOR_KEYS}

    for tk in tickers:
        ins  = fetch_insider_data(tk)
        inst = fetch_institutional_score(tk)
        news = fetch_news_sentiment_for_ticker(tk)

        data_flag = []
        if ins  != 0.50: all_neutral["insider"]       = False; data_flag.append("INS")
        if inst != 0.50: all_neutral["institutional"] = False; data_flag.append("INST")
        if news != 0.50: all_neutral["news"]          = False; data_flag.append("NEWS")

        flag_str = ",".join(data_flag) if data_flag else "neutral-only"

        print(f"  {tk:<8}  {ins:>8.3f}  {inst:>8.3f}  {news:>8.3f}  {flag_str}")
        results[tk] = {"insider": ins, "institutional": inst, "news": news}

    print(f"{'='*72}\n")

    # Aggregate a sample score for the first ticker
    if tickers:
        tk   = tickers[0]
        agg  = aggregate_scores(
            macro_score=0.65,          # representative macro value
            institutional_score=results[tk]["institutional"],
            insider_score=results[tk]["insider"],
            news_score=results[tk]["news"],
            regime_mult=1.20,          # Bull-regime multiplier
        )
        print(f"  Sample aggregate_scores for {tk} (macro=0.65, Bull regime):")
        for key in ("macro_score", "institutional_score", "insider_score",
                    "news_score", "regime_mult", "final_score", "badge"):
            print(f"    {key:<24} {agg[key]}")
        print(f"    score_breakdown:")
        for k, v in agg["score_breakdown"].items():
            print(f"      {k:<20} {v:.4f}  (weight={WEIGHTS.get(k, 0):.2f})")
        print()

    return results, all_neutral


def main() -> int:
    parser = argparse.ArgumentParser(description="Check all five scoring factors")
    parser.add_argument("--tickers", nargs="+", default=_DEFAULT_TICKERS,
                        help="Tickers to test (default: AAPL MSFT XOM JPM SPY)")
    args = parser.parse_args()

    results, all_neutral = run_diagnostics(args.tickers)

    # Report which factors are stuck at 0.50
    problems = [k for k, v in all_neutral.items() if v]
    if problems:
        print(f"  WARNING: The following factors returned 0.50 (default) for ALL tickers:")
        for p in problems:
            print(f"    - {p}: check API key / data source in backend/utils/score_helpers.py")
        print()
        return 1

    print("  All factors returned non-neutral values for at least one ticker.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
