"""research — local-only quant research sandbox (never imported by production).

Production boundary (CLAUDE.md §2): nothing under research/ is imported by the
live pipeline. The only artifact that crosses the boundary is research/ic_report.json,
which backend/market_intel/portfolio_optimizer.py reads as an advisory IC input.
"""
