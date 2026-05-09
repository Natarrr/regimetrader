"""discovery_scanner.py — backwards-compat shim.

The implementation now lives in :mod:`regime_trader.discovery_scanner`.

Existing callers using ``from discovery_scanner import X`` (e.g. local notebooks
or ad-hoc CLI invocations) keep working through these re-exports. New code
should import directly from the package:

    from regime_trader.discovery_scanner import get_top_alpha_picks_sync

CLI:
    python -m discovery_scanner
    # or:
    python -m regime_trader.discovery_scanner
"""
from __future__ import annotations

from regime_trader.discovery_scanner import (  # noqa: F401
    ScanResult,
    disc_get_json,
    enrich_with_momentum,
    explain_result,
    fmp_insider_buys,
    fmp_institutional_accumulation,
    fmp_profile,
    fmp_profile_batch,
    fmp_screener,
    force_refresh_sync,
    get_top_alpha_picks_sync,
    liquidity_filter,
    load_disc_cache,
    load_json_safe,
    run_scan,
    run_scan_async,
    save_disc_cache,
    save_json_atomic,
    select_candidates,
)

__all__ = [
    "ScanResult",
    "disc_get_json",
    "enrich_with_momentum",
    "explain_result",
    "fmp_insider_buys",
    "fmp_institutional_accumulation",
    "fmp_profile",
    "fmp_profile_batch",
    "fmp_screener",
    "force_refresh_sync",
    "get_top_alpha_picks_sync",
    "liquidity_filter",
    "load_disc_cache",
    "load_json_safe",
    "run_scan",
    "run_scan_async",
    "save_disc_cache",
    "save_json_atomic",
    "select_candidates",
]


if __name__ == "__main__":
    results = run_scan(n=10)
    for r in results:
        print(explain_result(r))
        print()
