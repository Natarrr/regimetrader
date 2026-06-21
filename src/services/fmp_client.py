# Path: src/services/fmp_client.py
"""FMP Ultimate client — compatibility shim.

The implementation now lives in the per-category package ``src/services/fmp/``
(comparison plan, Track C — modular split mirroring simonpierreboucher02/fmp-mcp's
per-category layout, adapted to a single shared-core class):

    src/services/fmp/core.py          shared HTTP / cache / rate-limit / breaker
    src/services/fmp/market.py        prices, quotes, batch-quote, screener
    src/services/fmp/ownership.py     congress, insider, 13F institutional flow
    src/services/fmp/estimates.py     news, earnings, analyst, price-target, transcript
    src/services/fmp/fundamentals.py  ratios, EV, DCF, sector P/E, statements

This module re-exports the public names so existing imports
(`from src.services.fmp_client import FMPClient, FMPEndpointError,
fmp_prices_to_arrays`) and module-path monkeypatches
(`src.services.fmp_client.time.*`, `src.services.fmp_client.FMPClient.<method>`)
keep working unchanged. `import time` is kept so those patch targets resolve.
"""
from __future__ import annotations

import time  # noqa: F401 — kept for `src.services.fmp_client.time.*` patch targets

from src.services.fmp import (  # noqa: F401
    FMPClient,
    FMPEndpointError,
    _DEAD_ENDPOINTS,
    _STABLE,
    _TTL,
    fmp_prices_to_arrays,
)

__all__ = [
    "FMPClient",
    "FMPEndpointError",
    "fmp_prices_to_arrays",
    "_DEAD_ENDPOINTS",
    "_TTL",
    "_STABLE",
]
