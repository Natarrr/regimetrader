# Code Style — Regime Trader

A short, opinionated guide. The goal is consistency and reviewability, not perfection.

---

## Python version & types

- **Python 3.11+**. Use `from __future__ import annotations` at the top of every module.
- **Type hints required** on every public function. Internal helpers (`_name`) can skip them but shouldn't.
- Use built-in generics: `list[int]`, `dict[str, Any]` (PEP 585) — not `List[int]`.

---

## Imports

```python
from __future__ import annotations

# stdlib
import json
import os
from pathlib import Path
from typing import Any, Optional

# third-party
import numpy as np
import pandas as pd

# project
from regime_trader.utils.io import save_json_atomic
```

Three groups, blank line between each. No wildcard imports except in
`__init__.py` re-export shims.

---

## Logging

**Always**:

```python
import logging
log = logging.getLogger(__name__)
```

**Never** `print()` in library code (CLI scripts may use `print` at the
boundary). Configure once at the entry point:

```python
from regime_trader.utils.logging_cfg import configure_logging
configure_logging(level="INFO")
```

### Don't leak secrets

Don't log `os.environ`, `request.headers`, full HTTP request bodies, or
anything that contains an API key. The test
[test_streamlit_app_smoke.py::test_configure_logging_does_not_emit_environment](../tests/test_streamlit_app_smoke.py)
enforces this for `configure_logging`.

When you must log a request:

```python
log.info("fmp request", extra={"endpoint": url.split("?")[0], "params_count": len(params)})
# NOT: log.info(f"fmp request: {url}?{params}")  # apikey ends up in the log
```

---

## Atomic file I/O

Any file that other code reads concurrently — caches, snapshots, run summaries —
must be written atomically. Use `save_json_atomic`:

```python
from regime_trader.utils.io import save_json_atomic, load_json_safe

save_json_atomic(cache_file, payload)         # write
data = load_json_safe(cache_file, default={}) # read, never raises
```

Under the hood: write to a tempfile in the same directory, then `Path.replace()`
(POSIX `rename` semantics — atomic on the same filesystem).

Test for atomic behavior in any new cache module — see
[tests/test_atomic_write.py](../tests/test_atomic_write.py) and the round-trip
test in [tests/test_streamlit_app_smoke.py](../tests/test_streamlit_app_smoke.py).

---

## Async vs sync

Streamlit is fully synchronous. Library functions follow this rule:

| Caller | Function flavour |
| ------ | ---------------- |
| Streamlit, pytest, CLI | `*_sync` wrapper |
| GitHub Actions / FastAPI | async coroutine |

Pattern:

```python
async def run_scan_async(n: int = 20) -> list[ScanResult]:
    ...

def run_scan_sync(n: int = 20) -> list[ScanResult]:
    return asyncio.run(run_scan_async(n))
```

For network I/O inside sync code, prefer a bounded thread pool over `asyncio.run`
in tight loops:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

with ThreadPoolExecutor(max_workers=8) as pool:
    futures = {pool.submit(fetch, sym): sym for sym in symbols}
    for fut in as_completed(futures, timeout=30):
        ...
```

`max_workers` should be configurable via env or function arg, not hardcoded
deep in the call stack.

---

## HTTP

Use a single `requests.Session()` per module with a `urllib3.util.retry.Retry`
adapter:

```python
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

_RETRY = Retry(total=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))
_SESSION = requests.Session()
_SESSION.mount("https://", HTTPAdapter(max_retries=_RETRY))
```

Every external call gets an explicit `timeout=`. Default 15 s for low-latency
data feeds, 30 s for SEC EDGAR.

---

## Errors

- `raise SpecificError("context")` — never bare `except:` or `except Exception: pass`.
- Wrap external calls in narrow try/except, log at WARNING, return a sentinel:

  ```python
  try:
      return _fmp_client.fetch(symbol, timeout=15)
  except (requests.RequestException, ValueError) as exc:
      log.warning("fmp fetch failed: symbol=%s err=%s", symbol, exc)
      return None
  ```

- Module-level: validate inputs at the boundary with `pydantic` schemas, not
  with hand-rolled `if not isinstance(...)`.

---

## Tests

- One test file per source module: `regime/foo.py` → `tests/test_foo.py`.
- Use `pytest`, not `unittest.TestCase` (existing code mostly uses pytest classes
  with `Test...` prefix — that's fine).
- Mock the network. `requests-mock` for `requests`, `monkeypatch.setattr` on
  `yfinance.download` for yf.
- Don't assert on log strings unless the log itself is the contract under test.

For a complete acceptance run before pushing:

```bash
bash scripts/lint_and_test.sh
```

---

## Docstrings

Public functions get a short docstring with a one-line summary. Optional Args /
Returns / Raises sections only when the signature is non-obvious.

The "Laureate Quant" persona in [CLAUDE.md](../CLAUDE.md) requires Nobel-laureate
attribution in financial-model docstrings. That's a project-specific convention,
not a general rule. Apply it for `regime/`, `analysis/`, `backend/quant_models/`
modules; skip for plumbing (`utils/`, `ui/`, scripts).

---

## Don't add to working code

- No empty `__init__.py` re-export gymnastics. Callers use the full module path.
- No backwards-compat shims for code that has no external callers.
- No pre-emptive `try: import X except ImportError: X = None` — declare the
  dependency in `requirements.txt` and let it fail loudly if absent.
- No comments restating the code (`# increment counter` above `i += 1`).

When in doubt, look at how it's done in `regime/regime_detector.py` or
`analysis/claude_client.py` — those are the reference modules.
