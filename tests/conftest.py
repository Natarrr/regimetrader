"""tests/conftest.py
CI network isolation.

When the CI environment variable is set (GitHub Actions sets it
automatically), any test that makes a live HTTP call without an
explicit mock raises a clear RuntimeError rather than making a
flaky or secret-leaking network request.

Per-test overrides (unittest.mock.patch, monkeypatch.setattr) are
unaffected — they replace the method at a scope closer to the call
site and take precedence over this fixture.
"""
from __future__ import annotations

import os
from typing import Generator

import pytest

_IN_CI = os.environ.get("CI", "").lower() in ("1", "true")


if _IN_CI:
    @pytest.fixture(autouse=True)
    def _ci_block_live_http(monkeypatch: pytest.MonkeyPatch) -> Generator[None, None, None]:
        """Block unmocked HTTP in CI so forgotten mocks fail fast and loudly."""
        import requests

        def _blocked(self: object, prepared: object, **_kw: object) -> None:
            url = getattr(prepared, "url", "?")
            method = getattr(prepared, "method", "?")
            raise RuntimeError(
                f"[CI] Unmocked live HTTP call blocked: {method} {url}\n"
                "Add unittest.mock.patch or monkeypatch to this test."
            )

        monkeypatch.setattr(requests.Session, "send", _blocked)
        yield
