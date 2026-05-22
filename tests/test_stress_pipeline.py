"""tests/test_stress_pipeline.py
Stress test: run_pipeline.run() against a "toxic" dataset where every ticker
has a zero market_cap.  validate_raw() must raise PipelineIntegrityError
(Stage 1 gate) and main() must return exit code 1.

All external I/O is mocked — no network calls, no API keys required.
"""
from __future__ import annotations

from contextlib import contextmanager, ExitStack
from pathlib import Path
from unittest.mock import patch

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
TOXIC_CSV = FIXTURES / "toxic_universe.csv"

# ── mock implementations ──────────────────────────────────────────────────────

def _zero_mktcap_profiles(tickers):
    """All market caps are 0 — triggers MISSING_AMOUNT on every ticker."""
    return {t: 0.0 for t in tickers}


@contextmanager
def _toxic_patches():
    """Context manager that applies all mocks needed for the toxic dataset."""
    with ExitStack() as stack:
        stack.enter_context(patch("scripts.run_pipeline.fetch_fmp_profiles",
                                  side_effect=_zero_mktcap_profiles))
        stack.enter_context(patch("scripts.run_pipeline.fetch_congress_buys",
                                  return_value={}))
        stack.enter_context(patch("scripts.run_pipeline._fetch_spy_return",
                                  return_value=0.0))
        stack.enter_context(patch("scripts.run_pipeline.fetch_fmp_insider_all",
                                  return_value={}))
        stack.enter_context(patch("scripts.run_pipeline.fetch_price_data",
                                  return_value={"return_20d": 0.0, "spy_return_20d": 0.0, "volume_spike": 1.0}))
        stack.enter_context(patch("scripts.run_pipeline.score_news_fmp",
                                  return_value=0.0))
        stack.enter_context(patch("scripts.run_pipeline._score_news_yfinance",
                                  return_value=0.0))
        stack.enter_context(patch("scripts.run_pipeline.fetch_edgar_data",
                                  return_value=(0, 0.0, False, 0)))
        stack.enter_context(patch("scripts.run_pipeline._load_cik_map",
                                  return_value={}))
        stack.enter_context(patch("scripts.run_pipeline._sec_get",
                                  side_effect=Exception("mocked — no network")))
        yield


# ── tests ─────────────────────────────────────────────────────────────────────

class TestStressPipeline:

    def test_toxic_dataset_raises_integrity_error(self, tmp_path):
        """run() on an all-zero-cap universe must raise PipelineIntegrityError."""
        from backend.market_intel.validator import PipelineIntegrityError
        from scripts.run_pipeline import run

        with _toxic_patches():
            with pytest.raises(PipelineIntegrityError) as exc_info:
                run(TOXIC_CSV, tmp_path)

        assert "failed validation" in str(exc_info.value).lower()

    def test_toxic_dataset_integrity_error_mentions_missing_amount(self, tmp_path):
        """The error message must name the failure code so ops can triage."""
        from backend.market_intel.validator import PipelineIntegrityError
        from scripts.run_pipeline import run

        with _toxic_patches():
            with pytest.raises(PipelineIntegrityError) as exc_info:
                run(TOXIC_CSV, tmp_path)

        assert "MISSING_AMOUNT" in str(exc_info.value)

    def test_main_returns_exit_code_1_on_toxic_dataset(self, tmp_path):
        """main() must translate PipelineIntegrityError into exit code 1."""
        from scripts.run_pipeline import main

        with _toxic_patches():
            code = main(["--tickers-file", str(TOXIC_CSV), "--log-dir", str(tmp_path)])

        assert code == 1

    def test_toxic_dataset_does_not_write_status_file(self, tmp_path):
        """The pipeline must not silently write a poisoned intel_source_status.json."""
        from backend.market_intel.validator import PipelineIntegrityError
        from scripts.run_pipeline import run

        with _toxic_patches():
            with pytest.raises(PipelineIntegrityError):
                run(TOXIC_CSV, tmp_path)

        assert not (tmp_path / "intel_source_status.json").exists()
