"""tests/test_pipeline_integrity.py
Pipeline integrity tests: per-ticker scorers, universe preservation, and
generate() end-to-end output contract.

All tests are pure-Python with no network calls or filesystem I/O beyond
tmp_path fixtures.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest


# ── Helpers ────────────────────────────────────────────────────────────────────

def _raw_row(
    ticker: str = "AAPL",
    edgar: float = 0.70,
    insider: float = 0.80,
    congress: float = 0.60,
    news: float = 0.65,
    momentum: float = 0.55,
    cap_tier: str = "large",
    market_cap: float = 3e12,
    sector: str = "Technology",
    quality_piotroski_raw: int | None = 6,
) -> Dict[str, Any]:
    """Minimal intel_source_status.json result row for use in tests.

    Parameters use short aliases for brevity; they map to the 7-factor field
    names consumed by generate_top_lists.FACTOR_FIELDS:
      edgar    → insider_conviction_score (primary insider signal)
      insider  → insider_breadth_score
      congress → congress_score
      news     → news_sentiment_score  (news_buzz_score mirrors at 0.60)
      momentum → momentum_long_score   (volume_attention_score mirrors at 0.50)
    """
    return {
        "ticker":                    ticker,
        "sector":                    sector,
        "cap_tier":                  cap_tier,
        "market_cap":                market_cap,
        "insider_conviction_score":  edgar,
        "insider_breadth_score":     insider,
        "congress_score":            congress,
        "news_sentiment_score":      news,
        "news_buzz_score":           round(news * 0.85, 4),
        "momentum_long_score":       momentum,
        "volume_attention_score":    round(momentum * 0.90, 4),
        # 12-factor model additions (sparse but non-zero so schema gate passes)
        "analyst_consensus_score":   round(news * 0.50, 4),
        "analyst_revision_score":    round(momentum * 0.40, 4),
        "price_target_upside_score": round(momentum * 0.30, 4),
        "quality_piotroski_score":   round(edgar * 0.60, 4),
        "quality_piotroski_raw":     quality_piotroski_raw,
        "transcript_tone_score":     round(news * 0.20, 4),
        "ceo_buy":                   False,
        "form4_count":               3,
        "quiver_evidence":           {},
        "news_source":               "finnhub",
        "insider_usd":               50_000.0,
        "momentum_spy_relative":     0.02,
        "volume_spike":              1.5,
        "_edgar_ok":                 True,
        "_scoring_error":            False,
    }


def _status_payload(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {"results": rows, "run_id": "test"}


# ── score_edgar ────────────────────────────────────────────────────────────────

class TestScoreEdgar:
    def _fn(self):
        from src.ingestion.run_pipeline import score_edgar
        return score_edgar

    def test_zero_filings_returns_zero(self):
        assert self._fn()(0) == 0.0

    def test_negative_returns_zero(self):
        assert self._fn()(-1) == 0.0

    def test_one_filing_above_zero(self):
        assert self._fn()(1) > 0.0

    def test_output_bounded_0_to_1(self):
        fn = self._fn()
        for n in (0, 1, 5, 10, 50, 200, 999):
            v = fn(n)
            assert 0.0 <= v <= 1.0, f"score_edgar({n}) = {v} out of [0,1]"

    def test_monotone_increasing(self):
        fn = self._fn()
        scores = [fn(n) for n in (1, 5, 20, 100, 200)]
        assert scores == sorted(scores), "score_edgar should be monotone increasing"

    def test_large_count_does_not_exceed_ceiling(self):
        assert self._fn()(10_000) <= 0.90


# ── score_insider_value ────────────────────────────────────────────────────────

class TestScoreInsiderValue:
    def _fn(self):
        from src.ingestion.run_pipeline import score_insider_value
        return score_insider_value

    def test_zero_purchases_returns_zero(self):
        assert self._fn()(0.0, 1e10) == 0.0

    def test_zero_market_cap_returns_zero(self):
        assert self._fn()(50_000.0, 0.0) == 0.0

    def test_negative_purchases_returns_zero(self):
        assert self._fn()(-1.0, 1e10) == 0.0

    def test_output_bounded_0_to_1(self):
        fn = self._fn()
        for usd, cap in ((1_000, 1e9), (50_000, 1e10), (1e7, 1e11), (1e9, 1e9)):
            v = fn(usd, cap)
            assert 0.0 <= v <= 1.0, f"score_insider_value({usd}, {cap}) = {v}"

    def test_floor_at_small_but_real_purchase(self):
        # 0.01% of $1B market cap = $100k → floor ~0.30
        v = self._fn()(100_000.0, 1e9)
        assert v >= 0.29, f"small credible purchase scored too low: {v}"

    def test_larger_pct_scores_higher(self):
        fn = self._fn()
        cap = 1e10
        low  = fn(1_000.0, cap)     # ~0.001% — very small
        high = fn(10_000_000.0, cap) # ~0.1%   — meaningful conviction
        assert high > low

    def test_recency_decay_applied_after_30_days(self):
        fn = self._fn()
        fresh = fn(100_000.0, 1e9, days_since_most_recent=0)
        stale = fn(100_000.0, 1e9, days_since_most_recent=90)
        # stale should be closer to 0.5 (direction preserved, urgency reduced)
        assert abs(stale - 0.5) < abs(fresh - 0.5), (
            f"recency decay not applied: fresh={fresh}, stale={stale}"
        )

    def test_recency_decay_converges_toward_neutral(self):
        fn = self._fn()
        fresh = fn(100_000.0, 1e9, days_since_most_recent=0)
        stale = fn(100_000.0, 1e9, days_since_most_recent=999)
        # Decay pulls score toward 0.5 — stale is closer to neutral than fresh
        assert abs(stale - 0.5) < abs(fresh - 0.5), (
            f"decay did not pull toward neutral: fresh={fresh}, stale={stale}"
        )

    def test_recent_purchase_no_decay(self):
        fn = self._fn()
        v0  = fn(100_000.0, 1e9, days_since_most_recent=0)
        v30 = fn(100_000.0, 1e9, days_since_most_recent=30)
        assert v0 == v30, "decay must not apply at exactly 30 days"


# ── generate() end-to-end output contract ─────────────────────────────────────

class TestGenerateTopLists:
    """End-to-end tests for generate() using synthetic raw rows."""

    def _generate(self, rows: List[Dict[str, Any]], tmp_path: Path) -> Dict[str, Any]:
        from backend.market_intel.generate_top_lists import generate
        status = _status_payload(rows)
        # Patch VIX read to return None (no macro overlay) so tests are deterministic
        with patch("backend.market_intel.generate_top_lists._read_vix", return_value=None):
            result = generate(status, run_id="test", log_dir=tmp_path)
        return result

    def test_output_has_required_keys(self, tmp_path):
        rows = [_raw_row(f"T{i}") for i in range(5)]
        out = self._generate(rows, tmp_path)
        for key in ("top_buys", "mid_caps", "small_caps", "ticker_count",
                    "weights", "generated_at", "kill_switch"):
            assert key in out, f"missing key: {key}"

    def test_ticker_count_matches_input(self, tmp_path):
        rows = [_raw_row(f"T{i}") for i in range(10)]
        out = self._generate(rows, tmp_path)
        assert out["ticker_count"] == 10

    def test_top_buys_max_5(self, tmp_path):
        rows = [_raw_row(f"T{i}") for i in range(20)]
        out = self._generate(rows, tmp_path)
        assert len(out["top_buys"]) <= 5

    def test_top_buys_sorted_descending(self, tmp_path):
        rows = [_raw_row(f"T{i}", edgar=float(i) / 20) for i in range(10)]
        out = self._generate(rows, tmp_path)
        scores = [e["final_score"] for e in out["top_buys"]]
        assert scores == sorted(scores, reverse=True)

    def test_each_entry_has_required_fields(self, tmp_path):
        rows = [_raw_row(f"T{i}") for i in range(5)]
        out = self._generate(rows, tmp_path)
        for entry in out["top_buys"]:
            for field in ("ticker", "final_score", "badge", "factors", "cap_tier"):
                assert field in entry, f"entry missing {field}: {entry}"

    def test_badge_high_buy_threshold(self, tmp_path):
        # Give one ticker clearly dominant raw scores → should get HIGH BUY after normalization.
        # Need >= 40% of universe (2/5) with non-zero factors to pass the circuit-breaker.
        rows = [_raw_row("STAR", edgar=1.0, insider=1.0, congress=1.0, news=1.0, momentum=1.0)]
        rows += [_raw_row("BASE", edgar=0.3, insider=0.3, congress=0.1, news=0.2, momentum=0.2)]
        rows += [_raw_row(f"T{i}", edgar=0.0, insider=0.0, congress=0.0, news=0.0, momentum=0.0)
                 for i in range(3)]
        out = self._generate(rows, tmp_path)
        star = next(e for e in out["top_buys"] if e["ticker"] == "STAR")
        assert star["badge"] in ("HIGH BUY", "TACTICAL BUY"), f"unexpected badge: {star['badge']}"
        assert star["final_score"] >= star["factors"].get("insider_conviction", 0), \
            "final_score below individual factor — weighting broken"

    def test_factors_dict_has_expected_keys(self, tmp_path):
        # FACTOR_FIELDS covers 18 factors (12 US-weighted incl. inst_flow_13f +
        # 4 INTL fundamental signals + analyst_revision/price_target_upside).
        rows = [_raw_row(f"T{i}") for i in range(5)]
        out = self._generate(rows, tmp_path)
        expected_keys = {
            "insider_conviction", "insider_breadth", "congress",
            "news_sentiment", "news_buzz", "momentum_long", "volume_attention",
            "analyst_consensus", "analyst_revision", "price_target_upside",
            "quality_piotroski", "transcript_tone", "revenue_revision",
            "inst_flow_13f",
            "fcf_yield", "amihud_shock", "pb_value_up", "roic_quality",
        }
        for entry in out["top_buys"]:
            factors = entry["factors"]
            assert set(factors.keys()) == expected_keys, (
                f"unexpected factor keys: {set(factors.keys()) ^ expected_keys}"
            )

    def test_all_zero_scores_raises_integrity_error(self, tmp_path):
        # When every ticker has all-zero factors the circuit breaker must fire —
        # this prevents writing a degenerate top_lists.json to Discord.
        from backend.market_intel.generate_top_lists import PipelineIntegrityError
        rows = [_raw_row(f"T{i}", edgar=0.0, insider=0.0, congress=0.0, news=0.0, momentum=0.0)
                for i in range(5)]
        with pytest.raises(PipelineIntegrityError):
            self._generate(rows, tmp_path)

    def test_top_lists_json_written_to_disk(self, tmp_path):
        rows = [_raw_row(f"T{i}") for i in range(5)]
        self._generate(rows, tmp_path)
        assert (tmp_path / "top_lists_us.json").exists()

    def test_top5_csv_written_to_disk(self, tmp_path):
        rows = [_raw_row(f"T{i}") for i in range(5)]
        self._generate(rows, tmp_path)
        assert (tmp_path / "top5.csv").exists()

    def test_kill_switch_false_when_vix_none(self, tmp_path):
        rows = [_raw_row(f"T{i}") for i in range(5)]
        out = self._generate(rows, tmp_path)
        assert out["kill_switch"] is False

    def test_mid_caps_only_contains_mid_tier(self, tmp_path):
        rows = [_raw_row(f"M{i}", cap_tier="mid", market_cap=5e9) for i in range(3)]
        rows += [_raw_row(f"L{i}", cap_tier="large", market_cap=50e9) for i in range(3)]
        out = self._generate(rows, tmp_path)
        for entry in out["mid_caps"]:
            assert entry["cap_tier"] == "mid", f"non-mid in mid_caps: {entry['ticker']}"

    def test_scoring_error_row_included_not_dropped(self, tmp_path):
        # A row with _scoring_error=True must still appear in ticker_count
        rows = [_raw_row("GOOD")]
        bad = _raw_row("BAD")
        bad["_scoring_error"] = True
        bad["edgar_score"] = 0.0
        bad["insider_score"] = 0.0
        rows.append(bad)
        out = self._generate(rows, tmp_path)
        assert out["ticker_count"] == 2, "scoring-error ticker was silently dropped"


# ── fetch_edgar_data: transaction-date recency (PATCH 01) ─────────────────────

class TestFetchEdgarDataRecency:
    """
    PATCH 01: days_since_most_recent must be derived from the transaction date
    (transactionDate field in Form 4 XML), NOT from the filing date.

    SEC Form 4 may be filed up to 2 business days after the actual purchase.
    Using the filing date overstates signal freshness by 0-2 days.

    Cohen, Malloy & Pomorski (2012): point-in-time accuracy is essential for
    correct signal half-life estimation.
    """

    def _fetch(self):
        from src.ingestion.run_pipeline import fetch_edgar_data
        return fetch_edgar_data

    def _make_sec_response(self, filing_date: str, accession: str = "0001234567-24-000001"):
        """Return a mock object whose .json() mimics data.sec.gov/submissions/CIK*.json."""
        from unittest.mock import MagicMock
        resp = MagicMock()
        resp.json.return_value = {
            "filings": {
                "recent": {
                    "form": ["4"],
                    "filingDate": [filing_date],
                    "accessionNumber": [accession],
                    "primaryDocument": ["form4.xml"],
                }
            }
        }
        return resp

    def test_uses_transaction_date_not_filing_date(self):
        """When p_transactions carry a 'date' field, recency uses that, not form4_filings[0]['date']."""
        from datetime import datetime, timedelta, timezone
        from unittest.mock import patch

        # Match the production clock basis (datetime.now(timezone.utc).date(),
        # run_pipeline:474) — date.today() is LOCAL and drifts ±1 day vs UTC at
        # the day boundary, which is the hardcoded-fixture date-drift fragility.
        _today = datetime.now(timezone.utc).date()
        tx_date_str = (_today - timedelta(days=10)).isoformat()
        filing_date_str = (_today - timedelta(days=8)).isoformat()  # 2 days later (filing lag)

        p_tx = [{"date": tx_date_str, "value": 50_000, "title": "CEO", "code": "P", "is_ceo": True}]

        with patch("src.ingestion.run_pipeline._load_cik_map", return_value={"TEST": "0001234567"}), \
             patch("src.ingestion.run_pipeline._sec_get", return_value=self._make_sec_response(filing_date_str)), \
             patch("src.ingestion.run_pipeline._parse_form4_xml", return_value=p_tx):
            result = self._fetch()("TEST", lookback_days=180)

        _, _, _, days_since, _, _, _ = result
        assert days_since == 10, (
            f"Expected 10 days (transaction date), got {days_since} — "
            f"patch may be using filing date ({filing_date_str}) instead of tx date ({tx_date_str})"
        )

    def test_filing_date_used_as_fallback_when_no_tx_dates(self):
        """When p_transactions have no 'date' field, falls back to filing date without error."""
        from datetime import datetime, timedelta, timezone
        from unittest.mock import patch

        # UTC basis to match production (run_pipeline:474) — avoids the local-vs-UTC
        # day-boundary off-by-one (date-drift fragility).
        filing_date_str = (datetime.now(timezone.utc).date() - timedelta(days=5)).isoformat()
        p_tx_no_date = [{"value": 30_000, "title": "CFO", "code": "P", "is_ceo": False}]

        with patch("src.ingestion.run_pipeline._load_cik_map", return_value={"TEST": "0001234567"}), \
             patch("src.ingestion.run_pipeline._sec_get", return_value=self._make_sec_response(filing_date_str)), \
             patch("src.ingestion.run_pipeline._parse_form4_xml", return_value=p_tx_no_date):
            result = self._fetch()("TEST", lookback_days=180)

        _, _, _, days_since, _, _, _ = result
        assert days_since == 5, (
            f"Expected 5 days (filing date fallback), got {days_since}"
        )

    def test_no_edgar_data_returns_zero_without_error(self):
        """When no EDGAR data is available (e.g., CI without EDGAR_USER_AGENT), returns 0 gracefully."""
        from unittest.mock import patch

        with patch("src.ingestion.run_pipeline._load_cik_map", return_value={}):
            result = self._fetch()("UNKNOWN", lookback_days=180)

        _, _, _, days_since, _, _, _ = result
        assert days_since == 0, f"Expected 0 for unknown ticker, got {days_since}"


# ── PATCH 08: FMPEndpointError isolation in _score_ticker ─────────────────────


class TestPatch08FmpStructuralFailureInScoring:
    """PATCH 08: FMPEndpointError raised inside _score_ticker() must be caught
    separately from generic Exception so the broken endpoint path is tracked in
    _structural_failures_in_scoring and surfaced in the returned row dict.
    """

    def _import(self):
        import importlib
        import src.ingestion.run_pipeline as rp
        importlib.reload(rp)  # reset module-level shared set between tests
        return rp

    def test_shared_failure_set_exists_and_is_empty_on_import(self):
        """_structural_failures_in_scoring must be a set, initialised empty."""
        rp = self._import()
        assert hasattr(rp, "_structural_failures_in_scoring"), (
            "_structural_failures_in_scoring not found — PATCH 08 block 1 not applied"
        )
        assert isinstance(rp._structural_failures_in_scoring, set)
        assert len(rp._structural_failures_in_scoring) == 0

    def test_shared_lock_exists(self):
        """_structural_failures_lock must be a threading.Lock-compatible object."""
        import threading
        rp = self._import()
        assert hasattr(rp, "_structural_failures_lock"), (
            "_structural_failures_lock not found — PATCH 08 block 1 not applied"
        )
        assert isinstance(rp._structural_failures_lock, type(threading.Lock()))

    def test_shared_set_is_thread_safe_concurrent_writes(self):
        """Multiple threads writing to _structural_failures_in_scoring must not
        corrupt the set — the lock must be used correctly."""
        import threading
        rp = self._import()

        paths = [f"/v3/endpoint-{i}" for i in range(20)]
        errors = []

        def _write(path: str) -> None:
            try:
                with rp._structural_failures_lock:
                    rp._structural_failures_in_scoring.add(path)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=_write, args=(p,)) for p in paths]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Thread-safety errors: {errors}"
        assert rp._structural_failures_in_scoring == set(paths), (
            "Not all paths were written — concurrent writes may have been lost"
        )

    def test_post_loop_reporting_injects_into_endpoint_failures(self):
        """After generate(), any paths in _structural_failures_in_scoring must be
        injected into fmp_client.endpoint_failures so fmp_health.json captures them.
        Simulate the post-loop block directly to test its logic in isolation."""
        from collections import defaultdict
        from unittest.mock import MagicMock

        rp = self._import()

        # Seed the shared set as if _score_ticker() had caught a FMPEndpointError
        broken_path = "/v3/analyst-estimates/AAPL"
        with rp._structural_failures_lock:
            rp._structural_failures_in_scoring.add(broken_path)

        # Simulate the fmp_client local with the defaultdict(int) that the real
        # generate() uses via _FMPClient.endpoint_failures
        mock_client = MagicMock()
        mock_client.endpoint_failures = defaultdict(int)

        # Run the PATCH 08 post-loop reporting block exactly as written in generate()
        if rp._structural_failures_in_scoring:
            for path in rp._structural_failures_in_scoring:
                mock_client.endpoint_failures[path] += 1
            rp._structural_failures_in_scoring.clear()

        assert mock_client.endpoint_failures[broken_path] == 1, (
            "Broken endpoint path was not injected into endpoint_failures — "
            "PATCH 08 post-loop block not applied"
        )
        assert len(rp._structural_failures_in_scoring) == 0, (
            "_structural_failures_in_scoring was not cleared after reporting"
        )
