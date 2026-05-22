import pytest
import re


def _make_entry(final_score: float, source_reliability: float) -> dict:
    return {"final_score": final_score, "source_reliability": source_reliability,
            "ticker": "TEST", "factors": {}}


def _apply_dampening(entries: list[dict]) -> list[dict]:
    """Mirror of the dampening block in generate_top_lists.py."""
    for e in entries:
        rel = float(e.get("source_reliability", 1.0))
        e["final_score"] = round(e["final_score"] * rel, 4)
    return entries


def test_dampening_reduces_low_reliability():
    entries = [_make_entry(0.8, 0.6)]
    result = _apply_dampening(entries)
    assert result[0]["final_score"] == pytest.approx(0.48, abs=1e-4)


def test_dampening_full_reliability_unchanged():
    entries = [_make_entry(0.8, 1.0)]
    result = _apply_dampening(entries)
    assert result[0]["final_score"] == pytest.approx(0.8, abs=1e-4)


def test_dampening_missing_field_defaults_to_1():
    entries = [{"final_score": 0.5, "ticker": "X", "factors": {}}]
    result = _apply_dampening(entries)
    assert result[0]["final_score"] == pytest.approx(0.5, abs=1e-4)


def test_dampening_zero_reliability_zeroes_score():
    entries = [_make_entry(0.9, 0.0)]
    result = _apply_dampening(entries)
    assert result[0]["final_score"] == pytest.approx(0.0, abs=1e-4)


# ── Validator regex tests ──────────────────────────────────────────────────────

_TICKER_RE_NEW = re.compile(r"^[A-Z0-9]{1,6}(\.[A-Z]{1,2})?$")
_TICKER_RE_OLD = re.compile(r"^[A-Z]{1,5}$")


def test_new_regex_accepts_us_ticker():
    assert _TICKER_RE_NEW.match("AAPL")
    assert _TICKER_RE_NEW.match("PLTR")


def test_new_regex_accepts_eu_ticker():
    assert _TICKER_RE_NEW.match("SAP.DE")
    assert _TICKER_RE_NEW.match("ASML.AS")


def test_new_regex_accepts_asia_ticker():
    assert _TICKER_RE_NEW.match("7203.T")
    assert _TICKER_RE_NEW.match("9984.T")


def test_new_regex_rejects_invalid():
    assert not _TICKER_RE_NEW.match("TOOLONG7")
    assert not _TICKER_RE_NEW.match("SAP.DEU")
    assert not _TICKER_RE_NEW.match("sa.de")


def test_old_regex_would_reject_new_formats():
    assert not _TICKER_RE_OLD.match("SAP.DE")
    assert not _TICKER_RE_OLD.match("7203.T")
