# Path: tests/test_regional_weights.py
#
# pytest — validates the regional weight system end-to-end.
# Run: pytest tests/test_regional_weights.py -v

import pytest
from regime_trader.config.weights import (
    WEIGHTS_US,
    WEIGHTS_GLOBAL,
    get_region,
    get_weights,
    is_congress_eligible,
)
from backend.market_intel._score_compositor import (
    compute_composite_score,
    _piotroski_gate_multiplier,
)


# ── Weight integrity ─────────────────────────────────────────────────────────

def test_weights_us_sum():
    assert abs(sum(WEIGHTS_US.values()) - 1.0) < 1e-6

def test_weights_global_sum():
    assert abs(sum(WEIGHTS_GLOBAL.values()) - 1.0) < 1e-6

def test_weights_global_congress_zero():
    assert WEIGHTS_GLOBAL["congress"] == 0.0

def test_weights_us_congress_nonzero():
    assert WEIGHTS_US["congress"] > 0.0

def test_weights_us_unchanged():
    """US weights must be identical to the pre-patch WEIGHTS dict."""
    expected = {
        "insider_conviction": 0.30,
        "insider_breadth":    0.15,
        "congress":           0.22,
        "news_sentiment":     0.10,
        "news_buzz":          0.05,
        "momentum_long":      0.15,
        "volume_attention":   0.03,
        "analyst_consensus":  0.00,
        "quality_piotroski":  0.00,
    }
    assert WEIGHTS_US == expected

def test_weights_global_redistribution_correct():
    """The 0.22 freed from congress must be redistributed correctly."""
    delta = {k: WEIGHTS_GLOBAL[k] - WEIGHTS_US[k] for k in WEIGHTS_US}
    assert delta["congress"] == pytest.approx(-0.22)
    assert delta["insider_conviction"] == pytest.approx(+0.08)
    assert delta["analyst_consensus"]  == pytest.approx(+0.07)
    assert delta["momentum_long"]      == pytest.approx(+0.04)
    assert delta["quality_piotroski"]  == pytest.approx(+0.03)
    # Everything else unchanged
    for factor in ("insider_breadth", "news_sentiment", "news_buzz", "volume_attention"):
        assert delta[factor] == pytest.approx(0.0)


# ── Region classifier ────────────────────────────────────────────────────────

@pytest.mark.parametrize("ticker,expected", [
    ("AAPL",     "US"),
    ("MSFT",     "US"),
    ("BRK.A",    "US"),   # dot but not an exchange suffix
    ("SAP.DE",   "EU"),
    ("AIR.PA",   "EU"),
    ("SHEL.L",   "EU"),
    ("ASML.AS",  "EU"),
    ("ENI.MI",   "EU"),
    ("7203.T",   "ASIA"),
    ("005930.KS","ASIA"),
    ("0700.HK",  "ASIA"),
    ("RELIANCE.NS","ASIA"),
    ("600519.SS","ASIA"),
    ("UNKNOWN.XY","US"),   # unrecognised suffix → US default
])
def test_get_region(ticker, expected):
    assert get_region(ticker) == expected

def test_congress_eligible_us():
    assert is_congress_eligible("AAPL") is True

def test_congress_not_eligible_eu():
    assert is_congress_eligible("SAP.DE") is False

def test_congress_not_eligible_asia():
    assert is_congress_eligible("7203.T") is False

def test_get_weights_us_returns_copy():
    w = get_weights("AAPL")
    w["congress"] = 999.0
    assert WEIGHTS_US["congress"] == 0.22   # original not mutated

def test_get_weights_global_returns_copy():
    w = get_weights("SAP.DE")
    w["insider_conviction"] = 999.0
    assert WEIGHTS_GLOBAL["insider_conviction"] == 0.38   # original not mutated


# ── Composite score computation ──────────────────────────────────────────────

def test_us_ticker_uses_us_weights():
    """US ticker with perfect scores should reach 1.0 (before piotroski gate)."""
    perfect = {k: 1.0 for k in WEIGHTS_US}
    score, meta = compute_composite_score("AAPL", perfect, piotroski_raw=9)
    assert score == pytest.approx(1.0, abs=1e-6)
    assert meta["weights_set"] == "US"
    assert meta["region"] == "US"
    assert meta["congress_masked"] is False

def test_eu_ticker_uses_global_weights():
    """EU ticker with perfect scores should reach 1.0 (WEIGHTS_GLOBAL sums to 1)."""
    perfect = {k: 1.0 for k in WEIGHTS_GLOBAL}
    perfect["congress"] = 0.0   # EU: congress structurally absent — not contamination
    score, meta = compute_composite_score("SAP.DE", perfect, piotroski_raw=9)
    assert score == pytest.approx(1.0, abs=1e-6)
    assert meta["weights_set"] == "GLOBAL"
    assert meta["region"] == "EU"
    assert meta["congress_masked"] is False   # congress was already 0.0 — no contamination

def test_eu_ticker_congress_contamination_guard():
    """If upstream scorer accidentally passes congress_score > 0 for an EU ticker,
    the compositor must zero it and log a warning."""
    contaminated = {k: 0.5 for k in WEIGHTS_GLOBAL}
    contaminated["congress"] = 0.8   # upstream contamination
    score, meta = compute_composite_score("SAP.DE", contaminated, piotroski_raw=7)
    assert meta["congress_masked"] is True
    # Score must equal what we'd get with congress=0.0
    clean = dict(contaminated)
    clean["congress"] = 0.0
    expected_score = sum(
        WEIGHTS_GLOBAL[f] * clean.get(f, 0.0) for f in WEIGHTS_GLOBAL
    ) * 1.0   # piotroski_raw=7 → gate=1.0
    assert score == pytest.approx(expected_score, abs=1e-6)

def test_us_score_unaffected_by_patch():
    """US scoring must be mathematically identical before and after patch."""
    factors = {
        "insider_conviction": 0.80,
        "insider_breadth":    0.60,
        "congress":           0.70,
        "news_sentiment":     0.50,
        "news_buzz":          0.40,
        "momentum_long":      0.65,
        "volume_attention":   0.30,
        "analyst_consensus":  0.00,
        "quality_piotroski":  0.00,
    }
    score, meta = compute_composite_score("AAPL", factors, piotroski_raw=None)
    expected = sum(WEIGHTS_US[f] * factors.get(f, 0.0) for f in WEIGHTS_US)
    assert score == pytest.approx(expected, abs=1e-6)

def test_eu_score_higher_than_penalty_score():
    """An EU ticker must score higher with WEIGHTS_GLOBAL (proper redistribution)
    than it would with WEIGHTS_US where congress=0.22 is dead weight."""
    factors = {k: 0.6 for k in WEIGHTS_US}
    factors["congress"] = 0.0   # absent

    # Score with WEIGHTS_GLOBAL (correct)
    score_global, _ = compute_composite_score("SAP.DE", factors, piotroski_raw=None)

    # Score with WEIGHTS_US applied naively (old behaviour — penalised)
    score_us_naive = sum(WEIGHTS_US[f] * factors.get(f, 0.0) for f in WEIGHTS_US)

    assert score_global > score_us_naive, (
        f"WEIGHTS_GLOBAL ({score_global:.4f}) should exceed "
        f"naive US score ({score_us_naive:.4f}) for EU ticker with congress=0"
    )


# ── Piotroski gate ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected_mult", [
    (None, 1.0),
    (0,    0.0),
    (2,    0.0),
    (3,    0.6),
    (5,    0.6),
    (6,    1.0),
    (9,    1.0),
])
def test_piotroski_gate(raw, expected_mult):
    assert _piotroski_gate_multiplier(raw) == expected_mult

def test_piotroski_gate_suppresses_eu_buy():
    """Low F-Score EU ticker must have BUY suppressed regardless of other signals."""
    strong_factors = {k: 0.9 for k in WEIGHTS_GLOBAL}
    strong_factors["congress"] = 0.0
    score, meta = compute_composite_score("SAP.DE", strong_factors, piotroski_raw=1)
    assert score == pytest.approx(0.0)
    assert meta["piotroski_gate"] == 0.0


# ── Soft failure ─────────────────────────────────────────────────────────────

def test_soft_failure_on_invalid_input():
    """compute_composite_score must never raise — returns (0.0, minimal_meta)."""
    score, meta = compute_composite_score(None, None)   # type: ignore
    assert score == 0.0
    assert "weights_set" in meta
