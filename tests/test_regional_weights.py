# Path: tests/test_regional_weights.py
#
# pytest — validates the regional weight system end-to-end.
# Run: pytest tests/test_regional_weights.py -v

import pytest
from src.config.weights import (
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

def test_weights_us_sprint_v24():
    """US weights must reflect v2.4/v2.5 sprint allocation (transcript_tone, revenue_revision, inst_flow_13f activated)."""
    expected = {
        "insider_conviction": 0.30,
        "insider_breadth":    0.12,   # reduced 0.15→0.12 (donor for revenue_revision)
        "congress":           0.01,   # reduced 0.04→0.01 (donor for transcript_tone)
        "news_sentiment":     0.10,
        "news_buzz":          0.01,   # reduced 0.05→0.01 (donor for inst_flow_13f)
        "momentum_long":      0.15,
        "volume_attention":   0.01,   # reduced 0.03→0.01 (donor for transcript_tone)
        "analyst_consensus":  0.10,
        "quality_piotroski":  0.08,
        "transcript_tone":    0.05,   # Huang et al. 2018
        "revenue_revision":   0.03,   # Zacks 2003
        "inst_flow_13f":      0.04,   # NEW v2.5 — 13F QoQ delta; SIGNED
    }
    assert WEIGHTS_US == expected

def test_weights_global_redistribution_correct():
    """WEIGHTS_GLOBAL vs WEIGHTS_US deltas (v2.4-global)."""
    # Only compare factors present in BOTH weight sets
    common = {k for k in WEIGHTS_US if k in WEIGHTS_GLOBAL}
    delta = {k: WEIGHTS_GLOBAL[k] - WEIGHTS_US[k] for k in common}
    # congress: US 0.01 vs GLOBAL 0.00 (structurally absent outside US)
    assert delta["congress"]           == pytest.approx(-0.01)
    assert delta["insider_conviction"] == pytest.approx(-0.02)   # MAR Art.19 parity
    assert delta["insider_breadth"]    == pytest.approx(+0.02)   # GLOBAL 0.14 vs US 0.12
    assert delta["news_buzz"]          == pytest.approx(+0.01)   # GLOBAL 0.02 vs US 0.01 (v2.5: US donor for inst_flow_13f)
    assert delta["volume_attention"]   == pytest.approx(+0.03)   # GLOBAL 0.04 vs US 0.01
    assert delta["analyst_consensus"]  == pytest.approx(0.00)    # same in both (0.10)
    assert delta["momentum_long"]      == pytest.approx(+0.02)   # Rouwenhorst EU premium
    assert delta["quality_piotroski"]  == pytest.approx(-0.03)   # GLOBAL 0.05 < US 0.08
    assert delta["news_sentiment"]     == pytest.approx(+0.03)   # Tetlock 2007 global corpus
    # transcript_tone: US 0.05, GLOBAL 0.00 (FMP transcripts US-only)
    assert delta["transcript_tone"]    == pytest.approx(-0.05)
    # revenue_revision: both have it; GLOBAL 0.02 < US 0.03
    assert delta["revenue_revision"]   == pytest.approx(-0.01)
    # Factors present in GLOBAL only (not in common set)
    assert WEIGHTS_GLOBAL["analyst_revision"]    == pytest.approx(0.02)
    assert WEIGHTS_GLOBAL["price_target_upside"] == pytest.approx(0.03)


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
    assert WEIGHTS_US["congress"] == 0.01   # original not mutated

def test_get_weights_eu_returns_copy():
    from src.config.weights import WEIGHTS_EU
    w = get_weights("SAP.DE")
    w["insider_conviction"] = 999.0
    assert WEIGHTS_EU["insider_conviction"] == 0.08   # original not mutated (WEIGHTS_EU, not GLOBAL)

def test_get_weights_global_returns_copy():
    w = get_weights("SAP.DE")
    w["insider_conviction"] = 999.0
    assert WEIGHTS_GLOBAL["insider_conviction"] == 0.28   # WEIGHTS_GLOBAL itself not mutated


# ── Composite score computation ──────────────────────────────────────────────

def test_us_ticker_uses_us_weights():
    """US ticker with perfect scores should reach 1.0 (before piotroski gate)."""
    perfect = {k: 1.0 for k in WEIGHTS_US}
    score, meta = compute_composite_score("AAPL", perfect, piotroski_raw=9)
    assert score == pytest.approx(1.0, abs=1e-6)
    assert meta["weights_set"] == "US"
    assert meta["region"] == "US"
    assert meta["congress_masked"] is False

def test_eu_ticker_uses_eu_weights():
    """EU ticker with perfect scores should reach 1.0 (WEIGHTS_EU sums to 1)."""
    from src.config.weights import WEIGHTS_EU
    perfect = {k: 1.0 for k in WEIGHTS_EU}
    perfect["congress"] = 0.0   # EU: congress structurally absent — not contamination
    score, meta = compute_composite_score("SAP.DE", perfect, piotroski_raw=9)
    assert score == pytest.approx(1.0, abs=1e-6)
    assert meta["weights_set"] == "EU"
    assert meta["region"] == "EU"
    assert meta["congress_masked"] is False   # congress was already 0.0 — no contamination

def test_eu_ticker_congress_contamination_guard():
    """If upstream scorer accidentally passes congress_score > 0 for an EU ticker,
    the compositor must zero it and log a warning."""
    from src.config.weights import WEIGHTS_EU
    contaminated = {k: 0.5 for k in WEIGHTS_EU}
    contaminated["congress"] = 0.8   # upstream contamination
    score, meta = compute_composite_score("SAP.DE", contaminated, piotroski_raw=7)
    assert meta["congress_masked"] is True
    # Score must equal what we'd get with congress=0.0 (using WEIGHTS_EU)
    clean = dict(contaminated)
    clean["congress"] = 0.0
    expected_score = sum(
        WEIGHTS_EU[f] * clean.get(f, 0.0) for f in WEIGHTS_EU
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
    """An EU ticker must score higher with WEIGHTS_EU (proper redistribution)
    than it would with WEIGHTS_US where congress is present but absent for EU."""
    from src.config.weights import WEIGHTS_EU
    factors = {k: 0.6 for k in WEIGHTS_EU}
    factors["congress"] = 0.0   # absent

    # Score with WEIGHTS_EU (correct — congress weight redistributed to quality factors)
    score_eu, _ = compute_composite_score("SAP.DE", factors, piotroski_raw=None)

    # Score with WEIGHTS_US applied naively (old behaviour — penalised by congress=0)
    score_us_naive = sum(WEIGHTS_US[f] * factors.get(f, 0.0) for f in WEIGHTS_US)

    assert score_eu > score_us_naive, (
        f"WEIGHTS_EU ({score_eu:.4f}) should exceed "
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


# ── WEIGHTS_EU (Quality-Core Model) ─────────────────────────────────────────

class TestWeightsEU:
    def test_sums_to_one(self):
        from src.config.weights import WEIGHTS_EU
        assert abs(sum(WEIGHTS_EU.values()) - 1.0) < 1e-6

    def test_congress_absent(self):
        from src.config.weights import WEIGHTS_EU
        assert WEIGHTS_EU["congress"] == 0.0

    def test_transcript_tone_absent(self):
        from src.config.weights import WEIGHTS_EU
        assert WEIGHTS_EU["transcript_tone"] == 0.0

    def test_quality_pillar(self):
        from src.config.weights import WEIGHTS_EU
        # EU quality-core pillar: piotroski + analyst + PT + roic + fcf + pb >= 0.55
        pillar = (WEIGHTS_EU["quality_piotroski"] + WEIGHTS_EU["analyst_revision"]
                  + WEIGHTS_EU["price_target_upside"] + WEIGHTS_EU["roic_quality"]
                  + WEIGHTS_EU["fcf_yield"] + WEIGHTS_EU["pb_value_up"])
        assert pillar >= 0.55

    def test_fcf_yield_is_dominant_fundamental(self):
        from src.config.weights import WEIGHTS_EU
        # FCF yield is the top fundamental signal in EU (Damodaran) — must be >= 0.10
        assert WEIGHTS_EU["fcf_yield"] >= 0.10


# ── WEIGHTS_ASIA (Liquidity/Momentum Model) ──────────────────────────────────

class TestWeightsAsia:
    def test_sums_to_one(self):
        from src.config.weights import WEIGHTS_ASIA
        assert abs(sum(WEIGHTS_ASIA.values()) - 1.0) < 1e-6

    def test_congress_absent(self):
        from src.config.weights import WEIGHTS_ASIA
        assert WEIGHTS_ASIA["congress"] == 0.0

    def test_transcript_tone_absent(self):
        from src.config.weights import WEIGHTS_ASIA
        assert WEIGHTS_ASIA["transcript_tone"] == 0.0

    def test_momentum_pillar(self):
        from src.config.weights import WEIGHTS_ASIA
        # Asia momentum pillar: momentum_long + volume_attention + news_sentiment + news_buzz
        pillar = (WEIGHTS_ASIA["momentum_long"] + WEIGHTS_ASIA["volume_attention"]
                  + WEIGHTS_ASIA["news_sentiment"] + WEIGHTS_ASIA["news_buzz"])
        assert pillar >= 0.30

    def test_momentum_long_dominant(self):
        from src.config.weights import WEIGHTS_ASIA
        # momentum_long is the single largest momentum signal in WEIGHTS_ASIA
        assert WEIGHTS_ASIA["momentum_long"] >= 0.12


# ── 3-way get_weights() ──────────────────────────────────────────────────────

class TestGetWeightsThreeWay:
    def test_eu_ticker_returns_eu_weights(self):
        from src.config.weights import WEIGHTS_EU
        assert get_weights("SAP.DE") == dict(WEIGHTS_EU)

    def test_asia_ticker_returns_asia_weights(self):
        from src.config.weights import WEIGHTS_ASIA
        assert get_weights("9984.T") == dict(WEIGHTS_ASIA)

    def test_us_ticker_returns_us_weights(self):
        assert get_weights("AAPL") == dict(WEIGHTS_US)

    def test_eu_weights_differ_from_global(self):
        from src.config.weights import WEIGHTS_EU
        # Quality-core model must differ significantly from WEIGHTS_GLOBAL
        assert WEIGHTS_EU["quality_piotroski"] > WEIGHTS_GLOBAL["quality_piotroski"]

    def test_asia_weights_differ_from_global(self):
        from src.config.weights import WEIGHTS_ASIA
        # Asia v2.3 adds FCF/Amihud/PB/ROIC not in WEIGHTS_GLOBAL
        assert WEIGHTS_ASIA.get("fcf_yield", 0) > 0
        assert "fcf_yield" not in WEIGHTS_GLOBAL


# ── Soft failure ─────────────────────────────────────────────────────────────

def test_soft_failure_on_invalid_input():
    """compute_composite_score must never raise — returns (0.0, minimal_meta)."""
    score, meta = compute_composite_score(None, None)   # type: ignore
    assert score == 0.0
    assert "weights_set" in meta
