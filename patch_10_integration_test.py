"""patch_10_integration_test.py
Integration tests for PATCH-01 through PATCH-09.
Run after all patches are applied:
    python patch_10_integration_test.py
"""
import sys
import json
import math
import inspect
from pathlib import Path
from datetime import datetime, timezone

FAILURES = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  PASS  {name}")
    else:
        msg = f"  FAIL  {name}" + (f": {detail}" if detail else "")
        print(msg)
        FAILURES.append(msg)


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────────────────────
# Section 1: WEIGHTS integrity (PATCH-08)
# ─────────────────────────────────────────────────────────────────────────────
section("1. WEIGHTS integrity (PATCH-08)")
try:
    from scripts.run_pipeline import WEIGHTS as RP_WEIGHTS
    total = sum(RP_WEIGHTS.values())
    check("run_pipeline.WEIGHTS sums to 1.0",
          abs(total - 1.0) < 1e-6, f"got {total:.8f}")
    check("momentum_long >= 0.24",
          RP_WEIGHTS.get("momentum_long", 0) >= 0.24,
          f"got {RP_WEIGHTS.get('momentum_long')}")
    check("quality_piotroski >= 0.07",
          RP_WEIGHTS.get("quality_piotroski", 0) >= 0.07,
          f"got {RP_WEIGHTS.get('quality_piotroski')}")
    check("insider_conviction <= 0.16",
          RP_WEIGHTS.get("insider_conviction", 1) <= 0.16,
          f"got {RP_WEIGHTS.get('insider_conviction')}")
    check("all 12 factors present",
          len(RP_WEIGHTS) == 12, f"got {len(RP_WEIGHTS)}")
except Exception as e:
    check("run_pipeline WEIGHTS import", False, str(e))

try:
    from backend.market_intel.generate_top_lists import WEIGHTS as GTL_WEIGHTS
    total_gtl = sum(GTL_WEIGHTS.values())
    check("generate_top_lists.WEIGHTS sums to 1.0",
          abs(total_gtl - 1.0) < 1e-6, f"got {total_gtl:.8f}")
    for k in RP_WEIGHTS:
        check(f"generate_top_lists.WEIGHTS['{k}'] matches run_pipeline",
              abs(GTL_WEIGHTS.get(k, -1) - RP_WEIGHTS[k]) < 1e-6,
              f"gtl={GTL_WEIGHTS.get(k)} rp={RP_WEIGHTS[k]}")
except Exception as e:
    check("generate_top_lists WEIGHTS import", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Function signatures (PATCH-02)
# ─────────────────────────────────────────────────────────────────────────────
section("2. Shared FMP client threading (PATCH-02)")
try:
    from scripts.run_pipeline import score_analyst_consensus, score_transcript_tone
    sig1 = inspect.signature(score_analyst_consensus)
    sig2 = inspect.signature(score_transcript_tone)
    check("score_analyst_consensus has 'client' param",
          "client" in sig1.parameters)
    check("score_transcript_tone has 'client' param",
          "client" in sig2.parameters)
    p1 = sig1.parameters.get("client")
    p2 = sig2.parameters.get("client")
    check("score_analyst_consensus client defaults to None",
          p1 is not None and p1.default is None)
    check("score_transcript_tone client defaults to None",
          p2 is not None and p2.default is None)
except Exception as e:
    check("scoring function signatures", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Momentum regime constants (PATCH-03)
# ─────────────────────────────────────────────────────────────────────────────
section("3. Momentum regime propagation (PATCH-03)")
try:
    from backend.market_intel.generate_top_lists import _MOMENTUM_REGIME_MULTIPLIERS, _to_entry
    check("_MOMENTUM_REGIME_MULTIPLIERS exists",
          isinstance(_MOMENTUM_REGIME_MULTIPLIERS, dict))
    check("BEAR_CRASH multiplier = 0.30",
          _MOMENTUM_REGIME_MULTIPLIERS.get("BEAR_CRASH") == 0.30)
    check("BEAR_MOMENTUM multiplier = 0.55",
          _MOMENTUM_REGIME_MULTIPLIERS.get("BEAR_MOMENTUM") == 0.55)
    check("NORMAL multiplier = 1.00",
          _MOMENTUM_REGIME_MULTIPLIERS.get("NORMAL") == 1.00)
    sig = inspect.signature(_to_entry)
    check("_to_entry has momentum_multiplier param",
          "momentum_multiplier" in sig.parameters)
    mp = sig.parameters.get("momentum_multiplier")
    check("momentum_multiplier defaults to 1.0",
          mp is not None and mp.default == 1.0)
except Exception as e:
    check("momentum regime constants", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Section 4: Piotroski 9-point score (PATCH-09)
# ─────────────────────────────────────────────────────────────────────────────
section("4. Piotroski 9-point implementation (PATCH-09)")
try:
    from regime_trader.scoring.momentum_signals import score_quality_piotroski

    # High quality: MSFT-like fundamentals
    high_q = {
        "returnOnAssetsTTM": 0.18,
        "operatingCashFlowPerShareTTM": 8.5,
        "debtEquityRatioTTM": 0.35,
        "currentRatioTTM": 2.1,
        "grossProfitMarginTTM": 0.69,
        "netProfitMarginTTM": 0.35,
        "operatingProfitMarginTTM": 0.42,
    }
    score_hq = score_quality_piotroski(high_q)
    check("high-quality MSFT-like scores >= 0.75",
          score_hq >= 0.75, f"got {score_hq}")

    # Distressed: negative ROA, negative D/E (negative equity)
    distressed = {
        "returnOnAssetsTTM": -0.05,
        "operatingCashFlowPerShareTTM": -0.2,
        "debtEquityRatioTTM": -0.3,
        "currentRatioTTM": 0.8,
        "grossProfitMarginTTM": 0.15,
        "netProfitMarginTTM": -0.08,
        "operatingProfitMarginTTM": -0.05,
    }
    score_d = score_quality_piotroski(distressed)
    check("distressed company scores <= 0.25",
          score_d <= 0.25, f"got {score_d}")

    # Spec verification: ratios from patch spec
    spec_ratios = {
        "returnOnAssetsTTM": 0.18,
        "debtEquityRatioTTM": 0.3,
        "currentRatioTTM": 2.0,
        "grossProfitMarginTTM": 0.65,
        "netProfitMarginTTM": 0.30,
        "operatingProfitMarginTTM": 0.40,
    }
    spec_score = score_quality_piotroski(spec_ratios)
    check("spec verification: score >= 0.66",
          spec_score >= 0.66, f"got {spec_score}")

    check("None input returns 0.0",
          score_quality_piotroski(None) == 0.0)
    check("empty dict returns 0.0",
          score_quality_piotroski({}) == 0.0)
    check("score is in [0, 1]",
          0.0 <= score_hq <= 1.0 and 0.0 <= score_d <= 1.0)
    check("score uses 9-point denominator (not 8)",
          abs(score_hq - round(score_hq * 9) / 9.0) < 1e-6)
except Exception as e:
    check("Piotroski 9-point", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Section 5: check_score_distribution (PATCH-01 + PATCH-07)
# ─────────────────────────────────────────────────────────────────────────────
section("5. check_score_distribution substance (PATCH-01 + PATCH-07)")
try:
    import tempfile
    from monitoring.check_metrics import check_score_distribution, _check_insider_feed_density

    # Test: missing file → False
    with tempfile.TemporaryDirectory() as tmpdir:
        result = check_score_distribution(Path(tmpdir))
        check("missing intel_source_status.json returns False", result is False)

    # Test: fewer than 10 US results → False
    with tempfile.TemporaryDirectory() as tmpdir:
        fake = {"results": [
            {"market": "USA", "final_score": 0.5, "momentum_long_score": 0.6}
            for _ in range(5)
        ]}
        p = Path(tmpdir) / "intel_source_status.json"
        p.write_text(json.dumps(fake))
        result = check_score_distribution(Path(tmpdir))
        check("< 10 US results returns False", result is False)

    # Test: all-zero scores → False
    with tempfile.TemporaryDirectory() as tmpdir:
        fake = {"results": [
            {"market": "USA", "final_score": 0.0,
             "insider_conviction_score": 0.0, "momentum_long_score": 0.0}
            for _ in range(15)
        ]}
        p = Path(tmpdir) / "intel_source_status.json"
        p.write_text(json.dumps(fake))
        result = check_score_distribution(Path(tmpdir))
        check("all-zero scores returns False", result is False)

    # Test: healthy distribution → True
    import random
    random.seed(42)
    with tempfile.TemporaryDirectory() as tmpdir:
        healthy = {"results": [
            {
                "market": "USA",
                "final_score": random.uniform(0.1, 0.8),
                "insider_conviction_score": random.uniform(0, 0.5) if random.random() > 0.88 else 0.0,
                "insider_breadth_score": random.uniform(0, 0.4) if random.random() > 0.85 else 0.0,
                "momentum_long_score": random.uniform(0.2, 0.8),
                "news_sentiment_score": random.uniform(0.3, 0.7),
                "congress_score": random.uniform(0, 0.6) if random.random() > 0.95 else 0.0,
            }
            for _ in range(40)
        ]}
        p = Path(tmpdir) / "intel_source_status.json"
        p.write_text(json.dumps(healthy))
        result = check_score_distribution(Path(tmpdir))
        check("healthy distribution returns True", result is True)

    # Test: _check_insider_feed_density always returns True
    import logging
    log = logging.getLogger("test")
    fake_results = [
        {"market": "USA", "insider_conviction_score": 0.0, "insider_breadth_score": 0.0}
        for _ in range(20)
    ]
    result = _check_insider_feed_density(fake_results, log)
    check("_check_insider_feed_density returns True always", result is True)

except Exception as e:
    check("check_score_distribution tests", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Section 6: audit_payload EU/Asia ceiling (PATCH-05)
# ─────────────────────────────────────────────────────────────────────────────
section("6. audit_payload EU/Asia ceiling (PATCH-05)")
try:
    from scripts.audit_payload import audit, InternationalScoreOverflowError, PipelineAuditError
    check("InternationalScoreOverflowError inherits PipelineAuditError",
          issubclass(InternationalScoreOverflowError, PipelineAuditError))

    # Valid EU score (0.55 — well within ceiling)
    valid_eu = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "vix": 17.5,
        "kill_switch": False,
        "top_buys": [],
        "top_buys_usa": [],
        "top_buys_europe": [{
            "ticker": "SAP.DE",
            "market": "EUROPE",
            "final_score": 0.55,
            "badge": "WATCHLIST",
            "factors": {"congress": 0.0, "momentum_long": 0.72},
        }],
        "top_buys_asia": [],
    }
    try:
        result = audit(valid_eu)
        check("valid EU score 0.55 passes audit", result is True)
    except Exception as e:
        check("valid EU score 0.55 passes audit", False, str(e))

    # Over-ceiling EU score (0.99 — impossible with only momentum+volume)
    overflow_eu = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "vix": 17.5,
        "kill_switch": False,
        "top_buys": [],
        "top_buys_usa": [],
        "top_buys_europe": [{
            "ticker": "SAP.DE",
            "market": "EUROPE",
            "final_score": 0.99,
            "badge": "HIGH BUY",
            "factors": {"congress": 0.0, "momentum_long": 1.0},
        }],
        "top_buys_asia": [],
    }
    try:
        audit(overflow_eu)
        check("overflow EU score raises InternationalScoreOverflowError", False,
              "Expected exception was not raised")
    except InternationalScoreOverflowError:
        check("overflow EU score raises InternationalScoreOverflowError", True)
    except Exception as e:
        check("overflow EU score raises correct exception", False, str(e))

except Exception as e:
    check("audit_payload EU ceiling tests", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Section 7: FMP client congress probe (PATCH-06)
# ─────────────────────────────────────────────────────────────────────────────
section("7. FMP client congress probe (PATCH-06)")
try:
    from regime_trader.services.fmp_client import FMPClient
    src = inspect.getsource(FMPClient.get_congress_trades)
    check("congress probe key in get_congress_trades",
          "_fmp_congress_probe_done" in src)
    check("senate-trading probe call in get_congress_trades",
          '"senate-trading"' in src)
except Exception as e:
    check("FMP congress probe", False, str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Final summary
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
if FAILURES:
    print(f"  RESULT: {len(FAILURES)} test(s) FAILED")
    for f in FAILURES:
        print(f"    {f}")
    sys.exit(1)
else:
    print("  RESULT: ALL INTEGRATION TESTS PASSED")
    sys.exit(0)
