"""monitoring/check_metrics.py — threshold gate for the canary.

Reads `<log-dir>/metrics.json` and exits:
    0 — all thresholds passed
    2 — ALERT (errors > 0  OR  edgar_count / ticker_count < min_coverage
               OR fmp_health.json reports has_structural_failure=true)

On a failed gate, posts to Discord via DISCORD_WEBHOOK_URL when set.

Defaults match the canary spec:
    --min-coverage  0.6   (≥60% of tickers must come back from EDGAR)
    --max-errors    0     (any error fails the gate)

Usage:
    python -m monitoring.check_metrics
    python -m monitoring.check_metrics --log-dir logs --min-coverage 0.5
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

import statistics

from .alert_state import update_after_evaluation
from .evaluate import evaluate
from .slack_notifier import send_discord_alert as send_slack_alert

log = logging.getLogger("monitoring.check_metrics")


def check_score_distribution(
    log_dir: Path,
    min_stdev: float = 0.05,
    min_max_score: float = 0.40,
    min_entries: int = 3,
) -> bool:
    """Assert that score distributions are non-degenerate.

    Reads intel_source_status.json when available (full pipeline output with
    per-factor data), otherwise falls back to top_lists.json (canary artifact).
    Missing or unreadable files are treated as skips (return True) so the
    canary does not fail on a missing artifact.

    When intel_source_status.json is present, applies 4 strict gates:
      1. Population: >= 10 US results.
      2. Density: >= 20% of US tickers have final_score > 0.
      3. Variance: stdev of final_scores > min_stdev.
      4. Per-factor liveness: >= 1 factor has > 5% non-zero density.

    When only top_lists.json is available, applies the original 2 gates:
      1. Variance: stdev < min_stdev → degenerate.
      2. Max score: max < min_max_score → dead signal.

    Returns True when healthy or when no data is available to check.
    Returns False when a degenerate distribution is detected.
    Never raises.
    """
    # ── Try intel_source_status.json first (full pipeline output) ─────────────
    status_path = Path(log_dir) / "intel_source_status.json"
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("check_score_distribution: cannot parse intel_source_status.json — %s", exc)
        else:
            results = [
                r for r in status.get("results", [])
                if r.get("market", "USA") in ("USA", "US")
            ]
            if len(results) >= 10:
                return _check_status_distribution(results, log)
            # < 10 results: fall through to top_lists.json check
            log.debug(
                "check_score_distribution: intel_source_status.json has only %d US rows — "
                "falling back to top_lists.json",
                len(results),
            )

    # ── Fallback: top_lists.json (canary artifact) ─────────────────────────────
    tl_path = Path(log_dir) / "top_lists.json"
    if not tl_path.exists():
        log.warning("check_score_distribution: no artifact found — skipping")
        return True

    try:
        data = json.loads(tl_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("check_score_distribution: cannot parse top_lists.json — %s", exc)
        return True

    _BUCKETS = (
        "top_buys", "top_buys_usa", "top_buys_europe",
        "top_buys_asia", "mid_caps", "small_caps",
    )
    scores: list[float] = []
    for bucket in _BUCKETS:
        for entry in data.get(bucket, []):
            s = entry.get("final_score")
            if s is not None:
                scores.append(float(s))

    scores = list(set(scores))  # deduplicate same ticker across lists

    if len(scores) < min_entries:
        log.warning(
            "check_score_distribution: only %d unique scores (need >= %d) — skipping",
            len(scores), min_entries,
        )
        return True

    stdev = statistics.stdev(scores) if len(scores) > 1 else 0.0
    max_score = max(scores)
    mean_score = statistics.mean(scores)
    degenerate = False

    if stdev < min_stdev:
        log.error(
            "::error::SCORE DISTRIBUTION DEGENERATE: stdev=%.4f < %.4f threshold. "
            "All tickers scoring near %.4f — data feeds may be dead.",
            stdev, min_stdev, mean_score,
        )
        degenerate = True

    if max_score < min_max_score:
        log.error(
            "::error::DEAD SIGNAL DETECTED: max_score=%.4f < %.4f threshold. "
            "No ticker reached the minimum actionable score.",
            max_score, min_max_score,
        )
        degenerate = True

    if not degenerate:
        log.info(
            "check_score_distribution: OK — stdev=%.4f max=%.4f mean=%.4f (%d entries)",
            stdev, max_score, mean_score, len(scores),
        )

    return not degenerate


def _check_status_distribution(results: list, log) -> bool:
    """4-gate check against intel_source_status.json results (US rows only)."""
    final_scores = [float(r.get("final_score", 0.0) or 0.0) for r in results]

    # Gate 2: non-zero density
    nonzero_count = sum(1 for s in final_scores if s > 0.0)
    nonzero_density = nonzero_count / len(final_scores)
    if nonzero_density < 0.20:
        log.error(
            "check_score_distribution: DEGENERATE — only %d/%d US tickers "
            "have final_score > 0 (density=%.1f%% < 20%%). "
            "Dead signal feeds likely. Check fmp_health.json.",
            nonzero_count, len(final_scores), nonzero_density * 100,
        )
        return False

    # Gate 3: variance
    std_dev = statistics.stdev(final_scores) if len(final_scores) > 1 else 0.0
    if std_dev < 0.05:
        log.error(
            "check_score_distribution: DEGENERATE — std_dev=%.4f < 0.05. "
            "All tickers scoring nearly identically.",
            std_dev,
        )
        return False

    # Gate 4: per-factor density
    _FACTOR_SCORE_KEYS = [
        "insider_conviction_score", "insider_breadth_score",
        "congress_score", "news_sentiment_score", "momentum_long_score",
    ]
    live_factors = []
    for key in _FACTOR_SCORE_KEYS:
        nz = sum(1 for r in results if float(r.get(key, 0.0) or 0.0) > 0.0)
        density = nz / len(results)
        if density > 0.05:
            live_factors.append((key, density))

    if not live_factors:
        log.error(
            "check_score_distribution: ALL core factor feeds dead "
            "(<5%% non-zero density) across %d US tickers.",
            len(results),
        )
        return False

    log.info(
        "check_score_distribution PASSED: n=%d, nonzero=%.1f%%, std=%.4f, live_factors=%d",
        len(results), nonzero_density * 100, std_dev, len(live_factors),
    )
    _check_insider_feed_density(results, log)
    return True


def _check_insider_feed_density(results: list, log) -> bool:
    """Warn when insider feed appears silently dead (HTTP 200 but all-zeros).

    Distinguishes between:
      - Structurally sparse (e.g. 5–15% non-zero on S&P 500): EXPECTED, no warn
      - Apparently dead (< 1% non-zero across 10+ US tickers): WARN — possible
        HTTP 200 empty-array response from FMP insider endpoint

    Does NOT hard-fail — sparsity is a known property of the insider signal
    on large-cap universes (Cohen, Malloy & Pomorski 2012: ~11% of S&P 500
    tickers have key-officer purchases in any 90-day window).

    Returns True (always) — this is a monitoring signal, not a gate.
    """
    us_results = [r for r in results if r.get("market", "USA") in ("USA", "US")]
    if len(us_results) < 10:
        return True

    conviction_nonzero = sum(
        1 for r in us_results
        if float(r.get("insider_conviction_score", 0.0) or 0.0) > 0.0
    )
    breadth_nonzero = sum(
        1 for r in us_results
        if float(r.get("insider_breadth_score", 0.0) or 0.0) > 0.0
    )
    n = len(us_results)

    conviction_density = conviction_nonzero / n
    breadth_density = breadth_nonzero / n

    if conviction_density < 0.01 and breadth_density < 0.01:
        log.warning(
            "INSIDER FEED LIKELY DEAD (HTTP 200 empty-array): "
            "conviction_density=%.1f%% breadth_density=%.1f%% across %d US tickers. "
            "FMP insider-trading/search may be returning [] for all tickers. "
            "Check fmp_health.json for endpoint call counts vs failures.",
            conviction_density * 100,
            breadth_density * 100,
            n,
        )
    elif conviction_density < 0.03:
        log.warning(
            "Insider conviction density unusually low: %.1f%% across %d US tickers "
            "(expected 5–15%% on S&P 500 universe). "
            "Possible feed degradation — check fmp_health.json.",
            conviction_density * 100,
            n,
        )
    else:
        log.info(
            "Insider feed density OK: conviction=%.1f%% breadth=%.1f%% "
            "across %d US tickers.",
            conviction_density * 100,
            breadth_density * 100,
            n,
        )

    return True


# Factors that must never be all-zeros when the canary universe has price history.
# momentum_long and volume_attention are sourced from FMP historical-price-eod/full
# which is always available for any listed security — an all-zero result is a
# definitive dead-feed indicator, not sparsity.
_ALWAYS_NONZERO_FACTORS: dict[str, str] = {
    "momentum_long_score":      "momentum_long",
    "volume_attention_score":   "volume_attention",
    "analyst_consensus_score":  "analyst_consensus",  # bulk snapshot always fetched
}


def check_per_factor_distribution(log_dir: "Path") -> bool:
    """Hard gate: factors that must never be all-zeros for any canary run.

    momentum_long and volume_attention are computed from price/volume history,
    which is structurally always available for listed securities.  If either
    is all-zeros across the entire top_buys list the price data feed is dead.

    Returns False and logs ERROR if a mandatory-nonzero factor is all-zeros.
    Returns True (pass) otherwise, including when top_lists.json is absent.
    """
    import logging as _log  # noqa: PLC0415
    _logger = _log.getLogger(__name__)

    tl = log_dir / "top_lists.json"
    if not tl.exists():
        return True  # can't check without artifact

    try:
        d = json.loads(tl.read_text(encoding="utf-8"))
    except Exception as exc:
        _logger.warning("check_per_factor_distribution: could not parse top_lists.json: %s", exc)
        return True

    buys = d.get("top_buys", [])
    if not buys:
        return True

    ok = True
    for score_key, factor_name in _ALWAYS_NONZERO_FACTORS.items():
        scores = [float(t.get("factors", {}).get(factor_name, 0.0) or 0.0) for t in buys]
        if all(s == 0.0 for s in scores):
            _logger.error(
                "DEAD SIGNAL: factor %r is 0.0 for ALL %d top_buys tickers. "
                "Price data feed (FMP historical-price-eod/full) may be dead.",
                factor_name, len(scores),
            )
            ok = False

    return ok


def check_fmp_telemetry(
    metrics: dict,
    *,
    max_error_rate: float = 0.05,
    min_cache_hit_rate: float = 0.50,
    min_cache_lookups: int = 50,
) -> List[str]:
    """Advisory soft signals from the WS4 FMP telemetry rollup.

    Returns human-readable warning strings (empty when healthy or when no
    telemetry block is present). These are advisory ONLY — they never fail the
    canary; a genuinely dead route is already a hard gate via fmp_health.json.

    - error_rate is evaluated only once there are real calls.
    - cache_hit_rate is evaluated only once there is meaningful cache traffic
      (``cache_lookups >= min_cache_lookups``), so a cold-cache or tiny
      on-demand run does not trip a false "0% hit rate" warning.
    """
    warnings: List[str] = []
    calls = int(metrics.get("calls_per_run", 0) or 0)
    if calls <= 0:
        return warnings  # older artifact / no telemetry — nothing to say

    err = float(metrics.get("error_rate", 0.0) or 0.0)
    if err > max_error_rate:
        warnings.append(
            f"error_rate {err:.1%} > {max_error_rate:.0%} threshold over {calls} calls"
        )

    lookups = int(metrics.get("cache_lookups", 0) or 0)
    hit = float(metrics.get("cache_hit_rate", 0.0) or 0.0)
    if lookups >= min_cache_lookups and hit < min_cache_hit_rate:
        warnings.append(
            f"cache_hit_rate {hit:.1%} < {min_cache_hit_rate:.0%} over {lookups} "
            f"lookups — more live calls than expected (stale snapshots?)"
        )
    return warnings


def _format_alert_body(metrics: dict, reasons: List[str]) -> str:
    lines = ["🚨 EDGAR canary failed:"]
    lines.extend(f"  • {r}" for r in reasons)
    lines.append("")
    lines.append("Metrics snapshot:")
    lines.append("```")
    lines.append(json.dumps(metrics, indent=2))
    lines.append("```")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Canary threshold check")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"),
                        help="Directory containing metrics.json (default: logs)")
    parser.add_argument("--min-coverage", type=float, default=0.60,
                        help="Minimum EDGAR coverage ratio (default: 0.60)")
    parser.add_argument("--max-errors", type=int, default=0,
                        help="Maximum tolerated error_count (default: 0)")
    parser.add_argument("--webhook", type=str, default=None,
                        help="Discord webhook URL (defaults to env DISCORD_WEBHOOK_URL)")
    parser.add_argument("--no-slack", action="store_true",
                        help="Do not send Discord alert even if a webhook is set")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    metrics_path = args.log_dir / "metrics.json"
    if not metrics_path.exists():
        log.error("metrics.json not found at %s", metrics_path)
        return 2

    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.error("could not parse metrics.json: %s", exc)
        return 2

    if metrics.get("pipeline_failed"):
        log.error("ALERT — pipeline aborted before intel_source_status.json was written")
        return 2

    ok, reasons = evaluate(metrics, min_coverage=args.min_coverage, max_errors=args.max_errors)

    # ── FMP structural-failure gate ────────────────────────────────────────────
    # A non-zero failure count means a factor feed is dead — the pipeline is
    # producing zeroed scores, not sparse ones. This must fail the canary so it
    # surfaces loudly rather than being masked by a lowered circuit-breaker.
    fmp_health_path = args.log_dir / "fmp_health.json"
    if fmp_health_path.exists():
        try:
            fmp_health = json.loads(fmp_health_path.read_text(encoding="utf-8"))
            quarantined = set(fmp_health.get("quarantined_endpoints", []))
            quarantined |= set(fmp_health.get("runtime_quarantined", []))
            if quarantined:
                log.info("FMP quarantined endpoints (expected, no alarm): %s", sorted(quarantined))
            if fmp_health.get("has_structural_failure"):
                # Only alarm on failures that are NOT in the quarantine list.
                failed_routes = {
                    k: v for k, v in fmp_health.get("failures", {}).items()
                    if k not in quarantined and v > 0
                }
                if failed_routes:
                    reason = f"FMP structural failure on route(s): {failed_routes}"
                    reasons.append(reason)
                    ok = False
                    log.error("FMP structural failure detected: %s", failed_routes)
                else:
                    log.info(
                        "FMP has_structural_failure=true but all failures are quarantined — no alarm"
                    )
        except Exception as exc:
            log.warning("could not read fmp_health.json: %s", exc)

    # ── Score distribution gate (PATCH 11) ────────────────────────────────────
    # Catches dead-feed scenarios where all factors return 0.0 and the
    # cross-sectional normaliser collapses every final_score to ~0.18.
    if not check_score_distribution(args.log_dir):
        reasons.append(
            "Score distribution degenerate: stdev < 0.05 or max_score < 0.40 "
            "— insider/news/congress feeds may all be returning 0.0"
        )
        ok = False

    # ── FMP telemetry soft signals (WS4) — advisory, never fail the canary ────
    for _w in check_fmp_telemetry(metrics):
        log.warning("FMP telemetry: %s", _w)

    decision = update_after_evaluation(ok)

    if ok:
        log.info("OK — coverage %d/%d, errors=%d",
                 metrics.get("edgar_count", 0), metrics.get("ticker_count", 0),
                 metrics.get("error_count", 0))
        return 0

    log.error("ALERT — %s (consecutive_failures=%d, escalate=%s)",
              "; ".join(reasons), decision.consecutive_failures, decision.escalate)
    if not args.no_slack:
        webhook: Optional[str] = args.webhook or os.getenv("DISCORD_WEBHOOK_URL")
        if webhook:
            sent = send_slack_alert(
                webhook=webhook,
                title="EDGAR Canary FAILED",
                body=_format_alert_body(metrics, reasons),
                escalate=decision.escalate,
            )
            log.info("discord alert sent=%s", sent)
        else:
            log.warning("DISCORD_WEBHOOK_URL not set — skipping notification")
    return 2


if __name__ == "__main__":
    sys.exit(main())
