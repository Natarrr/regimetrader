"""regime_trader.research.ic_backtest — IC backtest orchestrator.

Entry point: run_ic_backtest()

Produces an advisory Markdown report. Does NOT modify WEIGHTS automatically.
Weight changes are human decisions based on the report's recommendations.

References:
  López de Prado (2018) AFML ch. 7-8
  Grinold & Kahn (2000) Active Portfolio Management ch. 6
  Tetlock (2007), Barber & Odean (2008), Cohen et al. (2012)
"""
from __future__ import annotations

import logging
import math
import warnings
from datetime import date
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# v2 factor names in declaration order (matches WEIGHTS in run_pipeline.py)
_V2_FACTORS = [
    "insider_conviction_score",
    "insider_breadth_score",
    "congress_score",
    "news_sentiment_score",
    "news_buzz_score",
    "momentum_long_score",
    "volume_attention_score",
]

# Schema migration warnings per factor (empty = no warning)
_SCHEMA_WARNINGS: dict[str, str] = {
    "news_sentiment_score": (
        "Mapped from v1 news_score (contains ~40% buzz contamination). "
        "IC estimates may be upward-biased for v1 snapshots."
    ),
    "insider_conviction_score": "MISSING in v1 — only v2 snapshots contribute.",
    "insider_breadth_score":    "MISSING in v1 — only v2 snapshots contribute.",
    "momentum_long_score":      "MISSING in v1 — 20d reversal ≠ Jegadeesh 12-1m.",
    "volume_attention_score":   "MISSING in v1 — no v1 counterpart.",
    "news_buzz_score":          "MISSING in v1 — no v1 counterpart.",
}


def run_ic_backtest(
    log_dir: Path,
    output_dir: Path,
    score_variant: Literal["raw", "neutralized", "both"] = "raw",
    horizon_days: int = 21,
    n_folds: int = 5,
    embargo_days: int = 5,
    min_snapshots: int = 60,
    cache_root: Path | None = None,
) -> Path:
    """Run IC backtest and write Markdown report to output_dir.

    Args:
        log_dir: Directory containing logs/historical/ subdirs.
        output_dir: Directory where report (and optional PNG) is written.
        score_variant: "raw" uses *_score fields; "neutralized" uses *_score_neutral.
        horizon_days: Forward return horizon in calendar days (~21 = 1 month).
        n_folds: Purged k-fold count.
        embargo_days: Embargo window in days around fold boundaries.
        min_snapshots: Abort if fewer qualifying snapshots.
        cache_root: Project root for forward-return cache.

    Returns:
        Path to the written Markdown report.

    Raises:
        RuntimeError: If < min_snapshots exist (actionable message).
        ValueError: If neutralized requested but *_score_neutral fields absent.
    """
    from .historical_loader import load_historical_snapshots, detect_schema_version
    from .forward_returns import fetch_forward_returns
    from .ic_metrics import purged_kfold_ic, ICResult

    historical_dir = log_dir / "historical"

    # ── Load snapshots ───────────────────────────────────────────────────────
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        snapshots = list(
            load_historical_snapshots(
                historical_dir,
                min_tickers_per_snapshot=80,
                min_snapshots=min_snapshots,
            )
        )

    schema_versions = {detect_schema_version(snapshots[0][1][0])} if snapshots else set()
    for _, rows in snapshots[1:]:
        if rows:
            schema_versions.add(detect_schema_version(rows[0]))

    v1_count = sum(
        1 for _, rows in snapshots
        if rows and detect_schema_version(rows[0]) == "v1_legacy_5factor"
    )
    v2_count = len(snapshots) - v1_count

    logger.info(
        "IC backtest: %d snapshots loaded (%d v1, %d v2)",
        len(snapshots), v1_count, v2_count,
    )

    # ── Validate score_variant ───────────────────────────────────────────────
    factors_to_run = _resolve_factors(snapshots, score_variant)

    # ── Collect all tickers across all snapshots ──────────────────────────────
    all_tickers: set[str] = set()
    for _, rows in snapshots:
        for row in rows:
            t = row.get("ticker", "")
            if t:
                all_tickers.add(t)

    # ── Fetch forward returns per snapshot date ───────────────────────────────
    forward_return_map: dict[date, dict[str, float]] = {}
    snap_dates = [d for d, _ in snapshots]

    logger.info("IC backtest: fetching forward returns for %d dates…", len(snap_dates))
    for snap_date in snap_dates:
        fwd = fetch_forward_returns(
            list(all_tickers),
            snap_date,
            horizon_days=horizon_days,
            cache_root=cache_root or log_dir.parent,
        )
        forward_return_map[snap_date] = fwd

    # ── Compute IC per factor ─────────────────────────────────────────────────
    ic_results: list[ICResult] = []
    for factor_name in factors_to_run:
        schema_warn = _SCHEMA_WARNINGS.get(factor_name, "")
        try:
            result = purged_kfold_ic(
                snapshots=snapshots,
                factor_name=factor_name,
                forward_return_map=forward_return_map,
                n_folds=n_folds,
                embargo_days=embargo_days,
                schema_warning=schema_warn,
            )
            ic_results.append(result)
        except Exception as exc:
            logger.error("IC backtest: factor %s failed: %s", factor_name, exc)

    # ── Generate report ───────────────────────────────────────────────────────
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"ic_backtest_{date.today().isoformat()}.md"

    _try_write_chart(ic_results, output_dir)

    report_md = _render_report(
        ic_results=ic_results,
        snapshots=snapshots,
        v1_count=v1_count,
        v2_count=v2_count,
        horizon_days=horizon_days,
        n_folds=n_folds,
        embargo_days=embargo_days,
        score_variant=score_variant,
        schema_warnings_list=caught_warnings,
    )
    report_path.write_text(report_md, encoding="utf-8")
    logger.info("IC backtest report → %s", report_path)
    return report_path


# ── Internal helpers ─────────────────────────────────────────────────────────

def _resolve_factors(
    snapshots: list[tuple[date, list[dict]]],
    score_variant: Literal["raw", "neutralized", "both"],
) -> list[str]:
    """Return list of factor field names to evaluate."""
    if score_variant == "raw":
        return list(_V2_FACTORS)

    # Check that neutralized fields exist in at least one snapshot
    sample_row = snapshots[0][1][0] if snapshots and snapshots[0][1] else {}
    neutral_fields = [f"{f}_neutral" for f in _V2_FACTORS]
    present = [f for f in neutral_fields if f in sample_row]

    if score_variant == "neutralized" and not present:
        raise ValueError(
            "score_variant='neutralized' requested but no *_score_neutral fields found "
            "in snapshots. Run the pipeline with neutralization enabled first."
        )

    if score_variant == "both":
        return list(_V2_FACTORS) + present

    return neutral_fields


def _try_write_chart(ic_results: list, output_dir: Path) -> None:
    """Write IC bar chart PNG if matplotlib is available; silently skip otherwise."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        names = [r.factor_name.replace("_score", "").replace("_", "\n") for r in ic_results]
        means = [r.ic_mean if not math.isnan(r.ic_mean) else 0.0 for r in ic_results]
        ci_lo = [
            r.ic_mean - r.ci_lower if not (math.isnan(r.ic_mean) or math.isnan(r.ci_lower)) else 0.0
            for r in ic_results
        ]
        ci_hi = [
            r.ci_upper - r.ic_mean if not (math.isnan(r.ic_mean) or math.isnan(r.ci_upper)) else 0.0
            for r in ic_results
        ]

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#2ecc71" if m > 0 else "#e74c3c" for m in means]
        ax.bar(names, means, color=colors, alpha=0.85)
        ax.errorbar(range(len(means)), means, yerr=[ci_lo, ci_hi],
                    fmt="none", color="black", capsize=4, linewidth=1.2)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"Factor IC (Spearman, {date.today().isoformat()})")
        ax.set_ylabel("IC mean ± 95% CI")
        ax.set_xlabel("Factor")
        plt.tight_layout()
        chart_path = output_dir / f"ic_backtest_{date.today().isoformat()}.png"
        fig.savefig(chart_path, dpi=120)
        plt.close(fig)
        logger.info("IC chart → %s", chart_path)
    except ImportError:
        logger.info("matplotlib not available — IC chart skipped")
    except Exception as exc:
        logger.warning("IC chart generation failed: %s", exc)


def _fmt_float(v: float) -> str:
    if math.isnan(v):
        return "N/A"
    return f"{v:+.4f}"


def _significance_badge(p: float) -> str:
    if math.isnan(p):
        return ""
    if p < 0.01:
        return " ⭐⭐ (p<0.01)"
    if p < 0.05:
        return " ⭐ (p<0.05)"
    return " (n.s.)"


def _render_report(
    ic_results: list,
    snapshots: list,
    v1_count: int,
    v2_count: int,
    horizon_days: int,
    n_folds: int,
    embargo_days: int,
    score_variant: str,
    schema_warnings_list: list,
) -> str:
    today = date.today().isoformat()
    n_total = len(snapshots)
    date_range = (
        f"{snapshots[0][0].isoformat()} – {snapshots[-1][0].isoformat()}"
        if snapshots else "N/A"
    )

    lines: list[str] = []
    lines += [
        f"# IC Backtest Report — {today}",
        "",
        "> **Advisory only.** This report does NOT modify WEIGHTS automatically.",
        "> Weight changes are human decisions based on empirical evidence below.",
        "",
        "## Methodology",
        "",
        "- **IC measure:** Spearman rank correlation (Grinold & Kahn 2000, ch. 6)",
        f"- **Forward return horizon:** {horizon_days} calendar days (~1 month)",
        f"- **Validation:** Purged {n_folds}-fold CV, embargo={embargo_days} days",
        "  (López de Prado 2018 AFML ch. 7 — prevents leakage from overlapping return windows)",
        f"- **Score variant:** `{score_variant}`",
        "- **Bootstrap CI:** 1 000 samples, 95% two-sided",
        "",
        "## Dataset",
        "",
        f"- **Snapshots:** {n_total} ({v1_count} v1-legacy, {v2_count} v2-orthogonal)",
        f"- **Date range:** {date_range}",
        "",
        "## Results",
        "",
        "| Factor | IC mean | IC std | IR | 95% CI | p-value | N snapshots | Avg tickers |",
        "|--------|---------|--------|----|--------|---------|-------------|-------------|",
    ]

    for r in ic_results:
        ci = f"[{_fmt_float(r.ci_lower)}, {_fmt_float(r.ci_upper)}]"
        p_str = f"{r.p_value:.4f}" if not math.isnan(r.p_value) else "N/A"
        badge = _significance_badge(r.p_value)
        lines.append(
            f"| `{r.factor_name}` "
            f"| {_fmt_float(r.ic_mean)} "
            f"| {_fmt_float(r.ic_std)} "
            f"| {_fmt_float(r.ir)} "
            f"| {ci} "
            f"| {p_str}{badge} "
            f"| {r.n_snapshots} "
            f"| {r.n_tickers_avg:.0f} |"
        )

    lines += [
        "",
        "## Grinold-Kahn Optimal Weights (advisory)",
        "",
        "> `weight_i ∝ |IC_i| / Σ|IC_j|`  — proportional to absolute IC.",
        "> Only statistically significant factors (p<0.05) are included.",
        "",
    ]

    sig = [(r.factor_name, r.ic_mean) for r in ic_results
           if not math.isnan(r.ic_mean) and not math.isnan(r.p_value) and r.p_value < 0.05]
    total_abs_ic = sum(abs(ic) for _, ic in sig)

    if not sig:
        lines.append("_No factors achieved p<0.05. Cannot compute optimal weights._\n")
        lines.append("_Possible causes: insufficient history, noisy factors, or market regime changes._\n")
    else:
        lines += [
            "| Factor | |IC_i| | Suggested weight |",
            "|--------|--------|-----------------|",
        ]
        for fname, ic in sig:
            w = abs(ic) / total_abs_ic
            lines.append(f"| `{fname}` | {abs(ic):.4f} | {w:.4f} |")
        lines.append("")
        lines.append(
            "> **Human decision required.** Compare suggested weights against current WEIGHTS "
            "in `scripts/run_pipeline.py`. Consider portfolio construction constraints "
            "(turnover, liquidity, regulatory) before applying any changes."
        )

    lines += [
        "",
        "## Schema Migration Impact",
        "",
        "The historical dataset spans two factor schema versions:",
        "",
        f"- **v1_legacy_5factor:** {v1_count} snapshots — fields: `edgar_score`, `insider_score`, "
        f"`news_score`, `momentum_score`, `congress_score`",
        f"- **v2_orthogonal_7factor:** {v2_count} snapshots — fields: {', '.join(f'`{f}`' for f in _V2_FACTORS)}",
        "",
        "**Cross-version mapping decisions:**",
        "",
        "| v2 Factor | v1 Source | Status |",
        "|-----------|-----------|--------|",
        "| `congress_score` | `congress_score` | ✅ Safe — identical semantics |",
        "| `news_sentiment_score` | `news_score` | ⚠️ Warning — v1 contaminated by ~40% buzz |",
        "| `insider_conviction_score` | — | ❌ MISSING in v1 — `edgar_score` has different semantics |",
        "| `insider_breadth_score` | — | ❌ MISSING in v1 — no counterpart |",
        "| `momentum_long_score` | — | ❌ MISSING in v1 — 20d reversal ≠ Jegadeesh 12-1m |",
        "| `volume_attention_score` | — | ❌ MISSING in v1 — no counterpart |",
        "| `news_buzz_score` | — | ❌ MISSING in v1 — no counterpart |",
        "",
        "> ⚠️ IC estimates for factors with MISSING v1 data are computed **only from v2 snapshots**.",
        "> With fewer data points, confidence intervals will be wider. Accumulate more v2 snapshots",
        "> by running the pipeline daily.",
    ]

    if schema_warnings_list:
        lines += [
            "",
            "### Runtime schema warnings",
            "",
        ]
        for w in schema_warnings_list:
            lines.append(f"- {w.category.__name__}: {w.message}")

    lines += [
        "",
        "## Interpretation Guide",
        "",
        "- **IC > 0.05:** Economically meaningful signal (Grinold-Kahn threshold)",
        "- **IC > 0.10:** Strong signal — consider overweighting",
        "- **IR > 0.5:** Consistent signal across time (not just one lucky period)",
        "- **p < 0.05:** Statistically significant at 5% level",
        "- **p < 0.01 (⭐⭐):** Statistically significant at 1% level",
        "",
        "---",
        f"_Generated {today} by `regime_trader.research.ic_backtest`. "
        f"Do not share without human review._",
    ]

    return "\n".join(lines) + "\n"
