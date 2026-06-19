# Path: src/engine/engine.py
import os
import json
import logging
from typing import Dict, List, Any

from src.config.factor_matrix import FACTOR_MATRIX_V3
from src.config.weights import get_region
from src.ingestion.v3_shadow import _V3_OUTPUT_KEYS, v3_shadow_enabled
from src.scoring.engine_v3 import score_universe_v3

logger = logging.getLogger("QuantEngine")

# Raw FMPFetcher metrics forwarded verbatim for Discord display / cook joins.
# These are NOT scoring inputs — cook_toplists._normalize_intl_entry reads them
# by exactly these names. market_cap is listing-currency (no FX normalization
# upstream) and must not be used for USD math downstream; insider_usd is Form 4
# USD by definition; the remaining keys are dimensionless or used as ratios.
_DISPLAY_META_KEYS = (
    "insider_usd",
    "market_cap",
    "return_12_1m",
    "return_5d",     # recent run-up — freshness/extension gate (send_discord)
    "return_21d",
    "target_price",
    "current_price",
    "price_to_book",
    "beta_30d",       # P2.1 — CAPITULATION low-beta gate (INTL producer)
    "earnings_surprise_pct",
    "earnings_surprise_days",
    "analyst_consensus_source",
    "analyst_revision_score",
    "analyst_revision_n_analysts",
    "price_target_upside_score",
    "quality_piotroski_score",
    # Candidate shadow factors (A1 valuation breadth + A2 growth/earnings-quality).
    # Forwarded verbatim into ranking rows for the de-overlapped IC gate
    # (src/research/ic_metrics) — NOT in active_factors, so live composite is
    # unchanged until a measured IC justifies a weight.
    "earnings_yield_score",
    "ev_ebitda_score",
    "revenue_growth_score",
    "eps_growth_score",
    "accruals_score",
    # Exit-anchor / liquidity inputs — forwarded when the fetcher produces
    # them (atr_14 / adv_20d_usd are not fetched yet; keys are forward-compat).
    "atr_14",
    "adv_20d_usd",
)

# v3.0 shadow: engine factor name → INTL metrics column carrying its raw value.
# Superset across EU+ASIA; each region's engine input only reads its own 9.
_V3_INTL_INPUT_MAP: Dict[str, str] = {
    "quality_piotroski": "quality_piotroski_score",
    "fcf_yield": "fcf_yield_score",
    "pb_value_up": "pb_value_up_score",
    "roic_quality": "roic_quality_score",
    "amihud_shock": "amihud_shock_score",
    "analyst_consensus": "analyst_consensus_score",
    "margin_expansion": "margin_expansion_score",
    "inst_concentration": "inst_concentration_score",
    "dividend_sustain": "dividend_sustain_score",
    "revision_velocity": "revision_velocity_score",
    # v3-specific keys — v2.2 same-named factors use different math:
    "analyst_revision": "analyst_revision_score_v3",
    # v2.2 stores 0.0 for a missing PT (downward bias on a signed factor);
    # the _v3 key preserves None = unavailable.
    "price_target_upside": "price_target_upside_score_v3",
}

class StrategyEngine:
    def __init__(self, profile_path: str):
        with open(profile_path, 'r') as f:
            self.profile = json.load(f)

        self.region = self.profile["region"]
        self.active_factors = self.profile["active_factors"]
        self.output_filename = self.profile["output_filename"]

        # Verify weights sum exactly to 1.0 (100%) — 1e-6 per CLAUDE.md §3
        total_weight = sum(self.active_factors.values())
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Profile {self.region} weights sum to {total_weight}, must be exactly 1.0")

    def score_ticker_pool(self, raw_universe_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process raw ticker metrics, scoring only the active factors in the regional profile.

        Score formula: Σ(w_i · s_i) / Σ(w_i)  for i in factors where data is present.
        None in raw_metrics means API failure — excluded from both numerator and denominator.
        0.0 means genuine zero signal (API succeeded, no signal) — included normally.
        """
        processed_rankings = []
        total_declared_weight = sum(self.active_factors.values())

        for asset in raw_universe_data:
            ticker = asset.get("ticker")
            raw_metrics = asset.get("metrics", {})

            weighted_score = 0.0
            available_weight = 0.0  # accumulates per-ticker: only live factors count
            factor_breakdown = {}

            for factor, weight in self.active_factors.items():
                raw_key = f"{factor}_score" if not factor.endswith("_score") else factor
                try:
                    raw_val = raw_metrics.get(raw_key)  # no default — None means absent
                    if raw_val is None:
                        factor_breakdown[factor] = None  # preserve absence signal
                        continue                         # don't count weight in denominator
                    if isinstance(raw_val, str):
                        stripped = raw_val.strip()
                        if (stripped.startswith("[") and stripped.endswith("]")) or \
                                (stripped.startswith("{") and stripped.endswith("}")):
                            try:
                                raw_val = json.loads(stripped)
                            except Exception:
                                pass
                    if isinstance(raw_val, dict):
                        metric_value = float(raw_val.get("score", 0.0) or 0.0)
                    elif isinstance(raw_val, list):
                        metric_value = float(raw_val[0]) if raw_val else 0.0
                    else:
                        metric_value = float(raw_val or 0.0)
                except Exception:
                    metric_value = 0.0

                weighted_score += metric_value * weight
                available_weight += weight  # only live factors count
                factor_breakdown[factor] = metric_value

            composite_score = (
                round(weighted_score / available_weight, 4)
                if available_weight > 1e-9
                else 0.0
            )

            none_factors = [f for f, v in factor_breakdown.items() if v is None]
            zero_factors = [f for f, v in factor_breakdown.items()
                            if v is not None and float(v) == 0.0]
            if none_factors:
                logger.info(
                    "%s: %d factors absent (None/no coverage): %s",
                    ticker, len(none_factors), none_factors,
                )
            if zero_factors:
                logger.debug(
                    "%s: %d factors zero signal (covered, no activity): %s",
                    ticker, len(zero_factors), zero_factors,
                )

            weight_coverage = (
                round(available_weight / total_declared_weight, 4)
                if total_declared_weight > 1e-9 else 0.0
            )

            # Coverage penalty: prevents momentum-only tickers from scoring
            # as high as fully-covered tickers cross-sectionally.
            # Full score at coverage >= 70%; linear penalty below that floor.
            # This stops a ticker with only MO:1.00 from outranking a ticker
            # with MO:0.80 + NS:0.60 + AC:0.70 + AR:0.40 + PT:0.55.
            coverage_penalty = min(1.0, weight_coverage / 0.70)
            penalized_score = round(composite_score * coverage_penalty, 4)

            low_coverage = weight_coverage < 0.40
            if low_coverage:
                logger.warning(
                    "%s: weight_coverage=%.2f < 0.40 — score unreliable "
                    "(only %s factors active). Marking _low_coverage=True.",
                    ticker, weight_coverage,
                    [f for f, v in factor_breakdown.items() if v is not None and float(v or 0) > 0],
                )

            processed_rankings.append({
                "ticker":           ticker,
                "composite_score":  penalized_score,
                "raw_composite_score": composite_score,  # pre-penalty, for diagnostics
                "region_applied":   self.region,
                "factor_snapshots": factor_breakdown,
                "pipeline":         "INTL",
                "weight_coverage":  weight_coverage,
                "_low_coverage":    low_coverage,
                "sector":           asset.get("sector", ""),
                # Display-meta passthrough: only keys present in raw_metrics are
                # forwarded (absent keys stay absent — no None spray).
                **{k: raw_metrics[k] for k in _DISPLAY_META_KEYS if k in raw_metrics},
            })

        # Sort universe descending by final quantitative output ranking
        processed_rankings.sort(key=lambda x: x["composite_score"], reverse=True)

        # ── v3.0 shadow (SCORING_V3_SHADOW=1): split the co-mingled INTL ──
        # pool into separate EU/ASIA engine_v3 runs and attach final_score_v3
        # alongside the untouched composite_score (v2.2 authoritative).
        if v3_shadow_enabled():
            self._attach_v3_shadow(raw_universe_data, processed_rankings)

        return processed_rankings

    def _attach_v3_shadow(
        self,
        raw_universe_data: List[Dict[str, Any]],
        rankings: List[Dict[str, Any]],
    ) -> None:
        """Score EU and ASIA as SEPARATE engine_v3 pools (never co-mingled —
        bucket statistics across regions are meaningless) and merge the v3
        output columns onto the matching v2.2 ranking rows in place."""
        by_ticker = {r.get("ticker"): r for r in rankings}
        pools: Dict[str, List[Dict[str, Any]]] = {"EU": [], "ASIA": []}

        for asset in raw_universe_data:
            ticker = asset.get("ticker") or ""
            region = get_region(ticker)
            if region not in pools:
                logger.warning(
                    "v3 shadow: %s resolves to region %s — not an INTL "
                    "ticker, skipped (v2.2 composite unaffected)",
                    ticker, region,
                )
                continue
            metrics = asset.get("metrics", {}) or {}
            row: Dict[str, Any] = {
                "ticker": ticker,
                "sector": (asset.get("sector")
                           or metrics.get("_v3_sector") or "Unknown"),
                "cap_tier": metrics.get("_v3_cap_tier") or "large",
                "market": "EUROPE" if region == "EU" else "ASIA",
                "quality_piotroski_raw": metrics.get("quality_piotroski_raw"),
            }
            for factor in FACTOR_MATRIX_V3[region]:
                row[f"{factor}_score"] = metrics.get(_V3_INTL_INPUT_MAP[factor])
            pools[region].append(row)

        for region, rows in pools.items():
            if not rows:
                continue
            try:
                scored = score_universe_v3(rows, region)
            except ValueError as exc:
                logger.error(
                    "v3 shadow %s pool failed: %s — composite_score "
                    "unaffected", region, exc,
                )
                continue
            for s in scored:
                target = by_ticker.get(s.get("ticker"))
                if target is None:
                    continue
                for key in _V3_OUTPUT_KEYS:
                    target[key] = s.get(key)
                for key in ("_factor_blackout", "_contamination_masked"):
                    if key in s:
                        target[key] = s[key]

    def save_results(self, output_dir: str, data: List[Dict[str, Any]]):
        os.makedirs(output_dir, exist_ok=True)
        target_path = os.path.join(output_dir, self.output_filename)
        with open(target_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Successfully serialized {self.region} rankings to {target_path}")
