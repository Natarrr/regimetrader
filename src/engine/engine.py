# Path: backend/market_intel/engine.py
import os
import json
import logging
from typing import Dict, List, Any

logger = logging.getLogger("QuantEngine")

class StrategyEngine:
    def __init__(self, profile_path: str):
        with open(profile_path, 'r') as f:
            self.profile = json.load(f)

        self.region = self.profile["region"]
        self.active_factors = self.profile["active_factors"]
        self.output_filename = self.profile["output_filename"]

        # Verify weights sum exactly to 1.0 (100%)
        total_weight = sum(self.active_factors.values())
        if abs(total_weight - 1.0) > 1e-4:
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
            })

        # Sort universe descending by final quantitative output ranking
        processed_rankings.sort(key=lambda x: x["composite_score"], reverse=True)
        return processed_rankings

    def save_results(self, output_dir: str, data: List[Dict[str, Any]]):
        os.makedirs(output_dir, exist_ok=True)
        target_path = os.path.join(output_dir, self.output_filename)
        with open(target_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Successfully serialized {self.region} rankings to {target_path}")
