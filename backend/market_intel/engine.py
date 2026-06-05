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
        """
        Processes a raw array of ticker metrics, extracting and scoring
        only the active factors allowed by the regional strategy profile.
        """
        processed_rankings = []

        for asset in raw_universe_data:
            ticker = asset.get("ticker")
            raw_metrics = asset.get("metrics", {})

            weighted_score = 0.0
            factor_breakdown = {}

            # Map and calculate only the factors required by the strategy profile
            for factor, weight in self.active_factors.items():
                # Normalize input key to factor_score suffix used across the codebase
                raw_key = f"{factor}_score" if not factor.endswith("_score") else factor
                try:
                    metric_value = float(raw_metrics.get(raw_key, 0.0) or 0.0)
                except Exception:
                    metric_value = 0.0

                # Apply strategy weight matrices
                weighted_score += metric_value * weight
                factor_breakdown[factor] = metric_value

            processed_rankings.append({
                "ticker": ticker,
                "composite_score": round(weighted_score, 4),
                "region_applied": self.region,
                "factor_snapshots": factor_breakdown
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
