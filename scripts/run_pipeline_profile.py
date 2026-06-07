#!/usr/bin/env python3
# Path: scripts/run_pipeline_profile.py
import argparse
import json
import sys
from src.engine.engine import StrategyEngine


def main():
    parser = argparse.ArgumentParser(
        description="Parallel Decoupled Quant Engine Entrypoint")
    parser.add_argument("--config", required=True,
                        help="Path to regional strategy JSON profile")
    parser.add_argument("--raw-data", required=True,
                        help="Path to raw fetched metrics payload")
    parser.add_argument("--out-dir", default="logs",
                        help="Output matrix destination")
    args = parser.parse_args()

    try:
        # Load Raw Metrics Data Frame
        with open(args.raw_data, 'r', encoding='utf-8') as f:
            raw_payload = json.load(f)

        # Initialize strategy based on loaded config profile
        engine = StrategyEngine(args.config)
        rankings = engine.score_ticker_pool(raw_payload)
        engine.save_results(args.out_dir, rankings)

        print(
            f"Pipeline executed successfully for profile region: {engine.region}")
    except Exception as e:
        print(f"CRITICAL: Pipeline processing exception encountered: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
