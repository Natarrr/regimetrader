"""scripts/run_ic_backtest.py — CLI for the IC backtest research tool.

Usage:
  python scripts/run_ic_backtest.py [--score-variant {raw,neutralized,both}]
                                    [--horizon-days N]
                                    [--log-dir PATH]
                                    [--output-dir PATH]

The report is advisory only. WEIGHTS are NOT modified automatically.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_ic_backtest")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run IC backtest to validate factor WEIGHTS (advisory report only)."
    )
    parser.add_argument(
        "--score-variant",
        choices=["raw", "neutralized", "both"],
        default="raw",
        help="Factor field variant to evaluate (default: raw).",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=21,
        help="Forward return horizon in calendar days (default: 21 ≈ 1 month).",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=ROOT / "logs",
        help="Directory containing intel_source_status.json and historical/ subdirs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "logs" / "research",
        help="Directory where Markdown report is written (default: logs/research/).",
    )
    parser.add_argument(
        "--min-snapshots",
        type=int,
        default=60,
        help="Minimum historical snapshots required (default: 60 ≈ 3 months).",
    )

    args = parser.parse_args(argv)

    from regime_trader.research.ic_backtest import run_ic_backtest

    try:
        report_path = run_ic_backtest(
            log_dir=args.log_dir,
            output_dir=args.output_dir,
            score_variant=args.score_variant,
            horizon_days=args.horizon_days,
            min_snapshots=args.min_snapshots,
            cache_root=ROOT,
        )
        logger.info("Report written to: %s", report_path)
        logger.info("")
        logger.info("IMPORTANT: This report is advisory only.")
        logger.info("Weight changes require human review and manual edit of")
        logger.info("  scripts/run_pipeline.py → WEIGHTS dict.")
        return 0

    except RuntimeError as exc:
        logger.error("IC backtest aborted:\n\n%s", exc)
        return 1

    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
