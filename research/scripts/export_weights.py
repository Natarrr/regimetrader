# Path: research/scripts/export_weights.py
"""Write calibrated weights from optimal_weights.json → config/weights.py.

Run from repo root after train_lgbm.py completes:
    python research/scripts/export_weights.py [--dry-run]

The script ONLY updates the WEIGHTS_US dict. WEIGHTS_GLOBAL, PIOTROSKI_GATE,
and all region-classifier code are preserved unchanged.

After running, review the diff with:
    git diff regime_trader/config/weights.py
Then commit explicitly once satisfied.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_OPTIMAL = Path("research/optimal_weights.json")
_WEIGHTS_PY = Path("regime_trader/config/weights.py")

# Factors present in WEIGHTS_US (the only ones export updates)
_US_FACTORS = [
    "insider_conviction", "insider_breadth", "congress",
    "news_sentiment", "news_buzz", "momentum_long",
    "volume_attention", "analyst_consensus", "quality_piotroski",
]


def load_final_weights(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    return {f: data["weights"][f]["final"] for f in _US_FACTORS}


def update_weights_py(weights: dict[str, float], source: Path, dry_run: bool) -> str:
    """Rewrite the WEIGHTS_US dict block in weights.py with new values.

    Preserves all other content exactly. Returns the new file content.
    """
    content = source.read_text()

    # Build the replacement WEIGHTS_US block
    lines = ["WEIGHTS_US: dict[str, float] = {"]
    for factor in _US_FACTORS:
        w = weights[factor]
        lines.append(f'    "{factor}": {w:.8f},')
    lines.append("}")
    new_block = "\n".join(lines)

    # Replace the existing WEIGHTS_US block (from "WEIGHTS_US: dict" to closing "}")
    pattern = re.compile(
        r'WEIGHTS_US:\s*dict\[str,\s*float\]\s*=\s*\{[^}]*\}',
        re.DOTALL
    )
    if not pattern.search(content):
        raise ValueError("Could not find WEIGHTS_US block in weights.py")

    new_content = pattern.sub(new_block, content)

    # Verify the weights sum to 1.0
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-5:
        raise ValueError(f"Calibrated weights sum to {total:.8f}, not 1.0 — aborting")

    if dry_run:
        print("── DRY RUN — would write the following WEIGHTS_US block: ──")
        print(new_block)
        return new_content

    source.write_text(new_content)
    return new_content


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print changes without writing")
    args = parser.parse_args()

    if not _OPTIMAL.exists():
        sys.exit(f"ERROR: {_OPTIMAL} not found — run train_lgbm.py first")

    weights = load_final_weights(_OPTIMAL)
    print(f"Loaded {len(weights)} calibrated weights from {_OPTIMAL}")
    for f, w in weights.items():
        print(f"  {f:<22} {w:.8f}")

    update_weights_py(weights, _WEIGHTS_PY, dry_run=args.dry_run)
    if not args.dry_run:
        print(f"\n✓ Wrote updated WEIGHTS_US to {_WEIGHTS_PY}")
        print("  Review with: git diff regime_trader/config/weights.py")
        print("  Then commit: git add regime_trader/config/weights.py && git commit")


if __name__ == "__main__":
    main()
