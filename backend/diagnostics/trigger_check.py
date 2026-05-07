"""backend/diagnostics/trigger_check.py
Verify Minsky trigger reproducibility for a set of known scenarios.

Run from repo root:
    python -m backend.diagnostics.trigger_check

Expected output:
    Scenario 1 (baseline SPY): conditions_met == 1 (persistence below 0.98)
    Scenario 2 (high vol):     conditions_met == 1
    Scenario 3 (2/3):          conditions_met == 2
    Scenario 4 (full Minsky):  conditions_met == 3
"""
from __future__ import annotations

import json
import sys
import os

# ── sys.path wiring ───────────────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
_root = os.path.dirname(os.path.dirname(_here))
if _root not in sys.path:
    sys.path.insert(0, _root)

from backend.utils.triggers import compute_minsky_conditions, minsky_ui_line

SCENARIOS = [
    {
        "label":       "Baseline SPY (persistence below threshold)",
        "persistence": 0.9446665,
        "cape_pct":    98.72,
        "yield_bps":   52.0,
        "expect":      1,   # cape_pct >= 95 fires
    },
    {
        "label":       "High vol only",
        "persistence": 0.99,
        "cape_pct":    70.0,
        "yield_bps":   30.0,
        "expect":      1,   # persistence fires
    },
    {
        "label":       "Vol + valuation (2/3)",
        "persistence": 0.99,
        "cape_pct":    96.0,
        "yield_bps":   20.0,
        "expect":      2,
    },
    {
        "label":       "Full Minsky (3/3)",
        "persistence": 0.99,
        "cape_pct":    96.0,
        "yield_bps":   -30.0,
        "expect":      3,
    },
    {
        "label":       "Clear (0/3)",
        "persistence": 0.95,
        "cape_pct":    80.0,
        "yield_bps":   50.0,
        "expect":      0,
    },
]


def main() -> int:
    print(f"\n{'='*72}")
    print(f"  MINSKY TRIGGER REPRODUCIBILITY CHECK")
    print(f"{'='*72}")

    all_pass = True
    for s in SCENARIOS:
        result = compute_minsky_conditions(s["persistence"], s["cape_pct"], s["yield_bps"])
        passed = result["conditions_met"] == s["expect"]
        status = "[PASS]" if passed else "[FAIL]"
        if not passed:
            all_pass = False
        print(f"\n  {status}  {s['label']}")
        print(f"    {minsky_ui_line(result)}")
        print(f"    conditions_met={result['conditions_met']}  expected={s['expect']}")
        if not passed:
            print(f"    *** MISMATCH — raw: {json.dumps(result, indent=6)}")

    print(f"\n{'='*72}")
    if all_pass:
        print("  All scenarios passed.")
    else:
        print("  SOME SCENARIOS FAILED — see above for details.")
    print(f"{'='*72}\n")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
