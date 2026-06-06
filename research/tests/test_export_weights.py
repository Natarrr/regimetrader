# Path: research/tests/test_export_weights.py
"""Tests for export_weights.py."""
import json
import re
import tempfile
from pathlib import Path

import pytest

from research.scripts.export_weights import load_final_weights, update_weights_py
from research.scripts.ic_engine import FACTORS

_SAMPLE_OPTIMAL = {
    "generated_at": "2026-06-06T00:00:00+00:00",
    "blend_alpha": 0.6,
    "weight_floor": 0.05,
    "lgbm_val_ic_per_fold": [0.038, 0.044],
    "lgbm_val_ic_mean": 0.041,
    "investigate_factors": [],
    "weights": {
        "insider_conviction": {"academic": 0.30, "final": 0.30},
        "insider_breadth":    {"academic": 0.15, "final": 0.15},
        "congress":           {"academic": 0.22, "final": 0.22},
        "news_sentiment":     {"academic": 0.10, "final": 0.10},
        "news_buzz":          {"academic": 0.05, "final": 0.05},
        "momentum_long":      {"academic": 0.15, "final": 0.15},
        "volume_attention":   {"academic": 0.03, "final": 0.03},
        "analyst_consensus":  {"academic": 0.00, "final": 0.00},
        "quality_piotroski":  {"academic": 0.00, "final": 0.00},
    },
}

_SAMPLE_WEIGHTS_PY = '''
WEIGHTS_US: dict[str, float] = {
    "insider_conviction": 0.30000000,
    "insider_breadth":    0.15000000,
    "congress":           0.22000000,
    "news_sentiment":     0.10000000,
    "news_buzz":          0.05000000,
    "momentum_long":      0.15000000,
    "volume_attention":   0.03000000,
    "analyst_consensus":  0.00000000,
    "quality_piotroski":  0.00000000,
}
assert abs(sum(WEIGHTS_US.values()) - 1.0) < 1e-6

WEIGHTS_GLOBAL = {"insider_conviction": 0.30}
'''


def _write_optimal(tmp_path: Path) -> Path:
    p = tmp_path / "optimal_weights.json"
    p.write_text(json.dumps(_SAMPLE_OPTIMAL))
    return p


def test_load_final_weights_keys(tmp_path):
    p = _write_optimal(tmp_path)
    weights = load_final_weights(p)
    from research.scripts.export_weights import _US_FACTORS
    assert set(weights.keys()) == set(_US_FACTORS)


def test_load_final_weights_sum_to_one(tmp_path):
    p = _write_optimal(tmp_path)
    weights = load_final_weights(p)
    assert abs(sum(weights.values()) - 1.0) < 1e-5


def test_update_weights_py_preserves_global(tmp_path):
    p = _write_optimal(tmp_path)
    weights_py = tmp_path / "weights.py"
    weights_py.write_text(_SAMPLE_WEIGHTS_PY)
    weights = load_final_weights(p)
    new_content = update_weights_py(weights, weights_py, dry_run=False)
    # WEIGHTS_GLOBAL must be preserved
    assert "WEIGHTS_GLOBAL" in new_content


def test_update_weights_py_new_values_written(tmp_path):
    p = _write_optimal(tmp_path)
    # Modify weights while keeping sum = 1.0:
    # insider_conviction=0.30, insider_breadth=0.17, congress=0.19,
    # news_sentiment=0.10, news_buzz=0.03, momentum_long=0.18,
    # volume_attention=0.03, analyst_consensus=0.00, quality_piotroski=0.00
    # sum = 0.30 + 0.17 + 0.19 + 0.10 + 0.03 + 0.18 + 0.03 + 0.00 + 0.00 = 1.00
    opt = json.loads(p.read_text())
    opt["weights"]["insider_breadth"]["final"] = 0.17
    opt["weights"]["congress"]["final"] = 0.19
    opt["weights"]["news_buzz"]["final"] = 0.03
    opt["weights"]["momentum_long"]["final"] = 0.18
    p.write_text(json.dumps(opt))

    weights_py = tmp_path / "weights.py"
    weights_py.write_text(_SAMPLE_WEIGHTS_PY)
    weights = load_final_weights(p)
    update_weights_py(weights, weights_py, dry_run=False)
    content = weights_py.read_text()
    assert "0.18000000" in content


def test_export_weights_assert_still_valid(tmp_path):
    """After export, the written file must pass the weight sum assert."""
    p = _write_optimal(tmp_path)
    weights_py = tmp_path / "weights.py"
    weights_py.write_text(_SAMPLE_WEIGHTS_PY)
    weights = load_final_weights(p)
    update_weights_py(weights, weights_py, dry_run=False)
    content = weights_py.read_text()
    # Evaluate just the WEIGHTS_US block and check sum
    local_ns = {}
    exec(content, local_ns)  # noqa: S102
    written_weights = local_ns["WEIGHTS_US"]
    assert abs(sum(written_weights.values()) - 1.0) < 1e-5
