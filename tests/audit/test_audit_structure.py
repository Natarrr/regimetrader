from pathlib import Path
from scripts.audit.audit_structure import score_pairs, _list_symbols


def _make_pkg(root: Path, name: str, functions=("foo",)) -> None:
    pkg = root / name
    pkg.mkdir(parents=True, exist_ok=True)
    body = "\n".join(f"def {fn}():\n    pass" for fn in functions)
    (pkg / "__init__.py").write_text(body)


def test_more_functions_wins(tmp_path):
    _make_pkg(tmp_path, "left",  functions=["a"])
    _make_pkg(tmp_path, "right", functions=["a", "b", "c"])
    decisions = score_pairs(tmp_path, graph={}, pairs=[("left", "right")])
    assert decisions[0].winner == "right"


def test_missing_left_package_picks_right(tmp_path):
    _make_pkg(tmp_path, "right", functions=["a", "b"])
    decisions = score_pairs(tmp_path, graph={}, pairs=[("missing", "right")])
    assert decisions[0].winner == "right"


def test_unique_symbols_in_loser_flagged(tmp_path):
    _make_pkg(tmp_path, "left",  functions=["shared", "unique_left"])
    _make_pkg(tmp_path, "right", functions=["shared", "r1", "r2", "r3"])
    decisions = score_pairs(tmp_path, graph={}, pairs=[("left", "right")])
    d = decisions[0]
    assert d.winner == "right"
    assert "unique_left" in d.unique_in_loser


def test_list_symbols_finds_functions(tmp_path):
    _make_pkg(tmp_path, "mypkg", functions=["alpha", "beta"])
    syms = _list_symbols(tmp_path, "mypkg")
    assert "alpha" in syms and "beta" in syms


def test_both_missing_skipped(tmp_path):
    decisions = score_pairs(tmp_path, graph={}, pairs=[("gone_a", "gone_b")])
    assert decisions == []
