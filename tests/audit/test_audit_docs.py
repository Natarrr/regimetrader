from pathlib import Path
from scripts.audit.audit_docs import (
    collect_symbols, extract_doc_symbols, find_dead_docs, find_duplicate_docs, jaccard,
)


def test_collect_symbols_finds_class_and_function(tmp_path):
    (tmp_path / "mod.py").write_text("class Foo:\n    pass\ndef bar(): pass\n")
    syms = collect_symbols(tmp_path)
    assert "Foo" in syms
    assert "bar" in syms


def test_extract_symbols_from_backtick_identifiers():
    text = "Use `FmpClient` to fetch data. Call `get_profile()` for details."
    syms = extract_doc_symbols(text)
    assert "FmpClient" in syms
    assert "get_profile" in syms


def test_dead_doc_flagged_when_symbols_missing(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "old.md").write_text(
        "Use `OldClass` to call `deprecated_func` and `another_gone`."
    )
    dead = find_dead_docs(tmp_path, threshold=0.5)
    assert "docs/old.md" in dead


def test_live_doc_not_flagged(tmp_path):
    (tmp_path / "pkg.py").write_text("class RealClass:\n    pass\ndef real_func(): pass\n")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "live.md").write_text("Use `RealClass` and `real_func`.")
    dead = find_dead_docs(tmp_path, threshold=0.5)
    assert "docs/live.md" not in dead


def test_jaccard_identical():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0


def test_jaccard_disjoint():
    assert jaccard({"a"}, {"b"}) == 0.0


def test_duplicate_docs_detected(tmp_path):
    (tmp_path / "docs").mkdir()
    content = "# Overview\n## Setup\n```python\nx = 1\n```\n"
    (tmp_path / "docs" / "a.md").write_text(content)
    (tmp_path / "docs" / "b.md").write_text(content)
    dups = find_duplicate_docs(tmp_path, threshold=0.7)
    pairs = {frozenset([a, b]) for a, b, _ in dups}
    assert frozenset(["docs/a.md", "docs/b.md"]) in pairs
