"""tests/test_atomic_write.py — Knuth: atomic write leaves no partial state."""
from __future__ import annotations

import json
from pathlib import Path

from utils.atomic_write import atomic_write_json


def test_writes_file_with_exact_content(tmp_path: Path) -> None:
    """Knuth: post-condition is bit-exact JSON content at the target path."""
    p = tmp_path / "out.json"
    payload = {"a": 1, "b": [1, 2, 3], "c": "héllo"}
    atomic_write_json(p, payload)
    assert p.exists()
    assert json.loads(p.read_text(encoding="utf-8")) == payload


def test_creates_missing_parent_dirs(tmp_path: Path) -> None:
    """Knuth: the function owns mkdir — callers must not have to pre-create the tree."""
    p = tmp_path / "deep" / "nested" / "dir" / "out.json"
    atomic_write_json(p, {"k": "v"})
    assert p.exists()
    assert json.loads(p.read_text(encoding="utf-8")) == {"k": "v"}


def test_overwrites_existing_file(tmp_path: Path) -> None:
    """Knuth: replace semantics — a stale target is silently overwritten."""
    p = tmp_path / "out.json"
    p.write_text('{"old": true}', encoding="utf-8")
    atomic_write_json(p, {"new": True})
    assert json.loads(p.read_text(encoding="utf-8")) == {"new": True}


def test_no_temp_files_left_after_success(tmp_path: Path) -> None:
    """Markowitz: only the target should remain — no orphan temp files."""
    p = tmp_path / "out.json"
    atomic_write_json(p, {"ok": True})
    leftovers = [f for f in tmp_path.iterdir() if f.name.startswith(f".{p.name}.")]
    assert leftovers == []


def test_no_temp_files_left_after_serialisation_failure(tmp_path: Path) -> None:
    """Knuth: if serialisation raises, the directory must remain untouched."""
    p = tmp_path / "out.json"
    unserialisable = {"f": lambda: 1}
    try:
        atomic_write_json(p, unserialisable)
    except TypeError:
        pass
    assert not p.exists()
    assert list(tmp_path.iterdir()) == []
