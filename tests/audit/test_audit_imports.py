from scripts.audit.audit_imports import (
    build_import_graph, find_orphans, find_broken_imports, discover_local_packages,
)


def test_graph_resolves_package_import(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "mod.py").write_text("x = 1")
    (tmp_path / "main.py").write_text("from pkg import mod\n")
    graph = build_import_graph(tmp_path)
    assert any("pkg" in p for p in graph["main.py"])


def test_orphan_file_detected(tmp_path):
    (tmp_path / "used.py").write_text("x = 1")
    (tmp_path / "unused.py").write_text("y = 2")
    (tmp_path / "consumer.py").write_text("from used import x\n")
    graph = build_import_graph(tmp_path)
    orphans = find_orphans(graph, entry_points={"consumer.py"})
    assert "unused.py" in orphans
    assert "used.py" not in orphans


def test_entry_points_excluded_from_orphans(tmp_path):
    pages = tmp_path / "pages"
    pages.mkdir()
    (pages / "1_Home.py").write_text("x = 1")
    graph = build_import_graph(tmp_path)
    orphans = find_orphans(graph, entry_points={"pages/1_Home.py"})
    assert "pages/1_Home.py" not in orphans


def test_init_files_never_flagged_as_orphans(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "__init__.py").write_text("")
    graph = build_import_graph(tmp_path)
    orphans = find_orphans(graph, entry_points=set())
    assert not any("__init__.py" in o for o in orphans)


def test_broken_import_detected(tmp_path):
    (tmp_path / "regime_trader").mkdir()
    (tmp_path / "regime_trader" / "__init__.py").write_text("")
    (tmp_path / "main.py").write_text("from regime_trader import nonexistent\n")
    broken = find_broken_imports(tmp_path, local_packages={"regime_trader"})
    assert "main.py" in broken


def test_stdlib_not_flagged_as_broken(tmp_path):
    (tmp_path / "main.py").write_text("import os\nimport sys\n")
    broken = find_broken_imports(tmp_path, local_packages=set())
    assert "main.py" not in broken


def test_discover_local_packages(tmp_path):
    (tmp_path / "mypkg").mkdir()
    (tmp_path / "mypkg" / "__init__.py").write_text("")
    pkgs = discover_local_packages(tmp_path)
    assert "mypkg" in pkgs
