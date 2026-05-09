"""
CI environment invariant tests.

Verify that the Python version, project layout, and pytest config match
what is expected before the full suite runs.

Inspired by environment-consistency checks from Sims (1980) rational-expectations
framework: the model's assumptions must be validated before results are trusted.
"""

import pathlib
import sys


ROOT = pathlib.Path(__file__).parent.parent


def test_python_version_at_least_311():
    """CI mandates Python 3.11+ (match PYTHON_VERSION in ci.yml)."""
    assert sys.version_info >= (3, 11), (
        f"Expected Python >= 3.11, got {sys.version_info.major}.{sys.version_info.minor}"
    )


def test_pytest_ini_present():
    """pytest.ini must be at the repo root for correct rootdir discovery."""
    assert (ROOT / "pytest.ini").exists(), "pytest.ini not found at repo root"


def test_pytest_ini_has_testpaths():
    """pytest.ini must declare testpaths so bare 'pytest' works in CI."""
    content = (ROOT / "pytest.ini").read_text()
    assert "testpaths" in content, "pytest.ini must declare testpaths"


def test_backend_package():
    """backend/ must be a Python package (has __init__.py)."""
    assert (ROOT / "backend" / "__init__.py").exists(), (
        "backend/__init__.py is missing — 'import backend' will fail in CI"
    )


def test_backend_tests_package():
    """backend/tests/ must be a Python package."""
    assert (ROOT / "backend" / "tests" / "__init__.py").exists(), (
        "backend/tests/__init__.py is missing"
    )


def test_analysis_package():
    """analysis/ module directory must exist."""
    assert (ROOT / "analysis").is_dir(), "analysis/ directory not found"


def test_regime_package():
    """regime/ module directory must exist."""
    assert (ROOT / "regime").is_dir(), "regime/ directory not found"


def test_requirements_ci_has_pydantic():
    """requirements-ci.txt must declare pydantic (guards against regression)."""
    content = (ROOT / "requirements-ci.txt").read_text()
    assert "pydantic" in content, "pydantic missing from requirements-ci.txt"


def test_requirements_ci_has_anthropic():
    """requirements-ci.txt must declare anthropic for claude_client tests."""
    content = (ROOT / "requirements-ci.txt").read_text()
    assert "anthropic" in content, "anthropic missing from requirements-ci.txt"


def test_requirements_ci_has_hmmlearn():
    """requirements-ci.txt must declare hmmlearn for regime detector tests."""
    content = (ROOT / "requirements-ci.txt").read_text()
    assert "hmmlearn" in content, "hmmlearn missing from requirements-ci.txt"


def test_ci_workflow_exists():
    """ci.yml must exist so that the test gate runs on every push."""
    assert (ROOT / ".github" / "workflows" / "ci.yml").exists(), (
        ".github/workflows/ci.yml not found"
    )
