"""Tests for Skills/pipeline_dependency_checker.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from pipeline_dependency_checker import DependencyInfo, check_dependencies, format_dependency_check


def test_basic_ok_or_degraded():
    r = check_dependencies()
    assert r.flag in ("OK", "DEGRADED", "MISSING_REQUIRED")


def test_python_version_set():
    r = check_dependencies()
    assert r.python_version
    assert "." in r.python_version


def test_n_available_positive():
    r = check_dependencies()
    assert r.n_available >= 0


def test_dependency_tuple():
    r = check_dependencies()
    assert isinstance(r.dependencies, tuple)
    assert len(r.dependencies) > 0


def test_dependency_info_fields():
    r = check_dependencies()
    for d in r.dependencies:
        assert isinstance(d, DependencyInfo)
        assert isinstance(d.name, str)
        assert isinstance(d.required, bool)
        assert isinstance(d.available, bool)
        assert isinstance(d.feature, str)


def test_feature_matrix_is_dict():
    r = check_dependencies()
    assert isinstance(r.feature_matrix, dict)
    assert len(r.feature_matrix) > 0


def test_feature_matrix_booleans():
    r = check_dependencies()
    for v in r.feature_matrix.values():
        assert isinstance(v, bool)


def test_numpy_present():
    # numpy is a required dep and should be available in this env
    r = check_dependencies()
    names = {d.name for d in r.dependencies}
    assert "numpy" in names


def test_extra_packages():
    r = check_dependencies(extra_packages=["json"])
    names = [d.name for d in r.dependencies]
    assert "json" in names


def test_extra_package_feature():
    r = check_dependencies(extra_packages=["mypkg_xyz_not_real"])
    d = next(dep for dep in r.dependencies if dep.name == "mypkg_xyz_not_real")
    assert not d.available
    assert d.version is None


def test_format_returns_string():
    r = check_dependencies()
    assert isinstance(format_dependency_check(r), str)


def test_format_contains_key_words():
    r = check_dependencies()
    text = format_dependency_check(r)
    assert "Dependency" in text
    assert "Flag" in text
    assert "Feature Matrix" in text


def test_counts_consistent():
    r = check_dependencies()
    total = r.n_available + r.n_required_missing + r.n_optional_missing
    assert total == len(r.dependencies)
