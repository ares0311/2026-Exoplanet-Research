"""Tests for Skills/data_freshness_checker.py"""
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from data_freshness_checker import (
    check_data_freshness,
    format_freshness_report,
)


def test_fresh_file(tmp_path):
    f = tmp_path / "data.json"
    f.write_text("{}")
    report = check_data_freshness({"data": f}, max_age_days=30.0)
    assert report.flag == "OK"


def test_missing_file(tmp_path):
    report = check_data_freshness({"missing": tmp_path / "nope.json"})
    assert report.flag == "MISSING"


def test_stale_file(tmp_path):
    f = tmp_path / "old.json"
    f.write_text("{}")
    old_mtime = time.time() - 40 * 86400
    os.utime(f, (old_mtime, old_mtime))
    report = check_data_freshness({"old": f}, max_age_days=30.0)
    assert report.flag == "STALE"


def test_n_counts(tmp_path):
    fresh = tmp_path / "fresh.json"
    fresh.write_text("{}")
    report = check_data_freshness(
        {"fresh": fresh, "missing": tmp_path / "x.json"},
        max_age_days=30.0
    )
    assert report.n_fresh == 1
    assert report.n_missing == 1


def test_per_artifact_max_age(tmp_path):
    f = tmp_path / "data.json"
    f.write_text("{}")
    old_mtime = time.time() - 5 * 86400
    os.utime(f, (old_mtime, old_mtime))
    # Default age OK, per-artifact stricter
    report = check_data_freshness(
        {"data": f},
        max_age_days=30.0,
        per_artifact_max_age={"data": 3.0},
    )
    assert report.flag == "STALE"


def test_empty_artifacts():
    report = check_data_freshness({})
    assert report.n_fresh == 0
    assert report.flag == "OK"


def test_multiple_files_all_fresh(tmp_path):
    for name in ["a", "b", "c"]:
        (tmp_path / f"{name}.json").write_text("{}")
    artifacts = {n: tmp_path / f"{n}.json" for n in ["a", "b", "c"]}
    report = check_data_freshness(artifacts, max_age_days=30.0)
    assert report.n_fresh == 3
    assert report.flag == "OK"


def test_format_returns_string(tmp_path):
    f = tmp_path / "x.json"
    f.write_text("{}")
    report = check_data_freshness({"x": f})
    text = format_freshness_report(report)
    assert isinstance(text, str)
    assert "Freshness" in text


def test_format_has_table(tmp_path):
    f = tmp_path / "x.json"
    f.write_text("{}")
    report = check_data_freshness({"x": f})
    text = format_freshness_report(report)
    assert "|" in text


def test_stale_overrides_missing(tmp_path):
    f = tmp_path / "data.json"
    f.write_text("{}")
    old_mtime = time.time() - 15 * 86400
    os.utime(f, (old_mtime, old_mtime))
    report = check_data_freshness(
        {"data": f, "missing": tmp_path / "none.json"},
        max_age_days=10.0,
    )
    assert report.flag == "MISSING"


def test_age_days_in_result(tmp_path):
    f = tmp_path / "data.json"
    f.write_text("{}")
    report = check_data_freshness({"data": f}, max_age_days=30.0)
    check = report.checks[0]
    assert check.age_days is not None
    assert check.age_days < 1.0
