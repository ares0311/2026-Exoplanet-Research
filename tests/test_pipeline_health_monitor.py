"""Tests for Skills/pipeline_health_monitor.py"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from pipeline_health_monitor import (
    check_pipeline_health,
    format_health_report,
)


def _write_labels(tmp_path: Path, n_pc: int, n_fp: int) -> Path:
    p = tmp_path / "labels.json"
    rows = ([{"label": "planet_candidate"}] * n_pc
            + [{"label": "false_positive"}] * n_fp)
    p.write_text(json.dumps(rows))
    return p


def _write_registry(tmp_path: Path) -> Path:
    p = tmp_path / "registry.json"
    p.write_text(json.dumps([{"model_id": "cnn_v1", "auc": 0.92}]))
    return p


def _write_calibration(tmp_path: Path) -> Path:
    p = tmp_path / "calibration.json"
    p.write_text(json.dumps({"a": 1.0, "b": 0.0}))
    return p


def test_all_none_is_warn():
    report = check_pipeline_health()
    assert report.overall in ("DEGRADED", "UNHEALTHY")


def test_healthy_with_all_ok(tmp_path):
    label_json = _write_labels(tmp_path, 3000, 2001)
    registry = _write_registry(tmp_path)
    calibration = _write_calibration(tmp_path)
    snippet_dir = tmp_path / "snippets"
    snippet_dir.mkdir()
    for i in range(6000):
        (snippet_dir / f"snip_{i}.json").write_text("{}")
    report = check_pipeline_health(
        label_json=label_json,
        snippet_dir=snippet_dir,
        registry_path=registry,
        calibration_path=calibration,
        min_labels=5000,
    )
    assert report.overall == "HEALTHY"


def test_labels_fail_when_too_few(tmp_path):
    label_json = _write_labels(tmp_path, 10, 10)
    report = check_pipeline_health(label_json=label_json, min_labels=5000)
    label_check = next(c for c in report.checks if c.name == "labels")
    assert label_check.status == "FAIL"


def test_labels_warn_when_partial(tmp_path):
    label_json = _write_labels(tmp_path, 1500, 1000)
    report = check_pipeline_health(label_json=label_json, min_labels=5000)
    label_check = next(c for c in report.checks if c.name == "labels")
    assert label_check.status == "WARN"


def test_labels_ok_when_enough(tmp_path):
    label_json = _write_labels(tmp_path, 3000, 2001)
    report = check_pipeline_health(label_json=label_json, min_labels=5000)
    label_check = next(c for c in report.checks if c.name == "labels")
    assert label_check.status == "OK"


def test_snippets_fail_missing_dir(tmp_path):
    report = check_pipeline_health(snippet_dir=tmp_path / "nonexistent")
    snip_check = next(c for c in report.checks if c.name == "snippets")
    assert snip_check.status == "FAIL"


def test_snippets_ok_enough_files(tmp_path):
    sd = tmp_path / "snippets"
    sd.mkdir()
    for i in range(6000):
        (sd / f"s_{i}.json").write_text("{}")
    report = check_pipeline_health(snippet_dir=sd, min_labels=5000)
    snip_check = next(c for c in report.checks if c.name == "snippets")
    assert snip_check.status == "OK"


def test_registry_fail_missing(tmp_path):
    report = check_pipeline_health(registry_path=tmp_path / "nofile.json")
    reg_check = next(c for c in report.checks if c.name == "registry")
    assert reg_check.status == "FAIL"


def test_registry_ok_fresh(tmp_path):
    reg = _write_registry(tmp_path)
    report = check_pipeline_health(registry_path=reg, max_model_age_days=30.0)
    reg_check = next(c for c in report.checks if c.name == "registry")
    assert reg_check.status == "OK"


def test_calibration_ok_fresh(tmp_path):
    cal = _write_calibration(tmp_path)
    report = check_pipeline_health(calibration_path=cal, max_model_age_days=30.0)
    cal_check = next(c for c in report.checks if c.name == "calibration")
    assert cal_check.status == "OK"


def test_calibration_warn_stale(tmp_path):
    cal = _write_calibration(tmp_path)
    old_mtime = time.time() - 40 * 86400
    import os
    os.utime(cal, (old_mtime, old_mtime))
    report = check_pipeline_health(calibration_path=cal, max_model_age_days=30.0)
    cal_check = next(c for c in report.checks if c.name == "calibration")
    assert cal_check.status == "WARN"


def test_n_ok_n_fail_counts(tmp_path):
    label_json = _write_labels(tmp_path, 10, 10)
    report = check_pipeline_health(label_json=label_json)
    assert report.n_fail >= 1


def test_overall_unhealthy_on_fail(tmp_path):
    report = check_pipeline_health(
        registry_path=tmp_path / "nofile.json",
    )
    assert report.overall == "UNHEALTHY"


def test_format_health_report_contains_overall(tmp_path):
    report = check_pipeline_health()
    text = format_health_report(report)
    assert "DEGRADED" in text or "UNHEALTHY" in text or "HEALTHY" in text


def test_format_has_table(tmp_path):
    report = check_pipeline_health()
    text = format_health_report(report)
    assert "|" in text
