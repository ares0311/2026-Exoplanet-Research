"""Tests for Skills/deployment_readiness_checker.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from deployment_readiness_checker import (
    ReadinessReport,
    check_deployment_readiness,
    format_readiness_report,
)


def _make_label_file(tmp_path: Path, n_pc: int, n_fp: int) -> Path:
    rows = [{"label": "planet_candidate"}] * n_pc + [{"label": "false_positive"}] * n_fp
    p = tmp_path / "labels.json"
    p.write_text(json.dumps(rows))
    return p


def _make_split_dir(tmp_path: Path) -> Path:
    sd = tmp_path / "splits"
    sd.mkdir()
    for name in ("train.json", "val.json", "test.json"):
        (sd / name).write_text("[]")
    return sd


def _make_checkpoint(tmp_path: Path) -> Path:
    p = tmp_path / "model.json"
    p.write_text(json.dumps({"epoch": 20, "val_auc": 0.92}))
    return p


def _make_calibration(tmp_path: Path) -> Path:
    p = tmp_path / "calibration.json"
    p.write_text(json.dumps({"a": 1.0, "b": 0.0}))
    return p


def _make_registry(tmp_path: Path, has_cnn: bool = True) -> Path:
    entries = [{"model_type": "cnn", "model_id": "cnn_v1"}] if has_cnn else []
    p = tmp_path / "registry.json"
    p.write_text(json.dumps(entries))
    return p


def test_all_pass(tmp_path):
    r = check_deployment_readiness(
        label_json=_make_label_file(tmp_path, 3000, 2001),
        split_dir=_make_split_dir(tmp_path),
        checkpoint_path=_make_checkpoint(tmp_path),
        calibration_path=_make_calibration(tmp_path),
        registry_path=_make_registry(tmp_path),
        min_labels=5000,
    )
    assert r.flag == "READY"
    assert r.ready


def test_label_count_fail(tmp_path):
    r = check_deployment_readiness(
        label_json=_make_label_file(tmp_path, 100, 100),
        min_labels=5000,
    )
    lc = next(c for c in r.checks if c.name == "label_count")
    assert not lc.passed


def test_missing_splits(tmp_path):
    r = check_deployment_readiness(
        split_dir=tmp_path / "no_such_dir",
        min_labels=0,
    )
    sc = next(c for c in r.checks if c.name == "split_dir")
    assert not sc.passed


def test_no_args_all_fail():
    r = check_deployment_readiness()
    assert r.n_failed == 5
    assert not r.ready


def test_five_checks():
    r = check_deployment_readiness()
    assert len(r.checks) == 5


def test_n_passed_n_failed_sum():
    r = check_deployment_readiness()
    assert r.n_passed + r.n_failed == len(r.checks)


def test_no_cnn_in_registry(tmp_path):
    r = check_deployment_readiness(
        registry_path=_make_registry(tmp_path, has_cnn=False),
        min_labels=0,
    )
    rc = next(c for c in r.checks if c.name == "registry_cnn")
    assert not rc.passed


def test_format_returns_string():
    r = check_deployment_readiness()
    s = format_readiness_report(r)
    assert isinstance(s, str)
    assert "Readiness" in s


def test_format_shows_fail():
    r = check_deployment_readiness()
    s = format_readiness_report(r)
    assert "FAIL" in s


def test_result_frozen():
    r = check_deployment_readiness()
    try:
        r.ready = True  # type: ignore[misc]
        raise AssertionError("Should be frozen")
    except Exception:
        pass


def test_label_file_not_found(tmp_path):
    r = check_deployment_readiness(
        label_json=tmp_path / "nonexistent.json",
        min_labels=0,
    )
    lc = next(c for c in r.checks if c.name == "label_count")
    assert not lc.passed
