"""Tests for Skills/pipeline_config_validator.py."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from pipeline_config_validator import (
    ConfigIssue,
    format_validation_result,
    load_and_validate,
    validate_pipeline_config,
)


def _valid_config():
    return {"min_snr": 7.0, "max_peaks": 5, "mission": "TESS"}


def test_valid_config_flag():
    result = validate_pipeline_config(_valid_config())
    assert result.flag == "VALID"
    assert result.n_errors == 0


def test_missing_required_key():
    result = validate_pipeline_config({}, required_keys=["min_snr"])
    assert result.flag == "INVALID"
    assert result.n_errors == 1


def test_multiple_missing_keys():
    result = validate_pipeline_config({}, required_keys=["a", "b", "c"])
    assert result.n_errors == 3


def test_path_key_warning_for_missing_path(tmp_path):
    cfg = {"data_dir": str(tmp_path / "nonexistent")}
    result = validate_pipeline_config(cfg, path_keys=["data_dir"])
    assert result.n_warnings == 1
    assert result.flag == "WARNINGS"


def test_path_key_no_warning_for_existing_path(tmp_path):
    cfg = {"data_dir": str(tmp_path)}
    result = validate_pipeline_config(cfg, path_keys=["data_dir"])
    assert result.n_warnings == 0


def test_numeric_range_in_bounds():
    cfg = {"min_snr": 7.0}
    result = validate_pipeline_config(cfg, numeric_ranges={"min_snr": (1.0, 50.0)})
    assert result.flag == "VALID"


def test_numeric_range_out_of_bounds():
    cfg = {"min_snr": 100.0}
    result = validate_pipeline_config(cfg, numeric_ranges={"min_snr": (1.0, 50.0)})
    assert result.flag == "INVALID"
    assert result.n_errors == 1


def test_numeric_range_non_numeric():
    cfg = {"min_snr": "bad"}
    result = validate_pipeline_config(cfg, numeric_ranges={"min_snr": (1.0, 50.0)})
    assert result.flag == "INVALID"


def test_load_and_validate_file(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"min_snr": 7.0}))
    result = load_and_validate(p, required_keys=["min_snr"])
    assert result.flag == "VALID"


def test_load_and_validate_missing_file(tmp_path):
    result = load_and_validate(tmp_path / "missing.json")
    assert result.flag == "INVALID"
    assert result.n_errors == 1


def test_load_and_validate_bad_json(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("not json {{{")
    result = load_and_validate(p)
    assert result.flag == "INVALID"


def test_issues_tuple():
    result = validate_pipeline_config({}, required_keys=["x"])
    assert isinstance(result.issues, tuple)
    assert isinstance(result.issues[0], ConfigIssue)


def test_format_contains_flag():
    result = validate_pipeline_config(_valid_config())
    md = format_validation_result(result)
    assert "VALID" in md


def test_format_no_issues():
    result = validate_pipeline_config(_valid_config())
    md = format_validation_result(result)
    assert "No issues" in md
