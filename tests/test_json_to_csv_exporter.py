"""Tests for Skills/json_to_csv_exporter.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from json_to_csv_exporter import (
    export_to_csv,
    flatten_candidate,
    format_export_result,
    load_and_export,
)


def _make_row(**kwargs):
    base = {
        "tic_id": 12345678,
        "period_days": 5.0,
        "pathway": "planet_hunters_discussion",
        "provenance_score": 0.85,
        "scores": {
            "false_positive_probability": 0.15,
            "detection_confidence": 0.90,
        },
        "posterior": {
            "planet_candidate": 0.65,
            "eclipsing_binary": 0.10,
        },
        "meta": {
            "toolkit_version": "0.1.0",
            "run_at": "2026-05-24T00:00:00Z",
            "scorer": "bayesian",
            "git_commit": "abc1234",
        },
    }
    base.update(kwargs)
    return base


def test_flatten_top_level_fields():
    flat = flatten_candidate(_make_row())
    assert flat["tic_id"] == 12345678
    assert flat["period_days"] == 5.0


def test_flatten_posterior_prefix():
    flat = flatten_candidate(_make_row())
    assert "posterior_planet_candidate" in flat
    assert abs(flat["posterior_planet_candidate"] - 0.65) < 1e-9


def test_flatten_scores_prefix():
    flat = flatten_candidate(_make_row())
    assert "scores_false_positive_probability" in flat
    assert abs(flat["scores_false_positive_probability"] - 0.15) < 1e-9


def test_flatten_meta_prefix():
    flat = flatten_candidate(_make_row())
    assert "meta_toolkit_version" in flat
    assert flat["meta_toolkit_version"] == "0.1.0"


def test_export_ok_single_row():
    r = export_to_csv([_make_row()])
    assert r.flag == "OK"
    assert r.n_rows == 1
    assert r.n_columns > 0


def test_export_empty_list():
    r = export_to_csv([])
    assert r.flag == "EMPTY"
    assert r.n_rows == 0


def test_export_invalid_not_list():
    r = export_to_csv("not a list")
    assert r.flag == "INVALID"


def test_export_to_file(tmp_path):
    out = tmp_path / "output.csv"
    r = export_to_csv([_make_row()], output_path=out)
    assert r.flag == "OK"
    assert r.output_path is not None
    content = out.read_text()
    assert "tic_id" in content
    assert "12345678" in content


def test_columns_tuple():
    r = export_to_csv([_make_row()])
    assert isinstance(r.columns, tuple)
    assert len(r.columns) == r.n_columns


def test_multiple_rows():
    rows = [_make_row(), _make_row(tic_id=99999, period_days=3.0)]
    r = export_to_csv(rows)
    assert r.n_rows == 2


def test_load_and_export_json_file(tmp_path):
    json_file = tmp_path / "data.json"
    json_file.write_text(json.dumps([_make_row()]))
    r = load_and_export(json_file)
    assert r.flag == "OK"
    assert r.n_rows == 1


def test_load_and_export_invalid_path(tmp_path):
    r = load_and_export(tmp_path / "nonexistent.json")
    assert r.flag == "INVALID"


def test_format_contains_keywords():
    r = export_to_csv([_make_row()])
    text = format_export_result(r)
    assert "CSV Exporter" in text
    assert "Rows exported" in text
    assert "OK" in text


def test_format_empty():
    r = export_to_csv([])
    text = format_export_result(r)
    assert "EMPTY" in text
