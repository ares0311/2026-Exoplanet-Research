"""Tests for Skills/candidate_label_exporter.py."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from candidate_label_exporter import (
    LabelExportRow,
    export_for_labeling,
    format_export_result,
    load_labeled,
)


def _row(tic_id=100, fpp=0.05, pathway="tfop_ready"):
    return {
        "tic_id": tic_id,
        "period_days": 3.0,
        "depth_ppm": 500.0,
        "false_positive_probability": fpp,
        "pathway": pathway,
    }


def test_export_creates_file(tmp_path):
    out = tmp_path / "labels.json"
    result = export_for_labeling([_row()], out)
    assert out.exists()
    assert result.flag == "OK"


def test_export_count(tmp_path):
    rows = [_row(i) for i in range(5)]
    result = export_for_labeling(rows, tmp_path / "out.json")
    assert result.n_exported == 5


def test_suggested_pc_low_fpp(tmp_path):
    result = export_for_labeling([_row(fpp=0.05)], tmp_path / "out.json")
    assert result.n_suggested_pc == 1


def test_suggested_fp_high_fpp(tmp_path):
    result = export_for_labeling([_row(fpp=0.80)], tmp_path / "out.json")
    assert result.n_suggested_fp == 1


def test_suggested_unknown_mid_fpp(tmp_path):
    result = export_for_labeling([_row(fpp=0.35)], tmp_path / "out.json")
    assert result.n_suggested_unknown == 1


def test_fpp_threshold_filters(tmp_path):
    rows = [_row(fpp=0.05), _row(tic_id=200, fpp=0.90)]
    result = export_for_labeling(rows, tmp_path / "out.json", fpp_threshold=0.50)
    assert result.n_exported == 1


def test_empty_input_flag(tmp_path):
    result = export_for_labeling([], tmp_path / "out.json")
    assert result.flag == "EMPTY"


def test_overwrite_raises_if_exists(tmp_path):
    out = tmp_path / "out.json"
    export_for_labeling([_row()], out)
    import pytest
    with pytest.raises(FileExistsError):
        export_for_labeling([_row()], out, overwrite=False)


def test_overwrite_succeeds(tmp_path):
    out = tmp_path / "out.json"
    export_for_labeling([_row()], out)
    result = export_for_labeling([_row()], out, overwrite=True)
    assert result.flag == "OK"


def test_load_labeled_roundtrip(tmp_path):
    out = tmp_path / "out.json"
    export_for_labeling([_row()], out)
    rows = load_labeled(out)
    assert len(rows) == 1
    assert isinstance(rows[0], LabelExportRow)


def test_load_labeled_label_field(tmp_path):
    out = tmp_path / "out.json"
    export_for_labeling([_row()], out)
    data = json.loads(out.read_text())
    data[0]["label"] = "planet_candidate"
    out.write_text(json.dumps(data))
    rows = load_labeled(out)
    assert rows[0].label == "planet_candidate"


def test_format_contains_exported_count(tmp_path):
    result = export_for_labeling([_row()], tmp_path / "out.json")
    md = format_export_result(result)
    assert "1" in md
