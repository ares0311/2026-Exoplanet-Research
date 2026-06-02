"""Tests for Skills/tier2_progress_reporter.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from tier2_progress_reporter import (
    _cli,
    build_tier2_status,
    format_tier2_report,
    status_to_dict,
    write_status_outputs,
)


def _make_labels(tmp_path: Path, n_pc: int, n_fp: int) -> Path:
    rows = [{"label": "planet_candidate"}] * n_pc + [{"label": "false_positive"}] * n_fp
    p = tmp_path / "labels.json"
    p.write_text(json.dumps(rows))
    return p


def _make_checkpoint(tmp_path: Path) -> Path:
    p = tmp_path / "model.json"
    p.write_text(json.dumps({"epoch": 20, "val_auc": 0.92}))
    return p


def _make_calibration(tmp_path: Path) -> Path:
    p = tmp_path / "calibration.json"
    p.write_text(json.dumps({"a": 1.0, "b": 0.0}))
    return p


def _make_registry(tmp_path: Path) -> Path:
    p = tmp_path / "registry.json"
    p.write_text(json.dumps([{"model_type": "cnn", "model_id": "cnn_v1"}]))
    return p


def test_blocked_when_insufficient_labels(tmp_path):
    s = build_tier2_status(label_json=_make_labels(tmp_path, 100, 100), min_labels=5000)
    assert s.flag == "BLOCKED"


def test_gate_passed_when_enough_labels(tmp_path):
    s = build_tier2_status(label_json=_make_labels(tmp_path, 3000, 2001), min_labels=5000)
    assert s.gate_passed


def test_in_progress_gate_but_no_checkpoint(tmp_path):
    s = build_tier2_status(label_json=_make_labels(tmp_path, 3000, 2001), min_labels=5000)
    assert s.flag == "IN_PROGRESS"


def test_ready_all_complete(tmp_path):
    s = build_tier2_status(
        label_json=_make_labels(tmp_path, 3000, 2001),
        checkpoint_path=_make_checkpoint(tmp_path),
        calibration_path=_make_calibration(tmp_path),
        registry_path=_make_registry(tmp_path),
        min_labels=5000,
    )
    assert s.flag == "READY"


def test_training_complete_with_checkpoint(tmp_path):
    s = build_tier2_status(checkpoint_path=_make_checkpoint(tmp_path))
    assert s.training_complete


def test_calibrated_with_file(tmp_path):
    s = build_tier2_status(calibration_path=_make_calibration(tmp_path))
    assert s.calibrated


def test_registered_with_registry(tmp_path):
    s = build_tier2_status(registry_path=_make_registry(tmp_path))
    assert s.registered


def test_no_labels_gives_zero(tmp_path):
    s = build_tier2_status()
    assert s.n_labels == 0


def test_next_actions_nonempty():
    s = build_tier2_status()
    assert len(s.next_actions) > 0


def test_format_returns_string():
    s = build_tier2_status()
    text = format_tier2_report(s)
    assert isinstance(text, str)
    assert "Tier 2" in text


def test_format_shows_status():
    s = build_tier2_status()
    text = format_tier2_report(s)
    assert "BLOCKED" in text or "IN PROGRESS" in text or "READY" in text


def test_result_frozen():
    s = build_tier2_status()
    try:
        s.n_labels = 9999  # type: ignore[misc]
        raise AssertionError("Should be frozen")
    except Exception:
        pass


def test_numeric_label_rows_count_for_ctoi_contract(tmp_path):
    rows = [{"label": 1}, {"label": 0}, {"label": "pc"}, {"label": None}]
    p = tmp_path / "ctoi_labels.json"
    p.write_text(json.dumps(rows))
    s = build_tier2_status(label_json=p, min_labels=2)
    assert s.n_labels == 2
    assert s.gate_passed


def test_wrapped_label_rows_count(tmp_path):
    p = tmp_path / "labels.json"
    p.write_text(json.dumps({"rows": [{"label": "CP"}, {"label": "EB"}, {"label": "PC"}]}))
    s = build_tier2_status(label_json=p, min_labels=2)
    assert s.n_labels == 2
    assert s.flag == "IN_PROGRESS"


def test_status_to_dict_contains_next_actions():
    d = status_to_dict(build_tier2_status())
    assert d["flag"] == "BLOCKED"
    assert isinstance(d["next_actions"], list)


def test_write_status_outputs(tmp_path):
    s = build_tier2_status()
    md = tmp_path / "tier2.md"
    js = tmp_path / "tier2.json"
    written = write_status_outputs(s, markdown_path=md, json_path=js)
    assert written == (md, js)
    assert "Tier 2" in md.read_text()
    assert json.loads(js.read_text())["flag"] == "BLOCKED"


def test_cli_writes_outputs(tmp_path):
    labels = _make_labels(tmp_path, 3, 2)
    md = tmp_path / "report.md"
    js = tmp_path / "status.json"
    code = _cli([
        "--labels", str(labels),
        "--min-labels", "5",
        "--output", str(md),
        "--json-output", str(js),
    ])
    assert code == 0
    assert md.exists()
    assert json.loads(js.read_text())["gate_passed"] is True
