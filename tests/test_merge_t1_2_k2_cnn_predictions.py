"""Tests for Skills/merge_t1_2_k2_cnn_predictions.py (offline only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Skills"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from merge_t1_2_k2_cnn_predictions import (  # noqa: E402
    filter_complete_rows,
    format_merge_summary,
    load_snippets_by_epic,
    merge_cnn_predictions,
    write_completed_predictions,
)

from exo_toolkit.ml.cnn_scorer import CnnScorer  # noqa: E402


def _partial_row(epic_id: int, label: int) -> dict:
    return {
        "epic_id": epic_id,
        "label": label,
        "xgb_prob": 0.3,
        "bayes_prob": 0.4,
        "cnn_prob": None,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _snippet_row(epic_id: int, label: int) -> dict:
    return {
        "epic_id": epic_id,
        "label": label,
        "flux": [0.1] * 201,
        "period_days": 10.0,
        "epoch_bjd": 2457100.0,
        "n_bins": 201,
        "source": "k2_native_calibration",
    }


def test_load_snippets_by_epic(tmp_path: Path) -> None:
    snippets_path = tmp_path / "snippets.jsonl"
    _write_jsonl(snippets_path, [_snippet_row(1, 1), _snippet_row(2, 0)])
    by_epic = load_snippets_by_epic(snippets_path)
    assert set(by_epic) == {1, 2}
    assert len(by_epic[1]) == 201


def test_load_snippets_by_epic_missing_file(tmp_path: Path) -> None:
    assert load_snippets_by_epic(tmp_path / "does_not_exist.jsonl") == {}


def test_merge_cnn_predictions_fills_in_available_snippets(tmp_path: Path) -> None:
    partial_path = tmp_path / "partial.jsonl"
    snippets_path = tmp_path / "snippets.jsonl"
    _write_jsonl(
        partial_path,
        [_partial_row(1, 1), _partial_row(2, 0), _partial_row(3, 1)],
    )
    # Only EPICs 1 and 2 have a fetched snippet -- 3 simulates a fetch failure.
    _write_jsonl(snippets_path, [_snippet_row(1, 1), _snippet_row(2, 0)])

    scorer = CnnScorer(model_fn=lambda snippet: 0.77)
    rows = merge_cnn_predictions(
        partial_path, snippets_path, training_mission="Kepler",
        target_mission="K2", cnn_scorer=scorer,
    )

    by_epic = {r["epic_id"]: r for r in rows}
    assert by_epic[1]["cnn_prob"] == 0.77
    assert by_epic[2]["cnn_prob"] == 0.77
    assert by_epic[3]["cnn_prob"] is None
    # Original xgb_prob/bayes_prob/label fields must survive untouched.
    assert by_epic[1]["xgb_prob"] == 0.3
    assert by_epic[1]["label"] == 1


def test_merge_cnn_predictions_annotates_cross_mission(tmp_path: Path) -> None:
    partial_path = tmp_path / "partial.jsonl"
    snippets_path = tmp_path / "snippets.jsonl"
    _write_jsonl(partial_path, [_partial_row(1, 1)])
    _write_jsonl(snippets_path, [_snippet_row(1, 1)])

    scorer = CnnScorer(model_fn=lambda snippet: 0.5)
    rows = merge_cnn_predictions(
        partial_path, snippets_path, training_mission="Kepler",
        target_mission="K2", cnn_scorer=scorer,
    )
    assert rows[0]["cnn_training_mission"] == "Kepler"
    assert rows[0]["cnn_cross_mission"] is True


def test_merge_cnn_predictions_same_mission_not_flagged(tmp_path: Path) -> None:
    partial_path = tmp_path / "partial.jsonl"
    snippets_path = tmp_path / "snippets.jsonl"
    _write_jsonl(partial_path, [_partial_row(1, 1)])
    _write_jsonl(snippets_path, [_snippet_row(1, 1)])

    scorer = CnnScorer(model_fn=lambda snippet: 0.5)
    rows = merge_cnn_predictions(
        partial_path, snippets_path, training_mission="K2",
        target_mission="K2", cnn_scorer=scorer,
    )
    assert rows[0]["cnn_cross_mission"] is False


def test_filter_complete_rows_drops_null_cnn_prob() -> None:
    rows = [
        {"epic_id": 1, "cnn_prob": 0.5},
        {"epic_id": 2, "cnn_prob": None},
        {"epic_id": 3, "cnn_prob": 0.9},
    ]
    filtered = filter_complete_rows(rows)
    assert [r["epic_id"] for r in filtered] == [1, 3]


def test_write_completed_predictions_roundtrip(tmp_path: Path) -> None:
    rows = [{"epic_id": 1, "cnn_prob": 0.5, "label": 1}]
    output_path = tmp_path / "out.jsonl"
    write_completed_predictions(rows, output_path)
    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["epic_id"] == 1


def test_format_merge_summary_reports_missing_and_cross_mission() -> None:
    rows = [
        {"epic_id": 1, "cnn_prob": 0.5},
        {"epic_id": 2, "cnn_prob": None},
    ]
    summary = format_merge_summary(rows, training_mission="Kepler", target_mission="K2")
    assert "Rows total**: 2" in summary
    assert "Scored with CNN (snippet found)**: 1" in summary
    assert "Dropped" in summary
    assert "Cross-mission note" in summary
    assert "Kepler" in summary and "K2" in summary


def test_format_merge_summary_no_cross_mission_note_when_same_mission() -> None:
    rows = [{"epic_id": 1, "cnn_prob": 0.5}]
    summary = format_merge_summary(rows, training_mission="K2", target_mission="K2")
    assert "Cross-mission note" not in summary


def test_merge_cnn_predictions_empty_snippets_leaves_all_null(tmp_path: Path) -> None:
    partial_path = tmp_path / "partial.jsonl"
    snippets_path = tmp_path / "snippets.jsonl"
    _write_jsonl(partial_path, [_partial_row(1, 1), _partial_row(2, 0)])
    _write_jsonl(snippets_path, [])

    scorer = CnnScorer(model_fn=lambda snippet: 0.9)
    rows = merge_cnn_predictions(
        partial_path, snippets_path, training_mission="Kepler",
        target_mission="K2", cnn_scorer=scorer,
    )
    assert all(r["cnn_prob"] is None for r in rows)
    assert filter_complete_rows(rows) == []
