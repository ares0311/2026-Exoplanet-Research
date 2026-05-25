"""Tests for Skills/prediction_batch_exporter.py"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

import pytest

from prediction_batch_exporter import (
    ExportResult,
    PredictionRow,
    export_predictions,
    format_export_result,
    load_predictions,
    make_row,
)


def test_make_row_score_above_threshold_positive():
    row = make_row("TIC1", "xgb", 0.8, threshold=0.5)
    assert row.threshold_decision == "POSITIVE"


def test_make_row_score_below_threshold_negative():
    row = make_row("TIC1", "xgb", 0.3, threshold=0.5)
    assert row.threshold_decision == "NEGATIVE"


def test_make_row_score_equal_threshold_positive():
    row = make_row("TIC1", "xgb", 0.5, threshold=0.5)
    assert row.threshold_decision == "POSITIVE"


def test_export_creates_csv():
    rows = [make_row("TIC1", "xgb", 0.9), make_row("TIC2", "bayes", 0.2)]
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "preds.csv"
        result = export_predictions(rows, out)
        assert out.exists()
        assert result.flag == "OK"


def test_csv_has_correct_header():
    rows = [make_row("TIC1", "xgb", 0.9)]
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "preds.csv"
        export_predictions(rows, out)
        text = out.read_text()
        assert text.startswith("tic_id,model,score,label,threshold_decision")


def test_load_predictions_round_trips():
    rows = [
        make_row("TIC1", "xgb", 0.9, label=1),
        make_row("TIC2", "bayes", 0.2, label=0),
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "preds.csv"
        export_predictions(rows, out)
        loaded = load_predictions(out)
        assert len(loaded) == 2
        assert loaded[0].tic_id == "TIC1"
        assert loaded[0].label == 1


def test_empty_list_returns_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "preds.csv"
        result = export_predictions([], out)
        assert result.flag == "EMPTY"


def test_n_positive_plus_n_negative_equals_n_rows():
    rows = [make_row(f"TIC{i}", "m", 0.6 if i % 2 == 0 else 0.4) for i in range(10)]
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "preds.csv"
        result = export_predictions(rows, out)
        assert result.n_positive_decisions + result.n_negative_decisions == result.n_rows


def test_prediction_row_frozen():
    row = make_row("TIC1", "xgb", 0.9)
    try:
        row.score = 0.0  # type: ignore[misc]
        raise AssertionError("Should have raised FrozenInstanceError")
    except Exception as exc:
        assert "frozen" in str(exc).lower() or "FrozenInstance" in type(exc).__name__


def test_export_result_frozen():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "preds.csv"
        result = export_predictions([make_row("TIC1", "xgb", 0.9)], out)
        try:
            result.flag = "MODIFIED"  # type: ignore[misc]
            raise AssertionError("Should have raised FrozenInstanceError")
        except Exception as exc:
            assert "frozen" in str(exc).lower() or "FrozenInstance" in type(exc).__name__


def test_label_none_written_as_empty_in_csv():
    rows = [make_row("TIC1", "xgb", 0.9, label=None)]
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "preds.csv"
        export_predictions(rows, out)
        text = out.read_text()
        # label column should be empty (two consecutive commas)
        assert "TIC1,xgb,0.9,,POSITIVE" in text or ",," in text


def test_format_returns_str():
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "preds.csv"
        result = export_predictions([make_row("TIC1", "xgb", 0.9)], out)
        md = format_export_result(result)
        assert isinstance(md, str)


def test_load_missing_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_predictions(Path("/nonexistent/path/preds.csv"))


def test_label_none_round_trips():
    rows = [make_row("TIC99", "bayes", 0.5, label=None)]
    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "preds.csv"
        export_predictions(rows, out)
        loaded = load_predictions(out)
        assert loaded[0].label is None
