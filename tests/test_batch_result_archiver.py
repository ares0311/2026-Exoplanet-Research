"""Tests for Skills/batch_result_archiver.py"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from batch_result_archiver import (
    archive_batch_results,
    format_archive_result,
)


def _make_files(tmp_path: Path, n: int) -> list[Path]:
    files = []
    for i in range(n):
        p = tmp_path / f"result_{i}.json"
        p.write_text(json.dumps({"tic_id": i}))
        files.append(p)
    return files


def test_empty_input(tmp_path):
    result = archive_batch_results([], tmp_path / "archive")
    assert result.flag == "EMPTY"


def test_archive_ok(tmp_path):
    files = _make_files(tmp_path, 3)
    result = archive_batch_results(files, tmp_path / "archive")
    assert result.flag == "OK"
    assert result.n_archived == 3


def test_files_exist_in_archive(tmp_path):
    files = _make_files(tmp_path, 2)
    result = archive_batch_results(files, tmp_path / "archive")
    for f in files:
        assert Path(result.archive_dir, f.name).exists()


def test_manifest_written(tmp_path):
    files = _make_files(tmp_path, 2)
    result = archive_batch_results(files, tmp_path / "archive", write_manifest=True)
    assert result.manifest_path is not None
    assert Path(result.manifest_path).exists()


def test_manifest_contains_file_count(tmp_path):
    files = _make_files(tmp_path, 3)
    result = archive_batch_results(files, tmp_path / "archive")
    manifest = json.loads(Path(result.manifest_path).read_text())
    assert manifest["n_files"] == 3


def test_missing_source_counted_as_failed(tmp_path):
    files = [tmp_path / "nonexistent.json"]
    result = archive_batch_results(files, tmp_path / "archive")
    assert result.n_failed == 1


def test_partial_flag_on_mixed(tmp_path):
    files = _make_files(tmp_path, 2)
    files.append(tmp_path / "missing.json")
    result = archive_batch_results(files, tmp_path / "archive")
    assert result.flag == "PARTIAL"


def test_session_label_in_dir(tmp_path):
    files = _make_files(tmp_path, 1)
    result = archive_batch_results(files, tmp_path / "archive", session_label="run1")
    assert "run1" in result.archive_dir


def test_total_bytes_positive(tmp_path):
    files = _make_files(tmp_path, 2)
    result = archive_batch_results(files, tmp_path / "archive")
    assert result.total_bytes > 0


def test_no_manifest(tmp_path):
    files = _make_files(tmp_path, 1)
    result = archive_batch_results(files, tmp_path / "archive", write_manifest=False)
    assert result.manifest_path is None


def test_format_returns_string(tmp_path):
    files = _make_files(tmp_path, 2)
    result = archive_batch_results(files, tmp_path / "archive")
    text = format_archive_result(result)
    assert isinstance(text, str)
    assert "Archiver" in text


def test_archive_dir_created(tmp_path):
    files = _make_files(tmp_path, 1)
    archive_root = tmp_path / "deep" / "archive"
    result = archive_batch_results(files, archive_root)
    assert Path(result.archive_dir).exists()
