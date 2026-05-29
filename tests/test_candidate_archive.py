"""Tests for Skills/candidate_archive.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from candidate_archive import (
    ArchiveRecord,
    CandidateArchive,
    format_archive_record,
)


def _archive(tmp_path):
    return CandidateArchive(tmp_path / "archive.json")


def _row(tic_id=12345, period=3.0, fpp=0.05, pathway="tfop_ready"):
    return {
        "tic_id": tic_id,
        "period_days": period,
        "false_positive_probability": fpp,
        "pathway": pathway,
        "meta": {"scorer": "bayesian"},
        "mission": "TESS",
    }


def test_insert_returns_archive_record(tmp_path):
    arch = _archive(tmp_path)
    rec = arch.insert(_row())
    assert isinstance(rec, ArchiveRecord)


def test_latest_returns_most_recent(tmp_path):
    arch = _archive(tmp_path)
    arch.insert(_row(fpp=0.40))
    arch.insert(_row(fpp=0.05))
    rec = arch.latest(12345, 3.0)
    assert rec is not None
    assert rec.fpp == 0.05


def test_latest_none_if_not_found(tmp_path):
    arch = _archive(tmp_path)
    assert arch.latest(99999, 1.0) is None


def test_history_returns_all_entries(tmp_path):
    arch = _archive(tmp_path)
    arch.insert(_row())
    arch.insert(_row())
    history = arch.history(12345, 3.0)
    assert len(history) == 2


def test_search_by_fpp_max(tmp_path):
    arch = _archive(tmp_path)
    arch.insert(_row(fpp=0.05))
    arch.insert(_row(tic_id=99999, period=5.0, fpp=0.80))
    results = arch.search(fpp_max=0.20)
    assert len(results) == 1
    assert results[0].tic_id == 12345


def test_search_by_pathway(tmp_path):
    arch = _archive(tmp_path)
    arch.insert(_row(pathway="tfop_ready"))
    arch.insert(_row(tic_id=2, period=5.0, pathway="github_only_reproducibility"))
    results = arch.search(pathway="tfop_ready")
    assert all(r.pathway == "tfop_ready" for r in results)


def test_search_by_mission(tmp_path):
    arch = _archive(tmp_path)
    arch.insert(_row())  # TESS
    results = arch.search(mission="TESS")
    assert len(results) == 1


def test_all_latest(tmp_path):
    arch = _archive(tmp_path)
    arch.insert(_row(tic_id=1))
    arch.insert(_row(tic_id=2))
    assert len(arch.all_latest()) == 2


def test_export_csv(tmp_path):
    arch = _archive(tmp_path)
    arch.insert(_row())
    csv_path = tmp_path / "out.csv"
    result = arch.export_csv(csv_path)
    assert result.exists()
    content = result.read_text()
    assert "tic_id" in content


def test_persistence(tmp_path):
    path = tmp_path / "arch.json"
    a1 = CandidateArchive(path)
    a1.insert(_row())
    a2 = CandidateArchive(path)
    assert len(a2.all_latest()) == 1


def test_note_stored(tmp_path):
    arch = _archive(tmp_path)
    arch.insert(_row(), note="good target")
    rec = arch.latest(12345, 3.0)
    assert rec is not None
    assert rec.note == "good target"


def test_format_contains_tic(tmp_path):
    arch = _archive(tmp_path)
    arch.insert(_row(tic_id=55555))
    rec = arch.latest(55555, 3.0)
    md = format_archive_record(rec)
    assert "55555" in md


def test_scores_dict_fpp(tmp_path):
    arch = _archive(tmp_path)
    row = {"tic_id": 1, "period_days": 2.0, "scores": {"false_positive_probability": 0.12}}
    rec = arch.insert(row)
    assert rec.fpp == 0.12
