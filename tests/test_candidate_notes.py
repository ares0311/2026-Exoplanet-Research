"""Tests for Skills.candidate_notes."""
from __future__ import annotations

from pathlib import Path

from Skills.candidate_notes import CandidateNotes


class TestCandidateNotes:
    def test_add_and_get(self, tmp_path: Path) -> None:
        notes = CandidateNotes(tmp_path / "notes.json")
        notes.add(1, "interesting signal")
        entries = notes.get(1)
        assert len(entries) == 1
        assert entries[0]["note"] == "interesting signal"

    def test_get_unknown_returns_empty(self, tmp_path: Path) -> None:
        notes = CandidateNotes(tmp_path / "notes.json")
        assert notes.get(999) == []

    def test_multiple_notes_per_tic(self, tmp_path: Path) -> None:
        notes = CandidateNotes(tmp_path / "notes.json")
        notes.add(1, "first")
        notes.add(1, "second")
        assert len(notes.get(1)) == 2

    def test_tag_stored(self, tmp_path: Path) -> None:
        notes = CandidateNotes(tmp_path / "notes.json")
        notes.add(1, "check centroid", tag="suspicious")
        assert notes.get(1)[0]["tag"] == "suspicious"

    def test_added_at_timestamp(self, tmp_path: Path) -> None:
        notes = CandidateNotes(tmp_path / "notes.json")
        notes.add(1, "test")
        ts = notes.get(1)[0]["added_at"]
        assert "T" in ts  # ISO format

    def test_remove_deletes_notes(self, tmp_path: Path) -> None:
        notes = CandidateNotes(tmp_path / "notes.json")
        notes.add(1, "a")
        notes.add(1, "b")
        n = notes.remove(1)
        assert n == 2
        assert notes.get(1) == []

    def test_remove_unknown_returns_zero(self, tmp_path: Path) -> None:
        notes = CandidateNotes(tmp_path / "notes.json")
        assert notes.remove(999) == 0

    def test_list_tic_ids(self, tmp_path: Path) -> None:
        notes = CandidateNotes(tmp_path / "notes.json")
        notes.add(3, "c")
        notes.add(1, "a")
        ids = notes.list_tic_ids()
        assert ids == [1, 3]

    def test_search_by_note_text(self, tmp_path: Path) -> None:
        notes = CandidateNotes(tmp_path / "notes.json")
        notes.add(1, "centroid offset detected")
        notes.add(2, "clean signal")
        results = notes.search("centroid")
        assert len(results) == 1
        assert results[0]["tic_id"] == 1

    def test_search_case_insensitive(self, tmp_path: Path) -> None:
        notes = CandidateNotes(tmp_path / "notes.json")
        notes.add(1, "Centroid Offset")
        results = notes.search("centroid")
        assert len(results) == 1

    def test_search_by_tag(self, tmp_path: Path) -> None:
        notes = CandidateNotes(tmp_path / "notes.json")
        notes.add(1, "note", tag="followup")
        results = notes.search("followup")
        assert len(results) == 1

    def test_summary_counts(self, tmp_path: Path) -> None:
        notes = CandidateNotes(tmp_path / "notes.json")
        notes.add(1, "a", tag="t1")
        notes.add(2, "b", tag="t1")
        s = notes.summary()
        assert s["n_targets"] == 2
        assert s["n_notes"] == 2
        assert s["tags"]["t1"] == 2

    def test_persistence(self, tmp_path: Path) -> None:
        p = tmp_path / "notes.json"
        notes = CandidateNotes(p)
        notes.add(7, "persisted")
        notes2 = CandidateNotes(p)
        assert len(notes2.get(7)) == 1
