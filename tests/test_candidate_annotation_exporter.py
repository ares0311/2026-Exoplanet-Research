"""Tests for candidate_annotation_exporter.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from candidate_annotation_exporter import (
    Annotation,
    AnnotationStore,
    export_annotations_csv,
    export_annotations_json,
)


def _make_ann(tic_id=12345, category="vetting", note="ok"):
    return Annotation(tic_id=tic_id, category=category, note=note)


class TestAnnotation:
    def test_frozen(self):
        a = _make_ann()
        try:
            a.note = "changed"  # type: ignore[misc]
            raise AssertionError()
        except (AttributeError, TypeError):
            pass

    def test_default_author(self):
        a = _make_ann()
        assert a.author == "auto"


class TestAnnotationStore:
    def test_add_and_get(self, tmp_path):
        store = AnnotationStore(tmp_path / "ann.json")
        a = _make_ann()
        store.add(a)
        results = store.get(12345)
        assert len(results) == 1
        assert results[0].note == "ok"

    def test_all_returns_all(self, tmp_path):
        store = AnnotationStore(tmp_path / "ann.json")
        store.add(_make_ann(tic_id=1, note="a"))
        store.add(_make_ann(tic_id=2, note="b"))
        assert len(store.all()) == 2

    def test_get_empty_for_missing_tic(self, tmp_path):
        store = AnnotationStore(tmp_path / "ann.json")
        store.add(_make_ann(tic_id=1))
        assert store.get(9999) == []

    def test_remove_existing(self, tmp_path):
        store = AnnotationStore(tmp_path / "ann.json")
        store.add(_make_ann(tic_id=1, note="first"))
        assert store.remove(1, 0)
        assert store.get(1) == []

    def test_remove_invalid_index(self, tmp_path):
        store = AnnotationStore(tmp_path / "ann.json")
        store.add(_make_ann(tic_id=1))
        assert not store.remove(1, 5)

    def test_summary_counts(self, tmp_path):
        store = AnnotationStore(tmp_path / "ann.json")
        store.add(_make_ann(tic_id=1, category="vetting"))
        store.add(_make_ann(tic_id=2, category="note"))
        s = store.summary()
        assert s["n_annotations"] == 2
        assert s["n_targets"] == 2

    def test_persist_across_reload(self, tmp_path):
        p = tmp_path / "ann.json"
        store = AnnotationStore(p)
        store.add(_make_ann(note="persisted"))
        store2 = AnnotationStore(p)
        assert len(store2.all()) == 1

    def test_created_at_auto_filled(self, tmp_path):
        store = AnnotationStore(tmp_path / "ann.json")
        store.add(_make_ann())
        a = store.all()[0]
        assert a.created_at != ""


class TestExportAnnotationsCSV:
    def test_creates_file(self, tmp_path):
        anns = [_make_ann(tic_id=1), _make_ann(tic_id=2)]
        p = export_annotations_csv(anns, tmp_path / "out.csv")
        assert p.exists()

    def test_row_count(self, tmp_path):
        anns = [_make_ann(tic_id=i) for i in range(5)]
        p = export_annotations_csv(anns, tmp_path / "out.csv")
        lines = p.read_text().strip().split("\n")
        assert len(lines) == 6  # header + 5 rows


class TestExportAnnotationsJSON:
    def test_creates_file(self, tmp_path):
        anns = [_make_ann()]
        p = export_annotations_json(anns, tmp_path / "out.json")
        assert p.exists()

    def test_json_contains_count(self, tmp_path):
        import json
        anns = [_make_ann(tic_id=i) for i in range(3)]
        p = export_annotations_json(anns, tmp_path / "out.json")
        data = json.loads(p.read_text())
        assert data["n_annotations"] == 3
