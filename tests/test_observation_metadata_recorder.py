"""Tests for Skills/observation_metadata_recorder.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from observation_metadata_recorder import (
    MetadataStore,
    ObservationRecord,
    format_metadata_record,
)


def _store(tmp_path):
    return MetadataStore(tmp_path / "meta.json")


def test_record_returns_observation_record(tmp_path):
    store = _store(tmp_path)
    rec = store.record(12345, instrument="TESS")
    assert isinstance(rec, ObservationRecord)


def test_flag_ok_with_instrument(tmp_path):
    store = _store(tmp_path)
    rec = store.record(12345, instrument="TESS/SPOC")
    assert rec.flag == "OK"


def test_flag_incomplete_without_instrument(tmp_path):
    store = _store(tmp_path)
    rec = store.record(12345)
    assert rec.flag == "INCOMPLETE"


def test_flag_flagged_with_quality_flags(tmp_path):
    store = _store(tmp_path)
    rec = store.record(12345, instrument="TESS", quality_flags=["scattered_light"])
    assert rec.flag == "FLAGGED"


def test_cadence_stored(tmp_path):
    store = _store(tmp_path)
    rec = store.record(12345, cadence_min=2.0)
    assert rec.cadence_min == 2.0


def test_sector_stored(tmp_path):
    store = _store(tmp_path)
    rec = store.record(12345, sector=22)
    assert rec.sector == 22


def test_notes_stored(tmp_path):
    store = _store(tmp_path)
    rec = store.record(12345, notes="good quality night")
    assert rec.notes == "good quality night"


def test_get_by_obs_id(tmp_path):
    store = _store(tmp_path)
    store.record(12345, obs_id="test-001")
    fetched = store.get("test-001")
    assert fetched is not None
    assert fetched.obs_id == "test-001"


def test_get_missing_returns_none(tmp_path):
    store = _store(tmp_path)
    assert store.get("nonexistent") is None


def test_list_by_tic(tmp_path):
    store = _store(tmp_path)
    store.record(12345, obs_id="a1")
    store.record(12345, obs_id="a2")
    store.record(99999, obs_id="b1")
    recs = store.list_by_tic(12345)
    assert len(recs) == 2


def test_all_records(tmp_path):
    store = _store(tmp_path)
    store.record(1)
    store.record(2)
    assert len(store.all_records()) == 2


def test_persistence(tmp_path):
    path = tmp_path / "m.json"
    s1 = MetadataStore(path)
    s1.record(12345, obs_id="x1")
    s2 = MetadataStore(path)
    assert len(s2.all_records()) == 1


def test_format_contains_tic_id(tmp_path):
    store = _store(tmp_path)
    rec = store.record(77777)
    md = format_metadata_record(rec)
    assert "77777" in md
