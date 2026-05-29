"""Tests for Skills/data_provenance_tracker.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from data_provenance_tracker import (
    ProvenanceLog,
    ProvenanceRecord,
    format_provenance_report,
)


def _log(tmp_path):
    return ProvenanceLog(tmp_path / "prov.json")


def test_record_returns_provenance_record(tmp_path):
    log = _log(tmp_path)
    rec = log.record("output.fits", "MAST")
    assert isinstance(rec, ProvenanceRecord)


def test_flag_unverified_for_missing_file(tmp_path):
    log = _log(tmp_path)
    rec = log.record("nonexistent_artifact.fits", "MAST")
    assert rec.flag == "UNVERIFIED"


def test_flag_ok_for_existing_file(tmp_path):
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}")
    log = _log(tmp_path)
    rec = log.record(str(artifact), "MAST")
    assert rec.flag == "OK"
    assert rec.checksum is not None


def test_get_returns_latest(tmp_path):
    log = _log(tmp_path)
    log.record("a.json", "source1", version="v1")
    log.record("a.json", "source2", version="v2")
    rec = log.get("a.json")
    assert rec is not None
    assert rec.version == "v2"


def test_get_missing_returns_none(tmp_path):
    log = _log(tmp_path)
    assert log.get("missing.json") is None


def test_history_returns_all(tmp_path):
    log = _log(tmp_path)
    log.record("a.json", "s1")
    log.record("a.json", "s2")
    h = log.history("a.json")
    assert len(h) == 2


def test_transform_chain_stored(tmp_path):
    log = _log(tmp_path)
    rec = log.record("a.json", "s", transform_chain=["clean", "normalize"])
    assert rec.transform_chain == ("clean", "normalize")


def test_summary_counts(tmp_path):
    log = _log(tmp_path)
    log.record("a.json", "s1")
    log.record("b.json", "s2")
    s = log.summary()
    assert s["n_records"] == 2
    assert s["n_artifacts"] == 2


def test_persistence(tmp_path):
    log1 = _log(tmp_path)
    log1.record("c.json", "src")
    log2 = ProvenanceLog(tmp_path / "prov.json")
    assert log2.summary()["n_records"] == 1


def test_format_contains_header(tmp_path):
    log = _log(tmp_path)
    log.record("x.json", "source")
    md = format_provenance_report(log)
    assert "Provenance" in md


def test_format_empty_log(tmp_path):
    log = _log(tmp_path)
    md = format_provenance_report(log)
    assert "No provenance" in md


def test_version_stored(tmp_path):
    log = _log(tmp_path)
    rec = log.record("v.json", "s", version="1.2.3")
    assert rec.version == "1.2.3"
