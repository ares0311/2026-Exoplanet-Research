"""Tests for Skills/build_t1_dr24_tce_expansion_manifest.py."""

from __future__ import annotations

import json
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Skills"))

from build_t1_dr24_tce_expansion_manifest import (  # noqa: E402
    build_dr24_expansion_manifest,
    load_existing_target_ids,
)


def _query(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    return urllib.parse.parse_qs(parsed.query).get("query", [""])[0].lower()


def _schema_csv() -> str:
    return (
        "column_name\nkepid\ntce_plnt_num\nav_training_set\n"
        "tce_period\ntce_time0bk\ntce_duration\n"
    )


def _tce_rows_csv() -> str:
    rows = ["kepid,tce_plnt_num,av_training_set,tce_period,tce_time0bk,tce_duration"]
    # New targets (not in the existing manifest fixture): a mix of PC/AFP/NTP/UNK.
    rows.append("200001,1,PC,3.0,100.0,2.5")
    rows.append("200002,1,AFP,5.0,101.0,2.0")
    rows.append("200003,1,NTP,7.0,102.0,3.0")
    rows.append("200004,1,UNK,9.0,103.0,1.5")  # excluded: ambiguous label
    # Multi-TCE star: both rows must land in the same split.
    rows.append("200005,1,PC,11.0,104.0,2.1")
    rows.append("200005,2,AFP,13.0,105.0,2.2")
    # Invalid period/epoch/duration rows: excluded by the sanity filters.
    rows.append("200006,1,PC,0.1,106.0,2.0")  # period too short
    rows.append("200007,1,PC,600.0,107.0,2.0")  # period too long
    rows.append("200008,1,PC,3.0,0.0,2.0")  # epoch <= 0
    rows.append("200009,1,PC,3.0,108.0,0.0")  # duration <= 0
    # Already-known target: excluded even though it has a usable label.
    rows.append("100001,1,PC,3.0,109.0,2.0")
    # Bulk of straightforward new targets so all three splits get populated with
    # both labels (mirrors the 60-target scale used in
    # tests/test_build_t1_training_manifest.py's leakage-free fixture).
    for index in range(1, 61):
        kepid = 400000 + index
        disposition = "PC" if index % 2 else "AFP"
        rows.append(f"{kepid},1,{disposition},4.0,110.0,2.0")
    return "\n".join(rows) + "\n"


def _tap(url: str) -> str:
    query = _query(url)
    if "tap_schema.columns" in query and "q1_q17_dr24_tce" in query:
        return _schema_csv()
    if "from q1_q17_dr24_tce" in query:
        return _tce_rows_csv()
    raise AssertionError(f"unexpected TAP query: {query}")


def _existing_manifest(tmp_path: Path) -> Path:
    path = tmp_path / "t1_1_kepler_training_manifest.jsonl"
    path.write_text(
        json.dumps({"target_id": 100001, "other": "field"}) + "\n",
        encoding="utf-8",
    )
    return path


def test_build_dr24_expansion_manifest_excludes_known_targets_and_unk(
    tmp_path: Path,
) -> None:
    existing = _existing_manifest(tmp_path)
    snapshot = tmp_path / "source_snapshots.json"
    snapshot.write_text(
        json.dumps({"created_at_utc": "2026-07-04T00:00:00Z", "sources": []}),
        encoding="utf-8",
    )

    rows, summary = build_dr24_expansion_manifest(
        tap_fn=_tap,
        existing_manifest_path=existing,
        source_snapshot_path=snapshot,
        created_at_utc="2026-07-04T00:00:00Z",
    )

    target_ids = {row.target_id for row in rows}
    assert {200001, 200002, 200003, 200005} <= target_ids
    assert 100001 not in target_ids  # already-known target excluded
    assert 200004 not in target_ids  # UNK label excluded
    assert 200006 not in target_ids  # period too short
    assert 200007 not in target_ids  # period too long
    assert 200008 not in target_ids  # epoch <= 0
    assert 200009 not in target_ids  # duration <= 0
    assert summary.flag == "OK"
    assert summary.leakage_errors == ()


def test_build_dr24_expansion_manifest_maps_labels_correctly(tmp_path: Path) -> None:
    rows, _ = build_dr24_expansion_manifest(
        tap_fn=_tap,
        existing_manifest_path=tmp_path / "missing.jsonl",
        source_snapshot_path=tmp_path / "missing_snapshot.json",
        created_at_utc="2026-07-04T00:00:00Z",
    )
    label_by_target = {row.target_id: row.label for row in rows}
    assert label_by_target[200001] == 1  # PC -> planet
    assert label_by_target[200002] == 0  # AFP -> not planet
    assert label_by_target[200003] == 0  # NTP -> not planet


def test_build_dr24_expansion_manifest_keeps_multi_tce_target_in_one_split(
    tmp_path: Path,
) -> None:
    rows, _ = build_dr24_expansion_manifest(
        tap_fn=_tap,
        existing_manifest_path=tmp_path / "missing.jsonl",
        source_snapshot_path=tmp_path / "missing_snapshot.json",
        created_at_utc="2026-07-04T00:00:00Z",
    )
    splits = {row.split for row in rows if row.target_id == 200005}
    assert len(splits) == 1


def test_build_dr24_expansion_manifest_sets_source_metadata(tmp_path: Path) -> None:
    rows, _ = build_dr24_expansion_manifest(
        tap_fn=_tap,
        existing_manifest_path=tmp_path / "missing.jsonl",
        source_snapshot_path=tmp_path / "missing_snapshot.json",
        created_at_utc="2026-07-04T00:00:00Z",
    )
    row = next(r for r in rows if r.target_id == 200001)
    assert row.source == "nasa_exoplanet_archive_dr24_tce"
    assert row.source_table == "Q1_Q17_DR24_TCE"
    assert row.mission == "Kepler"
    assert row.group_key == "kepler:kic:200001"
    assert row.lightcurve_search == {
        "target": "KIC 200001",
        "mission": "Kepler",
        "author": "Kepler",
        "exptime": 1800,
    }


def test_schema_failure_stops_before_manifest_rows() -> None:
    def _bad_tap(url: str) -> str:
        query = _query(url)
        if "tap_schema.columns" in query:
            return "column_name\nkepid\n"
        raise AssertionError("training rows should not be queried after schema failure")

    rows, summary = build_dr24_expansion_manifest(tap_fn=_bad_tap)

    assert rows == []
    assert summary.flag == "SCHEMA_FAIL"
    assert summary.leakage_errors


def test_load_existing_target_ids_missing_file_returns_empty(tmp_path: Path) -> None:
    assert load_existing_target_ids(tmp_path / "does_not_exist.jsonl") == frozenset()


def test_load_existing_target_ids_reads_real_manifest(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"target_id": tid, "other": "x"}) for tid in (10, 20, 20, 30)
        )
        + "\n",
        encoding="utf-8",
    )
    assert load_existing_target_ids(path) == frozenset({10, 20, 30})
