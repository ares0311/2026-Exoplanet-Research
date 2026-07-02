"""Tests for Skills/build_t1_training_manifest.py."""

from __future__ import annotations

import json
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Skills"))

from build_t1_training_manifest import (  # noqa: E402
    TrainingManifestRow,
    build_kepler_manifest,
    default_cleanup_policy,
    summarize_manifest,
    write_manifest_outputs,
)


def _query(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    return urllib.parse.parse_qs(parsed.query).get("query", [""])[0].lower()


def _schema_csv() -> str:
    return (
        "column_name\nkepid\nkepoi_name\nkoi_disposition\n"
        "koi_period\nkoi_time0bk\nkoi_duration\n"
    )


def _manifest_rows_csv(n_targets: int = 60) -> str:
    rows = ["kepid,kepoi_name,koi_disposition,koi_period,koi_time0bk,koi_duration"]
    for index in range(1, n_targets + 1):
        kepid = 100000 + index
        disposition = "CONFIRMED" if index % 2 else "FALSE POSITIVE"
        rows.append(f"{kepid},K{index:05d}.01,{disposition},3.0,100.0,2.5")
        if index % 10 == 0:
            rows.append(f"{kepid},K{index:05d}.02,{disposition},5.0,101.0,2.0")
    return "\n".join(rows) + "\n"


def _tap(url: str) -> str:
    query = _query(url)
    if "tap_schema.columns" in query and "cumulative" in query:
        return _schema_csv()
    if "from cumulative" in query:
        return _manifest_rows_csv()
    raise AssertionError(f"unexpected TAP query: {query}")


def test_build_kepler_manifest_groups_targets_without_leakage(tmp_path: Path) -> None:
    snapshot = tmp_path / "source_snapshots.json"
    snapshot.write_text(
        json.dumps({"created_at_utc": "2026-07-02T00:00:00Z", "sources": []}),
        encoding="utf-8",
    )

    rows, summary = build_kepler_manifest(
        tap_fn=_tap,
        source_snapshot_path=snapshot,
        created_at_utc="2026-07-02T00:00:00Z",
    )

    assert rows
    assert summary.flag == "OK"
    assert summary.row_count == len(rows)
    assert summary.target_count == 60
    assert summary.leakage_errors == ()
    by_group: dict[str, str] = {}
    for row in rows:
        previous = by_group.setdefault(row.group_key, row.split)
        assert previous == row.split
        assert row.lightcurve_search == {
            "target": row.target_name,
            "mission": "Kepler",
            "author": "Kepler",
            "exptime": 1800,
        }


def test_manifest_outputs_are_jsonl_json_and_report(tmp_path: Path) -> None:
    rows, summary = build_kepler_manifest(
        tap_fn=_tap,
        source_snapshot_path=tmp_path / "missing_snapshot.json",
        created_at_utc="2026-07-02T00:00:00Z",
    )
    manifest_path = tmp_path / "metadata" / "manifest.jsonl"
    summary_path = tmp_path / "metadata" / "summary.json"
    report_path = tmp_path / "reports" / "manifest.md"

    write_manifest_outputs(
        rows,
        summary,
        manifest_path=manifest_path,
        summary_path=summary_path,
        report_path=report_path,
    )

    assert manifest_path.read_text(encoding="utf-8").count("\n") == len(rows)
    assert json.loads(summary_path.read_text(encoding="utf-8"))["flag"] == "OK"
    assert "T1-1 Kepler Training Manifest" in report_path.read_text(encoding="utf-8")


def test_schema_failure_stops_before_manifest_rows() -> None:
    def _bad_tap(url: str) -> str:
        query = _query(url)
        if "tap_schema.columns" in query:
            return "column_name\nkepid\n"
        raise AssertionError("training rows should not be queried after schema failure")

    rows, summary = build_kepler_manifest(tap_fn=_bad_tap)

    assert rows == []
    assert summary.flag == "SCHEMA_FAIL"
    assert summary.leakage_errors


def test_summarize_manifest_detects_cross_split_leakage() -> None:
    base = {
        "manifest_version": "test",
        "source": "nasa_exoplanet_archive_dr25_koi",
        "source_table": "cumulative",
        "mission": "Kepler",
        "target_id": 123,
        "target_name": "KIC 123",
        "source_row_id": "K00001.01",
        "group_key": "kepler:kic:123",
        "label": 1,
        "label_name": "CONFIRMED",
        "period_days": 3.0,
        "epoch_bkjd": 100.0,
        "duration_hours": 2.0,
        "lightcurve_search": {"target": "KIC 123", "mission": "Kepler"},
    }
    rows = [
        TrainingManifestRow(**base, split="train"),
        TrainingManifestRow(**{**base, "source_row_id": "K00001.02"}, split="test"),
    ]

    summary = summarize_manifest(
        rows,
        created_at_utc="2026-07-02T00:00:00Z",
        source_snapshot="test",
        cleanup_policy=default_cleanup_policy(),
    )

    assert summary.flag == "LEAKAGE_FAIL"
    assert any("appears in both" in error for error in summary.leakage_errors)
