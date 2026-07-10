"""Tests for Skills/build_t1_2_k2_calibration_manifest.py."""

from __future__ import annotations

import json
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Skills"))

from build_t1_2_k2_calibration_manifest import (  # noqa: E402
    build_k2_calibration_manifest,
    format_summary,
    verify_k2_schema,
    write_manifest_outputs,
)


def _query(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    return urllib.parse.parse_qs(parsed.query).get("query", [""])[0].lower()


def _schema_csv() -> str:
    cols = [
        "epic_candname",
        "disposition",
        "pl_orbper",
        "pl_tranmid",
        "pl_trandep",
        "pl_trandur",
        "pl_rade",
        "sy_pnum",
        "hostname",
    ]
    return "column_name\n" + "\n".join(cols) + "\n"


def _rows_csv(n_confirmed: int = 40, n_fp: int = 10) -> str:
    header = "epic_candname,disposition,pl_orbper,pl_tranmid,pl_trandep,pl_trandur,pl_rade,sy_pnum"
    rows = [header]
    epic = 200000000
    for i in range(n_confirmed):
        epic += 1
        rows.append(
            f"EPIC {epic}.01,CONFIRMED,{3.0 + i * 0.1},{2457100.0 + i},0.05,0.1,1.8,1"
        )
    for i in range(n_fp):
        epic += 1
        rows.append(
            f"EPIC {epic}.01,FALSE POSITIVE,{5.0 + i * 0.1},{2457200.0 + i},0.5,0.05,20.0,1"
        )
    return "\n".join(rows) + "\n"


def _tap(url: str) -> str:
    query = _query(url)
    if "tap_schema.columns" in query and "k2pandc" in query:
        return _schema_csv()
    if "from k2pandc" in query:
        return _rows_csv()
    raise AssertionError(f"unexpected TAP query: {query}")


def _tap_missing_columns(url: str) -> str:
    query = _query(url)
    if "tap_schema.columns" in query:
        return "column_name\nepic_candname\ndisposition\n"
    raise AssertionError("should not query rows when schema is missing columns")


def test_verify_k2_schema_ok() -> None:
    ok, missing = verify_k2_schema(_tap)
    assert ok
    assert missing == []


def test_verify_k2_schema_reports_missing_columns() -> None:
    ok, missing = verify_k2_schema(_tap_missing_columns)
    assert not ok
    assert "pl_trandep" in missing


def test_build_k2_calibration_manifest_unit_conversions() -> None:
    rows, summary = build_k2_calibration_manifest(
        tap_fn=_tap, seed=42, sample_size=600, created_at_utc="2026-07-09T00:00:00Z"
    )
    assert summary.flag == "OK"
    assert rows
    row = next(r for r in rows if r.label_name == "FALSE POSITIVE")
    assert row.depth_ppm == 0.5 * 10_000.0
    # pl_trandur is already in hours (a real archive schema-metadata quirk —
    # see module docstring); the manifest must pass it through unconverted.
    assert row.duration_hours == 0.05
    # pl_tranmid is already full BJD_TDB in the live archive — the manifest
    # must pass it through unchanged, never adding a BKJD offset.
    assert row.epoch_bjd > 2450000.0
    assert row.mission == "K2"
    assert row.group_key == f"k2:epic:{row.epic_id}"


def test_epoch_bjd_is_not_offset_regression() -> None:
    """Regression test: pl_tranmid must pass through unchanged.

    An earlier version of this script (mirroring a real bug found live in
    ``Skills/fetch_tess_k2_overlap_snippets.py`` on 2026-07-10) incorrectly
    treated pl_tranmid as BKJD and added 2454833.0 to it, corrupting the
    phase-fold epoch. Both were fixed the same day.
    """
    rows, _ = build_k2_calibration_manifest(tap_fn=_tap, seed=42, sample_size=600)
    row = next(r for r in rows if r.label_name == "FALSE POSITIVE")
    # The fixture's first FALSE POSITIVE row has pl_tranmid == 2457200.0.
    assert row.epoch_bjd == 2457200.0


def test_build_k2_calibration_manifest_uses_all_false_positives_and_samples_confirmed() -> None:
    rows, summary = build_k2_calibration_manifest(
        tap_fn=_tap, seed=42, sample_size=30, created_at_utc="2026-07-09T00:00:00Z"
    )
    assert summary.flag == "OK"
    n_fp = sum(1 for r in rows if r.label == 0)
    n_conf = sum(1 for r in rows if r.label == 1)
    assert n_fp == 10  # all available false positives from the fixture
    assert n_conf == 20  # 30 - 10
    assert len(rows) == 30


def test_build_k2_calibration_manifest_is_deterministic() -> None:
    rows_a, _ = build_k2_calibration_manifest(tap_fn=_tap, seed=7, sample_size=30)
    rows_b, _ = build_k2_calibration_manifest(tap_fn=_tap, seed=7, sample_size=30)
    assert [r.epic_id for r in rows_a] == [r.epic_id for r in rows_b]


def test_build_k2_calibration_manifest_different_seed_differs() -> None:
    rows_a, _ = build_k2_calibration_manifest(tap_fn=_tap, seed=1, sample_size=30)
    rows_b, _ = build_k2_calibration_manifest(tap_fn=_tap, seed=2, sample_size=30)
    assert [r.epic_id for r in rows_a] != [r.epic_id for r in rows_b]


def test_build_k2_calibration_manifest_no_leakage_errors() -> None:
    rows, summary = build_k2_calibration_manifest(tap_fn=_tap, seed=42, sample_size=600)
    assert summary.leakage_errors == ()
    group_keys = [r.group_key for r in rows]
    assert len(group_keys) == len(set(group_keys))


def test_build_k2_calibration_manifest_group_keys_disjoint_from_kepler_namespace() -> None:
    rows, _ = build_k2_calibration_manifest(tap_fn=_tap, seed=42, sample_size=600)
    for row in rows:
        assert row.group_key.startswith("k2:epic:")
        assert not row.group_key.startswith("kepler:kic:")


def test_build_k2_calibration_manifest_schema_fail() -> None:
    rows, summary = build_k2_calibration_manifest(tap_fn=_tap_missing_columns)
    assert rows == []
    assert summary.flag == "SCHEMA_FAIL"
    assert "pl_trandep" in summary.leakage_errors


def test_build_k2_calibration_manifest_insufficient_when_too_few_rows() -> None:
    def _tiny_tap(url: str) -> str:
        query = _query(url)
        if "tap_schema.columns" in query:
            return _schema_csv()
        if "from k2pandc" in query:
            return _rows_csv(n_confirmed=2, n_fp=1)
        raise AssertionError(query)

    rows, summary = build_k2_calibration_manifest(tap_fn=_tiny_tap, sample_size=600)
    assert summary.flag == "INSUFFICIENT"


def test_duplicate_epic_source_rows_are_collapsed() -> None:
    def _dup_tap(url: str) -> str:
        query = _query(url)
        if "tap_schema.columns" in query:
            return _schema_csv()
        if "from k2pandc" in query:
            header = (
                "epic_candname,disposition,pl_orbper,pl_tranmid,"
                "pl_trandep,pl_trandur,pl_rade,sy_pnum"
            )
            # Same EPIC target with two planet candidate rows (.01 and .02).
            rows = [
                header,
                "EPIC 300000001.01,CONFIRMED,3.0,2457100.0,0.05,0.1,1.8,2",
                "EPIC 300000001.02,CONFIRMED,20.0,2457150.0,0.02,0.08,1.1,2",
            ] + [
                f"EPIC {300000010 + i}.01,FALSE POSITIVE,{5.0 + i},{2457200.0 + i},0.5,0.05,20.0,1"
                for i in range(25)
            ]
            return "\n".join(rows) + "\n"
        raise AssertionError(query)

    rows, summary = build_k2_calibration_manifest(tap_fn=_dup_tap, sample_size=600)
    epic_ids = [r.epic_id for r in rows]
    assert epic_ids.count(300000001) == 1
    assert summary.n_duplicate_epic_source_rows == 1


def test_write_manifest_outputs_writes_jsonl_json_and_report(tmp_path: Path) -> None:
    rows, summary = build_k2_calibration_manifest(
        tap_fn=_tap, seed=42, sample_size=30, created_at_utc="2026-07-09T00:00:00Z"
    )
    manifest_path = tmp_path / "metadata" / "manifest.jsonl"
    summary_path = tmp_path / "metadata" / "summary.json"
    report_path = tmp_path / "reports" / "report.md"

    write_manifest_outputs(
        rows,
        summary,
        manifest_path=manifest_path,
        summary_path=summary_path,
        report_path=report_path,
    )

    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(rows)
    parsed = json.loads(lines[0])
    assert "epic_id" in parsed
    assert "group_key" in parsed

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_payload["flag"] == "OK"

    report_text = report_path.read_text(encoding="utf-8")
    assert "T1-2 K2 Calibration Manifest" in report_text


def test_format_summary_includes_leakage_errors_section() -> None:
    rows, summary = build_k2_calibration_manifest(tap_fn=_tap_missing_columns)
    text = format_summary(summary)
    assert "Leakage Errors" in text
    assert "pl_trandep" in text
