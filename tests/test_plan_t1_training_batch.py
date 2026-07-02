"""Tests for Skills/plan_t1_training_batch.py."""

from __future__ import annotations

import json
import sys
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Skills"))

from plan_t1_training_batch import build_batch_plan, write_batch_plan


def _query(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    return urllib.parse.parse_qs(parsed.query).get("query", [""])[0].lower()


def _tap(url: str) -> str:
    query = _query(url)
    if "tap_schema.columns" in query and "cumulative" in query:
        return (
            "column_name\nkepid\nkepoi_name\nkoi_disposition\n"
            "koi_period\nkoi_time0bk\nkoi_duration\n"
        )
    if "tap_schema.columns" in query and "toi" in query:
        return "column_name\ntid\ntoi\ntfopwg_disp\npl_orbper\npl_tranmid\npl_trandurh\n"
    if "tap_schema.columns" in query and "pscomppars" in query:
        return "column_name\npl_name\nhostname\ndiscoverymethod\ndisc_year\n"
    if "count(*) as n_rows from cumulative" in query:
        return "n_rows\n3\n"
    if "count(*) as n_rows from toi" in query:
        return "n_rows\n2\n"
    if "count(*) as n_rows from pscomppars" in query:
        return "n_rows\n10\n"
    if "from cumulative" in query:
        return (
            "kepid,kepoi_name,koi_disposition,koi_period,koi_time0bk,koi_duration\n"
            "100,K00001.01,CONFIRMED,2.0,100.0,2.5\n"
            "101,K00002.01,FALSE POSITIVE,3.0,101.0,2.0\n"
            "101,K00002.02,FALSE POSITIVE,4.0,102.0,2.0\n"
        )
    if "from toi" in query:
        return (
            "tid,toi,tfopwg_disp,pl_orbper,pl_tranmid,pl_trandurh\n"
            "200,100.01,CP,5.0,2459000.0,3.0\n"
            "201,101.01,FP,6.0,2459001.0,2.0\n"
        )
    raise AssertionError(f"unexpected TAP query: {query}")


def _exofop(_url: str) -> str:
    return "TIC ID,TOI,Disposition\n200,100.01,CP\n201,101.01,FP\n"


class _FakeTable(list):
    colnames = ("productFilename", "dataURI", "author", "size")


class _FakeSearch:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.table = _FakeTable(rows)

    def __len__(self) -> int:
        return len(self.table)


def _search(target: str, *, mission: str, **_criteria: object) -> _FakeSearch:
    prefix = "kplr" if mission == "Kepler" else "tess"
    size = 100 if mission == "Kepler" else 200
    return _FakeSearch(
        [
            {
                "productFilename": f"{prefix}-{target}.fits",
                "dataURI": f"mast:{prefix}/{target}.fits",
                "author": mission,
                "size": size,
            }
        ]
    )


def test_build_batch_plan_uses_real_metadata_estimates() -> None:
    plan = build_batch_plan(
        sample_size=2,
        request_delay=0,
        tap_fn=_tap,
        exofop_fn=_exofop,
        lightkurve_search_fn=_search,
        created_at_utc="2026-07-02T00:00:00Z",
    )

    assert plan.flag == "OK"
    assert plan.koi_label_rows == 3
    assert plan.koi_unique_targets == 2
    assert plan.koi_positive_rows == 1
    assert plan.koi_negative_rows == 2
    assert plan.toi_unique_targets == 2
    assert plan.exofop_rows == 2
    assert plan.kepler_estimated_raw_bytes == 200
    assert plan.tess_estimated_raw_bytes == 400
    assert len(plan.download_manifest_rows) == 4


def test_write_batch_plan_outputs_source_snapshot_and_manifest(tmp_path: Path) -> None:
    plan = build_batch_plan(
        sample_size=1,
        request_delay=0,
        tap_fn=_tap,
        exofop_fn=_exofop,
        lightkurve_search_fn=_search,
        created_at_utc="2026-07-02T00:00:00Z",
    )
    metadata = tmp_path / "metadata"
    report = tmp_path / "report.md"

    write_batch_plan(plan, metadata_dir=metadata, report_path=report)

    snapshot = json.loads((metadata / "source_snapshots.json").read_text())
    assert snapshot["sources"][0]["name"] == "nasa_exoplanet_archive_cumulative"
    assert (metadata / "download_manifest_sample.jsonl").read_text().count("\n") == 2
    assert "T1-1 Training Data Batch Plan" in report.read_text()


def test_schema_failure_stops_before_lightkurve() -> None:
    def _bad_tap(url: str) -> str:
        query = _query(url)
        if "tap_schema.columns" in query:
            return "column_name\nkepid\n"
        return _tap(url)

    def _unexpected_search(*_args, **_kwargs):
        raise AssertionError("search should not run")

    plan = build_batch_plan(
        tap_fn=_bad_tap,
        exofop_fn=_exofop,
        lightkurve_search_fn=_unexpected_search,
    )

    assert plan.flag == "SCHEMA_FAIL"
    assert plan.download_manifest_rows == ()
