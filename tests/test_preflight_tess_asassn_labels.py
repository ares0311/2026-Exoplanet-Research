"""Offline tests for the metadata-only TESS/ASAS-SN overlap preflight."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest
from Skills.preflight_tess_asassn_labels import (
    aggregate_shards,
    build_query,
    load_contract,
    run_shard,
    select_shard,
    verify_source_metadata,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_contract(tmp_path: Path, *, target_count: int = 12) -> Path:
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text(
        "".join(
            json.dumps({"target_id": f"TIC {tic_id}"}) + "\n"
            for tic_id in range(1, target_count + 1)
        ),
        encoding="utf-8",
    )
    contract = {
        "schema_version": 1,
        "catalog": {
            "tap_endpoint": "https://example.invalid/tap",
            "table": "J/example/table",
            "gzip_url": "https://example.invalid/table.gz",
            "row_count": 100,
            "tic_row_count": 100,
        },
        "inventory": {
            "path": str(inventory.relative_to(Path.cwd()))
            if inventory.is_relative_to(Path.cwd())
            else str(inventory),
            "sha256": _sha256(inventory),
            "target_count": target_count,
        },
        "query": {
            "selected_columns": [
                "TIC",
                "ASAS-SN",
                "Class",
                "Prob",
                "Discovery",
                "RAJ2000",
                "DEJ2000",
            ],
            "batch_size": 3,
            "max_batch_size": 3,
            "retry_attempts": 1,
        },
        "source_verification_required": False,
        "parallel_shape": {"process_shards": 6, "workers_per_shard": 6},
        "evidence_gate": {
            "minimum_unique_tic_matches": 2,
            "minimum_known_variable_matches": 1,
            "minimum_probability": 0.9,
            "maximum_duplicate_tic_rows": 0,
            "maximum_duplicate_source_ids": 0,
            "training_authorized_on_pass": False,
        },
        "limitations": ["test-only"],
        "training_authorized": False,
    }
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(contract), encoding="utf-8")
    return path


def _fake_tap(_endpoint: str, query: str) -> dict[str, Any]:
    ids = [int(value) for value in re.findall(r"TIC (\d+)", query)]
    columns = ["TIC", "ASAS-SN", "Class", "Prob", "Discovery", "RAJ2000", "DEJ2000"]
    return {
        "metadata": [{"name": column} for column in columns],
        "data": [
            [f"TIC {tic_id}", f"ASASSN-{tic_id}", "ROT", 0.95, "False", 1.0, 2.0]
            for tic_id in ids
            if tic_id % 2 == 0
        ],
    }


def _no_report(*_args: object, **_kwargs: object) -> bool:
    return True


def test_merged_contract_preserves_automated_label_limit() -> None:
    contract = load_contract(Path("metadata/asassn_variability_label_source_contract_v1.json"))
    assert contract["catalog"]["row_count"] == 378861
    assert contract["catalog"]["label_quality_limit"].startswith("The Class field is machine")
    assert contract["training_authorized"] is False


def test_source_metadata_verification_checks_pinned_primary_metadata() -> None:
    contract = load_contract(Path("metadata/asassn_variability_label_source_contract_v1.json"))
    catalog = contract["catalog"]

    def fake_head(_url: str) -> dict[str, str]:
        return {
            "content-length": str(catalog["delivery"]["content_length_bytes"]),
            "etag": catalog["delivery"]["etag"],
            "last-modified": catalog["delivery"]["last_modified"],
            "content-type": catalog["delivery"]["content_type"],
        }

    def fake_tap(_endpoint: str, query: str) -> dict[str, Any]:
        if "TAP_SCHEMA.columns" in query:
            return {
                "metadata": [
                    {"name": "table_name"},
                    {"name": "column_name"},
                    {"name": "datatype"},
                    {"name": "description"},
                ],
                "data": [
                    [catalog["table"], f'"{name}"', datatype, "test"]
                    for name, datatype in catalog["required_columns"].items()
                ],
            }
        if 'GROUP BY "Class"' in query:
            return {
                "metadata": [{"name": "Class"}, {"name": "n"}],
                "data": [[name, count] for name, count in catalog["class_counts"].items()],
            }
        if 'GROUP BY "Discovery"' in query:
            return {
                "metadata": [{"name": "Discovery"}, {"name": "n"}],
                "data": [
                    [name, count] for name, count in catalog["discovery_counts"].items()
                ],
            }
        count = catalog["tic_row_count"] if "LIKE 'TIC %'" in query else catalog["row_count"]
        return {"metadata": [{"name": "n"}], "data": [[count]]}

    evidence = verify_source_metadata(contract, tap_fn=fake_tap, head_fn=fake_head)
    assert evidence["status"] == "pass"
    assert evidence["metadata_requests"] == 6
    assert evidence["full_catalog_payload_bytes_downloaded"] == 0


def test_select_shards_are_disjoint_and_complete() -> None:
    values = tuple(range(1, 101))
    shards = [set(select_shard(values, index, 6)) for index in range(6)]
    assert set.union(*shards) == set(values)
    assert all(
        not left & right
        for index, left in enumerate(shards)
        for right in shards[index + 1 :]
    )


def test_build_query_uses_exact_tic_ids(tmp_path: Path) -> None:
    contract = load_contract(_write_contract(tmp_path))
    query = build_query(contract, (12, 34))
    assert 'FROM "J/example/table"' in query
    assert "'TIC 12','TIC 34'" in query


def test_merged_contract_rejects_noncanonical_parallel_shape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="requires 6 shards x 6 workers"):
        run_shard(
            Path("metadata/asassn_variability_label_source_contract_v1.json"),
            tmp_path / "rows.jsonl",
            tmp_path / "summary.json",
            workers=1,
            batch_size=50,
            shard_index=0,
            shard_count=1,
            tap_fn=_fake_tap,
            report_fn=_no_report,
        )


def test_run_shard_writes_matched_and_unmatched_rows(tmp_path: Path) -> None:
    contract_path = _write_contract(tmp_path)
    summary = run_shard(
        contract_path,
        tmp_path / "rows.jsonl",
        tmp_path / "summary.json",
        workers=2,
        batch_size=3,
        shard_index=0,
        shard_count=1,
        tap_fn=_fake_tap,
        report_fn=_no_report,
    )
    rows = [json.loads(line) for line in (tmp_path / "rows.jsonl").read_text().splitlines()]
    assert summary["selected_tics"] == 12
    assert summary["matched_rows"] == 6
    assert {row["status"] for row in rows} == {"matched", "no_exact_tic_match"}
    assert all(row["training_authorized"] is False for row in rows)


def test_run_shard_rejects_duplicate_tic_rows(tmp_path: Path) -> None:
    contract_path = _write_contract(tmp_path)

    def duplicate_tap(endpoint: str, query: str) -> dict[str, Any]:
        response = _fake_tap(endpoint, query)
        if response["data"]:
            response["data"].append(response["data"][0])
        return response

    with pytest.raises(RuntimeError, match="duplicate rows"):
        run_shard(
            contract_path,
            tmp_path / "rows.jsonl",
            tmp_path / "summary.json",
            workers=1,
            batch_size=3,
            shard_index=0,
            shard_count=1,
            tap_fn=duplicate_tap,
            report_fn=_no_report,
        )


def test_aggregate_reconciles_six_shards_and_keeps_training_disabled(tmp_path: Path) -> None:
    contract_path = _write_contract(tmp_path)
    output = tmp_path / "rows.jsonl"
    summary = tmp_path / "summary.json"
    for shard_index in range(6):
        run_shard(
            contract_path,
            output,
            summary,
            workers=1,
            batch_size=3,
            shard_index=shard_index,
            shard_count=6,
            tap_fn=_fake_tap,
            report_fn=_no_report,
        )
    aggregate = aggregate_shards(
        contract_path,
        output,
        summary,
        tmp_path / "aggregate.json",
        shard_count=6,
        report_fn=_no_report,
    )
    assert aggregate["status"] == "pass"
    assert aggregate["matched_unique_tics"] == 6
    assert aggregate["duplicate_tic_rows"] == 0
    assert aggregate["duplicate_source_ids"] == 0
    assert aggregate["training_authorized"] is False
