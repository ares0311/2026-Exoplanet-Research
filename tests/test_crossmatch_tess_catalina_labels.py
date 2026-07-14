"""Offline tests for the bounded TESS-Catalina crossmatch pilot."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from Skills.crossmatch_tess_catalina_labels import (
    CatalinaRecord,
    TicRecord,
    crossmatch_records,
    ensure_catalog_cache,
    load_contract,
    parse_catalog,
    query_tic_metadata,
    run_crossmatch,
    select_tics,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _catalog_line(class_code: int, *, flag: str = " ") -> str:
    source = f"J0000{class_code:02d}+000000"
    line = (
        f"CSS_{source:<16} 00 00 {class_code:05.2f} +00 00 00.0 "
        f"{15 + class_code / 100:5.2f} {class_code / 10:11.7f}  0.25 "
        f"{class_code:2d} {flag:1}"
    )
    assert len(line) == 73
    return line


def _write_catalog(path: Path) -> None:
    with gzip.open(path, "wt", encoding="ascii") as handle:
        for class_code in range(1, 18):
            line = _catalog_line(class_code)
            handle.write((line.rstrip() if class_code == 2 else line) + "\n")


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    inventory = tmp_path / "inventory.jsonl"
    inventory.write_text(
        "".join(
            json.dumps({"target_id": f"TIC {tic_id}"}) + "\n"
            for tic_id in range(1, 13)
        ),
        encoding="utf-8",
    )
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "status": "success",
                "source_identity_authorized": True,
                "training_authorized": False,
            }
        ),
        encoding="utf-8",
    )
    catalog = tmp_path / "source.dat.gz"
    _write_catalog(catalog)
    return inventory, evidence, catalog


def _contract(tmp_path: Path, inventory: Path, evidence: Path, catalog: Path) -> Path:
    path = tmp_path / "contract.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "inventory": {
                    "path": str(inventory),
                    "sha256": _sha256(inventory),
                    "target_count": 12,
                },
                "source_evidence": {
                    "path": str(evidence),
                    "sha256": _sha256(evidence),
                },
                "catalog": {
                    "url": "https://example.invalid/catalog.gz",
                    "size_bytes": catalog.stat().st_size,
                    "sha256": _sha256(catalog),
                    "row_count": 17,
                    "class_counts": {str(index): 1 for index in range(1, 18)},
                },
                "tic_query": {
                    "columns": ["ID", "ra", "dec"],
                    "batch_size": 2,
                    "max_batch_size": 4,
                    "retry_attempts": 1,
                },
                "match_policy": {
                    "candidate_radius_arcsec": 3.0,
                    "accept_radius_arcsec": 1.0,
                    "max_vmag_delta": 2.0,
                    "max_tmag_delta": 5.0,
                },
                "pilot": {
                    "max_targets": 12,
                    "selection_seed": "test",
                    "full_run_authorized": False,
                },
                "training_authorized": False,
                "limitations": ["test"],
            }
        ),
        encoding="utf-8",
    )
    return path


def _query_rows(tic_ids: list[int] | tuple[int, ...]) -> list[dict[str, Any]]:
    return [
        {
            "ID": tic_id,
            "ra": float(tic_id),
            "dec": 0.0,
            "Tmag": 12.0,
            "Vmag": 12.5,
            "GAIA": str(1000 + tic_id),
            "pmRA": 1.0,
            "pmDEC": 2.0,
            "duplicate_i": None,
            "objType": "STAR",
        }
        for tic_id in tic_ids
    ]


def test_contract_and_catalog_parser_validate_counts(tmp_path: Path) -> None:
    inventory, evidence, catalog = _write_inputs(tmp_path)
    contract_path = _contract(tmp_path, inventory, evidence, catalog)
    contract = load_contract(contract_path)
    records = parse_catalog(catalog, contract)
    assert len(records) == 17
    assert records[0].class_code == 1
    assert records[1].class_flag is None
    assert records[-1].class_code == 17


def test_catalog_parser_rejects_rows_shorter_than_required_fields(tmp_path: Path) -> None:
    inventory, evidence, catalog = _write_inputs(tmp_path)
    with gzip.open(catalog, "wt", encoding="ascii") as handle:
        handle.write(_catalog_line(1)[:70] + "\n")
    contract = load_contract(_contract(tmp_path, inventory, evidence, catalog))

    with pytest.raises(ValueError, match="outside the supported 71-73 byte range"):
        parse_catalog(catalog, contract)


def test_select_tics_is_deterministic_and_shards_are_disjoint() -> None:
    tic_ids = tuple(range(1, 101))
    shards = [
        set(select_tics(tic_ids, seed="fixed", max_targets=60, shard_index=index, shard_count=6))
        for index in range(6)
    ]
    assert sum(len(shard) for shard in shards) == 60
    assert len(set.union(*shards)) == 60
    assert all(not left & right for i, left in enumerate(shards) for right in shards[i + 1 :])


def test_catalog_cache_downloads_once_and_validates_hash(tmp_path: Path) -> None:
    inventory, evidence, source = _write_inputs(tmp_path)
    contract = load_contract(_contract(tmp_path, inventory, evidence, source))
    cache = tmp_path / "cache" / "catalog.gz"
    calls: list[str] = []

    def _download(url: str, path: Path) -> None:
        calls.append(url)
        path.write_bytes(source.read_bytes())

    ensure_catalog_cache(contract, cache, download_fn=_download)
    ensure_catalog_cache(contract, cache, download_fn=_download)
    assert calls == ["https://example.invalid/catalog.gz"]
    assert _sha256(cache) == _sha256(source)


def test_query_tic_metadata_batches_exact_ids() -> None:
    seen: list[tuple[int, ...]] = []

    def _query(values: tuple[int, ...]) -> list[dict[str, Any]]:
        seen.append(values)
        return _query_rows(values)

    result = query_tic_metadata(
        (1, 2, 3, 4, 5),
        workers=2,
        batch_size=2,
        query_fn=_query,
        retry_attempts=1,
    )
    assert [row.tic_id for row in result] == [1, 2, 3, 4, 5]
    assert sorted(len(batch) for batch in seen) == [1, 2, 2]


def test_query_tic_metadata_fails_closed_on_missing_row() -> None:
    with pytest.raises(RuntimeError, match="batch mismatch"):
        query_tic_metadata(
            (1, 2),
            workers=1,
            batch_size=2,
            query_fn=lambda _values: _query_rows([1]),
            retry_attempts=1,
        )


def test_crossmatch_accepts_safe_match_and_rejects_blend() -> None:
    catalog = (
        CatalinaRecord("safe", 10.0, 0.0, 12.5, 1.0, 0.2, 4, None),
        CatalinaRecord("blend", 20.0, 0.0, 13.0, 2.0, 0.3, 1, "f"),
    )
    tics = (
        TicRecord(1, 10.0, 0.0, 12.0, 12.5, "1001", 1.0, 2.0, None, "STAR"),
        TicRecord(2, 20.0, 0.0, 12.0, 13.0, "1002", 1.0, 2.0, None, "STAR"),
        TicRecord(3, 30.0, 0.0, 12.0, 12.5, "1003", 1.0, 2.0, None, "STAR"),
    )
    rows = crossmatch_records(
        tics,
        catalog,
        {
            "candidate_radius_arcsec": 3.0,
            "accept_radius_arcsec": 1.0,
            "max_vmag_delta": 2.0,
            "max_tmag_delta": 5.0,
        },
    )
    assert rows[0]["match_status"] == "accepted_label"
    assert rows[0]["benchmark_family"] == "stellar_variability"
    assert rows[1]["match_status"] == "rejected_match"
    assert rows[1]["rejection_reasons"] == ["catalina_blend_flag"]
    assert rows[2]["match_status"] == "no_candidate_within_radius"
    assert all(row["training_authorized"] is False for row in rows)


def test_dry_run_writes_nothing_and_requires_pilot_bound(tmp_path: Path) -> None:
    inventory, evidence, catalog = _write_inputs(tmp_path)
    contract = _contract(tmp_path, inventory, evidence, catalog)
    output = tmp_path / "result.jsonl"
    summary = tmp_path / "summary.json"
    result = run_crossmatch(
        contract,
        tmp_path / "cache.gz",
        output,
        summary,
        workers=6,
        batch_size=2,
        max_targets=12,
        shard_index=0,
        shard_count=1,
        dry_run=True,
    )
    assert result["status"] == "dry_run"
    assert not output.exists()
    assert not summary.exists()

    with pytest.raises(ValueError, match="only a bounded pilot"):
        run_crossmatch(
            contract,
            tmp_path / "cache.gz",
            output,
            summary,
            workers=6,
            batch_size=2,
            max_targets=None,
            shard_index=0,
            shard_count=1,
            dry_run=True,
        )
