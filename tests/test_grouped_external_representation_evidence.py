"""Integrity checks for the merged grouped external-representation evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

ARTIFACT_DIR = Path("artifacts/manifests")
AGGREGATE = ARTIFACT_DIR / "grouped_external_representation_aggregate_v1.json"
CONTRACT = Path("metadata/grouped_external_representation_contract_v3.json")
AGGREGATE_SHA256 = "3d24363b29eefc1be7a6d9c69e163683e7c36e0dbca757a2ced7e33ebd4952bd"
CONTRACT_SHA256 = "24a42035490c79dfbd255e40a80d75d479b9f5830ae9bf3f016b36f20413279e"
SHARD_SHA256 = (
    "11861a6673a636c63d2e77a2082fea2c5d6b3745732a4d7038ef82e3de118df8",
    "1730b072e09fc80feb263ad258ec4aeb0056a0d9d95561e6d0fe26c103616a9c",
    "651ffb4f6da5d81d5eeef775c9085421d5944de404d0c63fd80212b1b0cbe2b0",
    "00db448150c34153ba7d4fa2cbef33680564d53415ef186a5a3f4e4e0f243337",
    "39413dcee6054b38fd0eb697b433701697e8691ab94fd15c830a8b84508ba199",
    "569871688046cdc347f75b57f3e9edec14c0b2eab04ea6063ba4f31298052e51",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_grouped_aggregate_identity_and_closed_authorization() -> None:
    artifact = json.loads(AGGREGATE.read_text())
    assert _sha256(AGGREGATE) == AGGREGATE_SHA256
    assert _sha256(CONTRACT) == CONTRACT_SHA256
    assert artifact["contract_sha256"] == CONTRACT_SHA256
    assert artifact["status"] == "pass"
    assert artifact["scientific_outcome"] == "no_external_added_value"
    assert artifact["rows"] == 1536
    assert artifact["split_rows"] == {"train": 1024, "val": 256, "test": 256}
    assert artifact["known_unreadable_cache_files_skipped"] == 111
    assert artifact["downloaded_bytes"] == 0
    assert artifact["temporary_embedding_files_removed"] == 6
    assert artifact["persisted_embeddings"] == 0
    assert artifact["test_opened_once"] is True
    assert artifact["training_authorized"] is False
    assert artifact["production_change_authorized"] is False


@pytest.mark.parametrize(
    ("name", "roc_auc", "average_precision", "top_k_positives"),
    [
        ("Chronos-Bolt tiny", 0.7227783203125, 0.6963441726520521, 71),
        ("Astromer2", 0.708984375, 0.659678655171027, 67),
        ("benchmark_cnn_v1", 0.923095703125, 0.8991844084288054, 91),
        ("statistical_ephemeris_baseline", 0.69940185546875, 0.6077804734905494, 67),
    ],
)
def test_grouped_test_metrics_are_frozen(
    name: str, roc_auc: float, average_precision: float, top_k_positives: int
) -> None:
    result = json.loads(AGGREGATE.read_text())["results"][name]["test"]
    assert result["roc_auc"] == pytest.approx(roc_auc)
    assert result["average_precision"] == pytest.approx(average_precision)
    assert result["top_k_positives"] == top_k_positives


def test_grouped_shards_and_run_reports_reconcile() -> None:
    aggregate = json.loads(AGGREGATE.read_text())
    summaries = []
    for shard_number, expected_sha in enumerate(SHARD_SHA256, 1):
        path = ARTIFACT_DIR / (
            f"grouped_external_representation_summary_v1.shard{shard_number}of6.json"
        )
        assert _sha256(path) == expected_sha
        summaries.append(json.loads(path.read_text()))
    assert summaries == aggregate["shard_summaries"]
    assert sum(summary["rows"] for summary in summaries) == 1536
    assert sum(
        summary["known_unreadable_cache_files_skipped"] for summary in summaries
    ) == 111
    assert {summary["downloaded_bytes"] for summary in summaries} == {0}
    assert {summary["production_change_authorized"] for summary in summaries} == {
        False
    }

    shard_reports = []
    for shard_index in range(6):
        path = ARTIFACT_DIR / "run_reports" / (
            f"benchmark_grouped_external_representations.shard{shard_index}of6.jsonl"
        )
        shard_reports.append(json.loads(path.read_text().splitlines()[-1]))
    assert sum(report["items_processed"] for report in shard_reports) == 1536
    assert {report["status"] for report in shard_reports} == {"success"}
    assert {report["items_failed"] for report in shard_reports} == {0}

    aggregate_report = ARTIFACT_DIR / "run_reports" / (
        "benchmark_grouped_external_representations_aggregate.jsonl"
    )
    report = json.loads(aggregate_report.read_text().splitlines()[-1])
    assert report["status"] == "success"
    assert report["items_processed"] == 1536
    assert report["items_written"] == 1
    assert report["items_failed"] == 0
