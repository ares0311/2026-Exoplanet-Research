"""Integrity checks for the merged TESS/ASAS-SN overlap evidence."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AGGREGATE_PATH = REPO_ROOT / "artifacts/manifests/tess_asassn_preflight_aggregate_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_tess_asassn_evidence_is_complete_and_training_disabled() -> None:
    aggregate = json.loads(AGGREGATE_PATH.read_text(encoding="utf-8"))
    outputs = sorted(
        REPO_ROOT.glob("artifacts/manifests/tess_asassn_preflight_v1.shard*of6.jsonl")
    )
    summaries = sorted(
        REPO_ROOT.glob(
            "artifacts/manifests/tess_asassn_preflight_summary_v1.shard*of6.json"
        )
    )
    rows = [
        json.loads(line)
        for path in outputs
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    matched = [row for row in rows if row["status"] == "matched"]

    assert len(outputs) == len(summaries) == 6
    assert len(rows) == 2790
    assert len({int(row["tic_id"]) for row in rows}) == 2790
    assert Counter(row["status"] for row in rows) == {
        "matched": 48,
        "no_exact_tic_match": 2742,
    }
    assert Counter(row["class_code"] for row in matched) == {
        "EA": 26,
        "EB": 9,
        "EW": 2,
        "ROT": 10,
        "SR": 1,
    }
    assert len({row["asassn_id"] for row in matched}) == 48
    assert sum(not row["discovery"] for row in matched) == 44
    assert min(float(row["probability"]) for row in matched) == 0.902
    assert all(row["training_authorized"] is False for row in rows)

    summary_rows = [json.loads(path.read_text(encoding="utf-8")) for path in summaries]
    assert sum(int(row["selected_tics"]) for row in summary_rows) == 2790
    assert sum(int(row["matched_rows"]) for row in summary_rows) == 48
    assert {int(row["shard_index"]) for row in summary_rows} == set(range(6))
    for summary in summary_rows:
        output = REPO_ROOT / summary["output_path"]
        assert _sha256(output) == summary["output_sha256"]

    assert aggregate["status"] == "pass"
    assert aggregate["matched_unique_tics"] == 48
    assert aggregate["known_variables"] == 44
    assert aggregate["discoveries"] == 4
    assert aggregate["duplicate_tic_rows"] == 0
    assert aggregate["duplicate_source_ids"] == 0
    assert all(aggregate["checks"].values())
    assert aggregate["source_verification"]["status"] == "pass"
    assert aggregate["catalog_payload_bytes_downloaded"] == 0
    assert aggregate["followup_benchmark_design_authorized"] is True
    assert aggregate["training_authorized"] is False
