"""Integrity checks for the merged TESS-Catalina crossmatch pilot evidence."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
AGGREGATE_PATH = (
    REPO_ROOT
    / "artifacts/manifests/tess_catalina_crossmatch_pilot_aggregate_v1.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_tess_catalina_pilot_evidence_is_complete_and_training_disabled() -> None:
    aggregate = json.loads(AGGREGATE_PATH.read_text(encoding="utf-8"))
    outputs = sorted(
        REPO_ROOT.glob(
            "artifacts/manifests/tess_catalina_crossmatch_pilot_v1.shard*of6.jsonl"
        )
    )
    summaries = sorted(
        REPO_ROOT.glob(
            "artifacts/manifests/tess_catalina_crossmatch_pilot_v1.shard*of6.json"
        )
    )
    rows = [
        json.loads(line)
        for path in outputs
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]

    assert len(outputs) == len(summaries) == 6
    assert len(rows) == 216
    assert len({int(row["tic_id"]) for row in rows}) == 216
    assert Counter(row["match_status"] for row in rows) == {
        "no_candidate_within_radius": 216
    }
    assert all(row["training_authorized"] is False for row in rows)
    assert not [row for row in rows if row.get("catalina_source_id")]

    summary_rows = [json.loads(path.read_text(encoding="utf-8")) for path in summaries]
    assert sum(int(row["queried_tics"]) for row in summary_rows) == 216
    assert sum(int(row["rows_written"]) for row in summary_rows) == 216
    assert {int(row["shard_index"]) for row in summary_rows} == set(range(6))

    for artifact in aggregate["artifacts"]:
        path = REPO_ROOT / artifact["path"]
        assert path.is_file()
        assert _sha256(path) == artifact["sha256"]

    reconciliation = aggregate["global_reconciliation"]
    assert reconciliation == {
        "duplicate_tic_ids": 0,
        "accepted_catalina_sources": 0,
        "cross_shard_duplicate_accepted_sources": 0,
        "passed": True,
    }
    assert aggregate["decision"]["full_2790_tic_execution_authorized"] is False
    assert aggregate["decision"]["training_authorized"] is False
