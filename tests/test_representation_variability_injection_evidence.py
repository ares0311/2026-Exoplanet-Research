"""Integrity checks for the merged representation variability/injection evidence."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_ROOT = REPO_ROOT / "artifacts/manifests"
AGGREGATE_PATH = ARTIFACT_ROOT / "representation_variability_injection_aggregate_v1.json"
AGGREGATE_SHA256 = "93ae6fb818054947ecfd485b3e74ec5cef1f88d1d5e18fd232e5aebd8303f59f"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_representation_variability_injection_evidence_is_complete() -> None:
    outputs = sorted(ARTIFACT_ROOT.glob("representation_variability_injection_v1.shard*of6.jsonl"))
    summaries = sorted(
        ARTIFACT_ROOT.glob("representation_variability_injection_summary_v1.shard*of6.json")
    )
    aggregate = json.loads(AGGREGATE_PATH.read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for path in outputs
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]

    assert len(outputs) == len(summaries) == 6
    assert _sha256(AGGREGATE_PATH) == AGGREGATE_SHA256
    assert aggregate["status"] == "pass"
    assert aggregate["checks"] == {
        "required_model_rows": True,
        "required_models": True,
        "required_tics": True,
        "required_trials": True,
        "zero_downloaded_bytes": True,
        "zero_duplicate_model_trials": True,
        "zero_persisted_embeddings": True,
    }

    keys = {(row["tic_id"], row["scenario_id"], row["model_name"]) for row in rows}
    trials = {(row["tic_id"], row["scenario_id"]) for row in rows}
    assert len(rows) == len(keys) == 384
    assert len(trials) == 192
    assert len({row["tic_id"] for row in rows}) == 48
    assert {row["model_name"] for row in rows} == {"Astromer2", "Chronos-Bolt tiny"}
    assert Counter(row["scenario_id"] for row in rows) == {
        "short_low": 96,
        "short_high": 96,
        "long_low": 96,
        "long_high": 96,
    }
    assert all(row["downloaded_bytes"] == 0 for row in rows)
    assert all(row["persisted_embedding"] is False for row in rows)
    assert all(row["training_authorized"] is False for row in rows)
    assert all(row["production_change_authorized"] is False for row in rows)
    assert all(math.isfinite(float(row["cosine_distance"])) for row in rows)
    assert all(math.isfinite(float(row["l2_distance"])) for row in rows)
    assert all(len(row["original_embedding_sha256"]) == 64 for row in rows)
    assert all(len(row["injected_embedding_sha256"]) == 64 for row in rows)

    summary_rows = [json.loads(path.read_text(encoding="utf-8")) for path in summaries]
    assert {int(summary["shard_index"]) for summary in summary_rows} == set(range(6))
    assert sum(int(summary["selected_tics"]) for summary in summary_rows) == 48
    assert sum(int(summary["unique_trials"]) for summary in summary_rows) == 192
    assert sum(int(summary["model_rows"]) for summary in summary_rows) == 384
    for summary in summary_rows:
        output_path = REPO_ROOT / summary["output_path"]
        assert output_path.is_file()
        assert _sha256(output_path) == summary["output_sha256"]

    assert aggregate["class_counts"] == {"EA": 26, "EB": 9, "EW": 2, "ROT": 10, "SR": 1}
    assert aggregate["depth_order"] == {
        "Astromer2": {
            "comparisons": 96,
            "fraction": 1.0,
            "higher_depth_larger_shift": 96,
        },
        "Chronos-Bolt tiny": {
            "comparisons": 96,
            "fraction": 1.0,
            "higher_depth_larger_shift": 96,
        },
    }
    assert sum(int(row["bls_recovered"]) for row in aggregate["by_scenario"]) == 13
    assert aggregate["training_authorized"] is False
    assert aggregate["production_change_authorized"] is False
