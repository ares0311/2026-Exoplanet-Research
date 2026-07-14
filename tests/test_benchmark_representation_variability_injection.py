"""Offline tests for the cache-only representation/injection benchmark."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from Skills.benchmark_representation_variability_injection import (
    cosine_distance,
    load_contract,
    load_matched_labels,
    select_products,
    select_shard,
    summarize_rows,
    thin_relative_magnitude,
)


def test_merged_contract_is_bounded_and_training_disabled() -> None:
    contract = load_contract(
        Path("metadata/representation_variability_injection_contract_v1.json")
    )
    assert contract["execution_gate"]["required_tics"] == 48
    assert contract["execution_gate"]["required_trials"] == 192
    assert contract["parallel_shape"] == {
        "process_shards": 6,
        "workers_per_shard": 6,
        "partition": "tic_id modulo shard_count",
        "worker_scope": "FITS preparation and blind BLS trials",
        "embedding_scope": (
            "one frozen session per model per shard; inference serialized within the "
            "shard to avoid 36 duplicate model sessions"
        ),
    }
    assert contract["training_authorized"] is False
    assert contract["production_change_authorized"] is False


def test_product_selection_prefers_largest_then_lowest_sector() -> None:
    labels = {1: {"class_code": "EA"}, 2: {"class_code": "ROT"}}
    inventory = [
        {"target_id": "TIC 1", "size_bytes": 10, "sector": 2, "cache_relative_path": "b"},
        {"target_id": "TIC 1", "size_bytes": 10, "sector": 1, "cache_relative_path": "a"},
        {"target_id": "TIC 2", "size_bytes": 12, "sector": 3, "cache_relative_path": "c"},
    ]
    selected = select_products(inventory, labels)
    assert selected[1]["sector"] == 1
    assert selected[2]["cache_relative_path"] == "c"


def test_shards_are_disjoint_and_complete() -> None:
    values = tuple(range(1, 49))
    shards = [set(select_shard(values, index, 6)) for index in range(6)]
    assert set.union(*shards) == set(values)
    assert all(
        not left & right
        for index, left in enumerate(shards)
        for right in shards[index + 1 :]
    )


def test_matched_labels_are_owned_by_pinned_aggregate_shards() -> None:
    overlap = json.loads(
        Path("artifacts/manifests/tess_asassn_preflight_aggregate_v1.json").read_text()
    )
    labels = load_matched_labels(overlap)
    assert len(labels) == 48
    assert {row["training_authorized"] for row in labels.values()} == {False}

    tampered = copy.deepcopy(overlap)
    tampered["shard_summaries"][0]["output_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="shard output hash changed"):
        load_matched_labels(tampered)


def test_even_thinning_preserves_full_baseline_and_shared_median() -> None:
    times = np.linspace(0.0, 27.0, 10_000)
    flux = np.ones_like(times)
    thinned_time, magnitude = thin_relative_magnitude(
        times, flux, reference_median=1.0, max_observations=2048
    )
    assert len(thinned_time) == len(magnitude) == 2048
    assert thinned_time[0] == pytest.approx(0.0)
    assert thinned_time[-1] == pytest.approx(27.0)
    assert np.all(magnitude == 0.0)


def test_cosine_distance_is_zero_for_identity_and_one_for_orthogonal() -> None:
    assert cosine_distance(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == pytest.approx(0.0)
    assert cosine_distance(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)


def test_summary_reports_bls_and_depth_order_without_training() -> None:
    rows = []
    for tic_id in (1, 2):
        for model in ("Astromer2", "Chronos-Bolt tiny"):
            for period, duration, prefix in ((3.0, 2.0, "short"), (10.0, 4.0, "long")):
                for depth, suffix, distance in ((500.0, "low", 0.1), (2000.0, "high", 0.2)):
                    rows.append(
                        {
                            "tic_id": tic_id,
                            "class_code": "EA" if tic_id == 1 else "ROT",
                            "scenario_id": f"{prefix}_{suffix}",
                            "period_days": period,
                            "depth_ppm": depth,
                            "duration_hours": duration,
                            "model_name": model,
                            "bls_recovered": depth == 2000.0,
                            "cosine_distance": distance,
                        }
                    )
    summary = summarize_rows(rows)
    assert summary["unique_tics"] == 2
    assert summary["unique_trials"] == 8
    assert summary["model_rows"] == 16
    assert summary["class_counts"] == {"EA": 1, "ROT": 1}
    assert all(item["fraction"] == 1.0 for item in summary["depth_order"].values())
    high = next(item for item in summary["by_scenario"] if item["scenario_id"] == "short_high")
    assert high["bls_recovery_rate"] == 1.0
