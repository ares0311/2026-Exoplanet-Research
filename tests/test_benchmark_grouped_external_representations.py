"""Offline tests for the grouped external-representation benchmark."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from Skills.benchmark_grouped_external_representations import (
    aggregate_shards,
    choose_threshold,
    evaluate_scores,
    fit_linear_probe,
    load_contract,
    phase_bin_relative_magnitude,
    select_population,
    select_shard,
)


def _canonical_sha256(rows: list[dict[str, object]]) -> str:
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def test_merged_contract_is_bounded_grouped_and_training_disabled() -> None:
    contract = load_contract(Path("metadata/grouped_external_representation_contract_v1.json"))
    assert contract["population"]["required_unique_kics"] == 1536
    assert contract["population"]["cache_inventory"]["bytes"] == 10_398_406_656
    assert contract["parallel_shape"]["process_shards"] == 6
    assert contract["parallel_shape"]["fits_workers_per_shard"] == 6
    assert contract["probe"]["minimum_absolute_improvement"] == pytest.approx(0.01)
    assert contract["storage"]["downloads_authorized"] is False
    assert contract["broad_extraction_authorized"] is False
    assert contract["training_authorized"] is False
    assert contract["production_change_authorized"] is False


def test_selection_is_deterministic_balanced_and_group_disjoint(tmp_path: Path) -> None:
    corpus_rows: list[dict[str, object]] = []
    target_id = 100
    for split in ("train", "val", "test"):
        for label in (0, 1):
            for candidate in range(2):
                target_id += 1
                corpus_rows.append(
                    {
                        "source_row_id": f"{split}-{label}-{candidate}",
                        "target_name": f"KIC {target_id}",
                        "group_key": f"kepler:kic:{target_id}",
                        "split": split,
                        "label": label,
                        "target_id": target_id,
                    }
                )
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("".join(json.dumps(row) + "\n" for row in corpus_rows))
    contract: dict[str, object] = {
        "population": {
            "selection_seed": "test-seed",
            "rows_per_label": {"train": 1, "val": 1, "test": 1},
            "selection_sha256": "pending",
        }
    }
    population = contract["population"]
    assert isinstance(population, dict)
    selected_records: list[dict[str, object]] = []
    for split in ("test", "train", "val"):
        for label in (0, 1):
            candidates = [
                row for row in corpus_rows if row["split"] == split and row["label"] == label
            ]
            candidates.sort(
                key=lambda row: hashlib.sha256(
                    f"test-seed|{row['group_key']}".encode()
                ).hexdigest()
            )
            row = candidates[0]
            selected_records.append(
                {
                    "source_row_id": row["source_row_id"],
                    "group_key": row["group_key"],
                    "split": row["split"],
                    "label": row["label"],
                    "target_id": row["target_id"],
                }
            )
    selected_records.sort(
        key=lambda row: (str(row["split"]), int(row["label"]), str(row["group_key"]))
    )
    population["selection_sha256"] = _canonical_sha256(selected_records)
    selected = select_population(corpus, {int(row["target_id"]) for row in corpus_rows}, contract)
    assert len(selected) == 6
    assert len({row["group_key"] for row in selected}) == 6
    assert {(row["split"], row["label"]) for row in selected} == {
        (split, label) for split in ("train", "val", "test") for label in (0, 1)
    }


def test_shards_are_disjoint_and_complete() -> None:
    rows = [{"target_id": value} for value in range(1, 97)]
    shards = [select_shard(rows, index, 6) for index in range(6)]
    ids = [{int(row["target_id"]) for row in shard} for shard in shards]
    assert set.union(*ids) == set(range(1, 97))
    assert all(not left & right for index, left in enumerate(ids) for right in ids[index + 1 :])


def test_phase_binning_produces_finite_relative_magnitude() -> None:
    phase = np.linspace(-0.5, 0.5, 1000, endpoint=False)
    flux = 1.0 - 0.01 * (np.abs(phase) < 0.05)
    centers, magnitude = phase_bin_relative_magnitude(
        phase, flux, n_bins=100, minimum_filled_fraction=0.95
    )
    assert centers.shape == magnitude.shape == (100,)
    assert np.all(np.isfinite(magnitude))
    assert float(np.max(magnitude)) > 0.0


def test_probe_threshold_and_metrics_use_validation_only() -> None:
    train_x = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    train_y = np.array([0.0, 0.0, 1.0, 1.0])
    validation_x = np.array([[-1.5], [-0.5], [0.5], [1.5]])
    validation_y = np.array([0.0, 0.0, 1.0, 1.0])
    weights, bias, _epoch, auc = fit_linear_probe(
        train_x,
        train_y,
        validation_x,
        validation_y,
        {"epochs": 50, "learning_rate": 0.1, "weight_decay": 0.001, "patience": 5},
        progress=False,
    )
    probabilities = 1.0 / (1.0 + np.exp(-(validation_x @ weights + bias)))
    threshold = choose_threshold(validation_y.astype(np.int8), probabilities)
    metrics = evaluate_scores(
        validation_y.astype(np.int8), probabilities, threshold=threshold, top_k=2
    )
    assert auc == pytest.approx(1.0)
    assert metrics["roc_auc"] == pytest.approx(1.0)
    assert metrics["top_k_positives"] == 2


def test_aggregate_reconciles_and_removes_temporary_embeddings(tmp_path: Path) -> None:
    contract_path = tmp_path / "contract.json"
    temp_root = tmp_path / "temp"
    summary_path = tmp_path / "summary.json"
    aggregate_path = tmp_path / "aggregate.json"
    contract = {
        "schema_version": 1,
        "inputs": {},
        "population": {},
        "preprocessing": {},
        "probe": {
            "epochs": 30,
            "learning_rate": 0.1,
            "weight_decay": 0.001,
            "patience": 5,
            "top_k": 2,
            "minimum_absolute_improvement": 0.01,
        },
        "parallel_shape": {"process_shards": 6},
        "execution_gate": {
            "required_rows": 12,
            "required_split_rows": {"train": 4, "val": 4, "test": 4},
            "required_split_label_rows": {
                split: {"0": 2, "1": 2} for split in ("train", "val", "test")
            },
        },
        "limitations": ["synthetic test"],
        "broad_extraction_authorized": False,
        "training_authorized": False,
        "production_change_authorized": False,
    }
    contract_path.write_text(json.dumps(contract))
    splits = np.array(["train"] * 4 + ["val"] * 4 + ["test"] * 4)
    labels = np.array([0, 1, 0, 1] * 3, dtype=np.int8)
    signal = labels.astype(np.float32) * 2.0 - 1.0
    temp_root.mkdir()
    for shard_index in range(6):
        indexes = np.array([shard_index * 2, shard_index * 2 + 1])
        temp_path = temp_root / f"features_v1.shard{shard_index + 1}of6.npz"
        np.savez_compressed(
            temp_path,
            source_row_id=np.array([f"row-{index}" for index in indexes]),
            group_key=np.array([f"kepler:kic:{index}" for index in indexes]),
            split=splits[indexes],
            label=labels[indexes],
            target_id=indexes,
            chronos=signal[indexes, None],
            astromer=signal[indexes, None],
            classical=np.zeros((2, 1), dtype=np.float32),
            cnn_probability=np.where(labels[indexes] == 1, 0.6, 0.4),
        )
        digest = hashlib.sha256(temp_path.read_bytes()).hexdigest()
        shard_summary = summary_path.with_name(f"summary.shard{shard_index + 1}of6.json")
        shard_summary.write_text(
            json.dumps(
                {
                    "status": "success",
                    "shard_index": shard_index,
                    "shard_count": 6,
                    "contract_sha256": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
                    "temp_sha256": digest,
                    "downloaded_bytes": 0,
                    "production_change_authorized": False,
                }
            )
        )
    artifact = aggregate_shards(
        contract_path,
        temp_root,
        summary_path,
        aggregate_path,
        shard_count=6,
        report_fn=lambda *_args, **_kwargs: True,
    )
    assert artifact["rows"] == 12
    assert artifact["persisted_embeddings"] == 0
    assert artifact["test_opened_once"] is True
    assert not list(temp_root.glob("*.npz"))
