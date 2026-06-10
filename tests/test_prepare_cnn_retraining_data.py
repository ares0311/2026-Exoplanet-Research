"""Tests for preparing a fresh sealed CNN promotion holdout."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from Skills.prepare_cnn_retraining_data import prepare_retraining_data


def _row(example_id: str, tic_id: int, label: int) -> dict:
    return {
        "example_id": example_id,
        "tic_id": tic_id,
        "label": label,
        "phase": [index / 200 for index in range(201)],
        "flux": [float(label)] * 201,
        "source": "test",
        "source_file": "fixture.jsonl",
        "augmentation": None,
        "normalization": "local_median_mad",
    }


def _write_source(path: Path) -> None:
    path.mkdir()
    train = [
        _row(f"train-{tic_id}", tic_id, tic_id % 2)
        for tic_id in range(1, 41)
    ]
    val = [
        _row(f"val-{tic_id}", tic_id, tic_id % 2)
        for tic_id in range(101, 121)
    ]
    opened_test = [
        _row(f"opened-{tic_id}", tic_id, tic_id % 2)
        for tic_id in range(201, 211)
    ]
    for name, rows in (("train", train), ("val", val), ("test", opened_test)):
        (path / f"{name}.json").write_text(
            json.dumps({"split": name, "examples": rows}),
            encoding="utf-8",
        )
    (path / "manifest.json").write_text(
        json.dumps({"split_counts": {"train": 40, "val": 20, "test": 10}}),
        encoding="utf-8",
    )


def test_retires_opened_test_and_seals_fresh_holdout(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    _write_source(source)
    old_test_hash = hashlib.sha256((source / "test.json").read_bytes()).hexdigest()

    manifest = prepare_retraining_data(
        source,
        output,
        seed=17,
        created_at="2026-06-10T00:00:00+00:00",
    )

    assert manifest["retired_test"]["sha256"] == old_test_hash
    assert manifest["retired_test"]["included_in_retraining"] is False
    assert manifest["test_role"] == "sealed_one_time_production_promotion_holdout"
    assert manifest["test_evaluated"] is False
    assert manifest["n_examples"] == 60

    split_rows = {
        name: json.loads((output / f"{name}.json").read_text())["examples"]
        for name in ("train", "val", "test")
    }
    all_ids = {
        str(row["example_id"])
        for rows in split_rows.values()
        for row in rows
    }
    assert not any(example_id.startswith("opened-") for example_id in all_ids)
    assert all(
        str(row["example_id"]).startswith("train-")
        for row in split_rows["test"]
    )


def test_output_has_no_tic_leakage_and_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_source(source)

    prepare_retraining_data(source, tmp_path / "one", seed=29, created_at="fixed")
    prepare_retraining_data(source, tmp_path / "two", seed=29, created_at="fixed")

    tic_sets: list[set[int]] = []
    for name in ("train", "val", "test"):
        first = (tmp_path / "one" / f"{name}.json").read_bytes()
        second = (tmp_path / "two" / f"{name}.json").read_bytes()
        assert first == second
        rows = json.loads(first)["examples"]
        assert {int(row["label"]) for row in rows} == {0, 1}
        tic_sets.append({int(row["tic_id"]) for row in rows})
    assert tic_sets[0].isdisjoint(tic_sets[1])
    assert tic_sets[0].isdisjoint(tic_sets[2])
    assert tic_sets[1].isdisjoint(tic_sets[2])

