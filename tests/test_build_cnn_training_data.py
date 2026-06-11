"""Tests for Skills.build_cnn_training_data."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from Skills.build_cnn_training_data import (
    SplitConfig,
    TrainingExample,
    format_split_summary,
    load_training_examples,
    split_examples,
    write_training_splits,
)


def _snippet(
    *,
    tic_id: int = 1,
    label: int = 1,
    n_bins: int = 5,
    source: str = "tess",
) -> dict:
    return {
        "tic_id": tic_id,
        "label": label,
        "period_days": 5.0,
        "epoch_bjd": 2458000.0,
        "phase": [round(-0.5 + (i + 0.5) / n_bins, 4) for i in range(n_bins)],
        "flux": [0.99 if i == n_bins // 2 else 1.0 for i in range(n_bins)],
        "source": source,
    }


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(f"{json.dumps(row)}\n" for row in rows),
        encoding="utf-8",
    )
    return path


def _examples(n_per_label: int = 4) -> list[TrainingExample]:
    rows = [
        _snippet(tic_id=100 + idx, label=label)
        for label in (0, 1)
        for idx in range(n_per_label)
    ]
    path = _write_json(Path("/private/tmp/cnn_examples.json"), {"snippets": rows})
    return load_training_examples([path])


def test_load_training_examples_from_collector_shape(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "dataset.json", {"snippets": [_snippet(tic_id=42)]})
    examples = load_training_examples([path])
    assert len(examples) == 1
    assert examples[0].tic_id == 42
    assert examples[0].source_file == str(path)


def test_load_training_examples_from_augmenter_shape(tmp_path: Path) -> None:
    augmented = _snippet(tic_id=12, label=0)
    augmented.pop("tic_id")
    augmented.pop("source")
    augmented["original_tic_id"] = 12
    augmented["augmentation"] = "noise"
    path = _write_json(
        tmp_path / "augmented.json",
        [augmented],
    )
    examples = load_training_examples([path])
    assert examples[0].tic_id == 12
    assert examples[0].augmentation == "noise"
    assert examples[0].source == "unknown"


def test_load_training_examples_from_jsonl_and_skip_error_rows(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "snippets.jsonl",
        [
            {**_snippet(tic_id=42), "status": "ok", "disposition": "CP"},
            {"tic_id": 43, "status": "error", "reason": "not found"},
        ],
    )

    examples = load_training_examples([path])

    assert len(examples) == 1
    assert examples[0].tic_id == 42
    assert examples[0].source == "tess"
    assert examples[0].normalization == "local_median_mad"


def test_load_training_examples_rejects_invalid_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"tic_id": 1}\nnot-json\n', encoding="utf-8")

    with pytest.raises(ValueError, match=r"bad\.jsonl:2"):
        load_training_examples([path])


def test_load_training_examples_normalizes_flux(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "dataset.json", {"snippets": [_snippet()]})

    example = load_training_examples([path])[0]

    assert example.flux != tuple(_snippet()["flux"])
    assert example.flux[2] < 0.0


def test_load_training_examples_skips_malformed_rows(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "bad.json",
        {"snippets": [_snippet(), {**_snippet(), "label": 7}, {**_snippet(), "flux": []}]},
    )
    examples = load_training_examples([path])
    assert len(examples) == 1


def test_load_training_examples_requires_matching_phase_flux_lengths(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "bad_lengths.json",
        {"snippets": [{**_snippet(), "phase": [0.0, 0.1], "flux": [1.0]}]},
    )
    assert load_training_examples([path]) == []


def test_split_examples_is_deterministic(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "dataset.json",
        {"snippets": [_snippet(tic_id=i, label=i % 2) for i in range(12)]},
    )
    examples = load_training_examples([path])
    config = SplitConfig(seed=123)
    first = split_examples(examples, config)
    second = split_examples(examples, config)
    assert first == second


def test_split_examples_stratifies_when_possible(tmp_path: Path) -> None:
    rows = [
        _snippet(tic_id=100 * label + idx, label=label)
        for label in (0, 1)
        for idx in range(4)
    ]
    path = _write_json(
        tmp_path / "dataset.json",
        {"snippets": rows},
    )
    splits = split_examples(load_training_examples([path]), SplitConfig(seed=1))
    for split_rows in splits.values():
        assert {example.label for example in split_rows} == {0, 1}


def test_split_examples_keeps_same_tic_in_one_split(tmp_path: Path) -> None:
    rows = [
        _snippet(tic_id=500, label=0),
        _snippet(tic_id=500, label=1),
        *[
            _snippet(tic_id=100 * label + idx + 1, label=label)
            for label in (0, 1)
            for idx in range(5)
        ],
    ]
    path = _write_json(tmp_path / "dataset.json", {"snippets": rows})

    splits = split_examples(load_training_examples([path]), SplitConfig(seed=8))

    containing_splits = {
        split_name
        for split_name, split_rows in splits.items()
        if any(example.tic_id == 500 for example in split_rows)
    }
    assert len(containing_splits) == 1
    assert sum(
        example.tic_id == 500 for split_rows in splits.values() for example in split_rows
    ) == 2


def test_split_examples_non_stratified_preserves_total(tmp_path: Path) -> None:
    path = _write_json(
        tmp_path / "dataset.json",
        {"snippets": [_snippet(tic_id=i, label=i % 2) for i in range(10)]},
    )
    splits = split_examples(load_training_examples([path]), SplitConfig(stratify=False))
    assert sum(len(rows) for rows in splits.values()) == 10


def test_split_config_rejects_bad_fraction_sum() -> None:
    with pytest.raises(ValueError, match="sum to 1.0"):
        split_examples([], SplitConfig(train_fraction=0.5, val_fraction=0.2, test_fraction=0.2))


def test_split_config_rejects_negative_fraction() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        split_examples([], SplitConfig(train_fraction=1.1, val_fraction=-0.1, test_fraction=0.0))


def test_write_training_splits_creates_files(tmp_path: Path) -> None:
    input_path = _write_json(
        tmp_path / "dataset.json",
        {"snippets": [_snippet(tic_id=i, label=i % 2) for i in range(8)]},
    )
    examples = load_training_examples([input_path])
    manifest = write_training_splits(
        examples,
        tmp_path / "splits",
        config=SplitConfig(seed=99),
        created_at="2026-05-20T00:00:00+00:00",
    )
    assert manifest["split_files"] == {
        "train": "train.json",
        "val": "val.json",
        "test": "test.json",
    }
    assert Path(tmp_path / "splits" / manifest["split_files"]["train"]).exists()
    assert Path(tmp_path / "splits" / "manifest.json").exists()


def test_write_training_splits_manifest_records_provenance(tmp_path: Path) -> None:
    input_path = _write_json(tmp_path / "dataset.json", {"snippets": [_snippet()]})
    manifest = write_training_splits(
        load_training_examples([input_path]),
        tmp_path / "splits",
        created_at="2026-05-20T00:00:00+00:00",
    )
    assert manifest["source_files"] == [str(input_path)]
    assert manifest["live_services"] is False
    assert "no candidate confirmation" in manifest["language_guardrail"]


def test_written_split_rows_are_json_ready(tmp_path: Path) -> None:
    input_path = _write_json(tmp_path / "dataset.json", {"snippets": [_snippet()]})
    manifest = write_training_splits(load_training_examples([input_path]), tmp_path / "splits")
    train_path = tmp_path / "splits" / manifest["split_files"]["train"]
    train_payload = json.loads(train_path.read_text())
    assert isinstance(train_payload["examples"], list)
    assert "phase" in train_payload["examples"][0]


def test_format_split_summary_contains_counts(tmp_path: Path) -> None:
    input_path = _write_json(
        tmp_path / "dataset.json",
        {"snippets": [_snippet(tic_id=i, label=i % 2) for i in range(6)]},
    )
    manifest = write_training_splits(load_training_examples([input_path]), tmp_path / "splits")
    summary = format_split_summary(manifest)
    assert "Total examples: 6" in summary
    assert "Live services: False" in summary


def test_load_training_examples_from_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "tess_snippets.jsonl"
    lines = [
        json.dumps(_snippet(tic_id=1, label=1)),
        json.dumps({"tic_id": 2, "status": "error", "reason": "no data"}),
        json.dumps(_snippet(tic_id=3, label=0)),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    examples = load_training_examples([path])
    assert len(examples) == 2
    assert {e.tic_id for e in examples} == {1, 3}


def test_load_training_examples_jsonl_201_bins(tmp_path: Path) -> None:
    n = 201
    row = {
        "tic_id": 99,
        "label": 1,
        "phase": [round(-0.5 + i / n, 6) for i in range(n)],
        "flux": [1.0] * n,
        "source": "tess",
    }
    path = tmp_path / "canonical.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    examples = load_training_examples([path])
    assert len(examples) == 1
    assert len(examples[0].phase) == 201
