from __future__ import annotations

import json
from pathlib import Path

import pytest
from Skills.representation_pilot import (
    load_config,
    load_predefined_splits,
    run_pilot,
)


def _row(split: str, group: str, label: int, offset: float, n_bins: int = 12) -> dict:
    flux = [offset + (index % 3) * 0.1 for index in range(n_bins)]
    if label:
        flux[n_bins // 2] -= 2.0
    return {
        "split": split,
        "group_key": group,
        "label": label,
        "period_days": 3.0 + offset,
        "duration_hours": 2.0,
        "flux": flux,
    }


def _write_corpus(path: Path, *, leak: bool = False) -> None:
    rows = []
    for split_index, split in enumerate(("train", "val", "test")):
        for index in range(8):
            group = "shared" if leak and index == 0 and split != "test" else f"{split}-{index}"
            rows.append(_row(split, group, index % 2, split_index * 0.01 + index * 0.001))
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _write_config(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "n_bins": 12,
                "d_model": 8,
                "n_heads": 2,
                "n_layers": 1,
                "dim_feedforward": 16,
                "dropout": 0.0,
                "mask_fraction": 0.25,
                "batch_size": 8,
                "pretrain_epochs": 1,
                "probe_epochs": 3,
                "learning_rate": 0.001,
                "probe_learning_rate": 0.03,
                "weight_decay": 0.0001,
                "patience": 1,
                "probe_patience": 2,
                "seed": 7,
                "device": "cpu",
                "top_k": 4,
            }
        ),
        encoding="utf-8",
    )


def test_load_config_rejects_incompatible_attention_width(tmp_path: Path) -> None:
    path = tmp_path / "config.json"
    _write_config(path)
    raw = json.loads(path.read_text())
    raw["d_model"] = 7
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="divisible"):
        load_config(path)


def test_load_splits_rejects_group_leakage(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    _write_corpus(path, leak=True)
    with pytest.raises(ValueError, match="group leakage"):
        load_predefined_splits(path, n_bins=12)


def test_pilot_writes_evidence_without_real_git_push(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    corpus = tmp_path / "corpus.jsonl"
    config = tmp_path / "config.json"
    output = tmp_path / "result.json"
    checkpoint = tmp_path / "checkpoint.pt"
    _write_corpus(corpus)
    _write_config(config)
    reports = []

    result = run_pilot(
        corpus,
        config,
        output,
        checkpoint,
        report_fn=lambda report, path: reports.append((report, path)) or True,
    )

    assert result["dataset_ids"]["pretrain"] == "t1_1_kepler_master_train"
    assert result["split_counts"] == {"train": 8, "val": 8, "test": 8}
    assert result["embedding_linear_probe"]["test"]["top_k"] == 4
    assert result["status"] in {"pass", "does_not_beat_cnn"}
    assert output.exists()
    assert checkpoint.exists()
    assert len(reports) == 1
    assert reports[0][0].script == "representation_pilot"
