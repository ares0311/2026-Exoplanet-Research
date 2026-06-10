"""Tests for development-only grouped CNN experiments."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import Skills.cnn_development_experiment as experiment
from Skills.cnn_training_config import default_config


def _row(example_id: str, tic_id: int, label: int) -> dict:
    return {
        "example_id": example_id,
        "tic_id": tic_id,
        "label": label,
        "flux": [float(label)] * 201,
    }


def test_grouped_folds_keep_tic_rows_together() -> None:
    rows = [
        _row("a1", 1, 0),
        _row("a2", 1, 0),
        _row("b", 2, 1),
        _row("c", 3, 0),
        _row("d", 4, 1),
    ]
    folds = experiment.grouped_folds(rows, folds=2, seed=42)
    fold_by_example = {
        str(row["example_id"]): fold_index
        for fold_index, fold in enumerate(folds)
        for row in fold
    }
    assert fold_by_example["a1"] == fold_by_example["a2"]


def test_experiment_does_not_read_sealed_test(
    tmp_path: Path,
    monkeypatch,
) -> None:
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    manifest = {
        "test_role": "sealed_one_time_production_promotion_holdout",
        "test_evaluated": False,
    }
    (split_dir / "manifest.json").write_text(json.dumps(manifest))
    rows = [
        _row(f"row-{tic_id}", tic_id, tic_id % 2)
        for tic_id in range(1, 13)
    ]
    for name, subset in (("train", rows[:8]), ("val", rows[8:])):
        (split_dir / f"{name}.json").write_text(
            json.dumps({"split": name, "examples": subset})
        )
    (split_dir / "test.json").write_text("THIS MUST NOT BE READ")

    monkeypatch.setattr(
        experiment,
        "train_cnn",
        lambda split, config, checkpoint_dir: SimpleNamespace(
            flag="OK",
            checkpoint_path=str(checkpoint_dir / "best.pt"),
            best_epoch=3,
        ),
    )
    monkeypatch.setattr(
        experiment,
        "run_cnn_inference",
        lambda fluxes, **kwargs: SimpleNamespace(
            flag="OK",
            probabilities=tuple(0.8 if flux[0] == 1.0 else 0.2 for flux in fluxes),
        ),
    )
    monkeypatch.setattr(
        experiment,
        "evaluate_ensemble",
        lambda labels, scores: SimpleNamespace(
            tiers=[SimpleNamespace(auc_roc=1.0, f1=1.0, brier=0.04, ece=0.2)]
        ),
    )

    report = experiment.run_development_experiment(
        split_dir,
        default_config(),
        tmp_path / "experiment",
        folds=2,
    )

    assert report["flag"] == "OK"
    assert report["promotion_holdout_accessed"] is False
    assert report["summary"]["mean_auc"] == 1.0


def test_experiment_requires_sealed_manifest(tmp_path: Path) -> None:
    split_dir = tmp_path / "splits"
    split_dir.mkdir()
    (split_dir / "manifest.json").write_text(
        json.dumps({"test_role": "ordinary_test", "test_evaluated": False})
    )

    try:
        experiment.run_development_experiment(
            split_dir,
            default_config(),
            tmp_path / "output",
        )
    except ValueError as error:
        assert "sealed promotion holdout" in str(error)
    else:
        raise AssertionError("expected the unsealed manifest to be rejected")

