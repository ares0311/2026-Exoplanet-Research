"""Run grouped CNN cross-validation without reading the promotion holdout."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

try:
    from Skills.cnn_inference_batcher import run_cnn_inference
    from Skills.cnn_training_config import CnnTrainingConfig, load_config
    from Skills.model_ensemble_evaluator import evaluate_ensemble
    from Skills.train_cnn import train_cnn
except ModuleNotFoundError:
    from cnn_inference_batcher import run_cnn_inference
    from cnn_training_config import CnnTrainingConfig, load_config
    from model_ensemble_evaluator import evaluate_ensemble
    from train_cnn import train_cnn


@dataclass(frozen=True)
class FoldResult:
    """Metrics from one grouped development fold."""

    fold: int
    n_train: int
    n_val: int
    auc: float
    f1: float
    brier: float
    ece: float
    best_epoch: int


def run_development_experiment(
    split_dir: Path,
    config: CnnTrainingConfig,
    output_dir: Path,
    *,
    folds: int = 3,
) -> dict[str, Any]:
    """Train grouped folds from train+val only and write an audit report."""
    if folds < 2:
        raise ValueError("folds must be at least 2")
    split_dir = Path(split_dir)
    manifest = _load_object(split_dir / "manifest.json")
    if manifest.get("test_role") != "sealed_one_time_production_promotion_holdout":
        raise ValueError("split manifest does not declare a sealed promotion holdout")
    if manifest.get("test_evaluated") is not False:
        raise ValueError("promotion holdout is not sealed")

    rows = _load_examples(split_dir / "train.json") + _load_examples(
        split_dir / "val.json"
    )
    fold_rows = grouped_folds(rows, folds=folds, seed=config.seed)
    output_dir = Path(output_dir)
    results: list[FoldResult] = []
    for fold_index, validation_rows in enumerate(fold_rows):
        validation_ids = {str(row["example_id"]) for row in validation_rows}
        training_rows = [
            row for row in rows if str(row["example_id"]) not in validation_ids
        ]
        fold_dir = output_dir / f"fold_{fold_index + 1}"
        split_work_dir = fold_dir / "split"
        _write_split(split_work_dir / "train.json", "train", training_rows)
        _write_split(split_work_dir / "val.json", "val", validation_rows)
        checkpoint_path, best_epoch = _train_or_resume(
            split_work_dir,
            config,
            checkpoint_dir=fold_dir / "checkpoints",
        )
        inference = run_cnn_inference(
            [list(row["flux"]) for row in validation_rows],
            model_path=checkpoint_path,
            batch_size=config.batch_size,
            n_bins=config.n_bins,
        )
        if inference.flag != "OK":
            raise RuntimeError(f"fold {fold_index + 1} inference failed")
        labels = [int(row["label"]) for row in validation_rows]
        metrics = evaluate_ensemble(labels, {"cnn": list(inference.probabilities)})
        score = metrics.tiers[0]
        results.append(
            FoldResult(
                fold=fold_index + 1,
                n_train=len(training_rows),
                n_val=len(validation_rows),
                auc=score.auc_roc,
                f1=score.f1,
                brier=score.brier,
                ece=score.ece,
                best_epoch=best_epoch,
            )
        )

    aucs = [result.auc for result in results]
    report = {
        "created_at": datetime.now(UTC).isoformat(),
        "split_manifest_sha256": _sha256(split_dir / "manifest.json"),
        "config": _config_payload(config),
        "folds": [asdict(result) for result in results],
        "summary": {
            "mean_auc": mean(aucs),
            "std_auc": pstdev(aucs),
            "mean_f1": mean(result.f1 for result in results),
            "mean_brier": mean(result.brier for result in results),
            "mean_ece": mean(result.ece for result in results),
        },
        "promotion_holdout_accessed": False,
        "flag": "OK",
    }
    _atomic_write_json(output_dir / "experiment.json", report)
    return report


def grouped_folds(
    rows: list[dict[str, Any]],
    *,
    folds: int,
    seed: int,
) -> list[list[dict[str, Any]]]:
    """Assign complete TIC groups to deterministic label-stratified folds."""
    import random

    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["tic_id"]), []).append(row)
    strata: dict[tuple[int, ...], list[list[dict[str, Any]]]] = {}
    for group in grouped.values():
        signature = tuple(sorted({int(row["label"]) for row in group}))
        strata.setdefault(signature, []).append(group)
    assigned: list[list[dict[str, Any]]] = [[] for _ in range(folds)]
    for index, signature in enumerate(sorted(strata)):
        groups = sorted(
            strata[signature],
            key=lambda group: int(group[0]["tic_id"]),
        )
        random.Random(seed + index).shuffle(groups)
        for group_index, group in enumerate(groups):
            assigned[group_index % folds].extend(group)
    for rows_in_fold in assigned:
        rows_in_fold.sort(key=lambda row: str(row["example_id"]))
    return assigned


def format_report(report: dict[str, Any]) -> str:
    """Format a concise experiment summary."""
    summary = report["summary"]
    return "\n".join(
        (
            "## CNN Development Experiment",
            "",
            f"- Mean AUC: {summary['mean_auc']:.4f} +/- {summary['std_auc']:.4f}",
            f"- Mean F1: {summary['mean_f1']:.4f}",
            f"- Mean Brier: {summary['mean_brier']:.4f}",
            f"- Mean ECE: {summary['mean_ece']:.4f}",
            "- Promotion holdout accessed: False",
            "",
        )
    )


def _config_payload(config: CnnTrainingConfig) -> dict[str, Any]:
    try:
        from Skills.cnn_training_config import _config_to_dict
    except ModuleNotFoundError:
        from cnn_training_config import _config_to_dict

    return _config_to_dict(config)


def _train_or_resume(
    split_dir: Path,
    config: CnnTrainingConfig,
    *,
    checkpoint_dir: Path,
) -> tuple[Path, int]:
    best_path = checkpoint_dir / "best.pt"
    metrics_path = checkpoint_dir / "metrics.json"
    if best_path.is_file() and metrics_path.is_file():
        metrics = _load_object(metrics_path).get("checkpoints")
        if isinstance(metrics, list) and metrics:
            valid = [row for row in metrics if isinstance(row, dict)]
            if valid:
                key = (
                    (lambda row: float(row["val_auc"]))
                    if config.selection_metric == "val_auc"
                    else (lambda row: -float(row["val_loss"]))
                )
                best = max(valid, key=key)
                return best_path, int(best["epoch"])

    training_result = train_cnn(
        split_dir,
        config,
        checkpoint_dir=checkpoint_dir,
    )
    if training_result.flag != "OK":
        raise RuntimeError("fold training failed")
    return Path(training_result.checkpoint_path), training_result.best_epoch


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _load_examples(path: Path) -> list[dict[str, Any]]:
    rows = _load_object(path).get("examples")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} must contain an examples list")
    return rows


def _write_split(path: Path, split: str, rows: list[dict[str, Any]]) -> None:
    _atomic_write_json(path, {"split": split, "examples": rows})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except Exception:
        with suppress(OSError):
            os.unlink(temporary)
        raise


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run grouped CNN development-only cross-validation."
    )
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=3)
    args = parser.parse_args(argv)
    report = run_development_experiment(
        args.split_dir,
        load_config(args.config),
        args.output_dir,
        folds=args.folds,
    )
    print(format_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
