"""Prepare a fresh CNN development split and sealed promotion holdout.

The source split's opened test partition is retired. A new promotion holdout is
selected only from the former training partition, while the remaining former
training rows and all former validation rows are repartitioned into development
train/validation sets. The new test partition must remain unopened until the
one-time production promotion evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def prepare_retraining_data(
    source_dir: Path,
    output_dir: Path,
    *,
    seed: int = 20_260_610,
    promotion_fraction: float = 0.20,
    validation_fraction: float = 0.20,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Write development splits and a sealed promotion holdout."""
    if not 0.05 <= promotion_fraction <= 0.40:
        raise ValueError("promotion_fraction must be between 0.05 and 0.40")
    if not 0.05 <= validation_fraction <= 0.40:
        raise ValueError("validation_fraction must be between 0.05 and 0.40")

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    source_manifest_path = source_dir / "manifest.json"
    source_train_path = source_dir / "train.json"
    source_val_path = source_dir / "val.json"
    retired_test_path = source_dir / "test.json"
    for path in (
        source_manifest_path,
        source_train_path,
        source_val_path,
        retired_test_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_manifest = _load_object(source_manifest_path)
    source_train = _load_examples(source_train_path)
    source_val = _load_examples(source_val_path)

    promotion_tics = _select_tic_groups(
        source_train,
        fraction=promotion_fraction,
        seed=seed,
    )
    promotion = [row for row in source_train if int(row["tic_id"]) in promotion_tics]
    development_pool = [
        row for row in source_train if int(row["tic_id"]) not in promotion_tics
    ] + source_val
    validation_tics = _select_tic_groups(
        development_pool,
        fraction=validation_fraction,
        seed=seed + 1,
    )
    validation = [
        row for row in development_pool if int(row["tic_id"]) in validation_tics
    ]
    training = [
        row for row in development_pool if int(row["tic_id"]) not in validation_tics
    ]

    splits = {
        "train": _sorted_rows(training),
        "val": _sorted_rows(validation),
        "test": _sorted_rows(promotion),
    }
    _validate_partition(splits)

    created_at = created_at or datetime.now(UTC).isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)
    split_files: dict[str, str] = {}
    for split_name, rows in splits.items():
        split_path = output_dir / f"{split_name}.json"
        _atomic_write_json(
            split_path,
            {"split": split_name, "created_at": created_at, "examples": rows},
        )
        split_files[split_name] = split_path.name

    all_rows = [row for rows in splits.values() for row in rows]
    manifest = {
        "created_at": created_at,
        "n_examples": len(all_rows),
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "label_counts": _label_counts(all_rows),
        "split_label_counts": {
            name: _label_counts(rows) for name, rows in splits.items()
        },
        "config": {
            "seed": seed,
            "promotion_fraction": promotion_fraction,
            "validation_fraction": validation_fraction,
            "stratify": True,
            "group_key": "tic_id",
        },
        "split_files": split_files,
        "live_services": False,
        "source_files": [str(source_train_path), str(source_val_path)],
        "source_manifest_sha256": _sha256(source_manifest_path),
        "source_split_counts": source_manifest.get("split_counts"),
        "retired_test": {
            "path": str(retired_test_path),
            "sha256": _sha256(retired_test_path),
            "status": "retired_after_opened_evaluation",
            "included_in_retraining": False,
        },
        "test_role": "sealed_one_time_production_promotion_holdout",
        "test_evaluated": False,
        "development_policy": (
            "hyperparameter selection may use train and val only; test must remain "
            "unopened until the final one-time production promotion evaluation"
        ),
        "language_guardrail": (
            "offline CNN preparation only; no candidate confirmation or discovery claim"
        ),
    }
    _atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest


def format_summary(manifest: dict[str, Any]) -> str:
    """Format a concise retraining split summary."""
    counts = manifest["split_counts"]
    return "\n".join(
        (
            "## CNN Retraining Split Summary",
            "",
            f"- Train/val/sealed promotion: {counts['train']} / {counts['val']} / "
            f"{counts['test']}",
            f"- Retired opened test: {manifest['retired_test']['sha256']}",
            f"- Promotion holdout evaluated: {manifest['test_evaluated']}",
            "- Live services: False",
            "",
        )
    )


def _load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _load_examples(path: Path) -> list[dict[str, Any]]:
    payload = _load_object(path)
    rows = payload.get("examples")
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} must contain an examples list")
    return rows


def _select_tic_groups(
    rows: list[dict[str, Any]],
    *,
    fraction: float,
    seed: int,
) -> set[int]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["tic_id"]), []).append(row)

    strata: dict[tuple[int, ...], list[int]] = {}
    for tic_id, tic_rows in grouped.items():
        signature = tuple(sorted({int(row["label"]) for row in tic_rows}))
        strata.setdefault(signature, []).append(tic_id)

    selected: set[int] = set()
    for stratum_index, signature in enumerate(sorted(strata)):
        tic_ids = sorted(strata[signature])
        random.Random(seed + stratum_index).shuffle(tic_ids)
        count = round(len(tic_ids) * fraction)
        if len(tic_ids) >= 2:
            count = min(max(count, 1), len(tic_ids) - 1)
        selected.update(tic_ids[:count])
    return selected


def _validate_partition(splits: dict[str, list[dict[str, Any]]]) -> None:
    seen_ids: set[str] = set()
    seen_tics: dict[int, str] = {}
    for split_name, rows in splits.items():
        if {int(row["label"]) for row in rows} != {0, 1}:
            raise ValueError(f"{split_name} must contain both labels")
        for row in rows:
            example_id = str(row["example_id"])
            if example_id in seen_ids:
                raise ValueError(f"duplicate example_id: {example_id}")
            seen_ids.add(example_id)
            tic_id = int(row["tic_id"])
            previous = seen_tics.get(tic_id)
            if previous is not None and previous != split_name:
                raise ValueError(f"TIC {tic_id} leaks between {previous} and {split_name}")
            seen_tics[tic_id] = split_name


def _sorted_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: str(row["example_id"]))


def _label_counts(rows: list[dict[str, Any]]) -> dict[int, int]:
    return {
        label: sum(int(row["label"]) == label for row in rows)
        for label in (0, 1)
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        description="Prepare CNN development data and a sealed promotion holdout."
    )
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20_260_610)
    parser.add_argument("--promotion-fraction", type=float, default=0.20)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    args = parser.parse_args(argv)
    manifest = prepare_retraining_data(
        args.source_dir,
        args.output_dir,
        seed=args.seed,
        promotion_fraction=args.promotion_fraction,
        validation_fraction=args.validation_fraction,
    )
    print(format_summary(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
