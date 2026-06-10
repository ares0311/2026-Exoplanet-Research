"""Assemble offline CNN train/validation/test splits from labelled snippets.

This Skill does not fetch live data. It consumes existing local JSON artifacts
from ``labelled_lc_collector.py`` or ``cnn_feature_augmenter.py``, validates
their phase/flux arrays, and writes deterministic split files for later CNN
training once the label-count gate opens.

Public API
----------
TrainingExample(...)
SplitConfig(...)
load_training_examples(paths) -> list[TrainingExample]
split_examples(examples, config) -> dict[str, tuple[TrainingExample, ...]]
write_training_splits(examples, output_dir, *, config, created_at) -> dict[str, Any]
format_split_summary(manifest) -> str
"""
from __future__ import annotations

import json
import os
import random
import tempfile
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from Skills.snippet_normalizer import normalize_snippet
except ModuleNotFoundError:  # Direct script execution adds Skills/ to sys.path.
    from snippet_normalizer import normalize_snippet


@dataclass(frozen=True)
class TrainingExample:
    """Validated CNN training example from an existing local snippet artifact."""

    example_id: str
    tic_id: int
    label: int
    phase: tuple[float, ...]
    flux: tuple[float, ...]
    source: str
    source_file: str
    augmentation: str | None = None
    normalization: str = "local_median_mad"


@dataclass(frozen=True)
class SplitConfig:
    """Deterministic split configuration."""

    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42
    stratify: bool = True


def load_training_examples(paths: list[Path | str]) -> list[TrainingExample]:
    """Load, normalize, and validate examples from local JSON or JSONL files."""
    examples: list[TrainingExample] = []
    for path_like in paths:
        path = Path(path_like)
        rows = _load_candidate_rows(path)
        for index, row in enumerate(rows):
            example = _training_example(row, source_file=str(path), row_index=index)
            if example is not None:
                examples.append(example)
    return examples


def split_examples(
    examples: list[TrainingExample],
    config: SplitConfig | None = None,
) -> dict[str, tuple[TrainingExample, ...]]:
    """Split examples into deterministic train/validation/test partitions."""
    config = config or SplitConfig()
    _validate_config(config)
    example_groups = _group_examples(examples)
    groups: list[list[list[TrainingExample]]]
    if config.stratify:
        by_label_signature: dict[tuple[int, ...], list[list[TrainingExample]]] = {}
        for example_group in example_groups:
            signature = tuple(sorted({example.label for example in example_group}))
            by_label_signature.setdefault(signature, []).append(example_group)
        groups = [by_label_signature[key] for key in sorted(by_label_signature)]
    else:
        groups = [example_groups]

    splits: dict[str, list[TrainingExample]] = {"train": [], "val": [], "test": []}
    for group_index, stratum in enumerate(groups):
        shuffled = list(stratum)
        random.Random(config.seed + group_index).shuffle(shuffled)
        train_n, val_n, test_n = _split_counts(len(shuffled), config)
        split_groups = {
            "train": shuffled[:train_n],
            "val": shuffled[train_n : train_n + val_n],
            "test": shuffled[train_n + val_n : train_n + val_n + test_n],
        }
        for split_name, assigned_groups in split_groups.items():
            for assigned_group in assigned_groups:
                splits[split_name].extend(assigned_group)

    for split_name in splits:
        splits[split_name].sort(key=lambda example: example.example_id)
    return {name: tuple(rows) for name, rows in splits.items()}


def write_training_splits(
    examples: list[TrainingExample],
    output_dir: Path | str,
    *,
    config: SplitConfig | None = None,
    created_at: str | None = None,
) -> dict[str, Any]:
    """Write split JSON files plus a manifest and return the manifest payload."""
    config = config or SplitConfig()
    created_at = created_at or datetime.now(UTC).isoformat()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    splits = split_examples(examples, config)
    split_files: dict[str, str] = {}

    for split_name, split_rows in splits.items():
        path = output / f"{split_name}.json"
        split_payload = {
            "split": split_name,
            "created_at": created_at,
            "examples": [_example_payload(example) for example in split_rows],
        }
        _atomic_write_json(path, split_payload)
        split_files[split_name] = path.name

    manifest = {
        "created_at": created_at,
        "n_examples": len(examples),
        "split_counts": {name: len(rows) for name, rows in splits.items()},
        "label_counts": _label_counts(examples),
        "split_label_counts": {
            name: _label_counts(list(rows)) for name, rows in splits.items()
        },
        "config": asdict(config),
        "split_files": split_files,
        "live_services": False,
        "source_files": sorted({example.source_file for example in examples}),
        "language_guardrail": (
            "offline CNN preparation only; no candidate confirmation or discovery claim"
        ),
    }
    _atomic_write_json(output / "manifest.json", manifest)
    return manifest


def format_split_summary(manifest: dict[str, Any]) -> str:
    """Format a compact Markdown summary for assembled CNN split artifacts."""
    counts = manifest["split_counts"]
    labels = manifest["label_counts"]
    lines = [
        "## CNN Training Split Summary",
        "",
        f"- Total examples: {manifest['n_examples']}",
        f"- Train/val/test: {counts['train']} / {counts['val']} / {counts['test']}",
        f"- Positive labels: {labels.get(1, 0)}",
        f"- Negative labels: {labels.get(0, 0)}",
        f"- Live services: {manifest['live_services']}",
    ]
    return "\n".join(lines) + "\n"


def _candidate_rows(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict) and isinstance(data.get("snippets"), list):
        data = data["snippets"]
    elif isinstance(data, dict) and isinstance(data.get("examples"), list):
        data = data["examples"]
    elif isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []
    return [row for row in data if isinstance(row, dict)]


def _load_candidate_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() != ".jsonl":
        return _candidate_rows(json.loads(text))

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: invalid JSONL row") from exc
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _training_example(
    row: dict[str, Any],
    *,
    source_file: str,
    row_index: int,
) -> TrainingExample | None:
    label = _label(row.get("label"))
    phase = _float_tuple(row.get("phase"))
    flux = _float_tuple(row.get("flux"))
    if label is None or not phase or not flux or len(phase) != len(flux):
        return None
    tic_id = _tic_id(row)
    source = str(
        row.get("source")
        or row.get("label_source")
        or ("tess" if row.get("disposition") else "unknown")
    )
    normalized = normalize_snippet(
        tic_id=str(tic_id),
        label=label,
        source=source,
        phase=list(phase),
        flux=list(flux),
        n_bins=len(phase),
        min_oot_points=min(10, max(1, len(phase) // 2)),
    )
    if normalized.flag != "OK":
        return None
    augmentation = row.get("augmentation")
    example_id = str(
        row.get("example_id")
        or row.get("candidate_id")
        or f"{Path(source_file).stem}_{row_index}_TIC{tic_id}_{label}"
    )
    return TrainingExample(
        example_id=example_id,
        tic_id=tic_id,
        label=label,
        phase=normalized.phase,
        flux=normalized.flux,
        source=source,
        source_file=source_file,
        augmentation=str(augmentation) if augmentation is not None else None,
        normalization=normalized.normalization,
    )


def _group_examples(examples: list[TrainingExample]) -> list[list[TrainingExample]]:
    grouped: dict[tuple[str, int | str], list[TrainingExample]] = {}
    for example in examples:
        key: tuple[str, int | str] = (
            ("tic", example.tic_id)
            if example.tic_id > 0
            else ("example", example.example_id)
        )
        grouped.setdefault(key, []).append(example)
    return [grouped[key] for key in sorted(grouped, key=lambda item: (item[0], str(item[1])))]


def _label(value: Any) -> int | None:
    try:
        label = int(value)
    except (TypeError, ValueError):
        return None
    return label if label in {0, 1} else None


def _tic_id(row: dict[str, Any]) -> int:
    value = row.get("tic_id", row.get("original_tic_id", 0))
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float_tuple(value: Any) -> tuple[float, ...]:
    if not isinstance(value, list | tuple):
        return ()
    result: list[float] = []
    for item in value:
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            return ()
    return tuple(result)


def _validate_config(config: SplitConfig) -> None:
    fractions = [config.train_fraction, config.val_fraction, config.test_fraction]
    if any(fraction < 0.0 for fraction in fractions):
        raise ValueError("split fractions must be non-negative")
    total = sum(fractions)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"split fractions must sum to 1.0, got {total:.6f}")


def _split_counts(n_examples: int, config: SplitConfig) -> tuple[int, int, int]:
    fractions = [config.train_fraction, config.val_fraction, config.test_fraction]
    raw = [n_examples * fraction for fraction in fractions]
    counts = [int(value) for value in raw]
    remaining = n_examples - sum(counts)
    order = sorted(range(3), key=lambda idx: (raw[idx] - counts[idx], -idx), reverse=True)
    for idx in order[:remaining]:
        counts[idx] += 1
    positive_splits = [idx for idx, fraction in enumerate(fractions) if fraction > 0.0]
    if n_examples >= len(positive_splits):
        for idx in positive_splits:
            if counts[idx] == 0:
                donor = max(range(3), key=lambda donor_idx: counts[donor_idx])
                if counts[donor] > 1:
                    counts[donor] -= 1
                    counts[idx] += 1
    return counts[0], counts[1], counts[2]


def _label_counts(examples: list[TrainingExample]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for example in examples:
        counts[example.label] = counts.get(example.label, 0) + 1
    return counts


def _example_payload(example: TrainingExample) -> dict[str, Any]:
    payload = asdict(example)
    payload["phase"] = list(example.phase)
    payload["flux"] = list(example.flux)
    return payload


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp, path)
    except Exception:
        with suppress(OSError):
            os.unlink(tmp)
        raise


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="build_cnn_training_data",
        description="Assemble offline CNN train/validation/test JSON splits.",
    )
    parser.add_argument("files", nargs="+", type=Path, metavar="JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--no-stratify", action="store_true")
    args = parser.parse_args(argv)

    config = SplitConfig(
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        stratify=not args.no_stratify,
    )
    examples = load_training_examples(args.files)
    manifest = write_training_splits(examples, args.output_dir, config=config)
    print(format_split_summary(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
