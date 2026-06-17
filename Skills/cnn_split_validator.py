"""Validate offline CNN train/validation/test split artifacts.

This Skill is intentionally local-only. It reads the manifest and split JSON
files emitted by ``build_cnn_training_data.py`` and checks that the artifacts
are internally consistent, provenance-preserving, and safe for later CNN
training once the label-count gate opens.

Public API
----------
ValidationIssue(...)
SplitValidationResult(...)
validate_split_dir(path) -> SplitValidationResult
validate_split_manifest(path) -> SplitValidationResult
format_validation_summary(result) -> str
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

Severity = Literal["error", "warning"]
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class ValidationIssue:
    """A single validation finding for a CNN split artifact."""

    severity: Severity
    code: str
    message: str
    split: str | None = None
    example_id: str | None = None


@dataclass(frozen=True)
class SplitValidationResult:
    """Validation result for one CNN split manifest."""

    manifest_path: Path
    issues: tuple[ValidationIssue, ...]
    split_counts: dict[str, int]
    label_counts: dict[str, dict[int, int]]

    @property
    def ok(self) -> bool:
        """Return ``True`` when no error-severity issues were found."""
        return not any(issue.severity == "error" for issue in self.issues)


def validate_split_dir(path: Path | str) -> SplitValidationResult:
    """Validate a directory containing ``manifest.json``."""
    return validate_split_manifest(Path(path) / "manifest.json")


def validate_split_manifest(path: Path | str) -> SplitValidationResult:
    """Validate a CNN split manifest and the split files it references."""
    manifest_path = Path(path)
    issues: list[ValidationIssue] = []
    split_counts: dict[str, int] = {}
    label_counts: dict[str, dict[int, int]] = {}
    seen_ids: dict[str, str] = {}
    seen_tic_ids: dict[int, str] = {}
    seen_group_ids: dict[str, str] = {}

    manifest = _load_json_object(manifest_path, issues)
    if manifest is None:
        return _result(manifest_path, issues, split_counts, label_counts)

    _validate_manifest_guardrails(manifest, issues)
    split_files = _split_files(manifest, manifest_path, issues)
    expected_split_counts = _dict_value(manifest.get("split_counts"))
    expected_split_label_counts = _nested_label_counts(manifest.get("split_label_counts"))

    for split_name in SPLIT_NAMES:
        split_path = split_files.get(split_name)
        if split_path is None:
            issues.append(
                ValidationIssue(
                    "error",
                    "missing_split_file_reference",
                    f"manifest does not reference {split_name}.json",
                    split=split_name,
                )
            )
            continue

        payload = _load_json_object(split_path, issues, split=split_name)
        if payload is None:
            continue

        if payload.get("split") != split_name:
            issues.append(
                ValidationIssue(
                    "error",
                    "split_name_mismatch",
                    f"split payload declares {payload.get('split')!r}, expected {split_name!r}",
                    split=split_name,
                )
            )

        rows = payload.get("examples")
        if not isinstance(rows, list):
            issues.append(
                ValidationIssue(
                    "error",
                    "examples_not_list",
                    "split payload must contain an examples list",
                    split=split_name,
                )
            )
            continue

        split_counts[split_name] = len(rows)
        label_counts[split_name] = {0: 0, 1: 0}
        for index, row in enumerate(rows):
            example_id = _example_id(row, split_name, index)
            _validate_example(
                row,
                split=split_name,
                example_id=example_id,
                issues=issues,
                label_counts=label_counts[split_name],
            )
            if example_id is None:
                continue
            previous_split = seen_ids.get(example_id)
            if previous_split is not None:
                issues.append(
                    ValidationIssue(
                        "error",
                        "duplicate_example_id",
                        f"example_id also appears in {previous_split}",
                        split=split_name,
                        example_id=example_id,
                    )
                )
            else:
                seen_ids[example_id] = split_name
            tic_id = _tic_id(row)
            if tic_id is None or tic_id <= 0:
                group_id = _group_id(row)
                if group_id is None:
                    continue
                previous_group_split = seen_group_ids.get(group_id)
                if previous_group_split is not None and previous_group_split != split_name:
                    issues.append(
                        ValidationIssue(
                            "error",
                            "group_id_leakage",
                            f"group_id {group_id!r} also appears in {previous_group_split}",
                            split=split_name,
                            example_id=example_id,
                        )
                    )
                else:
                    seen_group_ids[group_id] = split_name
                continue
            previous_tic_split = seen_tic_ids.get(tic_id)
            if previous_tic_split is not None and previous_tic_split != split_name:
                issues.append(
                    ValidationIssue(
                        "error",
                        "tic_id_leakage",
                        f"TIC {tic_id} also appears in {previous_tic_split}",
                        split=split_name,
                        example_id=example_id,
                    )
                )
            else:
                seen_tic_ids[tic_id] = split_name

    _validate_counts(
        manifest,
        _label_count_dict(manifest.get("label_counts")),
        expected_split_counts,
        expected_split_label_counts,
        split_counts,
        label_counts,
        issues,
    )
    return _result(manifest_path, issues, split_counts, label_counts)


def format_validation_summary(result: SplitValidationResult) -> str:
    """Format a compact Markdown validation summary."""
    status = "PASS" if result.ok else "FAIL"
    errors = sum(1 for issue in result.issues if issue.severity == "error")
    warnings = sum(1 for issue in result.issues if issue.severity == "warning")
    lines = [
        "## CNN Split Validation Summary",
        "",
        f"- Manifest: {result.manifest_path}",
        f"- Status: {status}",
        f"- Errors: {errors}",
        f"- Warnings: {warnings}",
    ]
    if result.split_counts:
        counts = result.split_counts
        lines.append(
            f"- Train/val/test: {counts.get('train', 0)} / "
            f"{counts.get('val', 0)} / {counts.get('test', 0)}"
        )
    for split_name in SPLIT_NAMES:
        labels = result.label_counts.get(split_name)
        if labels is not None:
            lines.append(
                f"- {split_name} labels: negative={labels.get(0, 0)}, "
                f"positive={labels.get(1, 0)}"
            )
    if result.issues:
        lines.extend(["", "### Issues"])
        for issue in result.issues:
            location = ""
            if issue.split is not None:
                location += f" [{issue.split}]"
            if issue.example_id is not None:
                location += f" {issue.example_id}"
            lines.append(
                f"- {issue.severity.upper()} {issue.code}{location}: {issue.message}"
            )
    return "\n".join(lines) + "\n"


def _result(
    manifest_path: Path,
    issues: list[ValidationIssue],
    split_counts: dict[str, int],
    label_counts: dict[str, dict[int, int]],
) -> SplitValidationResult:
    return SplitValidationResult(
        manifest_path=manifest_path,
        issues=tuple(issues),
        split_counts=split_counts,
        label_counts=label_counts,
    )


def _load_json_object(
    path: Path,
    issues: list[ValidationIssue],
    *,
    split: str | None = None,
) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        issues.append(
            ValidationIssue("error", "missing_json_file", f"missing JSON file: {path}", split=split)
        )
        return None
    except json.JSONDecodeError as exc:
        issues.append(
            ValidationIssue("error", "invalid_json", f"invalid JSON: {exc}", split=split)
        )
        return None
    if not isinstance(payload, dict):
        issues.append(
            ValidationIssue(
                "error",
                "json_not_object",
                "JSON payload must be an object",
                split=split,
            )
        )
        return None
    return payload


def _validate_manifest_guardrails(
    manifest: dict[str, Any],
    issues: list[ValidationIssue],
) -> None:
    required = (
        "n_examples",
        "split_counts",
        "label_counts",
        "split_label_counts",
        "config",
        "split_files",
        "live_services",
        "source_files",
        "language_guardrail",
    )
    for key in required:
        if key not in manifest:
            issues.append(
                ValidationIssue("error", "missing_manifest_field", f"manifest is missing {key!r}")
            )
    if manifest.get("live_services") is not False:
        issues.append(
            ValidationIssue(
                "error",
                "live_services_not_false",
                "CNN split artifacts must be produced without live services",
            )
        )
    source_files = manifest.get("source_files")
    if not isinstance(source_files, list) or not all(
        isinstance(item, str) for item in source_files
    ):
        issues.append(
            ValidationIssue(
                "error",
                "invalid_source_files",
                "manifest source_files must be a list of source path strings",
            )
        )
    guardrail = manifest.get("language_guardrail")
    if not isinstance(guardrail, str) or not guardrail.strip():
        issues.append(
            ValidationIssue(
                "error",
                "missing_language_guardrail",
                "manifest must include a conservative language guardrail",
            )
        )
    elif "confirmed planet" in guardrail.lower():
        issues.append(
            ValidationIssue(
                "error",
                "unsafe_language_guardrail",
                "language guardrail must not describe internally detected signals "
                "as confirmed planets",
            )
        )
    elif "no candidate confirmation" not in guardrail.lower():
        issues.append(
            ValidationIssue(
                "warning",
                "weak_language_guardrail",
                "language guardrail should explicitly prohibit candidate confirmation claims",
            )
        )


def _split_files(
    manifest: dict[str, Any],
    manifest_path: Path,
    issues: list[ValidationIssue],
) -> dict[str, Path]:
    raw = manifest.get("split_files")
    if not isinstance(raw, dict):
        issues.append(
            ValidationIssue(
                "error",
                "invalid_split_files",
                "manifest split_files must be an object",
            )
        )
        return {}
    split_files: dict[str, Path] = {}
    for split_name in SPLIT_NAMES:
        value = raw.get(split_name)
        if not isinstance(value, str) or not value.strip():
            continue
        path = Path(value)
        split_files[split_name] = path if path.is_absolute() else manifest_path.parent / path
    return split_files


def _validate_example(
    row: Any,
    *,
    split: str,
    example_id: str | None,
    issues: list[ValidationIssue],
    label_counts: dict[int, int],
) -> None:
    if not isinstance(row, dict):
        issues.append(
            ValidationIssue(
                "error",
                "example_not_object",
                "each example must be a JSON object",
                split=split,
                example_id=example_id,
            )
        )
        return

    if example_id is None:
        issues.append(
            ValidationIssue(
                "error",
                "missing_example_id",
                "example must include a non-empty example_id",
                split=split,
            )
        )
    for field in ("source_file", "source"):
        if not isinstance(row.get(field), str) or not row.get(field, "").strip():
            issues.append(
                ValidationIssue(
                    "error",
                    f"missing_{field}",
                    f"example must include provenance field {field!r}",
                    split=split,
                    example_id=example_id,
                )
            )

    label = row.get("label")
    if label not in (0, 1):
        issues.append(
            ValidationIssue(
                "error",
                "invalid_label",
                "label must be integer 0 or 1",
                split=split,
                example_id=example_id,
            )
        )
    else:
        label_counts[label] += 1

    phase = _numeric_sequence(row.get("phase"))
    flux = _numeric_sequence(row.get("flux"))
    if phase is None:
        issues.append(
            ValidationIssue(
                "error",
                "invalid_phase",
                "phase must be a non-empty finite numeric array",
                split=split,
                example_id=example_id,
            )
        )
    if flux is None:
        issues.append(
            ValidationIssue(
                "error",
                "invalid_flux",
                "flux must be a non-empty finite numeric array",
                split=split,
                example_id=example_id,
            )
        )
    if phase is not None and flux is not None and len(phase) != len(flux):
        issues.append(
            ValidationIssue(
                "error",
                "phase_flux_length_mismatch",
                f"phase length {len(phase)} does not match flux length {len(flux)}",
                split=split,
                example_id=example_id,
            )
        )


def _example_id(row: Any, split: str, index: int) -> str | None:
    if not isinstance(row, dict):
        return f"{split}[{index}]"
    value = row.get("example_id")
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def _tic_id(row: Any) -> int | None:
    if not isinstance(row, dict):
        return None
    value = row.get("tic_id")
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _group_id(row: Any) -> str | None:
    if not isinstance(row, dict):
        return None
    value = row.get("group_id")
    if not isinstance(value, str) or not value.strip():
        return None
    return value


def _numeric_sequence(value: Any) -> tuple[float, ...] | None:
    if not isinstance(value, list):
        return None
    if not value:
        return None
    numbers: list[float] = []
    for item in value:
        if isinstance(item, bool):
            return None
        try:
            number = float(item)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(number):
            return None
        numbers.append(number)
    return tuple(numbers)


def _dict_value(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    counts: dict[str, int] = {}
    for key, raw_count in value.items():
        if isinstance(key, str) and isinstance(raw_count, int) and not isinstance(raw_count, bool):
            counts[key] = raw_count
    return counts


def _nested_label_counts(value: Any) -> dict[str, dict[int, int]]:
    if not isinstance(value, dict):
        return {}
    counts: dict[str, dict[int, int]] = {}
    for split_name, raw_counts in value.items():
        if isinstance(split_name, str):
            counts[split_name] = _label_count_dict(raw_counts)
    return counts


def _label_count_dict(value: Any) -> dict[int, int]:
    if not isinstance(value, dict):
        return {}
    counts: dict[int, int] = {}
    for key, raw_count in value.items():
        try:
            label = int(key)
        except (TypeError, ValueError):
            continue
        if label in (0, 1) and isinstance(raw_count, int) and not isinstance(raw_count, bool):
            counts[label] = raw_count
    return counts


def _validate_counts(
    manifest: dict[str, Any],
    expected_label_counts: dict[int, int],
    expected_split_counts: dict[str, int],
    expected_split_label_counts: dict[str, dict[int, int]],
    split_counts: dict[str, int],
    label_counts: dict[str, dict[int, int]],
    issues: list[ValidationIssue],
) -> None:
    n_examples = manifest.get("n_examples")
    observed_total = sum(split_counts.values())
    if not isinstance(n_examples, int) or isinstance(n_examples, bool):
        issues.append(
            ValidationIssue("error", "invalid_n_examples", "manifest n_examples must be an integer")
        )
    elif observed_total != n_examples:
        issues.append(
            ValidationIssue(
                "error",
                "total_count_mismatch",
                f"manifest n_examples={n_examples}, observed {observed_total}",
            )
        )

    observed_label_counts = {
        label: sum(counts.get(label, 0) for counts in label_counts.values()) for label in (0, 1)
    }
    for label in (0, 1):
        expected_label_count = expected_label_counts.get(label)
        if expected_label_count is None:
            issues.append(
                ValidationIssue(
                    "error",
                    "missing_label_count",
                    f"manifest label_counts is missing label {label}",
                )
            )
        elif expected_label_count != observed_label_counts[label]:
            issues.append(
                ValidationIssue(
                    "error",
                    "label_count_mismatch",
                    f"manifest label {label} count={expected_label_count}, "
                    f"observed {observed_label_counts[label]}",
                )
            )

    for split_name in SPLIT_NAMES:
        expected = expected_split_counts.get(split_name)
        observed = split_counts.get(split_name, 0)
        if expected is None:
            issues.append(
                ValidationIssue(
                    "error",
                    "missing_split_count",
                    f"manifest split_counts is missing {split_name!r}",
                    split=split_name,
                )
            )
        elif expected != observed:
            issues.append(
                ValidationIssue(
                    "error",
                    "split_count_mismatch",
                    f"manifest split_counts={expected}, observed {observed}",
                    split=split_name,
                )
            )

        expected_labels = expected_split_label_counts.get(split_name, {})
        observed_labels = label_counts.get(split_name, {0: 0, 1: 0})
        for label in (0, 1):
            expected_label_count = expected_labels.get(label)
            if expected_label_count is None:
                issues.append(
                    ValidationIssue(
                        "error",
                        "missing_split_label_count",
                        f"manifest split_label_counts is missing label {label}",
                        split=split_name,
                    )
                )
            elif expected_label_count != observed_labels.get(label, 0):
                issues.append(
                    ValidationIssue(
                        "error",
                        "split_label_count_mismatch",
                        f"manifest label {label} count={expected_label_count}, "
                        f"observed {observed_labels.get(label, 0)}",
                        split=split_name,
                    )
                )


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cnn_split_validator",
        description="Validate offline CNN split manifest and train/val/test JSON files.",
    )
    parser.add_argument("path", type=Path, help="Split directory or manifest.json path")
    args = parser.parse_args(argv)

    result = (
        validate_split_dir(args.path)
        if args.path.is_dir()
        else validate_split_manifest(args.path)
    )
    print(format_validation_summary(result))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
