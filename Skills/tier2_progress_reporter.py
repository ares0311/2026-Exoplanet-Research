"""Unified Tier 2 CNN readiness report: labels, snippets, gate status, next actions.

Aggregates information from the label corpus, snippet cache, training progress,
and deployment gate to produce a single human-readable status dashboard.

Public API
----------
Tier2Status(n_labels, n_snippets, gate_passed, training_complete,
            calibrated, registered, next_actions, flag)
build_tier2_status(*, label_json, snippet_dir, training_log,
                   checkpoint_path, calibration_path, registry_path,
                   min_labels) -> Tier2Status
status_to_dict(status) -> dict
write_status_outputs(status, *, markdown_path, json_path) -> tuple[Path, ...]
format_tier2_report(status) -> str
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Tier2Status:
    n_labels: int
    n_snippets: int
    gate_passed: bool          # n_labels >= min_labels
    training_complete: bool    # checkpoint exists
    calibrated: bool           # calibration file exists
    registered: bool           # CNN entry in model registry
    next_actions: tuple[str, ...]
    flag: str  # "READY" | "IN_PROGRESS" | "BLOCKED"


def _is_supervised_label(value: object) -> bool:
    if value in (0, 1):
        return True
    if isinstance(value, str):
        return value.strip().lower() in {
            "0",
            "1",
            "cp",
            "fp",
            "eb",
            "planet_candidate",
            "false_positive",
        }
    return False


def _load_label_rows(path: Path) -> list[dict[str, object]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        rows = payload.get("rows") or payload.get("records") or payload.get("labels") or []
    else:
        rows = payload
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _count_labels(path: Path) -> int:
    try:
        return sum(1 for row in _load_label_rows(path) if _is_supervised_label(row.get("label")))
    except Exception:
        return 0


def build_tier2_status(
    *,
    label_json: Path | None = None,
    snippet_dir: Path | None = None,
    training_log: Path | None = None,
    checkpoint_path: Path | None = None,
    calibration_path: Path | None = None,
    registry_path: Path | None = None,
    min_labels: int = 5000,
) -> Tier2Status:
    """Gather Tier 2 progress from all available artifacts.

    Args:
        label_json: Assembled label rows JSON (for label count).
        snippet_dir: Phase-folded snippet cache directory (for snippet count).
        training_log: JSONL training curve log file (for completion check).
        checkpoint_path: Path to trained CNN checkpoint.
        calibration_path: Path to Platt calibration JSON.
        registry_path: Path to model registry JSON.
        min_labels: Label count threshold for the data gate.

    Returns:
        Tier2Status with counts, flags, and suggested next actions.
    """
    # Label count
    n_labels = 0
    if label_json is not None and Path(label_json).exists():
        n_labels = _count_labels(Path(label_json))

    # Snippet count
    n_snippets = 0
    if snippet_dir is not None and Path(snippet_dir).exists():
        n_snippets = sum(1 for f in Path(snippet_dir).glob("*.json"))

    gate_passed = n_labels >= min_labels

    # Training completion: check for checkpoint
    training_complete = (
        checkpoint_path is not None and Path(checkpoint_path).exists()
    )

    calibrated = (
        calibration_path is not None and Path(calibration_path).exists()
    )

    registered = False
    if registry_path is not None and Path(registry_path).exists():
        try:
            entries = json.loads(Path(registry_path).read_text())
            registered = any(e.get("model_type") == "cnn" for e in entries)
        except Exception:
            pass

    # Determine next actions
    actions: list[str] = []
    if not gate_passed:
        needed = min_labels - n_labels
        actions.append(
            f"Collect {needed:,} more labeled TESS examples "
            f"(run fetch_ctoi_table + assemble_labels)."
        )
    if gate_passed and n_snippets < n_labels:
        actions.append(
            "Extract phase-folded snippets for all labeled targets "
            "(run lc_snippet_batch_builder)."
        )
    if gate_passed and n_snippets >= min_labels and not training_complete:
        actions.append(
            "Train the CNN (run train_cnn.py with cnn_training_config)."
        )
    if training_complete and not calibrated:
        actions.append("Calibrate CNN predictions (run cnn_calibrator.py).")
    if calibrated and not registered:
        actions.append("Register the trained model (run model_registry.py register).")
    if registered and calibrated and training_complete:
        actions.append("Tune ensemble weights (run ensemble_weight_optimizer.py).")
        actions.append("Run full ensemble evaluation (model_ensemble_evaluator.py).")

    if not actions:
        actions.append("Tier 2 deployment complete. Monitor production metrics.")

    # Overall flag
    if registered and calibrated and training_complete:
        flag = "READY"
    elif gate_passed:
        flag = "IN_PROGRESS"
    else:
        flag = "BLOCKED"

    return Tier2Status(
        n_labels=n_labels,
        n_snippets=n_snippets,
        gate_passed=gate_passed,
        training_complete=training_complete,
        calibrated=calibrated,
        registered=registered,
        next_actions=tuple(actions),
        flag=flag,
    )


def status_to_dict(status: Tier2Status) -> dict[str, object]:
    """Convert status to a JSON-serializable mapping."""
    return {
        "n_labels": status.n_labels,
        "n_snippets": status.n_snippets,
        "gate_passed": status.gate_passed,
        "training_complete": status.training_complete,
        "calibrated": status.calibrated,
        "registered": status.registered,
        "next_actions": list(status.next_actions),
        "flag": status.flag,
    }


def write_status_outputs(
    status: Tier2Status,
    *,
    markdown_path: Path | None = None,
    json_path: Path | None = None,
) -> tuple[Path, ...]:
    """Write optional Markdown and JSON status artifacts."""
    written: list[Path] = []
    if markdown_path is not None:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(format_tier2_report(status))
        written.append(markdown_path)
    if json_path is not None:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(status_to_dict(status), indent=2))
        written.append(json_path)
    return tuple(written)


def format_tier2_report(status: Tier2Status) -> str:
    """Format a Markdown Tier 2 progress report.

    Args:
        status: Tier2Status to format.

    Returns:
        Markdown string.
    """
    flag_display = {
        "READY": "READY",
        "IN_PROGRESS": "IN PROGRESS",
        "BLOCKED": "BLOCKED — needs data",
    }.get(status.flag, status.flag)

    lines = [
        "## Tier 2 CNN Progress Report\n",
        f"**Status**: `{flag_display}`\n",
        "",
        "### Data Pipeline",
        "",
        "| Step | Status | Count |",
        "|---|---|---|",
        f"| Labeled examples | {'PASS' if status.gate_passed else 'FAIL'}"
        f" | {status.n_labels:,} |",
        f"| Extracted snippets | {'OK' if status.n_snippets > 0 else 'NONE'}"
        f" | {status.n_snippets:,} |",
        "",
        "### Model Pipeline",
        "",
        "| Step | Status |",
        "|---|---|",
        f"| CNN trained | {'DONE' if status.training_complete else 'PENDING'} |",
        f"| Calibration applied | {'DONE' if status.calibrated else 'PENDING'} |",
        f"| Model registered | {'DONE' if status.registered else 'PENDING'} |",
        "",
        "### Next Actions",
        "",
    ]
    for i, action in enumerate(status.next_actions, 1):
        lines.append(f"{i}. {action}")

    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Report Tier 2 CNN progress.")
    parser.add_argument("--labels", help="Path to assembled label JSON.")
    parser.add_argument("--snippets", help="Path to snippet cache directory.")
    parser.add_argument("--log", help="Path to training curve JSONL log.")
    parser.add_argument("--checkpoint", help="Path to CNN checkpoint file.")
    parser.add_argument("--calibration", help="Path to calibration JSON.")
    parser.add_argument("--registry", help="Path to model registry JSON.")
    parser.add_argument("--min-labels", type=int, default=5000)
    parser.add_argument("--output", help="Optional Markdown report path.")
    parser.add_argument("--json-output", help="Optional JSON status path.")
    args = parser.parse_args(argv)

    status = build_tier2_status(
        label_json=Path(args.labels) if args.labels else None,
        snippet_dir=Path(args.snippets) if args.snippets else None,
        training_log=Path(args.log) if args.log else None,
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        calibration_path=Path(args.calibration) if args.calibration else None,
        registry_path=Path(args.registry) if args.registry else None,
        min_labels=args.min_labels,
    )
    print(format_tier2_report(status))
    write_status_outputs(
        status,
        markdown_path=Path(args.output) if args.output else None,
        json_path=Path(args.json_output) if args.json_output else None,
    )
    return 0 if status.flag in ("READY", "IN_PROGRESS") else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
