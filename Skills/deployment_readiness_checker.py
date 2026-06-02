"""Gate check: verify all conditions for Tier 2 CNN production deployment.

Checks the full Tier 2 readiness chain:
  1. Label count meets the 5,000-example threshold
  2. Snippet split directory exists and is valid
  3. CNN checkpoint file exists
  4. Calibration file exists
  5. Model registry has a registered CNN entry

Public API
----------
ReadinessCheck(name, passed, detail)
ReadinessReport(checks, n_passed, n_failed, ready, flag)
check_deployment_readiness(*, label_json, split_dir, checkpoint_path,
                           calibration_path, registry_path,
                           min_labels) -> ReadinessReport
format_readiness_report(report) -> str
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class ReadinessReport:
    checks: tuple[ReadinessCheck, ...]
    n_passed: int
    n_failed: int
    ready: bool
    flag: str  # "READY" | "NOT_READY" | "INVALID"


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


def _count_usable_labels(path: Path) -> int:
    rows = _load_label_rows(path)
    return sum(1 for row in rows if _is_supervised_label(row.get("label")))


def check_deployment_readiness(
    *,
    label_json: Path | None = None,
    split_dir: Path | None = None,
    checkpoint_path: Path | None = None,
    calibration_path: Path | None = None,
    registry_path: Path | None = None,
    min_labels: int = 5000,
) -> ReadinessReport:
    """Run all Tier 2 deployment gate checks.

    Args:
        label_json: Path to assembled label JSON file.
        split_dir: Directory containing train/val/test splits.
        checkpoint_path: Path to trained CNN checkpoint file.
        calibration_path: Path to Platt calibration JSON file.
        registry_path: Path to model registry JSON file.
        min_labels: Minimum number of usable labels required.

    Returns:
        ReadinessReport with per-check results.
    """
    checks: list[ReadinessCheck] = []

    # Check 1: label count
    if label_json is not None:
        lp = Path(label_json)
        if lp.exists():
            try:
                usable = _count_usable_labels(lp)
                passed = usable >= min_labels
                detail = f"{usable} usable labels (threshold: {min_labels})"
            except Exception as exc:
                passed = False
                detail = f"Failed to parse: {exc}"
        else:
            passed = False
            detail = f"File not found: {label_json}"
        checks.append(ReadinessCheck("label_count", passed, detail))
    else:
        checks.append(ReadinessCheck("label_count", False, "label_json not specified"))

    # Check 2: split directory
    if split_dir is not None:
        sd = Path(split_dir)
        splits_exist = all((sd / f).exists() for f in ("train.json", "val.json", "test.json"))
        if splits_exist:
            passed = True
            detail = f"train/val/test splits found in {split_dir}"
        else:
            passed = False
            detail = f"Missing split files in {split_dir}"
        checks.append(ReadinessCheck("split_dir", passed, detail))
    else:
        checks.append(ReadinessCheck("split_dir", False, "split_dir not specified"))

    # Check 3: CNN checkpoint
    if checkpoint_path is not None:
        cp = Path(checkpoint_path)
        passed = cp.exists()
        detail = str(checkpoint_path) if passed else f"Not found: {checkpoint_path}"
        checks.append(ReadinessCheck("checkpoint", passed, detail))
    else:
        checks.append(ReadinessCheck("checkpoint", False, "checkpoint_path not specified"))

    # Check 4: calibration file
    if calibration_path is not None:
        cap = Path(calibration_path)
        passed = cap.exists()
        detail = str(calibration_path) if passed else f"Not found: {calibration_path}"
        checks.append(ReadinessCheck("calibration", passed, detail))
    else:
        checks.append(ReadinessCheck("calibration", False, "calibration_path not specified"))

    # Check 5: registry has CNN entry
    if registry_path is not None:
        rp = Path(registry_path)
        if rp.exists():
            try:
                entries = json.loads(rp.read_text())
                cnn_entries = [e for e in entries if e.get("model_type") == "cnn"]
                passed = len(cnn_entries) > 0
                detail = (
                    f"{len(cnn_entries)} CNN model(s) registered"
                    if passed
                    else "No CNN entries in registry"
                )
            except Exception as exc:
                passed = False
                detail = f"Failed to parse registry: {exc}"
        else:
            passed = False
            detail = f"Registry not found: {registry_path}"
        checks.append(ReadinessCheck("registry_cnn", passed, detail))
    else:
        checks.append(ReadinessCheck("registry_cnn", False, "registry_path not specified"))

    n_passed = sum(1 for c in checks if c.passed)
    n_failed = len(checks) - n_passed
    ready = n_failed == 0
    flag = "READY" if ready else "NOT_READY"

    return ReadinessReport(
        checks=tuple(checks),
        n_passed=n_passed,
        n_failed=n_failed,
        ready=ready,
        flag=flag,
    )


def format_readiness_report(report: ReadinessReport) -> str:
    """Format a Markdown deployment readiness report.

    Args:
        report: ReadinessReport to format.

    Returns:
        Markdown string.
    """
    status_icon = "READY" if report.ready else "NOT READY"
    lines = [
        "## Tier 2 Deployment Readiness\n",
        f"**Status**: `{status_icon}` | Passed: {report.n_passed}/{len(report.checks)}\n",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for c in report.checks:
        icon = "PASS" if c.passed else "FAIL"
        lines.append(f"| {c.name} | `{icon}` | {c.detail} |")

    if not report.ready:
        lines.append("\n> **Action required**: resolve all FAIL checks before deploying CNN.\n")
    else:
        lines.append("\n> **All checks passed** — CNN is ready for production deployment.\n")
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Check Tier 2 CNN deployment readiness.")
    parser.add_argument("--labels", help="Path to assembled label JSON.")
    parser.add_argument("--splits", help="Path to split directory.")
    parser.add_argument("--checkpoint", help="Path to CNN checkpoint file.")
    parser.add_argument("--calibration", help="Path to calibration JSON.")
    parser.add_argument("--registry", help="Path to model registry JSON.")
    parser.add_argument("--min-labels", type=int, default=5000)
    args = parser.parse_args(argv)

    report = check_deployment_readiness(
        label_json=Path(args.labels) if args.labels else None,
        split_dir=Path(args.splits) if args.splits else None,
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        calibration_path=Path(args.calibration) if args.calibration else None,
        registry_path=Path(args.registry) if args.registry else None,
        min_labels=args.min_labels,
    )
    print(format_readiness_report(report))
    return 0 if report.ready else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
