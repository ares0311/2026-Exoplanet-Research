"""Validate pipeline configuration files against expected schemas.

Checks that required keys are present, values are within valid ranges,
and paths referenced in config files actually exist.

Public API
----------
ConfigIssue(field, severity, message)
ValidationResult(config_path, n_errors, n_warnings, issues, flag)
validate_pipeline_config(config, *, required_keys, path_keys,
                          numeric_ranges) -> ValidationResult
load_and_validate(path, *, required_keys, path_keys,
                  numeric_ranges) -> ValidationResult
format_validation_result(result) -> str
"""
from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ConfigIssue:
    field: str
    severity: str   # "ERROR" | "WARNING"
    message: str


@dataclass(frozen=True)
class ValidationResult:
    config_path: str
    n_errors: int
    n_warnings: int
    issues: tuple[ConfigIssue, ...]
    flag: str  # "VALID" | "WARNINGS" | "INVALID"


def validate_pipeline_config(
    config: dict,
    *,
    required_keys: list[str] | None = None,
    path_keys: list[str] | None = None,
    numeric_ranges: dict[str, tuple[float, float]] | None = None,
    config_path: str = "<dict>",
) -> ValidationResult:
    """Validate a pipeline config dict.

    Args:
        config: Configuration dictionary to validate.
        required_keys: Keys that must be present.
        path_keys: Keys whose values should be existing filesystem paths.
        numeric_ranges: Mapping of key → (min, max) for numeric value checks.
        config_path: Label for the config source (for reporting).

    Returns:
        ValidationResult with issues list and overall flag.
    """
    issues: list[ConfigIssue] = []

    for key in (required_keys or []):
        if key not in config:
            issues.append(ConfigIssue(
                field=key,
                severity="ERROR",
                message=f"Required key '{key}' is missing",
            ))

    for key in (path_keys or []):
        val = config.get(key)
        if val is None:
            continue
        with contextlib.suppress(TypeError):
            if not Path(str(val)).exists():
                issues.append(ConfigIssue(
                    field=key,
                    severity="WARNING",
                    message=f"Path does not exist: {val}",
                ))

    for key, (lo, hi) in (numeric_ranges or {}).items():
        val = config.get(key)
        if val is None:
            continue
        num: float | None = None
        with contextlib.suppress(TypeError, ValueError):
            num = float(val)
        if num is None:
            issues.append(ConfigIssue(
                field=key, severity="ERROR",
                message=f"Expected numeric value, got: {val!r}",
            ))
        elif not (lo <= num <= hi):
            issues.append(ConfigIssue(
                field=key, severity="ERROR",
                message=f"Value {num} out of range [{lo}, {hi}]",
            ))

    n_errors = sum(1 for i in issues if i.severity == "ERROR")
    n_warnings = sum(1 for i in issues if i.severity == "WARNING")

    if n_errors > 0:
        flag = "INVALID"
    elif n_warnings > 0:
        flag = "WARNINGS"
    else:
        flag = "VALID"

    return ValidationResult(
        config_path=config_path,
        n_errors=n_errors,
        n_warnings=n_warnings,
        issues=tuple(issues),
        flag=flag,
    )


def load_and_validate(
    path: str | Path,
    *,
    required_keys: list[str] | None = None,
    path_keys: list[str] | None = None,
    numeric_ranges: dict[str, tuple[float, float]] | None = None,
) -> ValidationResult:
    """Load a JSON config file and validate it.

    Args:
        path: Path to the JSON config file.
        required_keys: Keys that must be present.
        path_keys: Keys whose values should be existing filesystem paths.
        numeric_ranges: Mapping of key → (min, max).

    Returns:
        ValidationResult; returns INVALID with parse error if file unreadable.
    """
    p = Path(path)
    try:
        config = json.loads(p.read_text())
    except FileNotFoundError:
        return ValidationResult(
            config_path=str(p),
            n_errors=1,
            n_warnings=0,
            issues=(ConfigIssue("file", "ERROR", f"File not found: {p}"),),
            flag="INVALID",
        )
    except json.JSONDecodeError as exc:
        return ValidationResult(
            config_path=str(p),
            n_errors=1,
            n_warnings=0,
            issues=(ConfigIssue("file", "ERROR", f"JSON parse error: {exc}"),),
            flag="INVALID",
        )

    return validate_pipeline_config(
        config,
        required_keys=required_keys,
        path_keys=path_keys,
        numeric_ranges=numeric_ranges,
        config_path=str(p),
    )


def format_validation_result(result: ValidationResult) -> str:
    """Format validation result as Markdown.

    Args:
        result: ValidationResult to format.

    Returns:
        Markdown string.
    """
    lines = [
        "## Pipeline Config Validator\n",
        f"**Config**: `{result.config_path}` | "
        f"**Status**: `{result.flag}` | "
        f"Errors: {result.n_errors} | Warnings: {result.n_warnings}\n",
    ]
    if not result.issues:
        lines.append("\n_No issues found._")
        return "\n".join(lines)

    lines += [
        "",
        "| Field | Severity | Message |",
        "|---|---|---|",
    ]
    for issue in result.issues:
        lines.append(f"| {issue.field} | `{issue.severity}` | {issue.message} |")
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Validate pipeline config.")
    parser.add_argument("config", help="JSON config file.")
    parser.add_argument("--require", nargs="*", default=[], help="Required keys.")
    parser.add_argument("--paths", nargs="*", default=[], help="Path keys to check.")
    args = parser.parse_args(argv)

    result = load_and_validate(args.config, required_keys=args.require,
                               path_keys=args.paths)
    print(format_validation_result(result))
    return 0 if result.flag == "VALID" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
