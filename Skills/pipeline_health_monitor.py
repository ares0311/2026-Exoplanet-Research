"""Monitor overall pipeline health: data freshness, model staleness, gate status.

Aggregates status from label corpus, snippet cache, model registry, and
calibration artifacts to produce a single health dashboard with actionable
alerts.

Public API
----------
HealthCheck(name, status, detail)    # "OK" | "WARN" | "FAIL"
PipelineHealthReport(checks, n_ok, n_warn, n_fail, overall, flag)
check_pipeline_health(*, label_json, snippet_dir, registry_path,
                      calibration_path, min_labels, max_model_age_days)
    -> PipelineHealthReport
format_health_report(report) -> str
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HealthCheck:
    name: str
    status: str   # "OK" | "WARN" | "FAIL"
    detail: str


@dataclass(frozen=True)
class PipelineHealthReport:
    checks: tuple[HealthCheck, ...]
    n_ok: int
    n_warn: int
    n_fail: int
    overall: str   # "HEALTHY" | "DEGRADED" | "UNHEALTHY"
    flag: str      # "OK" | "INVALID"


def _age_days(path: Path) -> float | None:
    try:
        return (time.time() - path.stat().st_mtime) / 86400
    except OSError:
        return None


def check_pipeline_health(
    *,
    label_json: Path | None = None,
    snippet_dir: Path | None = None,
    registry_path: Path | None = None,
    calibration_path: Path | None = None,
    min_labels: int = 5000,
    max_model_age_days: float = 30.0,
) -> PipelineHealthReport:
    """Run all pipeline health checks.

    Args:
        label_json: Assembled label rows JSON.
        snippet_dir: Snippet cache directory.
        registry_path: Model registry JSON.
        calibration_path: Platt calibration JSON.
        min_labels: Minimum label count for a healthy data gate.
        max_model_age_days: Warn if registered model is older than this.

    Returns:
        PipelineHealthReport with per-check results.
    """
    checks: list[HealthCheck] = []

    # Label corpus check
    if label_json is not None:
        lp = Path(label_json)
        if lp.exists():
            try:
                rows = json.loads(lp.read_text())
                n = sum(1 for r in rows if r.get("label") in
                        ("planet_candidate", "false_positive"))
                if n >= min_labels:
                    checks.append(HealthCheck("labels", "OK",
                                              f"{n:,} usable labels (≥ {min_labels})"))
                elif n >= min_labels * 0.5:
                    checks.append(HealthCheck("labels", "WARN",
                                              f"{n:,} labels — below {min_labels} threshold"))
                else:
                    checks.append(HealthCheck("labels", "FAIL",
                                              f"Only {n:,} labels — well below {min_labels}"))
            except Exception as exc:
                checks.append(HealthCheck("labels", "FAIL", f"Parse error: {exc}"))
        else:
            checks.append(HealthCheck("labels", "FAIL", f"File not found: {label_json}"))
    else:
        checks.append(HealthCheck("labels", "WARN", "label_json not configured"))

    # Snippet cache check
    if snippet_dir is not None:
        sd = Path(snippet_dir)
        if sd.exists():
            n_files = sum(1 for _ in sd.glob("*.json"))
            if n_files >= min_labels:
                checks.append(HealthCheck("snippets", "OK", f"{n_files:,} snippets cached"))
            elif n_files > 0:
                checks.append(HealthCheck("snippets", "WARN",
                                          f"Only {n_files:,} snippets — extraction incomplete"))
            else:
                checks.append(HealthCheck("snippets", "FAIL", "Snippet cache is empty"))
        else:
            checks.append(HealthCheck("snippets", "FAIL", f"Directory not found: {snippet_dir}"))
    else:
        checks.append(HealthCheck("snippets", "WARN", "snippet_dir not configured"))

    # Model registry check
    if registry_path is not None:
        rp = Path(registry_path)
        if rp.exists():
            try:
                entries = json.loads(rp.read_text())
                if not entries:
                    checks.append(HealthCheck("registry", "WARN", "Registry exists but is empty"))
                else:
                    best = max(entries, key=lambda e: e.get("auc") or 0.0)
                    age = _age_days(rp)
                    age_str = f"{age:.1f} days old" if age is not None else "unknown age"
                    if age is not None and age > max_model_age_days:
                        checks.append(HealthCheck("registry", "WARN",
                                                  f"Best model {best.get('model_id')} "
                                                  f"({age_str}) — consider retraining"))
                    else:
                        checks.append(HealthCheck("registry", "OK",
                                                  f"{len(entries)} model(s); best="
                                                  f"{best.get('model_id')} ({age_str})"))
            except Exception as exc:
                checks.append(HealthCheck("registry", "FAIL", f"Parse error: {exc}"))
        else:
            checks.append(HealthCheck("registry", "FAIL", "Registry file not found"))
    else:
        checks.append(HealthCheck("registry", "WARN", "registry_path not configured"))

    # Calibration check
    if calibration_path is not None:
        cp = Path(calibration_path)
        if cp.exists():
            age = _age_days(cp)
            age_str = f"{age:.1f} days old" if age is not None else "unknown age"
            if age is not None and age > max_model_age_days:
                checks.append(HealthCheck("calibration", "WARN",
                                          f"Calibration file {age_str} — may need refresh"))
            else:
                checks.append(HealthCheck("calibration", "OK",
                                          f"Calibration file present ({age_str})"))
        else:
            checks.append(HealthCheck("calibration", "FAIL", "Calibration file not found"))
    else:
        checks.append(HealthCheck("calibration", "WARN", "calibration_path not configured"))

    n_ok = sum(1 for c in checks if c.status == "OK")
    n_warn = sum(1 for c in checks if c.status == "WARN")
    n_fail = sum(1 for c in checks if c.status == "FAIL")

    if n_fail > 0:
        overall = "UNHEALTHY"
    elif n_warn > 0:
        overall = "DEGRADED"
    else:
        overall = "HEALTHY"

    return PipelineHealthReport(
        checks=tuple(checks),
        n_ok=n_ok,
        n_warn=n_warn,
        n_fail=n_fail,
        overall=overall,
        flag="OK",
    )


def format_health_report(report: PipelineHealthReport) -> str:
    """Format a Markdown pipeline health report.

    Args:
        report: PipelineHealthReport to format.

    Returns:
        Markdown string.
    """
    lines = [
        "## Pipeline Health Monitor\n",
        f"**Overall**: `{report.overall}` | "
        f"OK: {report.n_ok} | WARN: {report.n_warn} | FAIL: {report.n_fail}\n",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for c in report.checks:
        lines.append(f"| {c.name} | `{c.status}` | {c.detail} |")
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Check pipeline health.")
    parser.add_argument("--labels", help="Label JSON path.")
    parser.add_argument("--snippets", help="Snippet cache directory.")
    parser.add_argument("--registry", help="Model registry JSON path.")
    parser.add_argument("--calibration", help="Calibration JSON path.")
    parser.add_argument("--min-labels", type=int, default=5000)
    parser.add_argument("--max-age-days", type=float, default=30.0)
    args = parser.parse_args(argv)

    report = check_pipeline_health(
        label_json=Path(args.labels) if args.labels else None,
        snippet_dir=Path(args.snippets) if args.snippets else None,
        registry_path=Path(args.registry) if args.registry else None,
        calibration_path=Path(args.calibration) if args.calibration else None,
        min_labels=args.min_labels,
        max_model_age_days=args.max_age_days,
    )
    print(format_health_report(report))
    return 0 if report.overall == "HEALTHY" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
