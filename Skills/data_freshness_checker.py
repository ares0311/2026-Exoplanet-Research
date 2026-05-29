"""Check whether cached data artifacts are stale and need refreshing.

Public API
----------
FreshnessCheck(name, path, age_days, max_age_days, status, detail)
FreshnessReport(checks, n_fresh, n_stale, n_missing, flag)
check_data_freshness(artifacts, *, max_age_days) -> FreshnessReport
format_freshness_report(report) -> str
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FreshnessCheck:
    name: str
    path: str
    age_days: float | None
    max_age_days: float
    status: str   # "FRESH" | "STALE" | "MISSING"
    detail: str


@dataclass(frozen=True)
class FreshnessReport:
    checks: tuple[FreshnessCheck, ...]
    n_fresh: int
    n_stale: int
    n_missing: int
    flag: str  # "OK" | "STALE" | "MISSING"


def _age_days(path: Path) -> float | None:
    try:
        return (time.time() - path.stat().st_mtime) / 86400
    except OSError:
        return None


def check_data_freshness(
    artifacts: dict[str, str | Path],
    *,
    max_age_days: float = 7.0,
    per_artifact_max_age: dict[str, float] | None = None,
) -> FreshnessReport:
    """Check freshness of named data artifacts.

    Args:
        artifacts: Mapping of artifact name → file path.
        max_age_days: Default maximum age in days before marking stale.
        per_artifact_max_age: Optional per-artifact age overrides.

    Returns:
        FreshnessReport with per-check results.
    """
    checks: list[FreshnessCheck] = []
    for name, raw_path in artifacts.items():
        p = Path(raw_path)
        limit = (per_artifact_max_age or {}).get(name, max_age_days)
        age = _age_days(p)
        if age is None:
            checks.append(FreshnessCheck(
                name=name,
                path=str(p),
                age_days=None,
                max_age_days=limit,
                status="MISSING",
                detail=f"File not found: {p}",
            ))
        elif age > limit:
            checks.append(FreshnessCheck(
                name=name,
                path=str(p),
                age_days=round(age, 2),
                max_age_days=limit,
                status="STALE",
                detail=f"{age:.1f} days old (limit {limit:.0f} d)",
            ))
        else:
            checks.append(FreshnessCheck(
                name=name,
                path=str(p),
                age_days=round(age, 2),
                max_age_days=limit,
                status="FRESH",
                detail=f"{age:.1f} days old",
            ))

    n_fresh = sum(1 for c in checks if c.status == "FRESH")
    n_stale = sum(1 for c in checks if c.status == "STALE")
    n_missing = sum(1 for c in checks if c.status == "MISSING")

    if n_missing > 0:
        flag = "MISSING"
    elif n_stale > 0:
        flag = "STALE"
    else:
        flag = "OK"

    return FreshnessReport(
        checks=tuple(checks),
        n_fresh=n_fresh,
        n_stale=n_stale,
        n_missing=n_missing,
        flag=flag,
    )


def format_freshness_report(report: FreshnessReport) -> str:
    """Format freshness report as Markdown.

    Args:
        report: FreshnessReport to format.

    Returns:
        Markdown string.
    """
    lines = [
        "## Data Freshness Report\n",
        f"**Status**: `{report.flag}` | "
        f"Fresh: {report.n_fresh} | Stale: {report.n_stale} | Missing: {report.n_missing}\n",
        "",
        "| Artifact | Age (days) | Limit (days) | Status | Detail |",
        "|---|---|---|---|---|",
    ]
    for c in report.checks:
        age_str = f"{c.age_days:.1f}" if c.age_days is not None else "—"
        lines.append(
            f"| {c.name} | {age_str} | {c.max_age_days:.0f} | `{c.status}` | {c.detail} |"
        )
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Check data freshness.")
    parser.add_argument("paths", nargs="+", help="name:path pairs.")
    parser.add_argument("--max-age-days", type=float, default=7.0)
    args = parser.parse_args(argv)

    artifacts: dict[str, str] = {}
    for item in args.paths:
        if ":" in item:
            name, path = item.split(":", 1)
        else:
            name = Path(item).stem
            path = item
        artifacts[name] = path

    report = check_data_freshness(artifacts, max_age_days=args.max_age_days)
    print(format_freshness_report(report))
    return 0 if report.flag == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
