"""Check availability and versions of all optional/required pipeline dependencies.

Uses importlib.metadata for version detection; no pip calls required.

Public API
----------
DependencyInfo(name, required, available, version, feature)
DependencyCheckResult(python_version, n_available, n_required_missing,
                      n_optional_missing, dependencies, feature_matrix, flag)
check_dependencies(extra_packages=None) -> DependencyCheckResult
format_dependency_check(result) -> str
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

# (name, required, feature)
_KNOWN_DEPS: list[tuple[str, bool, str]] = [
    ("astropy", True, "search"),
    ("numpy", True, "clean"),
    ("lightkurve", False, "fetch"),
    ("xgboost", False, "xgboost"),
    ("torch", False, "cnn"),
    ("astroquery", False, "star_scanner"),
    ("matplotlib", False, "plots"),
    ("rich", False, "cli_display"),
]


@dataclass(frozen=True)
class DependencyInfo:
    name: str
    required: bool
    available: bool
    version: str | None
    feature: str


@dataclass(frozen=True)
class DependencyCheckResult:
    python_version: str
    n_available: int
    n_required_missing: int
    n_optional_missing: int
    dependencies: tuple[DependencyInfo, ...]
    feature_matrix: dict[str, bool]
    flag: str  # "OK" | "MISSING_REQUIRED" | "DEGRADED"


def _probe(name: str) -> tuple[bool, str | None]:
    """Return (available, version_str)."""
    try:
        import importlib.metadata as meta
        version = meta.version(name)
        return True, version
    except Exception:
        pass
    # Fallback: try importing
    try:
        mod = __import__(name)
        ver = getattr(mod, "__version__", None)
        return True, str(ver) if ver else None
    except ImportError:
        return False, None


def check_dependencies(
    extra_packages: list[str] | None = None,
) -> DependencyCheckResult:
    """Check availability of all pipeline dependencies.

    Args:
        extra_packages: Additional package names to check (all optional).

    Returns:
        :class:`DependencyCheckResult`.
    """
    deps_spec = list(_KNOWN_DEPS)
    if extra_packages:
        for pkg in extra_packages:
            # Extra packages are optional; feature defaults to pkg name
            deps_spec.append((pkg, False, pkg))

    infos: list[DependencyInfo] = []
    feature_matrix: dict[str, bool] = {}
    n_available = 0
    n_required_missing = 0
    n_optional_missing = 0

    for name, required, feature in deps_spec:
        available, version = _probe(name)
        info = DependencyInfo(
            name=name,
            required=required,
            available=available,
            version=version,
            feature=feature,
        )
        infos.append(info)
        if available:
            n_available += 1
        elif required:
            n_required_missing += 1
        else:
            n_optional_missing += 1

        # Update feature matrix: feature is enabled if all deps providing it are available
        if feature not in feature_matrix:
            feature_matrix[feature] = available
        else:
            feature_matrix[feature] = feature_matrix[feature] and available

    if n_required_missing > 0:
        flag = "MISSING_REQUIRED"
    elif n_optional_missing > 0:
        flag = "DEGRADED"
    else:
        flag = "OK"

    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    return DependencyCheckResult(
        python_version=py_ver,
        n_available=n_available,
        n_required_missing=n_required_missing,
        n_optional_missing=n_optional_missing,
        dependencies=tuple(infos),
        feature_matrix=feature_matrix,
        flag=flag,
    )


def format_dependency_check(result: DependencyCheckResult) -> str:
    """Format dependency check result as Markdown."""
    lines = [
        "## Pipeline Dependency Checker",
        "",
        f"- Python: {result.python_version}",
        f"- Available: {result.n_available}",
        f"- Required missing: {result.n_required_missing}",
        f"- Optional missing: {result.n_optional_missing}",
        f"- **Flag: {result.flag}**",
        "",
        "### Dependencies",
        "",
        "| Package | Required | Available | Version | Feature |",
        "|---------|----------|-----------|---------|---------|",
    ]
    for d in result.dependencies:
        req_s = "yes" if d.required else "no"
        avail_s = "yes" if d.available else "no"
        ver_s = d.version or "—"
        lines.append(f"| {d.name} | {req_s} | {avail_s} | {ver_s} | {d.feature} |")
    lines.append("")
    lines.append("### Feature Matrix")
    lines.append("")
    for feat, avail in sorted(result.feature_matrix.items()):
        lines.append(f"- {feat}: {'enabled' if avail else 'disabled'}")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="pipeline_dependency_checker",
        description="Check availability of pipeline dependencies.",
    )
    parser.add_argument("--extra", nargs="*", help="Additional packages to check")
    args = parser.parse_args(argv)

    result = check_dependencies(extra_packages=args.extra)
    print(format_dependency_check(result))
    return 0 if result.flag == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
