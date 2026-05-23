"""Adapter: convert boolean vetting flag dicts to structured triage results.

Thin bridge layer that translates a plain ``{flag_name: bool}`` dict (as
produced by many vetting scripts) into a list of :class:`FlagEntry` objects
plus a concise triage summary.

Public API
----------
FlagEntry(name, value, category, severity)
VettingTriageResult(n_flags, n_raised, n_critical, n_warning, n_info,
                    raised_flags, triage_decision, flag)
boolean_flags_to_entries(flag_dict, *, severity_map) -> list[FlagEntry]
run_vetting_triage(flag_dict, *, severity_map) -> VettingTriageResult
format_triage_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass

# ------------------------------------------------------------------
# Default severity mapping: flag_name -> (category, severity)
# severity: "critical" | "warning" | "info"
# ------------------------------------------------------------------
_DEFAULT_SEVERITY: dict[str, tuple[str, str]] = {
    # Critical — almost certain false positives
    "centroid_shift": ("centroid", "critical"),
    "secondary_eclipse": ("photometry", "critical"),
    "odd_even_depth_mismatch": ("photometry", "critical"),
    "known_false_positive": ("catalog", "critical"),
    "v_shaped_transit": ("morphology", "critical"),
    # Warnings
    "nearby_star_contamination": ("photometry", "warning"),
    "high_background": ("quality", "warning"),
    "momentum_dump_overlap": ("quality", "warning"),
    "stellar_variability": ("stellar", "warning"),
    "rms_elevated": ("quality", "warning"),
    "single_transit": ("coverage", "warning"),
    # Info
    "period_alias_possible": ("period", "info"),
    "crowded_field": ("photometry", "info"),
    "multi_sector_depth_variation": ("photometry", "info"),
    "gap_near_transit": ("quality", "info"),
}

_TRIAGE_PASS = "PASS"
_TRIAGE_WARN = "WARN"
_TRIAGE_FAIL = "FAIL"


@dataclass(frozen=True)
class FlagEntry:
    name: str
    value: bool
    category: str
    severity: str  # "critical" | "warning" | "info" | "unknown"


@dataclass(frozen=True)
class VettingTriageResult:
    n_flags: int
    n_raised: int       # number of True flags
    n_critical: int
    n_warning: int
    n_info: int
    raised_flags: tuple[FlagEntry, ...]
    triage_decision: str  # "PASS" | "WARN" | "FAIL"
    flag: str  # "OK" | "EMPTY" | "INVALID"


def boolean_flags_to_entries(
    flag_dict: dict,
    *,
    severity_map: dict[str, tuple[str, str]] | None = None,
) -> list[FlagEntry]:
    """Convert a ``{name: bool}`` flag dict to :class:`FlagEntry` objects.

    Args:
        flag_dict: Mapping of flag name → bool value.
        severity_map: Optional ``{name: (category, severity)}`` override.
            Unmapped flags receive ``category="unknown"``, ``severity="info"``.

    Returns:
        List of :class:`FlagEntry` in dict insertion order.
    """
    smap = _DEFAULT_SEVERITY.copy()
    if severity_map:
        smap.update(severity_map)

    entries: list[FlagEntry] = []
    for name, value in flag_dict.items():
        cat, sev = smap.get(name, ("unknown", "info"))
        entries.append(FlagEntry(name=name, value=bool(value), category=cat, severity=sev))
    return entries


def run_vetting_triage(
    flag_dict: dict,
    *,
    severity_map: dict[str, tuple[str, str]] | None = None,
) -> VettingTriageResult:
    """Run triage on a boolean flag dict.

    Decision rules (applied in order):
    - Any critical flag raised → ``FAIL``
    - Any warning flag raised → ``WARN``
    - Otherwise → ``PASS``

    Args:
        flag_dict: Mapping of flag name → bool.
        severity_map: Optional severity override (see :func:`boolean_flags_to_entries`).

    Returns:
        :class:`VettingTriageResult`.
    """
    if not isinstance(flag_dict, dict):
        return VettingTriageResult(0, 0, 0, 0, 0, (), _TRIAGE_FAIL, "INVALID")
    if not flag_dict:
        return VettingTriageResult(0, 0, 0, 0, 0, (), _TRIAGE_PASS, "EMPTY")

    entries = boolean_flags_to_entries(flag_dict, severity_map=severity_map)
    raised = [e for e in entries if e.value]

    n_critical = sum(1 for e in raised if e.severity == "critical")
    n_warning = sum(1 for e in raised if e.severity == "warning")
    n_info = sum(1 for e in raised if e.severity == "info")

    if n_critical > 0:
        decision = _TRIAGE_FAIL
    elif n_warning > 0:
        decision = _TRIAGE_WARN
    else:
        decision = _TRIAGE_PASS

    return VettingTriageResult(
        n_flags=len(entries),
        n_raised=len(raised),
        n_critical=n_critical,
        n_warning=n_warning,
        n_info=n_info,
        raised_flags=tuple(raised),
        triage_decision=decision,
        flag="OK",
    )


def format_triage_result(result: VettingTriageResult) -> str:
    """Format vetting triage result as Markdown."""
    lines = [
        "## Vetting Boolean Adapter",
        "",
        f"- Total flags: {result.n_flags}",
        f"- Raised: {result.n_raised}",
        f"  - Critical: {result.n_critical}",
        f"  - Warning: {result.n_warning}",
        f"  - Info: {result.n_info}",
        f"- **Triage decision: {result.triage_decision}**",
        f"- **Flag: {result.flag}**",
    ]
    if result.raised_flags:
        lines += ["", "### Raised Flags", ""]
        for e in result.raised_flags:
            lines.append(f"- `{e.name}` [{e.category} / {e.severity}]")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="vetting_boolean_adapter",
        description="Convert boolean vetting flag dict to structured triage.",
    )
    parser.add_argument(
        "--flags", type=str, default=None, help="JSON string of {name: bool} flags"
    )
    args = parser.parse_args(argv)

    flag_dict = json.loads(args.flags) if args.flags else {}
    result = run_vetting_triage(flag_dict)
    print(format_triage_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
