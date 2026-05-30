"""Auto-generate a vetting checklist from a candidate dict.

Public API:
    VettingChecklistResult  -- frozen dataclass
    build_vetting_checklist(candidate) -> VettingChecklistResult
    format_vetting_checklist(result) -> str
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

_REQUIRED_KEYS = [
    ("period_days", "Period measured"),
    ("fpp", "FPP computed"),
    ("depth_ppm", "Transit depth measured"),
    ("duration_hours", "Transit duration measured"),
    ("n_transits", "Transit count confirmed"),
    ("snr", "Detection SNR computed"),
    ("odd_even_significance", "Odd/even test run"),
    ("secondary_snr", "Secondary eclipse checked"),
    ("centroid_offset_arcsec", "Centroid shift checked"),
    ("stellar_radius_rsun", "Stellar parameters fetched"),
]


@dataclass(frozen=True)
class VettingChecklistResult:
    completed: list[str]
    missing: list[str]
    n_completed: int
    n_total: int
    completeness_fraction: float
    flag: str


def build_vetting_checklist(candidate: dict[str, object]) -> VettingChecklistResult:
    completed: list[str] = []
    missing: list[str] = []
    for key, label in _REQUIRED_KEYS:
        if key in candidate and candidate[key] is not None:
            completed.append(label)
        else:
            missing.append(label)
    n_total = len(_REQUIRED_KEYS)
    n_completed = len(completed)
    completeness_fraction = n_completed / n_total
    flag = "COMPLETE" if n_completed == n_total else "INCOMPLETE"
    return VettingChecklistResult(
        completed=completed,
        missing=missing,
        n_completed=n_completed,
        n_total=n_total,
        completeness_fraction=completeness_fraction,
        flag=flag,
    )


def format_vetting_checklist(result: VettingChecklistResult) -> str:
    lines = [
        "## Vetting Checklist",
        "",
        f"**Completeness:** {result.n_completed}/{result.n_total} "
        f"({result.completeness_fraction:.0%})",
        "",
        "### Completed",
    ]
    for item in result.completed:
        lines.append(f"- [x] {item}")
    lines.append("")
    lines.append("### Missing")
    for item in result.missing:
        lines.append(f"- [ ] {item}")
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Build vetting checklist.")
    parser.add_argument("candidate_file", help="JSON file with candidate dict.")
    args = parser.parse_args()
    with open(args.candidate_file) as fh:
        candidate = json.load(fh)
    result = build_vetting_checklist(candidate)
    print(format_vetting_checklist(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
