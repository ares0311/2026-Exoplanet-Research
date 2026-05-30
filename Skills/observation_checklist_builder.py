"""Build a structured transit observation checklist.

Public API
----------
ChecklistItem(phase, text)
ObsChecklist(target, n_pre_items, n_during_items, n_post_items, items, flag)
build_observation_checklist(target, period_days, epoch_bjd, duration_hours) -> ObsChecklist
format_observation_checklist(result) -> str
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


@dataclass(frozen=True)
class ChecklistItem:
    phase: str   # "pre" / "during" / "post"
    text: str


@dataclass(frozen=True)
class ObsChecklist:
    target: str
    n_pre_items: int
    n_during_items: int
    n_post_items: int
    items: tuple[ChecklistItem, ...]
    flag: str  # "OK"


def build_observation_checklist(
    target: str,
    period_days: float,
    epoch_bjd: float,
    duration_hours: float,
) -> ObsChecklist:
    """Build a structured transit observation checklist.

    Args:
        target: Target name or TIC ID string.
        period_days: Orbital period in days.
        epoch_bjd: Transit epoch in BJD.
        duration_hours: Transit duration in hours.

    Returns:
        :class:`ObsChecklist`.
    """
    pre = [
        ChecklistItem("pre", "Confirm target coordinates and finder chart"),
        ChecklistItem(
            "pre",
            f"Verify predicted transit mid-time (T0={epoch_bjd:.4f} BJD, P={period_days:.4f} d)",
        ),
        ChecklistItem(
            "pre",
            f"Allocate {duration_hours + 2:.1f} h total (transit + 1 h baseline each side)",
        ),
        ChecklistItem("pre", "Select comparison stars of similar brightness"),
        ChecklistItem("pre", "Set exposure time to keep target below saturation"),
        ChecklistItem("pre", "Start guiding and check focus"),
    ]
    during = [
        ChecklistItem("during", "Monitor airmass; flag if > 2.0"),
        ChecklistItem("during", "Check centroid stability every 30 min"),
        ChecklistItem("during", "Log any interruptions with timestamps"),
        ChecklistItem("during", "Monitor sky background for clouds"),
    ]
    post = [
        ChecklistItem("post", "Verify ingress, mid-transit, and egress are covered"),
        ChecklistItem("post", "Run aperture photometry with comparison ensemble"),
        ChecklistItem("post", "Record measured depth and timing offset"),
        ChecklistItem("post", "Upload raw data and reduction to follow-up log"),
    ]
    items = tuple(pre + during + post)
    return ObsChecklist(
        target=target,
        n_pre_items=len(pre),
        n_during_items=len(during),
        n_post_items=len(post),
        items=items,
        flag="OK",
    )


def format_observation_checklist(result: ObsChecklist) -> str:
    """Format observation checklist as Markdown."""
    lines = [f"## Observation Checklist — {result.target}", ""]
    for phase_label, phase_key in [
        ("Pre-Observation", "pre"),
        ("During", "during"),
        ("Post-Observation", "post"),
    ]:
        lines.append(f"### {phase_label}")
        for item in result.items:
            if item.phase == phase_key:
                lines.append(f"- [ ] {item.text}")
        lines.append("")
    lines.append(f"Flag: **{result.flag}**")
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", default="TIC-000000")
    p.add_argument("--period", type=float, default=3.0)
    p.add_argument("--epoch", type=float, default=2458000.0)
    p.add_argument("--duration", type=float, default=2.0)
    args = p.parse_args(argv)
    r = build_observation_checklist(args.target, args.period, args.epoch, args.duration)
    print(format_observation_checklist(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
