"""Build a ground-based observation report comparing measured vs predicted transit.

Quality grade:
  A: depth_ratio in [0.8, 1.2] AND |timing_offset_min| < 5
  B: depth_ratio in [0.6, 1.4] AND |timing_offset_min| < 15
  C: otherwise passes basic depth checks (ratio in [0.2, 5.0])
  D: depth_ratio < 0.2 OR depth_ratio > 5.0

Public API
----------
GroundObsReport(target, date, depth_ratio, timing_offset_min, quality_grade, flag)
build_ground_obs_report(obs_meta) -> GroundObsReport
format_ground_obs_report(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GroundObsReport:
    target: str
    date: str
    depth_ratio: float
    timing_offset_min: float
    quality_grade: str  # A / B / C / D
    flag: str  # "OK", "DEPTH_MISMATCH", "TIMING_OFFSET"


def build_ground_obs_report(obs_meta: dict) -> GroundObsReport:
    """Build a ground-based observation report.

    Args:
        obs_meta: Dict with keys:
            ``target`` (str), ``date`` (str),
            ``depth_ppm_measured`` (float), ``depth_ppm_expected`` (float),
            ``timing_offset_min`` (float).

    Returns:
        :class:`GroundObsReport`.
    """
    target = str(obs_meta.get("target", ""))
    date = str(obs_meta.get("date", ""))
    depth_measured = float(obs_meta.get("depth_ppm_measured", 0.0))
    depth_expected = float(obs_meta.get("depth_ppm_expected", 1.0))
    timing_offset = float(obs_meta.get("timing_offset_min", 0.0))

    depth_ratio = 0.0 if depth_expected <= 0 else depth_measured / depth_expected

    abs_timing = abs(timing_offset)

    # Grade assignment
    if depth_ratio < 0.2 or depth_ratio > 5.0:
        quality_grade = "D"
    elif 0.8 <= depth_ratio <= 1.2 and abs_timing < 5.0:
        quality_grade = "A"
    elif 0.6 <= depth_ratio <= 1.4 and abs_timing < 15.0:
        quality_grade = "B"
    else:
        quality_grade = "C"

    # Flag
    depth_mismatch = not (0.6 <= depth_ratio <= 1.4)
    timing_offset_flag = abs_timing >= 15.0

    if depth_mismatch:
        flag = "DEPTH_MISMATCH"
    elif timing_offset_flag:
        flag = "TIMING_OFFSET"
    else:
        flag = "OK"

    return GroundObsReport(
        target=target,
        date=date,
        depth_ratio=round(depth_ratio, 4),
        timing_offset_min=round(timing_offset, 3),
        quality_grade=quality_grade,
        flag=flag,
    )


def format_ground_obs_report(result: GroundObsReport) -> str:
    """Format ground observation report as Markdown."""
    lines = [
        "## Ground Observation Report",
        "",
        f"- Target: {result.target}",
        f"- Date: {result.date}",
        f"- Depth ratio (measured/expected): {result.depth_ratio:.4f}",
        f"- Timing offset: {result.timing_offset_min:.1f} min",
        f"- Quality grade: **{result.quality_grade}**",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", default="TIC-000000")
    p.add_argument("--date", default="2026-06-01")
    p.add_argument("--depth-measured", type=float, default=5000.0)
    p.add_argument("--depth-expected", type=float, default=5000.0)
    p.add_argument("--timing-offset-min", type=float, default=0.0)
    args = p.parse_args(argv)
    obs_meta = {
        "target": args.target,
        "date": args.date,
        "depth_ppm_measured": args.depth_measured,
        "depth_ppm_expected": args.depth_expected,
        "timing_offset_min": args.timing_offset_min,
    }
    r = build_ground_obs_report(obs_meta)
    print(format_ground_obs_report(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
