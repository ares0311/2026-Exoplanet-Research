"""Plan upcoming transit observation windows for multiple candidates.

Given a list of candidates with known period and epoch, generates a sorted
schedule of transit mid-times that fall within a specified time range, tagged
with a priority score and observable-window metadata.

Public API
----------
SurveyWindow(tic_id, period_days, epoch_bjd, mid_time, transit_number,
             ingress, egress, duration_hours, priority, flag)
SurveyPlanResult(n_windows, windows, t_start, t_end, flag)
plan_transit_windows(candidates, t_start, t_end, *,
                     default_duration_hours, max_windows) -> SurveyPlanResult
format_survey_plan(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SurveyWindow:
    tic_id: int | None
    period_days: float
    epoch_bjd: float
    mid_time: float          # BJD of transit mid-point
    transit_number: int      # integer transit count from epoch
    ingress: float           # mid_time - duration/2 (days)
    egress: float            # mid_time + duration/2 (days)
    duration_hours: float
    priority: float          # [0, 1]
    flag: str                # "OK" | "PARTIAL"


@dataclass(frozen=True)
class SurveyPlanResult:
    n_windows: int
    windows: tuple[SurveyWindow, ...]
    t_start: float
    t_end: float
    flag: str  # "OK" | "EMPTY" | "INVALID"


def _next_transit(epoch: float, period: float, t_ref: float) -> tuple[float, int]:
    """Return (mid_time, n) for the first transit at or after t_ref."""
    n = math.ceil((t_ref - epoch) / period)
    return epoch + n * period, int(n)


def plan_transit_windows(
    candidates: list[dict],
    t_start: float,
    t_end: float,
    *,
    default_duration_hours: float = 2.0,
    max_windows: int = 500,
) -> SurveyPlanResult:
    """Generate a sorted schedule of upcoming transit windows.

    Each candidate dict should contain:
        ``period_days``, ``epoch_bjd``, and optionally
        ``tic_id``, ``duration_hours``, ``priority``.

    Args:
        candidates: List of candidate signal dicts.
        t_start: Start of planning window (BJD).
        t_end: End of planning window (BJD).
        default_duration_hours: Duration used when not in candidate dict.
        max_windows: Cap on total windows returned.

    Returns:
        :class:`SurveyPlanResult`.
    """
    if t_end <= t_start:
        return SurveyPlanResult(0, (), t_start, t_end, "INVALID")

    all_windows: list[SurveyWindow] = []

    for cand in candidates:
        p = cand.get("period_days")
        ep = cand.get("epoch_bjd")
        if p is None or ep is None or p <= 0:
            continue

        dur_h = cand.get("duration_hours") or default_duration_hours
        dur_d = dur_h / 24.0
        priority = float(cand.get("priority", 0.5))
        tic_id = cand.get("tic_id")

        mid, n = _next_transit(ep, p, t_start)
        while mid <= t_end and len(all_windows) < max_windows:
            ingress = mid - dur_d / 2.0
            egress = mid + dur_d / 2.0
            # PARTIAL if window straddles t_start or t_end
            flag = "PARTIAL" if ingress < t_start or egress > t_end else "OK"
            all_windows.append(SurveyWindow(
                tic_id=tic_id,
                period_days=p,
                epoch_bjd=ep,
                mid_time=round(mid, 6),
                transit_number=n,
                ingress=round(ingress, 6),
                egress=round(egress, 6),
                duration_hours=dur_h,
                priority=round(priority, 4),
                flag=flag,
            ))
            mid += p
            n += 1

    all_windows.sort(key=lambda w: w.mid_time)

    return SurveyPlanResult(
        n_windows=len(all_windows),
        windows=tuple(all_windows),
        t_start=t_start,
        t_end=t_end,
        flag="OK",
    )


def format_survey_plan(result: SurveyPlanResult) -> str:
    """Format the transit survey plan as Markdown."""
    if result.flag == "INVALID":
        return "## Transit Survey Plan\n\n_Invalid input._\n"
    if result.flag == "EMPTY":
        return "## Transit Survey Plan\n\n_No transits in planning window._\n"

    lines = [
        "## Transit Survey Plan",
        "",
        f"- Planning window: {result.t_start:.2f} – {result.t_end:.2f} BJD",
        f"- Total transit windows: {result.n_windows}",
        "",
        "| TIC ID | Period (d) | Mid-time (BJD) | Duration (h) | Priority | Flag |",
        "|---|---|---|---|---|---|",
    ]
    for w in result.windows[:50]:
        tid = str(w.tic_id) if w.tic_id is not None else "—"
        lines.append(
            f"| {tid} | {w.period_days:.4f} | {w.mid_time:.4f}"
            f" | {w.duration_hours:.2f} | {w.priority:.3f} | {w.flag} |"
        )
    if result.n_windows > 50:
        lines.append(f"\n_... and {result.n_windows - 50} more windows._")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="transit_survey_planner",
        description="Plan upcoming transit windows for a set of candidates.",
    )
    parser.add_argument("t_start", type=float)
    parser.add_argument("t_end", type=float)
    parser.parse_args(argv)

    result = plan_transit_windows([], 0.0, 0.0)
    print(format_survey_plan(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
