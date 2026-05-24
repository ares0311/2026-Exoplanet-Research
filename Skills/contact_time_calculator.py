"""Compute absolute BJD contact times T1/T2/T3/T4 for a transit event.

Given the transit mid-time, total duration T14, and flat-bottom duration T23,
derives the four contact times.  Complements ``ephemeris_predictor`` (which
gives mid-times) and ``transit_ingress_timer`` (which gives durations).

Public API
----------
ContactTimesResult(t0_bjd, t1_bjd, t2_bjd, t3_bjd, t4_bjd,
                   ingress_duration_hours, flat_bottom_hours,
                   total_duration_hours, flag)
compute_contact_times(t0_bjd, t14_hours, t23_hours) -> ContactTimesResult
format_contact_times(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ContactTimesResult:
    t0_bjd: float
    t1_bjd: float | None   # first contact (ingress start)
    t2_bjd: float | None   # second contact (ingress end)
    t3_bjd: float | None   # third contact (egress start)
    t4_bjd: float | None   # fourth contact (egress end)
    ingress_duration_hours: float | None
    flat_bottom_hours: float | None
    total_duration_hours: float
    flag: str  # "OK" | "GRAZING" | "INVALID"


def compute_contact_times(
    t0_bjd: float,
    t14_hours: float,
    t23_hours: float | None = None,
) -> ContactTimesResult:
    """Compute T1, T2, T3, T4 contact times in BJD.

    T1 = T0 − T14/2  (first contact)
    T4 = T0 + T14/2  (fourth contact)
    T2 = T0 − T23/2  (second contact, if not grazing)
    T3 = T0 + T23/2  (third contact, if not grazing)

    Args:
        t0_bjd: Transit mid-time (BJD).
        t14_hours: Total transit duration T14 (hours).
        t23_hours: Flat-bottom duration T23 (hours).  ``None`` indicates a
            grazing transit — T2/T3 are undefined.

    Returns:
        :class:`ContactTimesResult`.
    """
    if not math.isfinite(t0_bjd) or not math.isfinite(t14_hours):
        return ContactTimesResult(t0_bjd, None, None, None, None, None, None, 0.0, "INVALID")
    if t14_hours <= 0:
        return ContactTimesResult(t0_bjd, None, None, None, None, None, None, 0.0, "INVALID")
    if t23_hours is not None and t23_hours > t14_hours:
        return ContactTimesResult(t0_bjd, None, None, None, None, None, None, t14_hours, "INVALID")

    t14_days = t14_hours / 24.0
    t1 = t0_bjd - t14_days / 2.0
    t4 = t0_bjd + t14_days / 2.0

    if t23_hours is None or t23_hours < 0:
        # Grazing — no flat bottom
        return ContactTimesResult(
            t0_bjd=t0_bjd,
            t1_bjd=round(t1, 8),
            t2_bjd=None,
            t3_bjd=None,
            t4_bjd=round(t4, 8),
            ingress_duration_hours=round(t14_hours / 4.0, 6),
            flat_bottom_hours=0.0,
            total_duration_hours=round(t14_hours, 6),
            flag="GRAZING",
        )

    t23_days = t23_hours / 24.0
    t2 = t0_bjd - t23_days / 2.0
    t3 = t0_bjd + t23_days / 2.0
    ingress = (t14_hours - t23_hours) / 2.0

    return ContactTimesResult(
        t0_bjd=t0_bjd,
        t1_bjd=round(t1, 8),
        t2_bjd=round(t2, 8),
        t3_bjd=round(t3, 8),
        t4_bjd=round(t4, 8),
        ingress_duration_hours=round(ingress, 6),
        flat_bottom_hours=round(t23_hours, 6),
        total_duration_hours=round(t14_hours, 6),
        flag="OK",
    )


def format_contact_times(result: ContactTimesResult) -> str:
    """Format contact times as Markdown."""
    def _fmt(v):
        return f"{v:.8f}" if v is not None else "—"

    lines = [
        "## Contact Time Calculator",
        "",
        f"- T0 (mid-transit): {result.t0_bjd:.8f} BJD",
        f"- **T1 (1st contact): {_fmt(result.t1_bjd)} BJD**",
        f"- T2 (2nd contact): {_fmt(result.t2_bjd)} BJD",
        f"- T3 (3rd contact): {_fmt(result.t3_bjd)} BJD",
        f"- **T4 (4th contact): {_fmt(result.t4_bjd)} BJD**",
        f"- Total duration T14: {result.total_duration_hours} hours",
        f"- Flat bottom T23: {result.flat_bottom_hours} hours",
        f"- Ingress duration: {result.ingress_duration_hours} hours",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="contact_time_calculator",
        description="Compute T1/T2/T3/T4 contact times from transit mid-time and durations.",
    )
    parser.add_argument("t0_bjd", type=float)
    parser.add_argument("t14_hours", type=float)
    parser.add_argument("--t23-hours", type=float, default=None)
    args = parser.parse_args(argv)

    result = compute_contact_times(args.t0_bjd, args.t14_hours, args.t23_hours)
    print(format_contact_times(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
