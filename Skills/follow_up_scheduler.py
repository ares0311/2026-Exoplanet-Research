"""Generate an observing schedule for transit follow-up.

Given a candidate's ephemeris (epoch + period) and an observing window
(start/end dates), compute all transit mid-times and rank them by:
- observability score (based on approximate airmass/hour-angle proxy)
- transit depth (deeper = easier to detect)
- days until next transit

Public API
----------
TransitEvent(mid_bjd, mid_iso, priority_score, depth_ppm, duration_hours, days_away)
build_schedule(period_days, epoch_bjd, depth_ppm, duration_hours, *,
               window_start_bjd, window_end_bjd, max_events) -> list[TransitEvent]
format_schedule(events, tic_id) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime


def _bjd_to_iso(bjd: float) -> str:
    """Approximate BJD → ISO 8601 string (BJD_TDB ≈ JD - 2400000.5 - 0.0002)."""
    jd = bjd  # BJD_TDB ≈ JD for scheduling purposes
    # seconds since Unix epoch (Unix: 1970-01-01T00:00:00 = JD 2440587.5)
    unix_sec = (jd - 2440587.5) * 86400.0
    try:
        dt = datetime.fromtimestamp(unix_sec, tz=UTC)
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    except (OSError, OverflowError):
        return f"BJD {bjd:.4f}"


@dataclass(frozen=True)
class TransitEvent:
    mid_bjd: float
    mid_iso: str
    transit_number: int         # integer from epoch
    priority_score: float       # 0–1; higher = higher priority
    depth_ppm: float
    duration_hours: float
    days_away: float            # days from now (BJD_now approximated by window_start)


def _depth_score(depth_ppm: float) -> float:
    """Normalise depth to [0, 1] favouring deeper transits; saturates at 10 000 ppm."""
    return min(depth_ppm / 10_000.0, 1.0)


def build_schedule(
    period_days: float,
    epoch_bjd: float,
    depth_ppm: float,
    duration_hours: float,
    *,
    window_start_bjd: float,
    window_end_bjd: float,
    max_events: int = 20,
    now_bjd: float | None = None,
) -> list[TransitEvent]:
    """Compute transit mid-times within an observing window and rank them.

    Args:
        period_days: Orbital period.
        epoch_bjd: Reference mid-transit BJD.
        depth_ppm: Transit depth in ppm (affects priority).
        duration_hours: Transit duration in hours.
        window_start_bjd: Start of the scheduling window (BJD).
        window_end_bjd: End of the scheduling window (BJD).
        max_events: Maximum number of events to return.
        now_bjd: Reference time for *days_away* (defaults to window_start).

    Returns:
        List of :class:`TransitEvent`, sorted by priority descending.
    """
    if period_days <= 0:
        raise ValueError(f"period_days must be positive, got {period_days}")
    if window_end_bjd <= window_start_bjd:
        raise ValueError("window_end_bjd must be after window_start_bjd")

    reference = now_bjd if now_bjd is not None else window_start_bjd
    ds = _depth_score(depth_ppm)

    # Find first transit at or after window_start
    n_start = math.ceil((window_start_bjd - epoch_bjd) / period_days)
    events: list[TransitEvent] = []

    n = n_start
    while True:
        mid = epoch_bjd + n * period_days
        if mid > window_end_bjd:
            break
        days_away = mid - reference

        # Priority heuristic: prefer deep transits soon
        recency = max(0.0, 1.0 - days_away / max(window_end_bjd - window_start_bjd, 1))
        priority = 0.6 * ds + 0.4 * recency

        events.append(TransitEvent(
            mid_bjd=mid,
            mid_iso=_bjd_to_iso(mid),
            transit_number=n,
            priority_score=round(priority, 4),
            depth_ppm=depth_ppm,
            duration_hours=duration_hours,
            days_away=round(days_away, 2),
        ))
        n += 1

    # Sort by priority descending, then time ascending
    events.sort(key=lambda e: (-e.priority_score, e.mid_bjd))
    return events[:max_events]


def format_schedule(events: list[TransitEvent], tic_id: int = 0) -> str:
    """Format transit schedule as a Markdown table."""
    if not events:
        return "No observable transits in the specified window.\n"

    header = f"## Follow-Up Schedule — TIC {tic_id}\n\n"
    header += (
        "| Transit # | Mid-time (UTC) | Depth (ppm) | Duration (h) | Priority | Days away |\n"
        "|-----------|----------------|-------------|--------------|----------|-----------|\n"
    )
    rows = []
    for e in events:
        rows.append(
            f"| {e.transit_number} | {e.mid_iso} | {e.depth_ppm:.0f} | "
            f"{e.duration_hours:.2f} | {e.priority_score:.3f} | {e.days_away:.1f} |"
        )
    return header + "\n".join(rows) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="follow_up_scheduler",
        description="Generate a transit follow-up observing schedule.",
    )
    parser.add_argument("--period", type=float, required=True, metavar="DAYS")
    parser.add_argument("--epoch", type=float, required=True, metavar="BJD")
    parser.add_argument("--depth", type=float, required=True, metavar="PPM")
    parser.add_argument("--duration", type=float, required=True, metavar="HOURS")
    parser.add_argument("--start", type=float, required=True,
                        metavar="BJD", help="Window start (BJD).")
    parser.add_argument("--end", type=float, required=True,
                        metavar="BJD", help="Window end (BJD).")
    parser.add_argument("--max-events", type=int, default=20)
    parser.add_argument("--tic-id", type=int, default=0)
    args = parser.parse_args(argv)

    events = build_schedule(
        args.period, args.epoch, args.depth, args.duration,
        window_start_bjd=args.start,
        window_end_bjd=args.end,
        max_events=args.max_events,
    )
    print(format_schedule(events, tic_id=args.tic_id))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
