"""Compare a transit signal across multiple missions (TESS, Kepler, K2).

Checks period consistency, depth consistency, and flags discrepancies
between mission-specific measurements of the same target.

Public API
----------
MissionMeasurement(mission, period_days, depth_ppm, duration_hours,
                   n_transits, snr, source)
MissionComparison(tic_id, measurements, period_consistent,
                  depth_consistent, best_mission, flag)
compare_multi_mission(tic_id, measurements, *, period_rtol,
                      depth_rtol) -> MissionComparison
format_mission_comparison(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MissionMeasurement:
    mission: str   # "TESS" | "Kepler" | "K2"
    period_days: float | None
    depth_ppm: float | None
    duration_hours: float | None
    n_transits: int | None
    snr: float | None
    source: str    # pipeline or dataset name


@dataclass(frozen=True)
class MissionComparison:
    tic_id: int | None
    measurements: tuple[MissionMeasurement, ...]
    period_consistent: bool
    depth_consistent: bool
    best_mission: str | None   # highest SNR mission
    period_spread_frac: float | None
    depth_spread_frac: float | None
    flag: str  # "CONSISTENT"|"PERIOD_DISCREPANT"|"DEPTH_DISCREPANT"|"SINGLE_MISSION"|"EMPTY"


def _frac_spread(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mn = sum(values) / len(values)
    if mn == 0:
        return 0.0
    return (max(values) - min(values)) / abs(mn)


def compare_multi_mission(
    tic_id: int | None,
    measurements: list[MissionMeasurement],
    *,
    period_rtol: float = 0.01,
    depth_rtol: float = 0.20,
) -> MissionComparison:
    """Compare transit measurements across missions.

    Args:
        tic_id: TIC identifier.
        measurements: List of per-mission measurements.
        period_rtol: Relative tolerance for period consistency.
        depth_rtol: Relative tolerance for depth consistency.

    Returns:
        MissionComparison with consistency flags and best mission.
    """
    if not measurements:
        return MissionComparison(
            tic_id=tic_id,
            measurements=(),
            period_consistent=False,
            depth_consistent=False,
            best_mission=None,
            period_spread_frac=None,
            depth_spread_frac=None,
            flag="EMPTY",
        )

    if len(measurements) == 1:
        return MissionComparison(
            tic_id=tic_id,
            measurements=tuple(measurements),
            period_consistent=True,
            depth_consistent=True,
            best_mission=measurements[0].mission,
            period_spread_frac=0.0,
            depth_spread_frac=0.0,
            flag="SINGLE_MISSION",
        )

    periods = [m.period_days for m in measurements if m.period_days is not None]
    depths = [m.depth_ppm for m in measurements if m.depth_ppm is not None]

    period_spread = _frac_spread(periods) if len(periods) >= 2 else None
    depth_spread = _frac_spread(depths) if len(depths) >= 2 else None

    period_consistent = period_spread is None or period_spread <= period_rtol
    depth_consistent = depth_spread is None or depth_spread <= depth_rtol

    best: MissionMeasurement | None = None
    for m in measurements:
        if m.snr is not None and (best is None or (best.snr is None) or m.snr > best.snr):
            best = m

    if not period_consistent:
        flag = "PERIOD_DISCREPANT"
    elif not depth_consistent:
        flag = "DEPTH_DISCREPANT"
    else:
        flag = "CONSISTENT"

    return MissionComparison(
        tic_id=tic_id,
        measurements=tuple(measurements),
        period_consistent=period_consistent,
        depth_consistent=depth_consistent,
        best_mission=best.mission if best else None,
        period_spread_frac=round(period_spread, 6) if period_spread is not None else None,
        depth_spread_frac=round(depth_spread, 6) if depth_spread is not None else None,
        flag=flag,
    )


def format_mission_comparison(result: MissionComparison) -> str:
    """Format mission comparison as Markdown.

    Args:
        result: MissionComparison to format.

    Returns:
        Markdown string.
    """
    tic_str = str(result.tic_id) if result.tic_id is not None else "Unknown"
    period_ok = "✓" if result.period_consistent else "✗"
    depth_ok = "✓" if result.depth_consistent else "✗"
    lines = [
        f"## Multi-Mission Comparison — TIC {tic_str}\n",
        f"**Status**: `{result.flag}` | "
        f"Period consistent: {period_ok} | Depth consistent: {depth_ok} | "
        f"Best mission: {result.best_mission or '—'}\n",
    ]
    if not result.measurements:
        lines.append("\n_No measurements._")
        return "\n".join(lines)

    lines += [
        "",
        "| Mission | Period (d) | Depth (ppm) | Duration (h) | N transits | SNR |",
        "|---|---|---|---|---|---|",
    ]
    for m in result.measurements:
        p_str = f"{m.period_days:.4f}" if m.period_days is not None else "—"
        d_str = f"{m.depth_ppm:.0f}" if m.depth_ppm is not None else "—"
        dur_str = f"{m.duration_hours:.2f}" if m.duration_hours is not None else "—"
        n_str = str(m.n_transits) if m.n_transits is not None else "—"
        snr_str = f"{m.snr:.1f}" if m.snr is not None else "—"
        lines.append(f"| {m.mission} | {p_str} | {d_str} | {dur_str} | {n_str} | {snr_str} |")
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Compare transit across missions.")
    parser.add_argument("input", help="JSON with list of mission measurements.")
    parser.add_argument("--tic-id", type=int, default=None)
    parser.add_argument("--period-rtol", type=float, default=0.01)
    parser.add_argument("--depth-rtol", type=float, default=0.20)
    args = parser.parse_args(argv)

    from pathlib import Path
    raw = json.loads(Path(args.input).read_text())
    meas = [MissionMeasurement(
        mission=m.get("mission", ""),
        period_days=m.get("period_days"),
        depth_ppm=m.get("depth_ppm"),
        duration_hours=m.get("duration_hours"),
        n_transits=m.get("n_transits"),
        snr=m.get("snr"),
        source=m.get("source", ""),
    ) for m in raw]
    result = compare_multi_mission(args.tic_id, meas,
                                   period_rtol=args.period_rtol,
                                   depth_rtol=args.depth_rtol)
    print(format_mission_comparison(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
