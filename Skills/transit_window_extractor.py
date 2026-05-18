"""Extract in-transit and out-of-transit windows from a light curve.

Given a known ephemeris, returns masked arrays for in-transit, out-of-transit,
and per-transit windows — useful for further vetting or modelling.

Public API
----------
TransitWindow(transit_number, t_mid, time, flux, flux_err)
TransitWindowResult(windows, n_windows, time_oot, flux_oot, flux_err_oot, flag)
extract_transit_windows(time, flux, period, epoch, *, flux_err,
                        duration_days, padding_factor) -> TransitWindowResult
format_window_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TransitWindow:
    transit_number: int
    t_mid: float
    time: tuple[float, ...]
    flux: tuple[float, ...]
    flux_err: tuple[float, ...] | None


@dataclass(frozen=True)
class TransitWindowResult:
    windows: tuple[TransitWindow, ...]
    n_windows: int
    time_oot: tuple[float, ...]
    flux_oot: tuple[float, ...]
    flux_err_oot: tuple[float, ...] | None
    flag: str                       # "OK", "PARTIAL", "NO_TRANSITS"


def extract_transit_windows(
    time: list[float],
    flux: list[float],
    period: float,
    epoch: float,
    *,
    flux_err: list[float] | None = None,
    duration_days: float = 0.1,
    padding_factor: float = 1.5,
    min_points_per_transit: int = 2,
) -> TransitWindowResult:
    """Extract per-transit and out-of-transit windows.

    Args:
        time: BJD time array.
        flux: Normalised flux array.
        period: Orbital period in days.
        epoch: Transit epoch (BJD).
        flux_err: Per-cadence uncertainties.
        duration_days: Transit duration in days.
        padding_factor: In-transit window = ±(duration/2 × padding_factor).
        min_points_per_transit: Minimum cadences to include a transit.

    Returns:
        :class:`TransitWindowResult`.
    """
    if not time or period <= 0:
        return TransitWindowResult((), 0, (), (), None, "NO_TRANSITS")

    t_arr = [float(t) for t in time]
    f_arr = [float(f) for f in flux]
    e_arr = ([float(e) for e in flux_err] if flux_err is not None else None)

    half_win = duration_days / 2.0 * padding_factor
    oot_half = duration_days * 2.0       # OOT zone: outside 2× duration

    t_min, t_max = t_arr[0], t_arr[-1]
    n_start = math.ceil((t_min - epoch) / period)
    n_end = math.floor((t_max - epoch) / period)

    windows: list[TransitWindow] = []
    in_transit_mask: list[bool] = [False] * len(t_arr)

    for n in range(int(n_start), int(n_end) + 1):
        t_mid = epoch + n * period
        if t_mid < t_min - half_win or t_mid > t_max + half_win:
            continue

        idx = [i for i, t in enumerate(t_arr) if abs(t - t_mid) <= half_win]
        if len(idx) < min_points_per_transit:
            continue

        for i in idx:
            in_transit_mask[i] = True

        t_win = tuple(t_arr[i] for i in idx)
        f_win = tuple(f_arr[i] for i in idx)
        e_win = tuple(e_arr[i] for i in idx) if e_arr is not None else None

        windows.append(TransitWindow(
            transit_number=int(n),
            t_mid=t_mid,
            time=t_win,
            flux=f_win,
            flux_err=e_win,
        ))

    # OOT: points more than oot_half days from any transit
    oot_idx = [
        i for i, t in enumerate(t_arr)
        if not in_transit_mask[i] and
        all(abs(t - (epoch + n * period)) > oot_half for n in range(int(n_start), int(n_end) + 1))
    ]
    t_oot = tuple(t_arr[i] for i in oot_idx)
    f_oot = tuple(f_arr[i] for i in oot_idx)
    e_oot = tuple(e_arr[i] for i in oot_idx) if e_arr is not None else None

    if not windows:
        flag = "NO_TRANSITS"
    elif any(len(w.time) < min_points_per_transit for w in windows):
        flag = "PARTIAL"
    else:
        flag = "OK"

    return TransitWindowResult(
        windows=tuple(windows),
        n_windows=len(windows),
        time_oot=t_oot,
        flux_oot=f_oot,
        flux_err_oot=e_oot,
        flag=flag,
    )


def format_window_result(result: TransitWindowResult) -> str:
    """Format transit window extraction result as Markdown."""
    lines = [
        "## Transit Window Extraction",
        "",
        f"- Windows extracted: {result.n_windows}",
        f"- OOT cadences: {len(result.time_oot)}",
        f"- Flag: **{result.flag}**",
    ]
    if result.windows:
        lines += [
            "",
            "| Transit # | T_mid (BJD) | N cadences |",
            "|---|---|---|",
        ]
        for w in result.windows:
            lines.append(f"| {w.transit_number} | {w.t_mid:.5f} | {len(w.time)} |")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="transit_window_extractor",
        description="Extract per-transit windows from a light curve.",
    )
    parser.add_argument("--lc", required=True, metavar="JSON")
    parser.add_argument("--period", type=float, required=True)
    parser.add_argument("--epoch", type=float, required=True)
    parser.add_argument("--duration", type=float, default=0.1)
    args = parser.parse_args(argv)

    lc = json.loads(Path(args.lc).read_text())
    result = extract_transit_windows(
        lc["time"], lc["flux"], args.period, args.epoch,
        duration_days=args.duration,
    )
    print(format_window_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
