"""Predict future transit windows and their observability from a ground site.

For a given ephemeris (period, epoch) and observer location, computes the
next N transit mid-times and evaluates observability by calling optional
injectable functions for airmass and moon separation.  When the injectables
are not provided the function still returns mid-time predictions without
observability scores.

Public API
----------
TransitWindow(transit_number, bjd_mid, utc_mid, duration_hours,
              airmass_mid, moon_separation_deg, moon_illumination,
              is_observable, observability_score, notes)
TransmissionWindowResult(tic_id, period_days, epoch_bjd, n_windows,
                         windows, n_observable, flag)
predict_transit_windows(period_days, epoch_bjd, bjd_start, bjd_end, *,
                        tic_id, duration_hours, obs_lat_deg,
                        obs_lon_deg, horizon_limit_deg,
                        min_moon_sep_deg, airmass_fn,
                        moon_fn) -> TransmissionWindowResult
format_transmission_window_result(result) -> str
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class TransitWindow:
    transit_number: int
    bjd_mid: float
    utc_mid: str             # approximate UTC string from BJD
    duration_hours: float
    airmass_mid: float | None
    moon_separation_deg: float | None
    moon_illumination: float | None
    is_observable: bool
    observability_score: float  # [0, 1]
    notes: str


@dataclass(frozen=True)
class TransmissionWindowResult:
    tic_id: int
    period_days: float
    epoch_bjd: float
    n_windows: int
    windows: tuple[TransitWindow, ...]
    n_observable: int
    flag: str  # "OK" | "NO_WINDOWS" | "INVALID"


_BJD_J2000 = 2451545.0  # BJD of J2000.0


def _bjd_to_utc_approx(bjd: float) -> str:
    """Very rough BJD → UTC string (ignores leap seconds and TDB correction)."""
    jd = bjd  # treat as JD for display purposes
    # Days since J2000.0
    d = jd - _BJD_J2000
    # Julian date 2451545.0 = 2000-01-01 12:00 UTC
    days_from_epoch = d
    year = 2000
    day_count = int(days_from_epoch)

    # Rough calendar conversion (not accounting for leap year edge cases precisely)
    def _is_leap(y: int) -> bool:
        return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)

    while day_count >= (diy := (366 if _is_leap(year) else 365)):
        day_count -= diy
        year += 1

    months = [31, 28 + (1 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 0),
              31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month = 1
    for m_days in months:
        if day_count < m_days:
            break
        day_count -= m_days
        month += 1

    day = day_count + 1
    frac = days_from_epoch - int(days_from_epoch)
    hour = int(frac * 24)
    minute = int((frac * 24 - hour) * 60)
    return f"{year:04d}-{month:02d}-{day:02d} {hour:02d}:{minute:02d} UTC"


def predict_transit_windows(
    period_days: float,
    epoch_bjd: float,
    bjd_start: float,
    bjd_end: float,
    *,
    tic_id: int = 0,
    duration_hours: float = 2.0,
    obs_lat_deg: float | None = None,
    obs_lon_deg: float | None = None,
    horizon_limit_deg: float = 20.0,
    min_moon_sep_deg: float = 30.0,
    airmass_fn: Callable | None = None,
    moon_fn: Callable | None = None,
) -> TransmissionWindowResult:
    """Predict transit windows in [bjd_start, bjd_end].

    Args:
        period_days: Orbital period in days.
        epoch_bjd: Reference mid-transit epoch (BJD).
        bjd_start: Start of prediction window (BJD).
        bjd_end: End of prediction window (BJD).
        tic_id: TIC identifier (for labelling only).
        duration_hours: Transit duration in hours.
        obs_lat_deg: Observer latitude (degrees).
        obs_lon_deg: Observer longitude (degrees, East positive).
        horizon_limit_deg: Minimum altitude for observability.
        min_moon_sep_deg: Minimum moon separation for observability.
        airmass_fn: Optional callable ``(ra, dec, lat, lon, bjd) -> AirmassResult``
            (signature matches ``airmass_calculator.compute_airmass``).
        moon_fn: Optional callable ``(ra, dec, bjd) -> MoonSeparationResult``
            (signature matches ``moon_separation_checker.check_moon_separation``).

    Returns:
        :class:`TransmissionWindowResult`.
    """
    if period_days <= 0 or bjd_end <= bjd_start:
        return TransmissionWindowResult(tic_id, period_days, epoch_bjd, 0, (), 0, "INVALID")

    # First transit number on or after bjd_start
    n_start = math.ceil((bjd_start - epoch_bjd) / period_days)
    windows: list[TransitWindow] = []

    n = n_start
    while True:
        mid = epoch_bjd + n * period_days
        if mid > bjd_end:
            break

        utc_str = _bjd_to_utc_approx(mid)
        airmass_mid: float | None = None
        moon_sep: float | None = None
        moon_illum: float | None = None
        is_observable = True
        notes_parts: list[str] = []

        if airmass_fn is not None and obs_lat_deg is not None and obs_lon_deg is not None:
            try:
                am_result = airmass_fn(0.0, 0.0, obs_lat_deg, obs_lon_deg, mid)
                airmass_mid = am_result.airmass
                if not am_result.is_observable:
                    is_observable = False
                    notes_parts.append("below horizon")
            except Exception:
                pass

        if moon_fn is not None:
            try:
                moon_result = moon_fn(0.0, 0.0, mid)
                moon_sep = moon_result.moon_separation_deg
                moon_illum = moon_result.moon_illumination_fraction
                if moon_result.is_problematic:
                    is_observable = False
                    notes_parts.append("moon interference")
            except Exception:
                pass

        obs_score = _observability_score(airmass_mid, moon_sep, moon_illum,
                                         horizon_limit_deg, min_moon_sep_deg)

        windows.append(TransitWindow(
            transit_number=n,
            bjd_mid=round(mid, 6),
            utc_mid=utc_str,
            duration_hours=duration_hours,
            airmass_mid=round(airmass_mid, 3) if airmass_mid is not None else None,
            moon_separation_deg=round(moon_sep, 2) if moon_sep is not None else None,
            moon_illumination=round(moon_illum, 3) if moon_illum is not None else None,
            is_observable=is_observable,
            observability_score=round(obs_score, 3),
            notes="; ".join(notes_parts) if notes_parts else "OK",
        ))
        n += 1

    if not windows:
        return TransmissionWindowResult(tic_id, period_days, epoch_bjd, 0, (), 0, "NO_WINDOWS")

    n_observable = sum(1 for w in windows if w.is_observable)
    return TransmissionWindowResult(
        tic_id=tic_id,
        period_days=period_days,
        epoch_bjd=epoch_bjd,
        n_windows=len(windows),
        windows=tuple(windows),
        n_observable=n_observable,
        flag="OK",
    )


def _observability_score(
    airmass: float | None,
    moon_sep: float | None,
    moon_illum: float | None,
    horizon_limit: float,
    min_moon_sep: float,
) -> float:
    """Heuristic observability score [0, 1]."""
    score = 1.0
    if airmass is not None:
        if airmass >= 10.0:
            return 0.0
        # Airmass penalty: 1.0 at X=1, 0 at X=3
        score *= max(0.0, 1.0 - (airmass - 1.0) / 2.0)
    if moon_sep is not None and moon_illum is not None:
        moon_penalty = max(0.0, 1.0 - moon_sep / max(min_moon_sep, 1.0)) * moon_illum
        score *= 1.0 - moon_penalty
    return max(0.0, min(1.0, score))


def format_transmission_window_result(result: TransmissionWindowResult) -> str:
    """Format predicted transit windows as Markdown."""
    lines = [
        "## Transit Window Predictions",
        "",
        f"- TIC ID: {result.tic_id}",
        f"- Period: {result.period_days:.6f} days",
        f"- Epoch: {result.epoch_bjd:.6f} BJD",
        f"- Windows found: {result.n_windows}",
        f"- Observable windows: {result.n_observable}",
        f"- **Flag: {result.flag}**",
        "",
    ]
    if result.windows:
        lines += [
            "| # | BJD Mid | UTC (approx) | Airmass | Moon Sep | Observable |",
            "|---|---|---|---|---|---|",
        ]
        for w in result.windows:
            am = f"{w.airmass_mid:.2f}" if w.airmass_mid is not None else "—"
            ms = f"{w.moon_separation_deg:.1f}°" if w.moon_separation_deg is not None else "—"
            obs = "✓" if w.is_observable else "✗"
            lines.append(
                f"| {w.transit_number} | {w.bjd_mid:.4f} | {w.utc_mid} "
                f"| {am} | {ms} | {obs} |"
            )
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="transmission_window_predictor",
        description="Predict future transit windows and observability.",
    )
    parser.add_argument("period_days", type=float)
    parser.add_argument("epoch_bjd", type=float)
    parser.add_argument("bjd_start", type=float)
    parser.add_argument("bjd_end", type=float)
    parser.add_argument("--tic-id", type=int, default=0)
    parser.add_argument("--duration-hours", type=float, default=2.0)
    parser.add_argument("--obs-lat", type=float, default=None)
    parser.add_argument("--obs-lon", type=float, default=None)
    args = parser.parse_args(argv)

    result = predict_transit_windows(
        args.period_days, args.epoch_bjd, args.bjd_start, args.bjd_end,
        tic_id=args.tic_id,
        duration_hours=args.duration_hours,
        obs_lat_deg=args.obs_lat,
        obs_lon_deg=args.obs_lon,
    )
    print(format_transmission_window_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
