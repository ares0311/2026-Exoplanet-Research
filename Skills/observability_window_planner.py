"""Plan per-night observable windows for a list of targets at a given site."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

_DEG = math.pi / 180.0


@dataclass(frozen=True)
class ObservabilityWindow:
    tic_id: str
    ra_deg: float
    dec_deg: float
    site_lat_deg: float
    max_elevation_deg: float
    hours_above_airmass: float   # hours target is above airmass limit that night
    best_ut_hour: float          # UT hour of transit (or culmination) if given
    observable: bool
    flag: str


def _hour_angle_at_airmass(dec_deg: float, lat_deg: float, airmass_limit: float) -> float:
    """Return the HA (hours) at which target reaches airmass_limit."""
    dec = dec_deg * _DEG
    lat = lat_deg * _DEG
    z_limit = math.acos(1.0 / airmass_limit)  # zenith angle at airmass limit
    cos_h = (math.cos(z_limit) - math.sin(lat) * math.sin(dec)) / (
        math.cos(lat) * math.cos(dec)
    )
    if abs(cos_h) > 1.0:
        return 0.0 if cos_h > 1.0 else 12.0
    ha_rad = math.acos(cos_h)
    return ha_rad / (_DEG * 15.0)  # convert rad to hours


def plan_observability(
    ra_deg: float,
    dec_deg: float,
    site_lat_deg: float,
    tic_id: str = "unknown",
    airmass_limit: float = 2.0,
    transit_ut_hour: float | None = None,
) -> ObservabilityWindow:
    """
    Compute observable window for a target at a given site.

    ra_deg / dec_deg: target coordinates.
    site_lat_deg: observatory geographic latitude.
    airmass_limit: maximum acceptable airmass.
    transit_ut_hour: UT hour of expected transit midpoint (optional).

    Returns hours above airmass limit (approximate, assumes culmination near midnight).
    """
    for name, val in [("ra_deg", ra_deg), ("dec_deg", dec_deg), ("site_lat_deg", site_lat_deg)]:
        if not math.isfinite(val):
            return ObservabilityWindow(
                tic_id=tic_id, ra_deg=ra_deg, dec_deg=dec_deg,
                site_lat_deg=site_lat_deg, max_elevation_deg=float("nan"),
                hours_above_airmass=0.0, best_ut_hour=float("nan"),
                observable=False, flag=f"INVALID_{name.upper()}",
            )

    # Maximum elevation = 90 - |lat - dec|  (culmination)
    max_alt_deg = 90.0 - abs(site_lat_deg - dec_deg)
    max_alt_deg = max(-90.0, min(90.0, max_alt_deg))

    # Check circumpolar (never sets)
    circumscribes = abs(site_lat_deg) + abs(dec_deg) >= 90.0

    # HA range above airmass limit
    ha_hours = _hour_angle_at_airmass(dec_deg, site_lat_deg, airmass_limit)
    hours_up = min(2.0 * ha_hours, 12.0) if ha_hours > 0 else 0.0
    if circumscribes:
        hours_up = 24.0

    observable = hours_up >= 1.0

    best_ut = transit_ut_hour if transit_ut_hour is not None and math.isfinite(transit_ut_hour) \
        else float("nan")

    return ObservabilityWindow(
        tic_id=tic_id,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        site_lat_deg=site_lat_deg,
        max_elevation_deg=round(max_alt_deg, 2),
        hours_above_airmass=round(hours_up, 2),
        best_ut_hour=round(best_ut, 2) if math.isfinite(best_ut) else float("nan"),
        observable=observable,
        flag="OK",
    )


def plan_observability_windows(
    targets: list[dict],
    site_lat_deg: float,
    airmass_limit: float = 2.0,
) -> list[ObservabilityWindow]:
    """Plan observability windows for multiple targets."""
    results = []
    for t in targets:
        results.append(plan_observability(
            ra_deg=float(t.get("ra_deg", 0.0)),
            dec_deg=float(t.get("dec_deg", 0.0)),
            site_lat_deg=site_lat_deg,
            tic_id=str(t.get("tic_id", "unknown")),
            airmass_limit=airmass_limit,
            transit_ut_hour=t.get("transit_ut_hour"),
        ))
    return results


def format_observability_windows(windows: list[ObservabilityWindow]) -> str:
    if not windows:
        return "No targets.\n"
    lines = [
        "| TIC ID | Dec (°) | Max El (°) | Hours up | Observable |",
        "|---|---|---|---|---|",
    ]
    for w in windows:
        el_str = f"{w.max_elevation_deg:.1f}" if math.isfinite(w.max_elevation_deg) else "N/A"
        lines.append(
            f"| {w.tic_id} | {w.dec_deg:.2f} | {el_str} | "
            f"{w.hours_above_airmass:.1f} | {'YES' if w.observable else 'NO'} |"
        )
    return "\n".join(lines)


def _cli() -> int:
    p = argparse.ArgumentParser(description="Plan observability windows for targets.")
    p.add_argument("targets_json", help="JSON array or @file")
    p.add_argument("site_lat_deg", type=float)
    p.add_argument("--airmass-limit", type=float, default=2.0)
    args = p.parse_args()
    import json
    raw = args.targets_json
    if raw.startswith("@"):
        with open(raw[1:]) as f:
            targets = json.load(f)
    else:
        targets = json.loads(raw)
    windows = plan_observability_windows(targets, args.site_lat_deg, args.airmass_limit)
    print(format_observability_windows(windows))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
