"""Assess composite habitability of a candidate planet.

Scores:
- hz_score  : 1.0 inner/outer HZ, 0.3 adjacent zones, 0.0 otherwise
- size_score: 1.0 if R in [0.7, 1.8], 0.5 if [0.5, 2.5], 0.0 otherwise
- stellar_score: 1.0 K/G/F (3700 < Teff <= 7000 K), 0.7 M (<=3700 K), 0.3 others
- lock_score: 1.0 unlocked, 0.5 uncertain (None), 0.0 locked
- overall = 0.35*hz + 0.30*size + 0.20*stellar + 0.15*lock

Public API
----------
HabitabilityReport(hz_class, size_class, stellar_class, tidal_status,
                   overall_score, flag)
assess_habitability(radius_rearth, insolation_searth, teff_k, t_lock_yr)
    -> HabitabilityReport
format_habitability_report(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass

_SOLAR_SYSTEM_AGE_YR = 4.5e9


@dataclass(frozen=True)
class HabitabilityReport:
    hz_class: str       # "hot_zone" / "inner_hz" / "outer_hz" / "cold_zone"
    size_class: str     # "rocky" / "sub_neptune" / "neptune" / "giant"
    stellar_class: str  # "F" / "G" / "K" / "M" / "other"
    tidal_status: str   # "locked" / "unlocked" / "uncertain"
    overall_score: float
    flag: str  # "PROMISING" / "MARGINAL" / "UNLIKELY"


def _hz_class(s: float) -> str:
    if s > 1.1:
        return "hot_zone"
    if s >= 0.36:
        return "inner_hz"
    if s >= 0.20:
        return "outer_hz"
    return "cold_zone"


def _size_class(r: float) -> str:
    if r < 2.0:
        return "rocky"
    if r < 4.0:
        return "sub_neptune"
    if r < 8.0:
        return "neptune"
    return "giant"


def _stellar_class(teff: float) -> str:
    if teff > 7500:
        return "other"
    if teff > 6000:
        return "F"
    if teff > 5200:
        return "G"
    if teff > 3700:
        return "K"
    return "M"


def assess_habitability(
    radius_rearth: float,
    insolation_searth: float,
    teff_k: float = 5778.0,
    t_lock_yr: float | None = None,
) -> HabitabilityReport:
    """Assess composite habitability of a candidate.

    Args:
        radius_rearth: Planet radius in Earth radii.
        insolation_searth: Insolation in Earth units.
        teff_k: Host star effective temperature in Kelvin.
        t_lock_yr: Tidal locking timescale in years (None = uncertain).

    Returns:
        :class:`HabitabilityReport`.
    """
    hz = _hz_class(insolation_searth)
    size = _size_class(radius_rearth)
    stellar = _stellar_class(teff_k)

    # Hz score
    if hz in ("inner_hz", "outer_hz"):
        hz_score = 1.0
    elif hz in ("hot_zone", "cold_zone"):
        hz_score = 0.3
    else:
        hz_score = 0.0

    # Size score
    r = float(radius_rearth)
    if 0.7 <= r <= 1.8:
        size_score = 1.0
    elif 0.5 <= r <= 2.5:
        size_score = 0.5
    else:
        size_score = 0.0

    # Stellar score
    if stellar in ("F", "G", "K"):
        stellar_score = 1.0
    elif stellar == "M":
        stellar_score = 0.7
    else:
        stellar_score = 0.3

    # Tidal lock score
    if t_lock_yr is None:
        tidal_status = "uncertain"
        lock_score = 0.5
    elif t_lock_yr < _SOLAR_SYSTEM_AGE_YR:
        tidal_status = "locked"
        lock_score = 0.0
    else:
        tidal_status = "unlocked"
        lock_score = 1.0

    overall = 0.35 * hz_score + 0.30 * size_score + 0.20 * stellar_score + 0.15 * lock_score

    if overall > 0.6:
        flag = "PROMISING"
    elif overall >= 0.3:
        flag = "MARGINAL"
    else:
        flag = "UNLIKELY"

    return HabitabilityReport(
        hz_class=hz,
        size_class=size,
        stellar_class=stellar,
        tidal_status=tidal_status,
        overall_score=round(overall, 4),
        flag=flag,
    )


def format_habitability_report(result: HabitabilityReport) -> str:
    """Format habitability report as Markdown."""
    lines = [
        "## Habitability Assessment",
        "",
        f"- HZ class: **{result.hz_class}**",
        f"- Size class: **{result.size_class}**",
        f"- Stellar class: **{result.stellar_class}**",
        f"- Tidal status: {result.tidal_status}",
        f"- Overall score: **{result.overall_score:.3f}**",
        f"- Flag: **{result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="candidate_habitability_reporter",
        description=__doc__,
    )
    p.add_argument("radius_rearth", type=float, help="Planet radius in Earth radii")
    p.add_argument("insolation_searth", type=float, help="Insolation in S_earth")
    p.add_argument("--teff", type=float, default=5778.0, help="Stellar Teff in K")
    p.add_argument("--t-lock-yr", type=float, default=None, help="Tidal lock timescale in yr")
    args = p.parse_args(argv)
    r = assess_habitability(args.radius_rearth, args.insolation_searth, args.teff, args.t_lock_yr)
    print(format_habitability_report(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
