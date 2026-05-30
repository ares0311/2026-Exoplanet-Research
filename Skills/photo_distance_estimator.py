"""Photometric distance from apparent magnitude and absolute magnitude.

d_pc = 10^((m - M_abs - A_v + 5) / 5)
sigma_d = d * ln(10)/5 * delta_m

Public API
----------
PhotoDistanceResult(distance_pc, distance_uncertainty_pc, m_apparent, M_absolute, flag)
estimate_photo_distance(m, M_abs, *, A_v, delta_m) -> PhotoDistanceResult
format_photo_distance(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass

_LN10_OVER5 = math.log(10.0) / 5.0


@dataclass(frozen=True)
class PhotoDistanceResult:
    distance_pc: float
    distance_uncertainty_pc: float
    m_apparent: float
    M_absolute: float
    flag: str  # "OK" or "UNCERTAIN" if uncertainty > 30%


def estimate_photo_distance(
    m: float,
    M_abs: float,
    *,
    A_v: float = 0.0,
    delta_m: float = 0.05,
) -> PhotoDistanceResult:
    """Estimate photometric distance from apparent and absolute magnitudes.

    Args:
        m: Apparent magnitude.
        M_abs: Absolute magnitude.
        A_v: Visual extinction in magnitudes (default 0.0).
        delta_m: Magnitude uncertainty for error propagation (default 0.05).

    Returns:
        :class:`PhotoDistanceResult`.
    """
    exponent = (m - M_abs - A_v + 5.0) / 5.0
    d_pc = 10.0**exponent
    sigma_d = d_pc * _LN10_OVER5 * abs(delta_m)
    frac = sigma_d / d_pc if d_pc > 0 else 1.0
    flag = "UNCERTAIN" if frac > 0.30 else "OK"
    return PhotoDistanceResult(
        distance_pc=round(d_pc, 4),
        distance_uncertainty_pc=round(sigma_d, 4),
        m_apparent=m,
        M_absolute=M_abs,
        flag=flag,
    )


def format_photo_distance(result: PhotoDistanceResult) -> str:
    """Format photometric distance result as Markdown."""
    lines = [
        "## Photometric Distance",
        "",
        f"- Apparent magnitude: {result.m_apparent:.3f}",
        f"- Absolute magnitude: {result.M_absolute:.3f}",
        f"- Distance: **{result.distance_pc:.1f} pc**",
        f"- Uncertainty: ±{result.distance_uncertainty_pc:.1f} pc",
        f"- Flag: {result.flag}",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="photo_distance_estimator",
        description=__doc__,
    )
    p.add_argument("m", type=float, help="Apparent magnitude")
    p.add_argument("M_abs", type=float, help="Absolute magnitude")
    p.add_argument("--A-v", type=float, default=0.0, help="Visual extinction (mag)")
    p.add_argument("--delta-m", type=float, default=0.05, help="Magnitude uncertainty")
    args = p.parse_args(argv)
    r = estimate_photo_distance(args.m, args.M_abs, A_v=args.A_v, delta_m=args.delta_m)
    print(format_photo_distance(r))
    return 0 if r.flag == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
