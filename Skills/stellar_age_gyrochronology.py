"""Estimate stellar age from rotation period via gyrochronology.

Uses the Barnes (2007) empirical relation between rotation period, B-V
colour, and stellar age.  Distinct from ``stellar_rotation`` (which detects
the rotation period from a light curve) — this converts an already-measured
period to an age estimate.

Public API
----------
GyrochronologyResult(p_rot_days, b_minus_v, age_myr, age_gyr,
                     sequence, flag)
estimate_stellar_age(p_rot_days, b_minus_v) -> GyrochronologyResult
format_gyro_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Barnes (2007) interface-sequence coefficients
_A = 0.7725
_B = 0.601
_N = 0.5189
_K = 0.0461   # Myr^-1 (age in Myr)

# Convective/interface boundary: P_C(B-V) = a × (B-V - 0.495)^b
# Stars on the convective sequence have P < P_interface

_BV_MIN = 0.495   # B-V colour floor for the relation


@dataclass(frozen=True)
class GyrochronologyResult:
    p_rot_days: float
    b_minus_v: float
    age_myr: float | None
    age_gyr: float | None
    sequence: str | None   # "I" (interface) or "C" (convective)
    flag: str  # "OK" | "INVALID"


def estimate_stellar_age(
    p_rot_days: float,
    b_minus_v: float,
) -> GyrochronologyResult:
    """Estimate stellar age via the Barnes (2007) gyrochronology relation.

    t = (1/k) × [P / (a × (B-V − 0.495)^b)]^(1/n)

    Args:
        p_rot_days: Measured stellar rotation period (days).
        b_minus_v: B-V photometric colour index.

    Returns:
        :class:`GyrochronologyResult`.
    """
    if not (math.isfinite(p_rot_days) and math.isfinite(b_minus_v)):
        return GyrochronologyResult(p_rot_days, b_minus_v, None, None, None, "INVALID")
    if p_rot_days <= 0:
        return GyrochronologyResult(p_rot_days, b_minus_v, None, None, None, "INVALID")
    if b_minus_v <= _BV_MIN:
        return GyrochronologyResult(p_rot_days, b_minus_v, None, None, None, "INVALID")

    color_factor = _A * (b_minus_v - _BV_MIN) ** _B
    if color_factor <= 0:
        return GyrochronologyResult(p_rot_days, b_minus_v, None, None, None, "INVALID")

    p_interface = color_factor  # P at which star transitions I/C sequence
    sequence = "I" if p_rot_days >= p_interface else "C"

    # Age formula (Barnes 2007, Eq. 3): P = a*(B-V-c)^b * t^n → t = (P/color_factor)^(1/n)
    age_myr = (p_rot_days / color_factor) ** (1.0 / _N)
    age_gyr = age_myr / 1000.0

    return GyrochronologyResult(
        p_rot_days=p_rot_days,
        b_minus_v=b_minus_v,
        age_myr=round(age_myr, 2),
        age_gyr=round(age_gyr, 4),
        sequence=sequence,
        flag="OK",
    )


def format_gyro_result(result: GyrochronologyResult) -> str:
    """Format gyrochronology result as Markdown."""
    lines = [
        "## Stellar Age (Gyrochronology — Barnes 2007)",
        "",
        f"- Rotation period: {result.p_rot_days} days",
        f"- B-V colour: {result.b_minus_v}",
        f"- Sequence: {result.sequence}",
        f"- **Age: {result.age_myr} Myr ({result.age_gyr} Gyr)**",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="stellar_age_gyrochronology",
        description="Estimate stellar age from rotation period and B-V colour.",
    )
    parser.add_argument("p_rot_days", type=float)
    parser.add_argument("b_minus_v", type=float)
    args = parser.parse_args(argv)

    result = estimate_stellar_age(args.p_rot_days, args.b_minus_v)
    print(format_gyro_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
