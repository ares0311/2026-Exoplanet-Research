"""Estimate orbital eccentricity from secondary eclipse timing and duration."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EccentricityFromEclipseResult:
    eclipse_offset_fraction: float
    eclipse_duration_ratio: float
    e_cos_omega: float
    e_sin_omega: float
    eccentricity_lower_bound: float
    omega_deg: float
    flag: str


def compute_eccentricity_from_eclipse(
    secondary_phase: float,
    transit_duration_hours: float,
    eclipse_duration_hours: float,
) -> EccentricityFromEclipseResult:
    """Estimate e*cos(ω) from secondary eclipse phase offset and e*sin(ω) from duration ratio.

    Uses the Charbonneau (2005) / Winn (2010) relations:
      e*cos(ω) ≈ π/2 * (t_sec/P - 0.5)
      e*sin(ω) ≈ (D_occ/D_tra - 1) / (D_occ/D_tra + 1)

    Args:
        secondary_phase: orbital phase of secondary eclipse centre (0–1); ~0.5 for circular orbit
        transit_duration_hours: duration of primary transit (hours)
        eclipse_duration_hours: duration of secondary eclipse (hours)
    """
    if not (0.0 < secondary_phase < 1.0):
        return EccentricityFromEclipseResult(
            eclipse_offset_fraction=secondary_phase,
            eclipse_duration_ratio=float("nan"),
            e_cos_omega=float("nan"),
            e_sin_omega=float("nan"),
            eccentricity_lower_bound=float("nan"),
            omega_deg=float("nan"),
            flag="INVALID_PHASE",
        )
    if transit_duration_hours <= 0.0:
        return EccentricityFromEclipseResult(
            eclipse_offset_fraction=secondary_phase,
            eclipse_duration_ratio=float("nan"),
            e_cos_omega=float("nan"),
            e_sin_omega=float("nan"),
            eccentricity_lower_bound=float("nan"),
            omega_deg=float("nan"),
            flag="INVALID_TRANSIT_DURATION",
        )
    if eclipse_duration_hours <= 0.0:
        return EccentricityFromEclipseResult(
            eclipse_offset_fraction=secondary_phase,
            eclipse_duration_ratio=float("nan"),
            e_cos_omega=float("nan"),
            e_sin_omega=float("nan"),
            eccentricity_lower_bound=float("nan"),
            omega_deg=float("nan"),
            flag="INVALID_ECLIPSE_DURATION",
        )

    offset = secondary_phase - 0.5
    e_cos_omega = math.pi / 2.0 * offset

    dur_ratio = eclipse_duration_hours / transit_duration_hours
    e_sin_omega = (dur_ratio - 1.0) / (dur_ratio + 1.0)

    e_lower = math.sqrt(e_cos_omega**2 + e_sin_omega**2)

    if abs(e_cos_omega) < 1e-12 and abs(e_sin_omega) < 1e-12:
        omega_deg = 0.0
    else:
        omega_deg = math.degrees(math.atan2(e_sin_omega, e_cos_omega)) % 360.0

    return EccentricityFromEclipseResult(
        eclipse_offset_fraction=offset,
        eclipse_duration_ratio=dur_ratio,
        e_cos_omega=e_cos_omega,
        e_sin_omega=e_sin_omega,
        eccentricity_lower_bound=min(e_lower, 0.9999),
        omega_deg=omega_deg,
        flag="OK",
    )


def format_eccentricity_result(r: EccentricityFromEclipseResult) -> str:
    if r.flag != "OK":
        return f"EccentricityFromEclipse | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Eclipse offset | {r.eclipse_offset_fraction:+.4f} (fraction of period) |\n"
        f"| Duration ratio (occ/tra) | {r.eclipse_duration_ratio:.4f} |\n"
        f"| e·cos(ω) | {r.e_cos_omega:+.4f} |\n"
        f"| e·sin(ω) | {r.e_sin_omega:+.4f} |\n"
        f"| e lower bound | {r.eccentricity_lower_bound:.4f} |\n"
        f"| ω (deg) | {r.omega_deg:.1f} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Eccentricity from secondary eclipse")
    p.add_argument("secondary_phase", type=float, help="Secondary eclipse phase (0-1)")
    p.add_argument("transit_duration_hours", type=float, help="Primary transit duration (hours)")
    p.add_argument("eclipse_duration_hours", type=float, help="Secondary eclipse duration (hours)")
    args = p.parse_args()
    r = compute_eccentricity_from_eclipse(
        args.secondary_phase, args.transit_duration_hours, args.eclipse_duration_hours
    )
    print(format_eccentricity_result(r))


if __name__ == "__main__":
    _cli()
