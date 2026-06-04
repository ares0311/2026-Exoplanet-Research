"""Compute eccentric-orbit transit duration correction factor."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EccentricDurationResult:
    period_days: float
    eccentricity: float
    omega_deg: float
    t14_circular_hours: float
    t14_eccentric_hours: float
    duration_factor: float   # t14_ecc / t14_circ
    flag: str


def compute_eccentric_t14(
    period_days: float,
    stellar_radius_rsun: float = 1.0,
    stellar_mass_msun: float = 1.0,
    impact_parameter: float = 0.0,
    planet_radius_rearth: float = 1.0,
    eccentricity: float = 0.0,
    omega_deg: float = 90.0,
) -> EccentricDurationResult:
    """
    Compute T14 with eccentric orbit correction.

    Circular T14 (Seager & Mallen-Ornelas 2003):
    T14 = (P/pi) * arcsin(sqrt((Rp+R*)^2 - (b*R*)^2) / a)

    Eccentric correction factor (Tingley & Sackett 2005):
    f_ecc = sqrt(1 - e^2) / (1 + e * sin(omega))

    T14_ecc = T14_circ * f_ecc
    """
    if not math.isfinite(period_days) or period_days <= 0:
        return EccentricDurationResult(
            period_days=period_days, eccentricity=eccentricity,
            omega_deg=omega_deg, t14_circular_hours=float("nan"),
            t14_eccentric_hours=float("nan"), duration_factor=float("nan"),
            flag="INVALID_PERIOD",
        )
    if not math.isfinite(eccentricity) or eccentricity < 0 or eccentricity >= 1.0:
        return EccentricDurationResult(
            period_days=period_days, eccentricity=eccentricity,
            omega_deg=omega_deg, t14_circular_hours=float("nan"),
            t14_eccentric_hours=float("nan"), duration_factor=float("nan"),
            flag="INVALID_ECCENTRICITY",
        )

    # Semi-major axis via Kepler's 3rd law (AU)
    a_au = (stellar_mass_msun * (period_days / 365.25) ** 2) ** (1.0 / 3.0)
    rsun_au = 0.00465047
    rearth_au = 4.2635e-5

    r_star_au = stellar_radius_rsun * rsun_au
    r_planet_au = planet_radius_rearth * rearth_au
    b_au = impact_parameter * r_star_au

    arg = ((r_star_au + r_planet_au) ** 2 - b_au ** 2) / a_au ** 2
    if arg <= 0:
        return EccentricDurationResult(
            period_days=period_days, eccentricity=eccentricity,
            omega_deg=omega_deg, t14_circular_hours=float("nan"),
            t14_eccentric_hours=float("nan"), duration_factor=float("nan"),
            flag="GRAZING_TRANSIT",
        )

    t14_circ_days = (period_days / math.pi) * math.asin(math.sqrt(arg))
    t14_circ_hours = t14_circ_days * 24.0

    omega_rad = math.radians(omega_deg)
    f_ecc = math.sqrt(1.0 - eccentricity ** 2) / (1.0 + eccentricity * math.sin(omega_rad))
    t14_ecc_hours = t14_circ_hours * f_ecc

    return EccentricDurationResult(
        period_days=period_days,
        eccentricity=eccentricity,
        omega_deg=omega_deg,
        t14_circular_hours=round(t14_circ_hours, 4),
        t14_eccentric_hours=round(t14_ecc_hours, 4),
        duration_factor=round(f_ecc, 6),
        flag="OK",
    )


def format_eccentric_duration_result(r: EccentricDurationResult) -> str:
    def _f(v: float) -> str:
        return f"{v:.4f}" if math.isfinite(v) else "N/A"

    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Period (days) | {_f(r.period_days)} |\n"
        f"| Eccentricity | {_f(r.eccentricity)} |\n"
        f"| omega (deg) | {_f(r.omega_deg)} |\n"
        f"| T14 circular (hours) | {_f(r.t14_circular_hours)} |\n"
        f"| T14 eccentric (hours) | {_f(r.t14_eccentric_hours)} |\n"
        f"| Duration factor | {r.duration_factor:.6f} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Compute eccentric orbit transit duration.")
    p.add_argument("period_days", type=float)
    p.add_argument("--eccentricity", type=float, default=0.0)
    p.add_argument("--omega-deg", type=float, default=90.0)
    p.add_argument("--stellar-radius-rsun", type=float, default=1.0)
    p.add_argument("--stellar-mass-msun", type=float, default=1.0)
    p.add_argument("--impact-parameter", type=float, default=0.0)
    args = p.parse_args()
    r = compute_eccentric_t14(
        args.period_days, args.stellar_radius_rsun, args.stellar_mass_msun,
        args.impact_parameter, eccentricity=args.eccentricity, omega_deg=args.omega_deg,
    )
    print(format_eccentric_duration_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
