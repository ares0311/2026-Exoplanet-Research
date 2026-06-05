"""Estimate tidal orbital decay timescale for hot Jupiters."""
from __future__ import annotations

import math
from dataclasses import dataclass

_G = 6.674e-11
_MSUN_KG = 1.989e30
_MJUP_KG = 1.898e27
_RSUN_M = 6.957e8
_SEC_PER_GYR = 3.156e16
_SEC_PER_DAY = 86400.0


@dataclass(frozen=True)
class OrbitalDecayResult:
    period_days: float
    semi_major_axis_au: float
    dp_dt_s_per_s: float
    decay_timescale_gyr: float
    decay_class: str   # STABLE / MARGINAL / DECAYING / RAPID_DECAY
    flag: str


def compute_orbital_decay_timescale(
    period_days: float,
    planet_mass_mjup: float,
    stellar_mass_msun: float = 1.0,
    stellar_radius_rsun: float = 1.0,
    tidal_quality_factor: float = 1e6,
) -> OrbitalDecayResult:
    """Estimate tidal orbital decay timescale using Rasio & Ford (1996).

    dP/dt = -(27π / Q'_★) × (Mp/Ms) × (Rs/a)^5 × P

    τ = P / |dP/dt| = (Q'_★ / 27π) × (Ms/Mp) × (a/Rs)^5

    Args:
        period_days: orbital period (days)
        planet_mass_mjup: planet mass (Jupiter masses)
        stellar_mass_msun: stellar mass (solar masses)
        stellar_radius_rsun: stellar radius (solar radii)
        tidal_quality_factor: modified stellar tidal quality factor Q'_★ (typ. 10^5–10^8)
    """
    if period_days <= 0.0:
        return OrbitalDecayResult(period_days, float("nan"), float("nan"),
                                   float("nan"), "UNKNOWN", "INVALID_PERIOD")
    if planet_mass_mjup <= 0.0:
        return OrbitalDecayResult(period_days, float("nan"), float("nan"),
                                   float("nan"), "UNKNOWN", "INVALID_PLANET_MASS")
    if stellar_mass_msun <= 0.0:
        return OrbitalDecayResult(period_days, float("nan"), float("nan"),
                                   float("nan"), "UNKNOWN", "INVALID_STELLAR_MASS")
    if stellar_radius_rsun <= 0.0:
        return OrbitalDecayResult(period_days, float("nan"), float("nan"),
                                   float("nan"), "UNKNOWN", "INVALID_STELLAR_RADIUS")
    if tidal_quality_factor <= 0.0:
        return OrbitalDecayResult(period_days, float("nan"), float("nan"),
                                   float("nan"), "UNKNOWN", "INVALID_Q_FACTOR")

    p_s = period_days * _SEC_PER_DAY
    ms_kg = stellar_mass_msun * _MSUN_KG
    mp_kg = planet_mass_mjup * _MJUP_KG
    rs_m = stellar_radius_rsun * _RSUN_M

    a_m = ((_G * ms_kg * p_s**2) / (4.0 * math.pi**2)) ** (1.0 / 3.0)
    a_au = a_m / 1.495978707e11

    mass_ratio = mp_kg / ms_kg
    radius_ratio = rs_m / a_m

    dp_dt = -(27.0 * math.pi / tidal_quality_factor) * mass_ratio * radius_ratio**5
    tau_s = abs(p_s / dp_dt) if dp_dt != 0.0 else float("inf")
    tau_gyr = tau_s / _SEC_PER_GYR

    if tau_gyr > 10.0:
        decay_class = "STABLE"
    elif tau_gyr > 1.0:
        decay_class = "MARGINAL"
    elif tau_gyr > 0.1:
        decay_class = "DECAYING"
    else:
        decay_class = "RAPID_DECAY"

    return OrbitalDecayResult(
        period_days=period_days,
        semi_major_axis_au=a_au,
        dp_dt_s_per_s=dp_dt,
        decay_timescale_gyr=tau_gyr,
        decay_class=decay_class,
        flag="OK",
    )


def format_orbital_decay_result(r: OrbitalDecayResult) -> str:
    if r.flag != "OK":
        return f"OrbitalDecay | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Period | {r.period_days:.4f} days |\n"
        f"| Semi-major axis | {r.semi_major_axis_au:.4f} AU |\n"
        f"| dP/dt | {r.dp_dt_s_per_s:.3e} s/s |\n"
        f"| Decay timescale | {r.decay_timescale_gyr:.3f} Gyr |\n"
        f"| Decay class | {r.decay_class} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Tidal orbital decay timescale")
    p.add_argument("period_days", type=float)
    p.add_argument("planet_mass_mjup", type=float)
    p.add_argument("--mstar", type=float, default=1.0)
    p.add_argument("--rstar", type=float, default=1.0)
    p.add_argument("--qstar", type=float, default=1e6)
    args = p.parse_args()
    r = compute_orbital_decay_timescale(
        args.period_days, args.planet_mass_mjup,
        stellar_mass_msun=args.mstar, stellar_radius_rsun=args.rstar,
        tidal_quality_factor=args.qstar,
    )
    print(format_orbital_decay_result(r))


if __name__ == "__main__":
    _cli()
