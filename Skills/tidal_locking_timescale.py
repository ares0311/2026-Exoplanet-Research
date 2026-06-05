"""Estimate tidal synchronization (spin-locking) timescale for a planet."""
from __future__ import annotations

import math
from dataclasses import dataclass

_G = 6.674e-11
_MSUN_KG = 1.989e30
_MJUP_KG = 1.898e27
_RJUP_M = 7.1492e7
_AU_M = 1.495978707e11
_SEC_PER_GYR = 3.1557e16


@dataclass(frozen=True)
class TidalLockingResult:
    sync_timescale_gyr: float       # spin synchronization timescale
    is_likely_locked: bool          # True if timescale < stellar age
    lock_ratio: float               # stellar_age / sync_timescale
    dominant_channel: str           # PLANETARY / STELLAR
    flag: str


def compute_tidal_locking_timescale(
    period_days: float,
    planet_mass_mjup: float = 1.0,
    stellar_mass_msun: float = 1.0,
    planet_radius_rjup: float = 1.0,
    stellar_radius_rsun: float = 1.0,
    initial_rotation_period_days: float = 0.5,
    tidal_quality_factor_planet: float = 1e5,
    stellar_age_gyr: float = 5.0,
) -> TidalLockingResult:
    """Estimate tidal spin-synchronization timescale.

    Goldreich & Soter (1966) / Peale (1977):
      τ_sync ≈ (4/9) * (Qp/k2p) * (Mp/Ms) * (a/Rp)^3 * (a^3/G*Ms)^(1/2)
    Simplified (Gladman et al. 1996):
      τ_sync = (ω_0 * a^6 * Mp * Qp) / (3 * G * Ms^2 * Rp^5)
    where ω_0 = 2π / P_rot_initial.

    Args:
        period_days: orbital period (days)
        planet_mass_mjup: planet mass (Jupiter masses)
        stellar_mass_msun: stellar mass (solar masses)
        planet_radius_rjup: planet radius (Jupiter radii)
        stellar_radius_rsun: stellar radius (solar radii)
        initial_rotation_period_days: planet initial rotation period (days)
        tidal_quality_factor_planet: planetary tidal Q
        stellar_age_gyr: system age (Gyr) for lock assessment
    """
    _RSUN_M = 6.957e8
    _K2_PLANET = 0.3   # Love number

    if period_days <= 0.0:
        return TidalLockingResult(float("nan"), False, float("nan"), "UNKNOWN", "INVALID_PERIOD")
    if planet_mass_mjup <= 0.0:
        return TidalLockingResult(float("nan"), False, float("nan"), "UNKNOWN", "INVALID_MASS")
    if planet_radius_rjup <= 0.0:
        return TidalLockingResult(float("nan"), False, float("nan"), "UNKNOWN", "INVALID_RADIUS")

    mp_kg = planet_mass_mjup * _MJUP_KG
    ms_kg = stellar_mass_msun * _MSUN_KG
    rp_m = planet_radius_rjup * _RJUP_M
    rs_m = stellar_radius_rsun * _RSUN_M
    p_s = period_days * 86400.0
    a_m = (_G * ms_kg * p_s**2 / (4.0 * math.pi**2)) ** (1.0 / 3.0)

    omega_0 = 2.0 * math.pi / (initial_rotation_period_days * 86400.0)

    # Planetary tide synchronization: τ = α × Mp × ω_0 × Qp × a^6 / (3 × k2 × G × Ms^2 × Rp^3)
    # α/k2 ≈ 0.3/0.3 = 1 for Earth-like; includes moment of inertia (Gladman+1996 corrected)
    tau_planet_s = (omega_0 * a_m**6 * mp_kg * tidal_quality_factor_planet /
                    (3.0 * _K2_PLANET * _G * ms_kg**2 * rp_m**3))

    # Stellar tide (star spinning down from planet torque); α_star/k2_star ≈ 0.1/0.1 = 1
    k2_star = 0.1
    q_star = 1e6
    tau_star_s = (omega_0 * a_m**6 * ms_kg * q_star /
                  (3.0 * k2_star * _G * mp_kg**2 * rs_m**3))

    if tau_planet_s <= tau_star_s:
        tau_s = tau_planet_s
        dominant = "PLANETARY"
    else:
        tau_s = tau_star_s
        dominant = "STELLAR"

    tau_gyr = tau_s / _SEC_PER_GYR
    locked = tau_gyr < stellar_age_gyr
    ratio = stellar_age_gyr / tau_gyr if tau_gyr > 0 else float("inf")

    return TidalLockingResult(
        sync_timescale_gyr=tau_gyr,
        is_likely_locked=locked,
        lock_ratio=ratio,
        dominant_channel=dominant,
        flag="OK",
    )


def format_tidal_locking_result(r: TidalLockingResult) -> str:
    if r.flag != "OK":
        return f"TidalLocking | flag={r.flag}"
    lock_str = "YES" if r.is_likely_locked else "NO"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Sync timescale | {r.sync_timescale_gyr:.3e} Gyr |\n"
        f"| Likely locked | {lock_str} |\n"
        f"| Age / τ_sync | {r.lock_ratio:.2f} |\n"
        f"| Dominant channel | {r.dominant_channel} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Tidal spin-locking timescale estimator")
    p.add_argument("period_days", type=float)
    p.add_argument("--mp", type=float, default=1.0, help="Planet mass (MJup)")
    p.add_argument("--rp", type=float, default=1.0, help="Planet radius (RJup)")
    p.add_argument("--age", type=float, default=5.0, help="Stellar age (Gyr)")
    args = p.parse_args()
    r = compute_tidal_locking_timescale(args.period_days, planet_mass_mjup=args.mp,
                                         planet_radius_rjup=args.rp, stellar_age_gyr=args.age)
    print(format_tidal_locking_result(r))


if __name__ == "__main__":
    _cli()
