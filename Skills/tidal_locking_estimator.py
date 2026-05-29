"""Estimate tidal locking timescale for a planet.

Simplified formula:
    t_lock_yr = 6e10 * (a_AU^6 * M_p_mearth * Q_factor) / (R_p_rearth^3 * M_star_msun^2)

A planet is considered likely tidally locked if t_lock_yr < 4.5e9 (age of solar system).

Public API
----------
TidalLockResult(t_lock_yr, is_likely_locked, flag)
estimate_tidal_locking(a_au, m_planet_mearth, r_planet_rearth, m_star_msun, q_factor)
    -> TidalLockResult
format_tidal_lock_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass

_SOLAR_SYSTEM_AGE_YR = 4.5e9
_DEFAULT_Q = 100.0  # typical rocky-planet tidal Q factor


@dataclass(frozen=True)
class TidalLockResult:
    t_lock_yr: float
    is_likely_locked: bool
    flag: str = "OK"


def estimate_tidal_locking(
    a_au: float,
    m_planet_mearth: float,
    r_planet_rearth: float,
    m_star_msun: float = 1.0,
    q_factor: float = _DEFAULT_Q,
) -> TidalLockResult:
    """Estimate tidal locking timescale.

    Args:
        a_au: Semi-major axis in AU.
        m_planet_mearth: Planet mass in Earth masses.
        r_planet_rearth: Planet radius in Earth radii.
        m_star_msun: Host star mass in solar masses.
        q_factor: Tidal quality factor (default 100).

    Returns:
        :class:`TidalLockResult`.
    """
    if any(v <= 0 for v in (a_au, m_planet_mearth, r_planet_rearth, m_star_msun, q_factor)):
        return TidalLockResult(t_lock_yr=0.0, is_likely_locked=True, flag="ERROR")

    a = float(a_au)
    mp = float(m_planet_mearth)
    rp = float(r_planet_rearth)
    ms = float(m_star_msun)
    q = float(q_factor)

    t_lock = 6e10 * (a**6 * mp * q) / (rp**3 * ms**2)
    is_locked = t_lock < _SOLAR_SYSTEM_AGE_YR
    return TidalLockResult(
        t_lock_yr=t_lock,
        is_likely_locked=is_locked,
        flag="OK",
    )


def format_tidal_lock_result(result: TidalLockResult) -> str:
    """Format tidal locking result as Markdown."""
    lock_str = "Yes (likely tidally locked)" if result.is_likely_locked else "No"
    t_str = f"{result.t_lock_yr:.3e}"
    lines = [
        "## Tidal Locking Estimate",
        "",
        f"- Locking timescale: **{t_str} yr**",
        f"- Likely locked: **{lock_str}**",
        f"- Flag: {result.flag}",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="tidal_locking_estimator",
        description="Estimate tidal locking timescale for a planet.",
    )
    parser.add_argument("a_au", type=float, help="Semi-major axis in AU")
    parser.add_argument("m_planet_mearth", type=float, help="Planet mass in Earth masses")
    parser.add_argument("r_planet_rearth", type=float, help="Planet radius in Earth radii")
    parser.add_argument("--m-star", type=float, default=1.0, help="Star mass in M_sun")
    parser.add_argument("--q-factor", type=float, default=_DEFAULT_Q, help="Tidal Q factor")
    args = parser.parse_args(argv)

    result = estimate_tidal_locking(
        args.a_au,
        args.m_planet_mearth,
        args.r_planet_rearth,
        m_star_msun=args.m_star,
        q_factor=args.q_factor,
    )
    print(format_tidal_lock_result(result))
    return 0 if result.flag == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
