"""Derive geometric albedo from reflected-light phase curve amplitude."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeometricAlbedoResult:
    reflected_amplitude_ppm: float
    geometric_albedo: float
    phase_integral: float
    spherical_albedo: float
    flag: str


def compute_geometric_albedo_from_phase(
    reflected_amplitude_ppm: float,
    orbital_distance_au: float,
    planet_radius_rjup: float,
    phase_integral: float = 1.5,
) -> GeometricAlbedoResult:
    """Derive geometric albedo from reflected phase curve amplitude.

    Reflected amplitude: A = Ag × (Rp/a)²
    → Ag = A × (a/Rp)²

    Spherical (Bond) albedo: AB = Ag × q  where q = phase integral.
    For Lambertian sphere: q = 1.5, Ag = 2/3.

    Args:
        reflected_amplitude_ppm: peak reflected phase curve amplitude (ppm)
        orbital_distance_au: orbital semi-major axis (AU)
        planet_radius_rjup: planet radius (Jupiter radii)
        phase_integral: phase integral q (1.5 for Lambertian, typically 1.0–2.0)
    """
    _RJUP_M = 7.1492e7
    _AU_M = 1.495978707e11

    if reflected_amplitude_ppm <= 0.0:
        return GeometricAlbedoResult(reflected_amplitude_ppm, float("nan"),
                                      phase_integral, float("nan"), "INVALID_AMPLITUDE")
    if orbital_distance_au <= 0.0:
        return GeometricAlbedoResult(reflected_amplitude_ppm, float("nan"),
                                      phase_integral, float("nan"), "INVALID_DISTANCE")
    if planet_radius_rjup <= 0.0:
        return GeometricAlbedoResult(reflected_amplitude_ppm, float("nan"),
                                      phase_integral, float("nan"), "INVALID_RADIUS")

    a_m = orbital_distance_au * _AU_M
    rp_m = planet_radius_rjup * _RJUP_M
    a_ppm = reflected_amplitude_ppm * 1e-6

    ag = a_ppm * (a_m / rp_m) ** 2
    ag = min(ag, 1.0)
    ab = ag * phase_integral / (3.0 / 2.0)
    ab = min(ab, 1.0)

    return GeometricAlbedoResult(
        reflected_amplitude_ppm=reflected_amplitude_ppm,
        geometric_albedo=ag,
        phase_integral=phase_integral,
        spherical_albedo=ab,
        flag="OK",
    )


def format_geometric_albedo_result(r: GeometricAlbedoResult) -> str:
    if r.flag != "OK":
        return f"GeometricAlbedo | flag={r.flag}"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Reflected amplitude | {r.reflected_amplitude_ppm:.2f} ppm |\n"
        f"| Geometric albedo Ag | {r.geometric_albedo:.4f} |\n"
        f"| Phase integral q | {r.phase_integral:.3f} |\n"
        f"| Spherical albedo AB | {r.spherical_albedo:.4f} |\n"
        f"| Flag | {r.flag} |"
    )


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Geometric albedo from phase curve")
    p.add_argument("amplitude_ppm", type=float)
    p.add_argument("distance_au", type=float)
    p.add_argument("radius_rjup", type=float)
    p.add_argument("--q", type=float, default=1.5, help="Phase integral")
    args = p.parse_args()
    r = compute_geometric_albedo_from_phase(args.amplitude_ppm, args.distance_au,
                                             args.radius_rjup, phase_integral=args.q)
    print(format_geometric_albedo_result(r))


if __name__ == "__main__":
    _cli()
