"""Compute habitable-zone flux limits using Kopparapu et al. (2013) coefficients.

Returns the insolation flux boundaries for four HZ limits (Recent Venus,
Runaway Greenhouse, Maximum Greenhouse, Early Mars) as a function of stellar
effective temperature.  Also classifies a planet's insolation flux relative
to these boundaries.

Public API
----------
HZBoundaries(teff_k, s_recent_venus, s_runaway_greenhouse,
             s_max_greenhouse, s_early_mars)
HZClassification(insolation_flux, hz_class, is_in_conservative_hz,
                 is_in_optimistic_hz, flag)
compute_hz_boundaries(teff_k) -> HZBoundaries
classify_hz_position(insolation_flux, teff_k) -> HZClassification
format_hz_result(boundaries, classification) -> str
"""
from __future__ import annotations

from dataclasses import dataclass

# Kopparapu+ (2013) polynomial coefficients for S_eff = S_eff_sun + a*T + b*T² + c*T³ + d*T⁴
# T = T_eff - 5780 K;  columns: [S_eff_sun, a, b, c, d]
# Rows: Recent Venus, Runaway Greenhouse, Maximum Greenhouse, Early Mars
_KOP_COEFFS: list[tuple[float, float, float, float, float]] = [
    # Recent Venus
    (1.7763, 1.4335e-4, 3.3954e-9, -7.6364e-12, -1.1950e-15),
    # Runaway Greenhouse
    (1.0385, 1.2456e-4, 4.8633e-9, -1.4051e-11, -3.5968e-15),
    # Maximum Greenhouse
    (0.3507, 5.9578e-5, 1.6707e-9, -3.0058e-12, -5.1925e-16),
    # Early Mars
    (0.3207, 5.4471e-5, 1.5275e-9, -2.1709e-12, -3.8282e-16),
]
_LIMIT_NAMES = ("recent_venus", "runaway_greenhouse", "max_greenhouse", "early_mars")

_HZ_CLASSES = (
    "too_hot",               # S > S_recent_venus
    "inner_edge_optimistic", # S_recent_venus >= S > S_runaway_greenhouse
    "conservative_hz",       # S_runaway_greenhouse >= S >= S_max_greenhouse
    "outer_edge_optimistic", # S_max_greenhouse > S >= S_early_mars
    "too_cold",              # S < S_early_mars
)


def _seff(teff_k: float, coeffs: tuple[float, float, float, float, float]) -> float:
    """Evaluate HZ insolation flux at given Teff."""
    t = teff_k - 5780.0
    s_sun, a, b, c, d = coeffs
    return s_sun + a * t + b * t ** 2 + c * t ** 3 + d * t ** 4


@dataclass(frozen=True)
class HZBoundaries:
    teff_k: float
    s_recent_venus: float       # S_eff relative to Earth (inner optimistic)
    s_runaway_greenhouse: float # S_eff inner conservative
    s_max_greenhouse: float     # S_eff outer conservative
    s_early_mars: float         # S_eff outer optimistic
    flag: str  # "OK" | "INVALID"


@dataclass(frozen=True)
class HZClassification:
    insolation_flux: float      # S_eff relative to Earth
    hz_class: str               # one of _HZ_CLASSES
    is_in_conservative_hz: bool
    is_in_optimistic_hz: bool
    flag: str  # "OK" | "INVALID"


def compute_hz_boundaries(teff_k: float) -> HZBoundaries:
    """Compute HZ flux boundaries for a given stellar effective temperature.

    Valid for 2600 K ≤ T_eff ≤ 7200 K.

    Args:
        teff_k: Stellar effective temperature (Kelvin).

    Returns:
        :class:`HZBoundaries`.
    """
    if teff_k <= 0:
        return HZBoundaries(teff_k, 0.0, 0.0, 0.0, 0.0, "INVALID")

    s_vals = [_seff(teff_k, c) for c in _KOP_COEFFS]
    return HZBoundaries(
        teff_k=teff_k,
        s_recent_venus=round(s_vals[0], 6),
        s_runaway_greenhouse=round(s_vals[1], 6),
        s_max_greenhouse=round(s_vals[2], 6),
        s_early_mars=round(s_vals[3], 6),
        flag="OK",
    )


def classify_hz_position(
    insolation_flux: float,
    teff_k: float,
) -> HZClassification:
    """Classify a planet's insolation flux relative to the HZ boundaries.

    Args:
        insolation_flux: Incident stellar flux relative to Earth (S_eff).
        teff_k: Host star effective temperature (Kelvin).

    Returns:
        :class:`HZClassification`.
    """
    if insolation_flux <= 0 or teff_k <= 0:
        return HZClassification(insolation_flux, "UNKNOWN", False, False, "INVALID")

    bnd = compute_hz_boundaries(teff_k)
    if bnd.flag != "OK":
        return HZClassification(insolation_flux, "UNKNOWN", False, False, "INVALID")

    s = insolation_flux
    if s > bnd.s_recent_venus:
        cls = "too_hot"
    elif s > bnd.s_runaway_greenhouse:
        cls = "inner_edge_optimistic"
    elif s >= bnd.s_max_greenhouse:
        cls = "conservative_hz"
    elif s >= bnd.s_early_mars:
        cls = "outer_edge_optimistic"
    else:
        cls = "too_cold"

    conservative = cls == "conservative_hz"
    optimistic = cls in ("conservative_hz", "inner_edge_optimistic", "outer_edge_optimistic")

    return HZClassification(
        insolation_flux=insolation_flux,
        hz_class=cls,
        is_in_conservative_hz=conservative,
        is_in_optimistic_hz=optimistic,
        flag="OK",
    )


def format_hz_result(
    boundaries: HZBoundaries,
    classification: HZClassification | None = None,
) -> str:
    """Format HZ boundaries and optional classification as Markdown."""
    lines = [
        "## Kopparapu HZ Calculator",
        "",
        f"- T_eff: {boundaries.teff_k} K",
        f"- Recent Venus (inner opt.): {boundaries.s_recent_venus:.4f} S_eff",
        f"- Runaway Greenhouse (inner cons.): {boundaries.s_runaway_greenhouse:.4f} S_eff",
        f"- Max Greenhouse (outer cons.): {boundaries.s_max_greenhouse:.4f} S_eff",
        f"- Early Mars (outer opt.): {boundaries.s_early_mars:.4f} S_eff",
        f"- **Flag: {boundaries.flag}**",
    ]
    if classification is not None and classification.flag == "OK":
        lines += [
            "",
            f"**Planet classification**: {classification.hz_class}",
            f"- In conservative HZ: {'Yes' if classification.is_in_conservative_hz else 'No'}",
            f"- In optimistic HZ: {'Yes' if classification.is_in_optimistic_hz else 'No'}",
        ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="kopparapu_hz_calculator",
        description="Compute habitable-zone boundaries (Kopparapu+ 2013).",
    )
    parser.add_argument("teff_k", type=float)
    parser.add_argument("--insolation", type=float, default=None)
    args = parser.parse_args(argv)

    bnd = compute_hz_boundaries(args.teff_k)
    cls = None
    if args.insolation is not None:
        cls = classify_hz_position(args.insolation, args.teff_k)
    print(format_hz_result(bnd, cls))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
