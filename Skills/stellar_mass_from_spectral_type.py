"""Estimate stellar mass range from spectral type classification."""
from __future__ import annotations

import argparse
from dataclasses import dataclass

# Pecaut & Mamajek (2013) + Allen's Astrophysical Quantities
# Keys: spectral class letter + luminosity class (V = main sequence default)
_MASS_TABLE: dict[str, tuple[float, float, float]] = {
    # spectral_type: (mass_min_msun, mass_typical_msun, mass_max_msun)
    "O": (16.0, 40.0, 150.0),
    "B": (2.1, 6.0, 16.0),
    "A": (1.4, 2.0, 2.1),
    "F": (1.04, 1.3, 1.4),
    "G": (0.80, 1.0, 1.04),
    "K": (0.45, 0.70, 0.80),
    "M": (0.08, 0.25, 0.45),
    # Luminosity class adjustments
    "Ia": (8.0, 20.0, 50.0),   # luminous supergiant
    "Ib": (8.0, 15.0, 40.0),   # less luminous supergiant
    "II": (4.0, 8.0, 20.0),    # bright giant
    "III": (1.5, 4.0, 10.0),   # giant
    "IV": (1.0, 1.5, 3.0),     # subgiant
    "V": (0.08, 1.0, 150.0),   # main sequence (wide range)
    "VI": (0.1, 0.5, 0.9),     # subdwarf
    "VII": (0.5, 0.6, 1.2),    # white dwarf (remnant mass)
}

_VALID_CLASSES = set("OBAFGKM")


@dataclass(frozen=True)
class SpectralMassResult:
    spectral_type: str
    spectral_class: str
    luminosity_class: str
    mass_min_msun: float
    mass_typical_msun: float
    mass_max_msun: float
    flag: str


def estimate_mass_from_spectral_type(
    spectral_type: str,
) -> SpectralMassResult:
    """
    Estimate stellar mass range from spectral type string (e.g. 'G2V', 'K5', 'M4V').

    Parses the leading letter as spectral class and trailing Roman numeral as
    luminosity class. Sub-type numbers are ignored; mass range is for the class.
    Returns (min, typical, max) mass in solar units.
    """
    if not spectral_type or not spectral_type.strip():
        return SpectralMassResult(
            spectral_type=spectral_type, spectral_class="", luminosity_class="",
            mass_min_msun=float("nan"), mass_typical_msun=float("nan"),
            mass_max_msun=float("nan"), flag="INVALID_SPECTRAL_TYPE",
        )

    stype = spectral_type.strip().upper()
    spectral_class = stype[0] if stype else ""

    if spectral_class not in _VALID_CLASSES:
        return SpectralMassResult(
            spectral_type=spectral_type, spectral_class=spectral_class, luminosity_class="",
            mass_min_msun=float("nan"), mass_typical_msun=float("nan"),
            mass_max_msun=float("nan"), flag="UNKNOWN_SPECTRAL_CLASS",
        )

    # Parse luminosity class: look for Roman numerals at end
    lum_class = "V"  # default main sequence
    for lc in ["VII", "VI", "Ia", "Ib", "IV", "III", "II", "I", "V"]:
        if stype.endswith(lc.upper()):
            lum_class = lc
            break

    # Use spectral class for mass estimate; refine with luminosity class if non-MS
    mass_min, mass_typ, mass_max = _MASS_TABLE[spectral_class]

    if lum_class in ("III", "II", "I", "Ia", "Ib") and spectral_class in "GKMO":
        lum_min, lum_typ, lum_max = _MASS_TABLE[lum_class]
        mass_min = max(mass_min, lum_min)
        mass_typ = max(mass_typ, lum_typ)
        mass_max = min(max(mass_max, lum_max), lum_max)

    if lum_class == "VII":
        mass_min, mass_typ, mass_max = _MASS_TABLE["VII"]

    return SpectralMassResult(
        spectral_type=spectral_type,
        spectral_class=spectral_class,
        luminosity_class=lum_class,
        mass_min_msun=round(mass_min, 3),
        mass_typical_msun=round(mass_typ, 3),
        mass_max_msun=round(mass_max, 3),
        flag="OK",
    )


def format_mass_result(r: SpectralMassResult) -> str:
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Spectral type | {r.spectral_type} |\n"
        f"| Spectral class | {r.spectral_class} |\n"
        f"| Luminosity class | {r.luminosity_class} |\n"
        f"| Mass min (M☉) | {r.mass_min_msun:.3f} |\n"
        f"| Mass typical (M☉) | {r.mass_typical_msun:.3f} |\n"
        f"| Mass max (M☉) | {r.mass_max_msun:.3f} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Estimate stellar mass from spectral type.")
    p.add_argument("spectral_type")
    args = p.parse_args()
    r = estimate_mass_from_spectral_type(args.spectral_type)
    print(format_mass_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
