"""Classify a star into OBAFGKM spectral type and luminosity class from Teff/logg.

Provides a quick, dependency-free classification used in vetting narratives
and submission-pathway decisions.

Public API
----------
SpectralTypeResult(teff_k, logg, spectral_type, luminosity_class,
                   type_string, is_dwarf, is_giant, flag)
classify_spectral_type(teff_k, logg) -> SpectralTypeResult
format_spectral_type(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpectralTypeResult:
    teff_k: float | None
    logg: float | None
    spectral_type: str    # "O" | "B" | "A" | "F" | "G" | "K" | "M" | "UNKNOWN"
    luminosity_class: str  # "V" | "III" | "I" | "UNKNOWN"
    type_string: str       # e.g. "G2V"
    is_dwarf: bool         # luminosity class V
    is_giant: bool         # luminosity class III or I
    flag: str  # "OK" | "UNKNOWN_TEFF" | "INVALID"


# Spectral type boundaries in K (lower bound of each bin)
_SPECTRAL_BINS: list[tuple[float, str]] = [
    (30000.0, "O"),
    (10000.0, "B"),
    (7500.0,  "A"),
    (6000.0,  "F"),
    (5200.0,  "G"),
    (3700.0,  "K"),
    (2400.0,  "M"),
]

# Sub-type within bin (rough, 10 sub-classes)
_SUBTYPE_BINS: list[tuple[float, str, float]] = [
    # (teff_lower, type, teff_upper)
    (30000.0, "O", 50000.0),
    (10000.0, "B", 30000.0),
    (7500.0,  "A", 10000.0),
    (6000.0,  "F", 7500.0),
    (5200.0,  "G", 6000.0),
    (3700.0,  "K", 5200.0),
    (2400.0,  "M", 3700.0),
]


def _teff_to_spectral(teff: float) -> tuple[str, str]:
    """Return (spectral_type, sub_number) from Teff."""
    for low, stype, high in _SUBTYPE_BINS:
        if teff >= low:
            sub = int(9 * (1.0 - (teff - low) / (high - low)))
            return stype, str(min(9, max(0, sub)))
    return "M", "9"


def _logg_to_luminosity(logg: float | None) -> str:
    if logg is None:
        return "UNKNOWN"
    if logg >= 4.0:
        return "V"
    if logg >= 2.5:
        return "III"
    return "I"


def classify_spectral_type(
    teff_k: float | None,
    logg: float | None = None,
) -> SpectralTypeResult:
    """Classify star spectral type and luminosity class.

    Args:
        teff_k: Effective temperature in Kelvin.
        logg: Surface gravity log g (cgs).

    Returns:
        :class:`SpectralTypeResult`.
    """
    if teff_k is not None and teff_k <= 0:
        return SpectralTypeResult(
            teff_k, logg, "UNKNOWN", "UNKNOWN", "UNKNOWN", False, False, "INVALID"
        )

    if teff_k is None:
        return SpectralTypeResult(
            None, logg, "UNKNOWN", "UNKNOWN", "UNKNOWN", False, False, "UNKNOWN_TEFF"
        )

    stype, sub = _teff_to_spectral(teff_k)
    lum_class = _logg_to_luminosity(logg)
    type_str = f"{stype}{sub}{lum_class}" if lum_class != "UNKNOWN" else f"{stype}{sub}"

    return SpectralTypeResult(
        teff_k=teff_k,
        logg=logg,
        spectral_type=stype,
        luminosity_class=lum_class,
        type_string=type_str,
        is_dwarf=lum_class == "V",
        is_giant=lum_class in ("III", "I"),
        flag="OK",
    )


def format_spectral_type(result: SpectralTypeResult) -> str:
    """Format spectral type result as Markdown."""
    lines = [
        "## Spectral Type Classification",
        "",
        f"- Teff: {result.teff_k} K",
        f"- log g: {result.logg}",
        f"- **Spectral type: {result.type_string}**",
        f"- Luminosity class: {result.luminosity_class}",
        f"- Dwarf: {'Yes' if result.is_dwarf else 'No'}",
        f"- Giant: {'Yes' if result.is_giant else 'No'}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="spectral_type_classifier",
        description="Classify star into spectral type from Teff/logg.",
    )
    parser.add_argument("teff_k", type=float, nargs="?", default=None)
    parser.add_argument("--logg", type=float, default=None)
    args = parser.parse_args(argv)

    result = classify_spectral_type(args.teff_k, args.logg)
    print(format_spectral_type(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
