"""Compute stellar insolation at planet orbit relative to Earth.

S_p = (L_star / L_sun) / a_AU^2

Public API
----------
InsolationResult(insolation_earth_units, hz_class, flag)
compute_insolation(luminosity_lsun, a_au) -> InsolationResult
format_insolation_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InsolationResult:
    insolation_earth_units: float
    hz_class: str  # "hot_zone" / "inner_hz" / "outer_hz" / "cold_zone"
    flag: str


def _hz_class(s: float) -> str:
    """Classify HZ position using Kopparapu-style boundaries."""
    if s > 1.1:
        return "hot_zone"
    if s >= 0.36:
        return "inner_hz"
    if s >= 0.20:
        return "outer_hz"
    return "cold_zone"


def compute_insolation(
    luminosity_lsun: float,
    a_au: float,
) -> InsolationResult:
    """Compute stellar insolation at a_au from a star of given luminosity.

    Args:
        luminosity_lsun: Stellar luminosity in solar luminosities.
        a_au: Orbital semi-major axis in AU.

    Returns:
        :class:`InsolationResult`.
    """
    if a_au <= 0 or luminosity_lsun <= 0:
        return InsolationResult(
            insolation_earth_units=0.0,
            hz_class="cold_zone",
            flag="ERROR",
        )
    s = float(luminosity_lsun) / (float(a_au) ** 2)
    hz = _hz_class(s)
    return InsolationResult(
        insolation_earth_units=round(s, 6),
        hz_class=hz,
        flag="OK",
    )


def format_insolation_result(result: InsolationResult) -> str:
    """Format insolation result as Markdown."""
    lines = [
        "## Insolation Flux",
        "",
        f"- Insolation: **{result.insolation_earth_units:.4f} S⊕**",
        f"- HZ class: **{result.hz_class}**",
        f"- Flag: {result.flag}",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="insolation_flux_calculator",
        description="Compute stellar insolation at a planet orbit.",
    )
    parser.add_argument("luminosity_lsun", type=float, help="Stellar luminosity in L_sun")
    parser.add_argument("a_au", type=float, help="Orbital semi-major axis in AU")
    args = parser.parse_args(argv)

    result = compute_insolation(args.luminosity_lsun, args.a_au)
    print(format_insolation_result(result))
    return 0 if result.flag == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
