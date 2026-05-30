"""Compute transit chord geometry for a given impact parameter.

Public API
----------
TransitChordResult(chord_rstar, b_used, grazing, flag)
compute_transit_chord(b, rp_rearth, rstar_rsun) -> TransitChordResult
format_transit_chord(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass

_R_EARTH_RSUN = 0.009168  # 1 Earth radius in solar radii


@dataclass(frozen=True)
class TransitChordResult:
    chord_rstar: float  # chord length in stellar radii
    b_used: float       # impact parameter used
    grazing: bool       # True if (b + Rp/Rs) > 1
    flag: str


def compute_transit_chord(
    b: float,
    rp_rearth: float = 1.0,
    rstar_rsun: float = 1.0,
) -> TransitChordResult:
    """Compute transit chord length for a given impact parameter.

    Args:
        b: Impact parameter in [0, 1+Rp/Rs]. Values outside range are clipped.
        rp_rearth: Planet radius in Earth radii.
        rstar_rsun: Stellar radius in solar radii.

    Returns:
        :class:`TransitChordResult`.
    """
    rp_rstar = (rp_rearth * _R_EARTH_RSUN) / rstar_rsun
    b_clipped = max(0.0, min(abs(b), 1.0 + rp_rstar))
    val = max(0.0, 1.0 - b_clipped**2)
    chord = 2.0 * math.sqrt(val)
    grazing = (b_clipped + rp_rstar) > 1.0
    flag = "GRAZING" if grazing else "OK"
    return TransitChordResult(
        chord_rstar=round(chord, 6),
        b_used=round(b_clipped, 6),
        grazing=grazing,
        flag=flag,
    )


def format_transit_chord(result: TransitChordResult) -> str:
    """Format transit chord result as Markdown."""
    lines = [
        "## Transit Chord",
        "",
        f"- Chord length: **{result.chord_rstar:.4f} R★**",
        f"- Impact parameter: {result.b_used:.3f}",
        f"- Grazing: {'yes' if result.grazing else 'no'}",
        f"- Flag: **{result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="transit_chord_calculator",
        description=__doc__,
    )
    p.add_argument("--b", type=float, default=0.0, help="Impact parameter")
    p.add_argument("--rp-rearth", type=float, default=1.0, help="Planet radius in Earth radii")
    p.add_argument("--rstar-rsun", type=float, default=1.0, help="Stellar radius in solar radii")
    args = p.parse_args(argv)
    r = compute_transit_chord(args.b, args.rp_rearth, args.rstar_rsun)
    print(format_transit_chord(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
