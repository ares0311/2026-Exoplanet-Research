"""Convert between common time systems used in TESS and Kepler pipelines.

TESS timestamps are stored as BTJD = BJD - 2457000.0.
Kepler timestamps use BKJD = BJD - 2454833.0.
Standard astronomical timestamps use JD or BJD.

This module provides lightweight, dependency-free conversions between all four
systems and can also propagate epoch uncertainty through the conversion.

Public API
----------
EphemerisConversionResult(input_value, input_system, output_value,
                          output_system, flag)
convert_epoch(value, from_system, to_system, *, uncertainty) -> EphemerisConversionResult
convert_time_array(times, from_system, to_system) -> list[float]
format_conversion_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass

# Offsets: BJD = system + OFFSET
_OFFSETS: dict[str, float] = {
    "BJD": 0.0,
    "JD": 0.0,   # BJD ≈ JD; rigorously differs by TDB-TT but treated as equal here
    "BTJD": 2457000.0,  # BJD = BTJD + 2457000
    "BKJD": 2454833.0,  # BJD = BKJD + 2454833
}

_VALID_SYSTEMS = frozenset(_OFFSETS)


@dataclass(frozen=True)
class EphemerisConversionResult:
    input_value: float
    input_system: str
    output_value: float
    output_system: str
    uncertainty_out: float | None   # propagated uncertainty (same magnitude)
    flag: str  # "OK" | "INVALID"


def convert_epoch(
    value: float,
    from_system: str,
    to_system: str,
    *,
    uncertainty: float | None = None,
) -> EphemerisConversionResult:
    """Convert a single epoch value between time systems.

    Args:
        value: Input epoch in ``from_system`` units.
        from_system: One of "BJD", "JD", "BTJD", "BKJD".
        to_system: One of "BJD", "JD", "BTJD", "BKJD".
        uncertainty: Optional epoch uncertainty (days); propagated unchanged.

    Returns:
        :class:`EphemerisConversionResult`.
    """
    fs = from_system.upper()
    ts = to_system.upper()
    if fs not in _VALID_SYSTEMS or ts not in _VALID_SYSTEMS:
        return EphemerisConversionResult(
            value, from_system, float("nan"), to_system, None, "INVALID"
        )

    # Convert to BJD then to target
    bjd = value + _OFFSETS[fs]
    out = bjd - _OFFSETS[ts]

    return EphemerisConversionResult(
        input_value=value,
        input_system=fs,
        output_value=round(out, 8),
        output_system=ts,
        uncertainty_out=uncertainty,  # shifts don't change uncertainty
        flag="OK",
    )


def convert_time_array(
    times: list[float],
    from_system: str,
    to_system: str,
) -> list[float]:
    """Batch-convert a time array between systems.

    Args:
        times: Input time values.
        from_system: Source time system.
        to_system: Target time system.

    Returns:
        List of converted values; empty list if systems are invalid.
    """
    fs = from_system.upper()
    ts = to_system.upper()
    if fs not in _VALID_SYSTEMS or ts not in _VALID_SYSTEMS:
        return []
    shift = _OFFSETS[fs] - _OFFSETS[ts]
    return [t + shift for t in times]


def format_conversion_result(result: EphemerisConversionResult) -> str:
    """Format ephemeris conversion result as Markdown."""
    lines = [
        "## Ephemeris Conversion",
        "",
        f"- Input:  {result.input_value:.6f} {result.input_system}",
        f"- Output: {result.output_value:.6f} {result.output_system}",
        f"- Uncertainty: {result.uncertainty_out}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="ephemeris_converter",
        description="Convert epoch between BJD/JD/BTJD/BKJD.",
    )
    parser.add_argument("value", type=float)
    parser.add_argument("from_system", choices=["BJD", "JD", "BTJD", "BKJD"])
    parser.add_argument("to_system", choices=["BJD", "JD", "BTJD", "BKJD"])
    parser.add_argument("--uncertainty", type=float, default=None)
    args = parser.parse_args(argv)

    result = convert_epoch(args.value, args.from_system, args.to_system,
                           uncertainty=args.uncertainty)
    print(format_conversion_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
