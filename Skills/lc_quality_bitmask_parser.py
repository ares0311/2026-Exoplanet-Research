"""Parse TESS quality bitmask flags for light curve cadences.

The TESS pipeline stores a bitmask integer for each cadence where set bits
indicate quality issues.  This module decodes those bitmasks into human-
readable flag names, groups cadences by quality level, and produces a
summary of how many cadences are affected by each issue.

Reference: TESS Science Data Products Description Document (Table 28).

Public API
----------
QualityBit(bit, name, description, severity)
QualityBitmaskResult(n_cadences, n_clean, n_flagged, flagged_fraction,
                     bit_counts, flag)
parse_quality_bitmask(quality_array) -> QualityBitmaskResult
get_clean_mask(quality_array, *, allowed_bits) -> list[bool]
format_bitmask_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QualityBit:
    bit: int          # bit position (0-indexed)
    name: str
    description: str
    severity: str     # "warn" | "bad"


# TESS quality flag definitions (SPOC pipeline)
TESS_QUALITY_BITS: tuple[QualityBit, ...] = (
    QualityBit(0,  "AttitudeTweak",       "Attitude tweak",                          "warn"),
    QualityBit(1,  "SafeMode",            "Safe mode",                               "bad"),
    QualityBit(2,  "CoarsePoint",         "Coarse pointing",                         "bad"),
    QualityBit(3,  "EarthPoint",          "Earth pointing",                          "bad"),
    QualityBit(4,  "ZeroCrossing",        "Momentum zero crossing",                  "warn"),
    QualityBit(5,  "Desat",               "Desaturation event",                      "warn"),
    QualityBit(6,  "Argabrightening",     "Argabrightening event",                   "bad"),
    QualityBit(7,  "ApertureCollision",   "Aperture collision",                      "bad"),
    QualityBit(8,  "SaturationRecov",     "Saturation recovery",                     "warn"),
    QualityBit(9,  "CosmicRay",           "Cosmic ray detected",                     "warn"),
    QualityBit(10, "ManualExclude",       "Manual exclusion",                        "bad"),
    QualityBit(12, "Discontinuity",       "Discontinuity corrected",                 "warn"),
    QualityBit(13, "ImpulsiveOutlier",    "Impulsive outlier",                       "warn"),
    QualityBit(14, "ACCrossing",          "AC crossing",                             "warn"),
    QualityBit(15, "NoiseFloor",          "Noise floor",                             "warn"),
)

_BIT_MAP: dict[int, QualityBit] = {qb.bit: qb for qb in TESS_QUALITY_BITS}

# Default "bad" bits that should be excluded from analysis
DEFAULT_BAD_BITS = frozenset(
    qb.bit for qb in TESS_QUALITY_BITS if qb.severity == "bad"
)


@dataclass(frozen=True)
class QualityBitmaskResult:
    n_cadences: int
    n_clean: int                         # no bad bits set
    n_flagged: int                       # at least one bad bit set
    flagged_fraction: float
    bit_counts: dict[str, int]           # name → count of affected cadences
    flag: str  # "OK" | "ALL_CLEAN" | "INVALID"


def parse_quality_bitmask(
    quality_array: list[int],
) -> QualityBitmaskResult:
    """Parse a TESS quality bitmask array and summarise flag counts.

    Args:
        quality_array: Integer quality value for each cadence.

    Returns:
        :class:`QualityBitmaskResult`.
    """
    n = len(quality_array)
    if n == 0:
        return QualityBitmaskResult(0, 0, 0, 0.0, {}, "INVALID")

    bit_counts: dict[str, int] = {qb.name: 0 for qb in TESS_QUALITY_BITS}
    n_flagged = 0

    bad_mask = 0
    for b in DEFAULT_BAD_BITS:
        bad_mask |= (1 << b)

    for q in quality_array:
        is_bad = bool(q & bad_mask)
        if is_bad:
            n_flagged += 1
        for qb in TESS_QUALITY_BITS:
            if q & (1 << qb.bit):
                bit_counts[qb.name] += 1

    n_clean = n - n_flagged
    frac = n_flagged / n

    return QualityBitmaskResult(
        n_cadences=n,
        n_clean=n_clean,
        n_flagged=n_flagged,
        flagged_fraction=round(frac, 4),
        bit_counts={k: v for k, v in bit_counts.items() if v > 0},
        flag="ALL_CLEAN" if n_flagged == 0 else "OK",
    )


def get_clean_mask(
    quality_array: list[int],
    *,
    allowed_bits: frozenset[int] | None = None,
) -> list[bool]:
    """Return a boolean mask where True means the cadence passes quality cuts.

    Args:
        quality_array: Integer quality values.
        allowed_bits: Bit positions that are tolerated (not excluded).
            Defaults to all warn-level bits being allowed.

    Returns:
        List of bool, same length as quality_array.
    """
    exclude_bits = DEFAULT_BAD_BITS
    if allowed_bits is not None:
        exclude_bits = DEFAULT_BAD_BITS - allowed_bits

    bad_mask = 0
    for b in exclude_bits:
        bad_mask |= (1 << b)

    return [not bool(q & bad_mask) for q in quality_array]


def format_bitmask_result(result: QualityBitmaskResult) -> str:
    """Format quality bitmask result as Markdown."""
    lines = [
        "## TESS Quality Bitmask Summary",
        "",
        f"- Cadences: {result.n_cadences}",
        f"- Clean: {result.n_clean}",
        f"- Flagged: {result.n_flagged} ({result.flagged_fraction:.1%})",
        f"- **Flag: {result.flag}**",
    ]
    if result.bit_counts:
        lines += ["", "| Quality Flag | Count |", "|---|---|"]
        for name, count in sorted(result.bit_counts.items()):
            lines.append(f"| {name} | {count} |")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="lc_quality_bitmask_parser",
        description="Parse TESS quality bitmask flags.",
    )
    parser.add_argument("quality_values", nargs="*", type=int)
    args = parser.parse_args(argv)

    result = parse_quality_bitmask(args.quality_values)
    print(format_bitmask_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
