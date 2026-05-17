"""Check whether a detected BLS period is an alias of known TESS systematics.

TESS has well-known spurious periods from the satellite's orbital and data-downlink
cadence.  A detected period that is close to — or a harmonic/sub-harmonic of — one of
these aliases should be flagged and down-ranked.

Known systematics (approximate):
- 13.7 d  — half the TESS orbital period
- 27.4 d  — full TESS orbital period / sector length
- 0.5 d   — scattered light from Earth/Moon
- 1.0 d   — diurnal alias (ground-based data corruption)

Public API
----------
AliasCheckResult(period_days, alias_period, ratio, is_alias, alias_name, confidence)
check_period_alias(period_days, *, tol=0.03) -> AliasCheckResult
format_alias_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass

# Known TESS / generic systematics that produce spurious BLS peaks.
_KNOWN_ALIASES: list[tuple[str, float]] = [
    ("TESS_half_orbit",    13.7),
    ("TESS_full_orbit",    27.4),
    ("scattered_light",     0.5),
    ("diurnal",             1.0),
    ("TESS_2yr_resonance", 54.8),  # 2 × full orbit
]

# Harmonics and sub-harmonics to check (multiples n × alias and alias/n)
_HARMONIC_ORDERS: tuple[int, ...] = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class AliasCheckResult:
    period_days: float
    alias_period: float | None      # the alias that was matched (None if clean)
    ratio: float | None             # detected / alias (close to an integer or 1/n)
    is_alias: bool
    alias_name: str                 # "" if not an alias
    confidence: float               # 0–1; higher = closer match to alias


def _ratio_distance(ratio: float) -> float:
    """Distance from the nearest integer or unit fraction."""
    # Check integers 1–5
    int_dist = min(abs(ratio - n) / n for n in range(1, 6))
    # Check 1/n fractions
    frac_dist = min(abs(ratio - 1.0 / n) / (1.0 / n) for n in range(1, 6))
    return min(int_dist, frac_dist)


def check_period_alias(
    period_days: float,
    *,
    tol: float = 0.03,
    known_aliases: list[tuple[str, float]] | None = None,
) -> AliasCheckResult:
    """Test whether *period_days* is a harmonic alias of a known systematic.

    Args:
        period_days: Detected BLS period.
        tol: Fractional tolerance — a ratio within ``tol`` of an integer/fraction
            is flagged (default 3%).
        known_aliases: Override the built-in alias table.

    Returns:
        :class:`AliasCheckResult`.
    """
    if period_days <= 0:
        raise ValueError(f"period_days must be positive, got {period_days}")

    aliases = known_aliases if known_aliases is not None else _KNOWN_ALIASES
    best_conf = 0.0
    best_name = ""
    best_alias_period: float | None = None
    best_ratio: float | None = None

    for name, alias_p in aliases:
        ratio = period_days / alias_p
        dist = _ratio_distance(ratio)
        if dist < tol:
            confidence = 1.0 - dist / tol
            if confidence > best_conf:
                best_conf = confidence
                best_name = name
                best_alias_period = alias_p
                best_ratio = ratio

    is_alias = best_conf > 0.0
    return AliasCheckResult(
        period_days=period_days,
        alias_period=best_alias_period,
        ratio=best_ratio,
        is_alias=is_alias,
        alias_name=best_name,
        confidence=best_conf,
    )


def format_alias_result(result: AliasCheckResult) -> str:
    """Human-readable one-liner for an alias check result."""
    if not result.is_alias:
        return f"P={result.period_days:.4f} d — no known alias (clean)"
    return (
        f"P={result.period_days:.4f} d — ALIAS of {result.alias_name} "
        f"({result.alias_period:.2f} d), ratio={result.ratio:.3f}, "
        f"confidence={result.confidence:.2f}"
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="period_alias_checker",
        description="Check whether a period is a TESS systematic alias.",
    )
    parser.add_argument("period", type=float, metavar="PERIOD_DAYS")
    parser.add_argument("--tol", type=float, default=0.03,
                        help="Fractional tolerance (default 0.03).")
    args = parser.parse_args(argv)

    result = check_period_alias(args.period, tol=args.tol)
    print(format_alias_result(result))
    return 1 if result.is_alias else 0


if __name__ == "__main__":
    raise SystemExit(_cli())
