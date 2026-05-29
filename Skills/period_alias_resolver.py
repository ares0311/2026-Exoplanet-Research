"""Resolve period aliases between independent search runs.

Given two period estimates, determines whether they are consistent, aliases
(harmonic/sub-harmonic), or genuinely different.

Public API
----------
AliasResolution(period_a, period_b, ratio, nearest_integer_ratio,
                alias_type, delta_frac, resolved_period, flag)
resolve_period_alias(period_a, period_b, *, period_rtol,
                     max_harmonic) -> AliasResolution
format_alias_resolution(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AliasResolution:
    period_a: float
    period_b: float
    ratio: float           # period_b / period_a
    nearest_integer_ratio: float  # nearest integer or half-integer ratio
    alias_type: str        # "CONSISTENT" | "HARMONIC" | "SUB_HARMONIC" | "UNRELATED"
    delta_frac: float      # |ratio - nearest_integer| / nearest_integer
    resolved_period: float  # the longer period (preferred)
    flag: str  # "OK" | "ALIAS_DETECTED" | "UNRESOLVED"


def resolve_period_alias(
    period_a: float,
    period_b: float,
    *,
    period_rtol: float = 0.01,
    max_harmonic: int = 5,
) -> AliasResolution:
    """Determine the relationship between two period estimates.

    Checks for exact agreement, harmonic (period_b = N × period_a),
    or sub-harmonic (period_a = N × period_b) relationships.

    Args:
        period_a: First period estimate (days).
        period_b: Second period estimate (days).
        period_rtol: Relative tolerance for declaring a match.
        max_harmonic: Maximum integer harmonic to check.

    Returns:
        AliasResolution with alias type and resolved (preferred) period.
    """
    if period_a <= 0 or period_b <= 0:
        return AliasResolution(
            period_a=period_a,
            period_b=period_b,
            ratio=0.0,
            nearest_integer_ratio=0.0,
            alias_type="UNRELATED",
            delta_frac=1.0,
            resolved_period=max(period_a, period_b),
            flag="UNRESOLVED",
        )

    # Ensure period_a <= period_b
    pa, pb = sorted([period_a, period_b])
    ratio = pb / pa  # >= 1.0

    # Check exact agreement
    if abs(ratio - 1.0) < period_rtol:
        return AliasResolution(
            period_a=period_a,
            period_b=period_b,
            ratio=round(ratio, 6),
            nearest_integer_ratio=1.0,
            alias_type="CONSISTENT",
            delta_frac=round(abs(ratio - 1.0), 6),
            resolved_period=round((pa + pb) / 2, 6),
            flag="OK",
        )

    # Check integer harmonics
    best_n = 1
    best_delta = float("inf")
    for n in range(2, max_harmonic + 1):
        for candidate_ratio in [float(n), n / 2.0]:
            delta = abs(ratio - candidate_ratio) / candidate_ratio
            if delta < best_delta:
                best_delta = delta
                best_n = candidate_ratio

    nearest = best_n
    delta_frac = abs(ratio - nearest) / nearest

    if delta_frac < period_rtol:
        alias_type = "HARMONIC" if ratio > 1.5 else "CONSISTENT"
        flag = "ALIAS_DETECTED"
        resolved = pb  # prefer the longer period
    else:
        alias_type = "UNRELATED"
        flag = "UNRESOLVED"
        resolved = pb

    return AliasResolution(
        period_a=period_a,
        period_b=period_b,
        ratio=round(ratio, 6),
        nearest_integer_ratio=round(nearest, 4),
        alias_type=alias_type,
        delta_frac=round(delta_frac, 6),
        resolved_period=round(resolved, 6),
        flag=flag,
    )


def format_alias_resolution(result: AliasResolution) -> str:
    """Format alias resolution as Markdown.

    Args:
        result: AliasResolution to format.

    Returns:
        Markdown string.
    """
    lines = [
        "## Period Alias Resolution\n",
        f"**Status**: `{result.flag}` | **Alias type**: `{result.alias_type}`\n",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Period A (days) | {result.period_a:.6f} |",
        f"| Period B (days) | {result.period_b:.6f} |",
        f"| Ratio (B/A) | {result.ratio:.4f} |",
        f"| Nearest integer ratio | {result.nearest_integer_ratio:.4f} |",
        f"| Δ fraction | {result.delta_frac:.6f} |",
        f"| Resolved period (days) | {result.resolved_period:.6f} |",
    ]
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Resolve period alias between two periods.")
    parser.add_argument("--period-a", type=float, required=True)
    parser.add_argument("--period-b", type=float, required=True)
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--max-harmonic", type=int, default=5)
    args = parser.parse_args(argv)

    result = resolve_period_alias(
        args.period_a, args.period_b,
        period_rtol=args.rtol, max_harmonic=args.max_harmonic,
    )
    print(format_alias_resolution(result))
    return 0 if result.flag == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
