"""Check whether multiple candidate periods are consistent with a multi-planet system.

For a list of candidate periods, tests whether any pair is a small-integer
harmonic ratio (alias) or whether they are truly independent periods.

Public API
----------
PeriodPairResult(period_a, period_b, ratio, nearest_integer_ratio,
                 ratio_deviation, is_harmonic, relationship)
MultiPlanetCheckResult(n_candidates, pairs, n_harmonic_pairs,
                       n_independent_periods, flag)
check_multi_planet_periods(periods_days, *, harmonic_tol,
                            max_ratio) -> MultiPlanetCheckResult
format_multi_planet_check(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PeriodPairResult:
    period_a: float
    period_b: float
    ratio: float              # period_b / period_a (always >= 1)
    nearest_integer_ratio: int
    ratio_deviation: float    # |ratio - nearest_integer| / nearest_integer
    is_harmonic: bool
    relationship: str         # "harmonic" | "independent" | "alias"


@dataclass(frozen=True)
class MultiPlanetCheckResult:
    n_candidates: int
    pairs: tuple[PeriodPairResult, ...]
    n_harmonic_pairs: int
    n_independent_periods: int
    flag: str  # "OK" | "SINGLE" | "INVALID"


def _check_pair(p_a: float, p_b: float, *, harmonic_tol: float, max_ratio: int) -> PeriodPairResult:
    """Classify a pair of periods."""
    # Ensure ratio >= 1
    if p_a > p_b:
        p_a, p_b = p_b, p_a
    ratio = p_b / p_a
    nearest = round(ratio)
    nearest = max(1, min(nearest, max_ratio))
    dev = abs(ratio - nearest) / nearest

    if nearest == 1:
        # Periods nearly identical — alias
        relationship = "alias" if dev < harmonic_tol else "independent"
        is_harmonic = dev < harmonic_tol
    elif dev < harmonic_tol:
        relationship = "harmonic"
        is_harmonic = True
    else:
        relationship = "independent"
        is_harmonic = False

    return PeriodPairResult(
        period_a=round(p_a, 6),
        period_b=round(p_b, 6),
        ratio=round(ratio, 4),
        nearest_integer_ratio=nearest,
        ratio_deviation=round(dev, 5),
        is_harmonic=is_harmonic,
        relationship=relationship,
    )


def check_multi_planet_periods(
    periods_days: list[float],
    *,
    harmonic_tol: float = 0.02,
    max_ratio: int = 8,
) -> MultiPlanetCheckResult:
    """Check whether candidate periods are harmonically related.

    Args:
        periods_days: List of candidate orbital periods (days).
        harmonic_tol: Fractional tolerance for harmonic match.
        max_ratio: Maximum integer ratio to check.

    Returns:
        :class:`MultiPlanetCheckResult`.
    """
    valid = [p for p in periods_days if p > 0]
    if not valid:
        return MultiPlanetCheckResult(0, (), 0, 0, "INVALID")
    if len(valid) == 1:
        return MultiPlanetCheckResult(1, (), 0, 1, "SINGLE")

    pairs: list[PeriodPairResult] = []
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            ratio = valid[j] / valid[i] if valid[i] < valid[j] else valid[i] / valid[j]
            if ratio > max_ratio + 0.5:
                continue
            pair = _check_pair(valid[i], valid[j], harmonic_tol=harmonic_tol, max_ratio=max_ratio)
            pairs.append(pair)

    n_harmonic = sum(1 for pr in pairs if pr.is_harmonic)

    # Count independent periods: periods not involved in any harmonic pair
    harmonic_indices: set[int] = set()
    for pr in pairs:
        if pr.is_harmonic:
            for k, p in enumerate(valid):
                if abs(p - pr.period_a) < 1e-9 or abs(p - pr.period_b) < 1e-9:
                    harmonic_indices.add(k)
    n_independent = len(valid) - len(harmonic_indices)

    return MultiPlanetCheckResult(
        n_candidates=len(valid),
        pairs=tuple(pairs),
        n_harmonic_pairs=n_harmonic,
        n_independent_periods=n_independent,
        flag="OK",
    )


def format_multi_planet_check(result: MultiPlanetCheckResult) -> str:
    """Format multi-planet period check as Markdown."""
    lines = [
        "## Multi-Planet Period Check",
        "",
        f"- Candidate periods: {result.n_candidates}",
        f"- Period pairs checked: {len(result.pairs)}",
        f"- Harmonic pairs: {result.n_harmonic_pairs}",
        f"- Independent periods: {result.n_independent_periods}",
        f"- **Flag: {result.flag}**",
    ]
    if result.pairs:
        lines.append("")
        lines.append("| Period A (d) | Period B (d) | Ratio | Relationship |")
        lines.append("|---|---|---|---|")
        for pr in result.pairs:
            lines.append(
                f"| {pr.period_a} | {pr.period_b} | {pr.ratio:.3f} | {pr.relationship} |"
            )
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="multi_planet_period_checker",
        description="Check whether candidate periods are harmonically related.",
    )
    parser.add_argument("periods", nargs="+", type=float)
    parser.add_argument("--harmonic-tol", type=float, default=0.02)
    args = parser.parse_args(argv)

    result = check_multi_planet_periods(args.periods, harmonic_tol=args.harmonic_tol)
    print(format_multi_planet_check(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
