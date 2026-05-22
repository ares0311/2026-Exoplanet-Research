"""Check orbital periods for near mean-motion resonance (MMR) commensurability.

Given a list of orbital periods, finds all pairs that lie within a tolerance
of integer or half-integer mean-motion resonance ratios.  Useful for identifying
potential resonant chains or alias pairs in multi-planet systems.

Public API
----------
CommensurabilityPair(period_a, period_b, ratio_p, ratio_q, actual_ratio,
                     deviation, is_near_resonance, flag)
CommensurabilityResult(n_periods, pairs, n_resonant, flag)
check_commensurability(periods, *, ratios, tol_frac) -> CommensurabilityResult
format_commensurability_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommensurabilityPair:
    period_a: float          # shorter period
    period_b: float          # longer period
    ratio_p: int             # numerator of nearest simple ratio (p:q, p > q)
    ratio_q: int             # denominator
    actual_ratio: float      # period_b / period_a
    deviation: float         # |actual - p/q| / (p/q)
    is_near_resonance: bool
    flag: str  # "OK" | "INVALID"


@dataclass(frozen=True)
class CommensurabilityResult:
    n_periods: int
    pairs: tuple[CommensurabilityPair, ...]
    n_resonant: int
    flag: str  # "OK" | "INSUFFICIENT" | "INVALID"


# Default MMR ratios to check: (p, q) with p > q
_DEFAULT_RATIOS: list[tuple[int, int]] = [
    (2, 1), (3, 2), (4, 3), (5, 4), (3, 1), (5, 2), (5, 3), (4, 1),
]


def _nearest_ratio(
    ratio: float,
    ratios: list[tuple[int, int]],
) -> tuple[int, int, float]:
    """Return (p, q, deviation) for the nearest MMR ratio."""
    best_p, best_q, best_dev = 1, 1, float("inf")
    for p, q in ratios:
        r = p / q
        dev = abs(ratio - r) / r
        if dev < best_dev:
            best_dev = dev
            best_p, best_q = p, q
    return best_p, best_q, best_dev


def check_commensurability(
    periods: list[float],
    *,
    ratios: list[tuple[int, int]] | None = None,
    tol_frac: float = 0.02,
) -> CommensurabilityResult:
    """Check all pairs of periods for near-MMR commensurability.

    Args:
        periods: List of orbital periods (days).  Order does not matter.
        ratios: List of (p, q) integer ratio tuples to check.
            Defaults to common resonances: 2:1, 3:2, 4:3, 5:4, 3:1, 5:2, 5:3, 4:1.
        tol_frac: Fractional deviation tolerance (0.02 = 2%).

    Returns:
        :class:`CommensurabilityResult`.
    """
    if ratios is None:
        ratios = _DEFAULT_RATIOS

    n = len(periods)
    if n < 2:
        return CommensurabilityResult(n, (), 0, "OK")
    if any(p <= 0 for p in periods):
        return CommensurabilityResult(n, (), 0, "INVALID")

    pairs: list[CommensurabilityPair] = []
    for i in range(n):
        for j in range(i + 1, n):
            pa = min(periods[i], periods[j])
            pb = max(periods[i], periods[j])
            actual = pb / pa
            best_p, best_q, dev = _nearest_ratio(actual, ratios)
            near = dev <= tol_frac
            pairs.append(CommensurabilityPair(
                period_a=round(pa, 6),
                period_b=round(pb, 6),
                ratio_p=best_p,
                ratio_q=best_q,
                actual_ratio=round(actual, 6),
                deviation=round(dev, 6),
                is_near_resonance=near,
                flag="OK",
            ))

    n_resonant = sum(1 for p in pairs if p.is_near_resonance)
    return CommensurabilityResult(
        n_periods=n,
        pairs=tuple(sorted(pairs, key=lambda p: p.deviation)),
        n_resonant=n_resonant,
        flag="OK",
    )


def format_commensurability_result(result: CommensurabilityResult) -> str:
    """Format commensurability result as Markdown."""
    if result.flag in ("INSUFFICIENT", "INVALID"):
        return f"## Period Commensurability\n\n_Flag: {result.flag}_\n"

    lines = [
        "## Period Commensurability Check",
        "",
        f"- Periods checked: {result.n_periods}",
        f"- Pairs near MMR: {result.n_resonant} / {len(result.pairs)}",
        "",
        "| P_a (d) | P_b (d) | Ratio | Actual | Deviation | Near MMR? |",
        "|---|---|---|---|---|---|",
    ]
    for pair in result.pairs:
        near = "Yes" if pair.is_near_resonance else "No"
        lines.append(
            f"| {pair.period_a:.4f} | {pair.period_b:.4f}"
            f" | {pair.ratio_p}:{pair.ratio_q}"
            f" | {pair.actual_ratio:.4f} | {pair.deviation:.4f} | {near} |"
        )
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="period_commensurability_checker",
        description="Check orbital periods for near MMR commensurability.",
    )
    parser.add_argument("periods", nargs="+", type=float)
    parser.add_argument("--tol", type=float, default=0.02)
    args = parser.parse_args(argv)

    result = check_commensurability(args.periods, tol_frac=args.tol)
    print(format_commensurability_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
