"""Identify mean-motion resonances in a multi-planet system."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResonancePair:
    inner_index: int
    outer_index: int
    period_ratio: float
    nearest_p: int
    nearest_q: int
    deviation_percent: float
    near_resonant: bool   # deviation < 1%


@dataclass(frozen=True)
class ResonanceChainResult:
    pairs: tuple[ResonancePair, ...]
    n_near_resonant: int
    chain_string: str      # e.g. "4:2:1" or "non-resonant"
    flag: str


def analyze_resonance_chain(
    periods_days: list[float],
    max_order: int = 5,
    threshold_percent: float = 2.0,
) -> ResonanceChainResult:
    """Find nearest MMRs for each adjacent planet pair.

    For each adjacent pair, searches p:q with p,q ≤ max_order and gcd(p,q)=1,
    finding the ratio closest to (P_outer/P_inner).

    Args:
        periods_days: list of orbital periods, sorted ascending
        max_order: maximum integer for p and q in p:q resonance
        threshold_percent: near-resonant threshold (%)
    """
    from math import gcd

    if len(periods_days) < 2:
        return ResonanceChainResult((), 0, "insufficient_planets", "INSUFFICIENT_PLANETS")

    for p in periods_days:
        if p <= 0.0:
            return ResonanceChainResult((), 0, "invalid", "INVALID_PERIOD")

    sorted_periods = sorted(periods_days)

    # Build candidate resonances
    candidates: list[tuple[int, int, float]] = []
    for q in range(1, max_order + 1):
        for p in range(q + 1, max_order * q + 1):
            if gcd(p, q) == 1:
                candidates.append((p, q, p / q))

    pairs: list[ResonancePair] = []
    for i in range(len(sorted_periods) - 1):
        ratio = sorted_periods[i + 1] / sorted_periods[i]
        best_dev = float("inf")
        best_p, best_q = 2, 1
        for cp, cq, cr in candidates:
            dev = abs(ratio - cr) / cr * 100.0
            if dev < best_dev:
                best_dev = dev
                best_p, best_q = cp, cq
        near = best_dev < threshold_percent
        pairs.append(ResonancePair(
            inner_index=i,
            outer_index=i + 1,
            period_ratio=ratio,
            nearest_p=best_p,
            nearest_q=best_q,
            deviation_percent=best_dev,
            near_resonant=near,
        ))

    n_near = sum(1 for pr in pairs if pr.near_resonant)

    if n_near == len(pairs) and len(pairs) > 0:
        chain_parts = [str(pairs[0].nearest_p)]
        for pr in pairs:
            chain_parts.append(str(pr.nearest_q))
        chain_str = ":".join(chain_parts)
    elif n_near > 0:
        chain_str = "partial_chain"
    else:
        chain_str = "non_resonant"

    return ResonanceChainResult(
        pairs=tuple(pairs),
        n_near_resonant=n_near,
        chain_string=chain_str,
        flag="OK",
    )


def format_resonance_chain_result(r: ResonanceChainResult) -> str:
    if r.flag != "OK":
        return f"ResonanceChain | flag={r.flag}"
    lines = [
        f"Chain: {r.chain_string} | Near-resonant pairs: {r.n_near_resonant} | "
        f"flag={r.flag}",
        "",
        "| Pair | Ratio | Nearest MMR | Δ (%) | Near? |",
        "|---|---|---|---|---|",
    ]
    for pr in r.pairs:
        near_str = "✓" if pr.near_resonant else ""
        lines.append(
            f"| {pr.inner_index+1}:{pr.outer_index+1} "
            f"| {pr.period_ratio:.4f} "
            f"| {pr.nearest_p}:{pr.nearest_q} "
            f"| {pr.deviation_percent:.2f} "
            f"| {near_str} |"
        )
    return "\n".join(lines)


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Planetary resonance chain analyzer")
    p.add_argument("periods", type=float, nargs="+", help="Orbital periods (days)")
    args = p.parse_args()
    r = analyze_resonance_chain(args.periods)
    print(format_resonance_chain_result(r))


if __name__ == "__main__":
    _cli()
