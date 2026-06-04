"""Find nearest mean-motion resonances for a pair of orbital periods."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MmrMatch:
    ratio_p: int
    ratio_q: int
    resonance_label: str
    period_ratio_exact: float
    deviation_percent: float


@dataclass(frozen=True)
class MmrResonanceResult:
    period_ratio: float
    nearest_resonance: MmrMatch
    all_matches: tuple[MmrMatch, ...]
    is_near_resonance: bool
    flag: str


def _build_resonances(max_order: int = 5) -> list[tuple[int, int]]:
    pairs = []
    for p in range(2, max_order + 2):
        for q in range(1, p):
            if math.gcd(p, q) == 1:
                pairs.append((p, q))
    return pairs


def find_mmr_resonances(
    period_inner_days: float,
    period_outer_days: float,
    max_order: int = 5,
    threshold_percent: float = 2.0,
) -> MmrResonanceResult:
    """Find mean-motion resonances within threshold for a planet pair.

    Checks p:q commensurabilities up to given order where p > q.
    Near-resonance flag set when closest match is within threshold_percent.

    Args:
        period_inner_days: orbital period of inner planet (days)
        period_outer_days: orbital period of outer planet (days)
        max_order: highest resonance order to check (p ≤ max_order+1)
        threshold_percent: deviation threshold for near-resonance flag (percent)
    """
    if period_inner_days <= 0.0:
        return MmrResonanceResult(float("nan"), None, (), False, "INVALID_INNER_PERIOD")  # type: ignore[arg-type]
    if period_outer_days <= 0.0:
        return MmrResonanceResult(float("nan"), None, (), False, "INVALID_OUTER_PERIOD")  # type: ignore[arg-type]
    if period_outer_days <= period_inner_days:
        return MmrResonanceResult(float("nan"), None, (), False, "OUTER_NOT_LONGER")  # type: ignore[arg-type]

    ratio = period_outer_days / period_inner_days
    resonances = _build_resonances(max_order)

    matches = []
    for p, q in resonances:
        exact = p / q
        dev = abs(ratio - exact) / exact * 100.0
        label = f"{p}:{q}"
        matches.append(
            MmrMatch(
                ratio_p=p,
                ratio_q=q,
                resonance_label=label,
                period_ratio_exact=exact,
                deviation_percent=dev,
            )
        )

    matches.sort(key=lambda m: m.deviation_percent)
    nearest = matches[0]
    is_near = nearest.deviation_percent <= threshold_percent

    return MmrResonanceResult(
        period_ratio=ratio,
        nearest_resonance=nearest,
        all_matches=tuple(matches[:10]),
        is_near_resonance=is_near,
        flag="OK",
    )


def format_mmr_result(r: MmrResonanceResult) -> str:
    if r.flag != "OK":
        return f"MmrResonance | flag={r.flag}"
    nr = r.nearest_resonance
    lines = [
        "| Resonance | Exact ratio | Deviation (%) | Near? |",
        "|---|---|---|---|",
    ]
    for m in r.all_matches[:6]:
        near_flag = "YES" if m.deviation_percent <= 2.0 else ""
        lines.append(
            f"| {m.resonance_label} | {m.period_ratio_exact:.4f} "
            f"| {m.deviation_percent:.3f} | {near_flag} |"
        )
    header = (
        f"Period ratio: {r.period_ratio:.4f} | "
        f"Nearest: {nr.resonance_label} ({nr.deviation_percent:.3f}%) | "
        f"Near-resonance: {'YES' if r.is_near_resonance else 'NO'} | "
        f"flag={r.flag}\n\n"
    )
    return header + "\n".join(lines)


def _cli() -> None:
    import argparse

    p = argparse.ArgumentParser(description="MMR resonance finder for planet pairs")
    p.add_argument("period_inner", type=float, help="Inner planet period (days)")
    p.add_argument("period_outer", type=float, help="Outer planet period (days)")
    p.add_argument("--max-order", type=int, default=5)
    p.add_argument("--threshold", type=float, default=2.0, help="Near-resonance threshold (%)")
    args = p.parse_args()
    r = find_mmr_resonances(args.period_inner, args.period_outer,
                            max_order=args.max_order, threshold_percent=args.threshold)
    print(format_mmr_result(r))


if __name__ == "__main__":
    _cli()
