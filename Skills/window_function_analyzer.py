"""Compute the spectral window function from observation times.

Gaps in the time series create sidelobes in the spectral window that cause
spurious peaks at alias periods.  This module computes the window power
spectrum and identifies the main alias periods to watch for.

Public API
----------
WindowFunctionResult(freq_grid, window_power, alias_periods_days,
                     duty_cycle, flag)
compute_window_function(time, *, freq_min, freq_max,
                        n_freqs) -> WindowFunctionResult
find_alias_periods(result, candidate_period_days, *,
                   alias_threshold) -> list[float]
format_window_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class WindowFunctionResult:
    freq_grid: tuple[float, ...]      # cycles/day
    window_power: tuple[float, ...]   # normalised [0, 1]
    alias_periods_days: tuple[float, ...]  # periods with window power > threshold
    duty_cycle: float                 # fraction of time span with observations
    flag: str  # "OK" | "INSUFFICIENT" | "INVALID"


def compute_window_function(
    time: list[float],
    *,
    freq_min: float = 0.01,
    freq_max: float = 2.0,
    n_freqs: int = 500,
    alias_threshold: float = 0.1,
) -> WindowFunctionResult:
    """Compute spectral window function from observation timestamps.

    Args:
        time: Time array (days) — only the timestamps matter, not flux.
        freq_min: Minimum frequency to evaluate (cycles/day).
        freq_max: Maximum frequency to evaluate (cycles/day).
        n_freqs: Number of frequency grid points.
        alias_threshold: Window power threshold for alias identification.

    Returns:
        :class:`WindowFunctionResult`.
    """
    n = len(time)
    if n < 3:
        return WindowFunctionResult((), (), (), 0.0, "INVALID")
    if freq_min <= 0 or freq_max <= freq_min or n_freqs < 2:
        return WindowFunctionResult((), (), (), 0.0, "INVALID")

    t_span = max(time) - min(time)
    if t_span < 1e-9:
        return WindowFunctionResult((), (), (), 0.0, "INVALID")

    # Duty cycle estimate from median gap
    sorted_t = sorted(time)
    gaps = [sorted_t[i + 1] - sorted_t[i] for i in range(n - 1)]
    med_gap = sorted(gaps)[len(gaps) // 2]
    med_cad = med_gap
    duty_cycle = min(1.0, (n * med_cad) / t_span)

    df = (freq_max - freq_min) / (n_freqs - 1)
    freqs = [freq_min + i * df for i in range(n_freqs)]
    powers: list[float] = []

    for freq in freqs:
        omega = 2.0 * math.pi * freq
        re = sum(math.cos(omega * t) for t in time)
        im = sum(math.sin(omega * t) for t in time)
        powers.append((re ** 2 + im ** 2) / (n * n))

    # Normalise so DC power = 1
    max_p = max(powers) if powers else 1.0
    if max_p < 1e-20:
        max_p = 1.0
    powers = [p / max_p for p in powers]

    # Find alias periods (peaks above threshold, excluding DC)
    alias_periods: list[float] = []
    for i in range(1, len(freqs) - 1):
        if (powers[i] > alias_threshold
                and powers[i] >= powers[i - 1]
                and powers[i] >= powers[i + 1]):
            alias_periods.append(round(1.0 / freqs[i], 4))

    return WindowFunctionResult(
        freq_grid=tuple(round(f, 6) for f in freqs),
        window_power=tuple(round(p, 6) for p in powers),
        alias_periods_days=tuple(sorted(alias_periods)),
        duty_cycle=round(duty_cycle, 4),
        flag="OK",
    )


def find_alias_periods(
    result: WindowFunctionResult,
    candidate_period_days: float,
    *,
    alias_threshold: float = 0.1,
) -> list[float]:
    """Find alias periods near a candidate period.

    Args:
        result: Window function result.
        candidate_period_days: Candidate orbital period.
        alias_threshold: Fractional tolerance for alias match.

    Returns:
        List of alias periods (days) within tolerance of candidate or its harmonics.
    """
    if not result.alias_periods_days or candidate_period_days <= 0:
        return []
    aliases = []
    for ap in result.alias_periods_days:
        ratio = ap / candidate_period_days
        nearest = round(ratio)
        if nearest > 0 and abs(ratio - nearest) / nearest < alias_threshold:
            aliases.append(ap)
    return aliases


def format_window_result(result: WindowFunctionResult) -> str:
    """Format window function result as Markdown."""
    lines = [
        "## Window Function Analysis",
        "",
        f"- Freq grid points: {len(result.freq_grid)}",
        f"- Alias periods found: {len(result.alias_periods_days)}",
        f"- Duty cycle: {result.duty_cycle:.3f}",
        f"- **Flag: {result.flag}**",
    ]
    if result.alias_periods_days:
        lines.append("")
        top5 = result.alias_periods_days[:5]
        lines.append("Top alias periods (days): " + ", ".join(f"{p:.3f}" for p in top5))
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="window_function_analyzer",
        description="Compute spectral window function from observation times.",
    )
    parser.add_argument("--freq-min", type=float, default=0.01)
    parser.add_argument("--freq-max", type=float, default=2.0)
    parser.add_argument("--n-freqs", type=int, default=500)
    args = parser.parse_args(argv)

    result = compute_window_function([], freq_min=args.freq_min,
                                     freq_max=args.freq_max, n_freqs=args.n_freqs)
    print(format_window_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
