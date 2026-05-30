"""Compute an upper limit on secondary eclipse depth from out-of-transit scatter.

Uses the median absolute deviation (MAD) of the out-of-transit flux to set a
noise floor, then returns n_sigma * noise_ppm as the upper limit on any
undetected secondary eclipse.

Public API
----------
SecondaryLimitResult(upper_limit_ppm, n_sigma, noise_ppm, flag)
compute_secondary_upper_limit(oot_flux, *, n_sigma, phase_window) -> SecondaryLimitResult
format_secondary_limit(result) -> str
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class SecondaryLimitResult:
    upper_limit_ppm: float
    n_sigma: float
    noise_ppm: float
    flag: str  # "OK" | "NO_DATA" | "INSUFFICIENT_DATA"


def _median(values: list[float]) -> float:
    """Return the median of a non-empty list."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 1:
        return sorted_vals[mid]
    return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0


def _mad(values: list[float]) -> float:
    """Return the median absolute deviation of a non-empty list."""
    med = _median(values)
    deviations = [abs(v - med) for v in values]
    return _median(deviations)


def compute_secondary_upper_limit(
    oot_flux: list[float],
    *,
    n_sigma: float = 3.0,
    phase_window: float = 0.1,
) -> SecondaryLimitResult:
    """Compute an n-sigma upper limit on secondary eclipse depth.

    Parameters
    ----------
    oot_flux:
        Out-of-transit flux values (normalised near 1.0).
    n_sigma:
        Number of sigma for the upper limit.
    phase_window:
        Reserved for future use (phase half-width around secondary phase).

    Returns
    -------
    SecondaryLimitResult
    """
    if len(oot_flux) == 0:
        return SecondaryLimitResult(
            upper_limit_ppm=0.0,
            n_sigma=n_sigma,
            noise_ppm=0.0,
            flag="NO_DATA",
        )

    if len(oot_flux) < 3:
        # Still compute a best-effort estimate but flag it
        mad_val = _mad(oot_flux)
        noise_ppm = mad_val * 1.4826 * 1e6
        upper_limit_ppm = n_sigma * noise_ppm
        return SecondaryLimitResult(
            upper_limit_ppm=upper_limit_ppm,
            n_sigma=n_sigma,
            noise_ppm=noise_ppm,
            flag="INSUFFICIENT_DATA",
        )

    mad_val = _mad(oot_flux)
    noise_ppm = mad_val * 1.4826 * 1e6
    upper_limit_ppm = n_sigma * noise_ppm

    return SecondaryLimitResult(
        upper_limit_ppm=upper_limit_ppm,
        n_sigma=n_sigma,
        noise_ppm=noise_ppm,
        flag="OK",
    )


def format_secondary_limit(result: SecondaryLimitResult) -> str:
    """Return a Markdown table summarising the secondary eclipse upper limit."""
    lines = [
        "| Metric | Value |",
        "| --- | --- |",
        f"| Upper Limit (ppm) | {result.upper_limit_ppm:.2f} |",
        f"| N-sigma | {result.n_sigma:.1f} |",
        f"| Noise (ppm) | {result.noise_ppm:.2f} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Compute secondary eclipse depth upper limit from OOT scatter."
    )
    parser.add_argument(
        "oot_flux",
        nargs="+",
        type=float,
        help="Out-of-transit flux values (normalised near 1.0).",
    )
    parser.add_argument(
        "--n-sigma",
        type=float,
        default=3.0,
        help="Number of sigma for the upper limit (default: 3.0).",
    )
    parser.add_argument(
        "--phase-window",
        type=float,
        default=0.1,
        help="Phase half-width around secondary phase (reserved; default: 0.1).",
    )
    args = parser.parse_args()

    result = compute_secondary_upper_limit(
        args.oot_flux,
        n_sigma=args.n_sigma,
        phase_window=args.phase_window,
    )
    print(format_secondary_limit(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
