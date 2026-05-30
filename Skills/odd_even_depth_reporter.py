"""Odd/even transit depth significance reporter.

Computes the weighted-mean depth for odd and even transits and tests whether
the difference is statistically significant — a key eclipsing-binary diagnostic.

Public API
----------
OddEvenResult(odd_mean, even_mean, delta, sigma, significance, flag)
compute_odd_even_significance(odd_depths, odd_errors, even_depths, even_errors)
    -> OddEvenResult
format_odd_even_result(result) -> str
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class OddEvenResult:
    odd_mean: float
    even_mean: float
    delta: float
    sigma: float
    significance: float
    flag: str


def compute_odd_even_significance(
    odd_depths: list[float],
    odd_errors: list[float],
    even_depths: list[float],
    even_errors: list[float],
) -> OddEvenResult:
    """Compute weighted-mean odd/even depths and their significance.

    Parameters
    ----------
    odd_depths:  per-transit depths for odd-numbered transits
    odd_errors:  per-transit depth uncertainties for odd transits
    even_depths: per-transit depths for even-numbered transits
    even_errors: per-transit depth uncertainties for even transits

    Returns
    -------
    OddEvenResult with flag one of:
    INSUFFICIENT_DATA, INVALID_ERRORS, SIGNIFICANT_ODD_EVEN, OK
    """
    if len(odd_depths) < 1 or len(even_depths) < 1:
        return OddEvenResult(
            odd_mean=0.0,
            even_mean=0.0,
            delta=0.0,
            sigma=0.0,
            significance=0.0,
            flag="INSUFFICIENT_DATA",
        )

    all_errors = list(odd_errors) + list(even_errors)
    if any(e <= 0 for e in all_errors):
        return OddEvenResult(
            odd_mean=0.0,
            even_mean=0.0,
            delta=0.0,
            sigma=0.0,
            significance=0.0,
            flag="INVALID_ERRORS",
        )

    # Weighted mean for odd transits
    odd_weights = [1.0 / (e * e) for e in odd_errors]
    odd_w_sum = sum(odd_weights)
    odd_mean = sum(d * w for d, w in zip(odd_depths, odd_weights, strict=False)) / odd_w_sum
    odd_var = 1.0 / odd_w_sum

    # Weighted mean for even transits
    even_weights = [1.0 / (e * e) for e in even_errors]
    even_w_sum = sum(even_weights)
    even_mean = sum(d * w for d, w in zip(even_depths, even_weights, strict=False)) / even_w_sum
    even_var = 1.0 / even_w_sum

    delta = abs(odd_mean - even_mean)
    sigma = math.sqrt(odd_var + even_var)
    significance = delta / sigma if sigma > 0.0 else 0.0

    flag = "SIGNIFICANT_ODD_EVEN" if significance > 3.0 else "OK"

    return OddEvenResult(
        odd_mean=odd_mean,
        even_mean=even_mean,
        delta=delta,
        sigma=sigma,
        significance=significance,
        flag=flag,
    )


def format_odd_even_result(result: OddEvenResult) -> str:
    """Return a Markdown table summarising the odd/even depth result."""
    lines = [
        "| Metric | Value |",
        "| --- | --- |",
        f"| Odd Mean | {result.odd_mean:.4f} |",
        f"| Even Mean | {result.even_mean:.4f} |",
        f"| Delta | {result.delta:.4f} |",
        f"| Sigma | {result.sigma:.4f} |",
        f"| Significance (sigma) | {result.significance:.4f} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Compute odd/even transit depth significance."
    )
    parser.add_argument(
        "--odd-depths",
        nargs="+",
        type=float,
        required=True,
        metavar="DEPTH",
        help="Odd-transit depths (same units as errors).",
    )
    parser.add_argument(
        "--odd-errors",
        nargs="+",
        type=float,
        required=True,
        metavar="ERR",
        help="Odd-transit depth uncertainties.",
    )
    parser.add_argument(
        "--even-depths",
        nargs="+",
        type=float,
        required=True,
        metavar="DEPTH",
        help="Even-transit depths.",
    )
    parser.add_argument(
        "--even-errors",
        nargs="+",
        type=float,
        required=True,
        metavar="ERR",
        help="Even-transit depth uncertainties.",
    )
    args = parser.parse_args()
    result = compute_odd_even_significance(
        args.odd_depths, args.odd_errors, args.even_depths, args.even_errors
    )
    print(format_odd_even_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
