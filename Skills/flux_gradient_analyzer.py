"""
Detects rapid flux gradients (ramps) by computing first differences of flux.

Public API:
    GradientResult        -- frozen dataclass holding gradient diagnostics
    analyze_flux_gradient(flux, threshold_ppm) -> GradientResult
    format_gradient(result) -> str
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class GradientResult:
    max_gradient_ppm_per_step: float
    mean_abs_gradient_ppm: float
    n_steep_steps: int
    flag: str


def analyze_flux_gradient(
    flux: list[float],
    threshold_ppm: float = 500.0,
) -> GradientResult:
    n = len(flux)
    if n < 2:
        return GradientResult(
            max_gradient_ppm_per_step=0.0,
            mean_abs_gradient_ppm=0.0,
            n_steep_steps=0,
            flag="INSUFFICIENT_DATA",
        )

    mean_flux = sum(flux) / n
    if mean_flux == 0.0:
        mean_flux = 1.0

    gradients = [abs(flux[i] - flux[i - 1]) / mean_flux * 1e6 for i in range(1, n)]

    max_grad = max(gradients)
    mean_abs_grad = sum(gradients) / len(gradients)
    n_steep = sum(1 for g in gradients if g > threshold_ppm)

    flag = "RAMP_DETECTED" if max_grad > threshold_ppm else "OK"

    return GradientResult(
        max_gradient_ppm_per_step=max_grad,
        mean_abs_gradient_ppm=mean_abs_grad,
        n_steep_steps=n_steep,
        flag=flag,
    )


def format_gradient(result: GradientResult) -> str:
    lines = [
        "## Flux Gradient Analysis",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Max gradient (ppm/step) | {result.max_gradient_ppm_per_step:.1f} |",
        f"| Mean |gradient| (ppm/step) | {result.mean_abs_gradient_ppm:.1f} |",
        f"| Steep steps (above threshold) | {result.n_steep_steps} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Detect rapid flux gradients from a JSON flux file."
    )
    parser.add_argument("flux_file", help="JSON file containing list of flux values")
    parser.add_argument(
        "--threshold-ppm",
        type=float,
        default=500.0,
        help="Gradient threshold in ppm/step (default 500.0)",
    )
    args = parser.parse_args()

    with open(args.flux_file) as fh:
        flux = json.load(fh)

    result = analyze_flux_gradient(flux, args.threshold_ppm)
    print(format_gradient(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
