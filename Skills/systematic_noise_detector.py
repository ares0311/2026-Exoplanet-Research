"""Detect systematics in a photometric flux series.

Computes Durbin-Watson statistic, lag-1 autocorrelation, and runs test
to identify correlated noise patterns indicative of instrumental systematics.

Public API
----------
SystematicNoiseResult(dw_statistic, lag1_autocorr, n_runs, expected_runs,
                      has_systematics, flag)
detect_systematic_noise(flux) -> SystematicNoiseResult
format_systematic_noise(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SystematicNoiseResult:
    dw_statistic: float
    lag1_autocorr: float
    n_runs: int
    expected_runs: float
    has_systematics: bool
    flag: str  # "OK" or "SYSTEMATICS_DETECTED"


def detect_systematic_noise(flux: list[float]) -> SystematicNoiseResult:
    """Detect systematics in a flux time series.

    Computes:
    - Durbin-Watson statistic: DW = sum((r[i]-r[i-1])^2) / sum(r[i]^2)
    - Lag-1 autocorrelation: sum(r[i]*r[i-1]) / sum(r[i]^2)
    - Runs test: count sign runs around median vs expected N/2 ± sqrt(N)/2

    Args:
        flux: Flux values (normalised or raw; mean-subtracted internally).

    Returns:
        SystematicNoiseResult.
    """
    if len(flux) < 4:
        return SystematicNoiseResult(
            dw_statistic=2.0,
            lag1_autocorr=0.0,
            n_runs=0,
            expected_runs=0.0,
            has_systematics=False,
            flag="OK",
        )

    n = len(flux)
    mean_f = sum(flux) / n
    r = [f - mean_f for f in flux]

    # Durbin-Watson statistic
    sum_diff_sq = sum((r[i] - r[i - 1]) ** 2 for i in range(1, n))
    sum_r_sq = sum(ri ** 2 for ri in r)
    dw = sum_diff_sq / sum_r_sq if sum_r_sq > 0 else 2.0

    # Lag-1 autocorrelation
    lag1 = sum(r[i] * r[i - 1] for i in range(1, n)) / sum_r_sq if sum_r_sq > 0 else 0.0

    # Runs test: count sign runs above/below median
    med = sorted(flux)[n // 2]
    signs = [1 if f >= med else -1 for f in flux]

    n_runs = 1
    for i in range(1, n):
        if signs[i] != signs[i - 1]:
            n_runs += 1

    expected_runs = n / 2.0
    tolerance = math.sqrt(n) / 2.0

    # Systematics: DW < 1.5 (positive autocorr) OR lag1 > 0.3 OR runs far from expected
    has_systematics = (
        dw < 1.5
        or lag1 > 0.3
        or n_runs < (expected_runs - tolerance)
    )
    flag = "SYSTEMATICS_DETECTED" if has_systematics else "OK"

    return SystematicNoiseResult(
        dw_statistic=round(dw, 4),
        lag1_autocorr=round(lag1, 4),
        n_runs=n_runs,
        expected_runs=round(expected_runs, 2),
        has_systematics=has_systematics,
        flag=flag,
    )


def format_systematic_noise(result: SystematicNoiseResult) -> str:
    """Format systematic noise detection as Markdown.

    Args:
        result: SystematicNoiseResult to format.

    Returns:
        Markdown string.
    """
    lines = [
        "## Systematic Noise Detection",
        "",
        f"- Durbin-Watson: {result.dw_statistic:.4f} (ideal ≈ 2.0)",
        f"- Lag-1 autocorrelation: {result.lag1_autocorr:.4f}",
        f"- Runs: {result.n_runs} (expected ≈ {result.expected_runs:.1f})",
        f"- Has systematics: {'**yes**' if result.has_systematics else 'no'}",
        f"- Status: `{result.flag}`",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Detect systematics in flux series.")
    parser.add_argument("flux_json", help="JSON file with a list of flux values.")
    args = parser.parse_args(argv)

    flux = json.loads(Path(args.flux_json).read_text())
    result = detect_systematic_noise(flux)
    print(format_systematic_noise(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
