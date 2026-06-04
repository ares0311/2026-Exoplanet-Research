"""Detect stellar flares in a TESS light curve using sigma-clipping."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FlareDetectionResult:
    n_flares: int
    n_cadences: int
    flare_indices: tuple[tuple[int, int], ...]  # (start, end) per flare
    max_amplitude: float | None          # peak flux excess above baseline
    total_flare_energy_proxy: float      # sum of flux excess over all flare cadences
    baseline_rms: float
    flag: str


def detect_stellar_flares(
    time: list[float],
    flux: list[float],
    flux_err: list[float] | None = None,
    sigma_threshold: float = 3.0,
    min_duration_cadences: int = 2,
) -> FlareDetectionResult:
    """
    Detect stellar flares from a normalised light curve.

    Algorithm:
    1. Validate inputs; compute robust baseline (median + 1.4826*MAD RMS).
    2. Flag cadences where flux > baseline + sigma_threshold * rms.
    3. Group consecutive flagged cadences; keep groups >= min_duration_cadences.

    Parameters
    ----------
    time:                   Array of observation times (any units).
    flux:                   Normalised flux array (same length as time).
    flux_err:               Optional flux uncertainties (used if provided).
    sigma_threshold:        Detection threshold in units of baseline RMS.
    min_duration_cadences:  Minimum run length to count as a flare.
    """
    n = len(flux)
    if n == 0 or len(time) != n:
        return FlareDetectionResult(
            n_flares=0, n_cadences=0, flare_indices=(),
            max_amplitude=None, total_flare_energy_proxy=0.0,
            baseline_rms=float("nan"), flag="INVALID",
        )
    if n < 5:
        return FlareDetectionResult(
            n_flares=0, n_cadences=n, flare_indices=(),
            max_amplitude=None, total_flare_energy_proxy=0.0,
            baseline_rms=float("nan"), flag="INSUFFICIENT",
        )

    finite = [f for f in flux if math.isfinite(f)]
    if len(finite) < 5:
        return FlareDetectionResult(
            n_flares=0, n_cadences=n, flare_indices=(),
            max_amplitude=None, total_flare_energy_proxy=0.0,
            baseline_rms=float("nan"), flag="INSUFFICIENT",
        )

    sorted_f = sorted(finite)
    mid = len(sorted_f) // 2
    median = sorted_f[mid] if len(sorted_f) % 2 else (sorted_f[mid - 1] + sorted_f[mid]) / 2.0
    mad = sorted([abs(f - median) for f in finite])[len(finite) // 2]
    rms = 1.4826 * mad if mad > 0 else 1e-9

    if flux_err is not None and len(flux_err) == n:
        finite_err = [e for e in flux_err if math.isfinite(e) and e > 0]
        if finite_err:
            rms = max(rms, sum(finite_err) / len(finite_err))

    threshold = median + sigma_threshold * rms
    flagged = [i for i, f in enumerate(flux) if math.isfinite(f) and f > threshold]

    if not flagged:
        return FlareDetectionResult(
            n_flares=0, n_cadences=n, flare_indices=(),
            max_amplitude=None, total_flare_energy_proxy=0.0,
            baseline_rms=round(rms, 6), flag="NO_FLARES",
        )

    groups: list[list[int]] = [[flagged[0]]]
    for idx in flagged[1:]:
        if idx == groups[-1][-1] + 1:
            groups[-1].append(idx)
        else:
            groups.append([idx])

    valid_groups = [g for g in groups if len(g) >= min_duration_cadences]
    if not valid_groups:
        return FlareDetectionResult(
            n_flares=0, n_cadences=n, flare_indices=(),
            max_amplitude=None, total_flare_energy_proxy=0.0,
            baseline_rms=round(rms, 6), flag="NO_FLARES",
        )

    flare_indices = tuple((g[0], g[-1]) for g in valid_groups)
    amplitudes = [
        max(flux[i] - median for i in g if math.isfinite(flux[i]))
        for g in valid_groups
    ]
    max_amp = max(amplitudes) if amplitudes else None
    total_energy = sum(
        flux[i] - median
        for g in valid_groups
        for i in g
        if math.isfinite(flux[i]) and flux[i] > median
    )

    return FlareDetectionResult(
        n_flares=len(valid_groups),
        n_cadences=n,
        flare_indices=flare_indices,
        max_amplitude=round(max_amp, 6) if max_amp is not None else None,
        total_flare_energy_proxy=round(total_energy, 6),
        baseline_rms=round(rms, 6),
        flag="OK",
    )


def format_flare_result(r: FlareDetectionResult) -> str:
    def _f(v: float | None, fmt: str = ".6f") -> str:
        if v is None:
            return "N/A"
        return format(v, fmt) if math.isfinite(v) else "N/A"

    lines = [
        "| Parameter | Value |\n|---|---|",
        f"| N flares | {r.n_flares} |",
        f"| N cadences | {r.n_cadences} |",
        f"| Max amplitude | {_f(r.max_amplitude)} |",
        f"| Total flare energy proxy | {_f(r.total_flare_energy_proxy)} |",
        f"| Baseline RMS | {_f(r.baseline_rms)} |",
        f"| Insolation flag | {r.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    p = argparse.ArgumentParser(description="Detect stellar flares in normalised flux.")
    p.add_argument("flux_json", help="JSON array of flux values")
    p.add_argument("--sigma-threshold", type=float, default=3.0)
    p.add_argument("--min-duration-cadences", type=int, default=2)
    args = p.parse_args()
    import json
    flux = json.loads(args.flux_json)
    time = list(range(len(flux)))
    r = detect_stellar_flares(time, flux, sigma_threshold=args.sigma_threshold,
                               min_duration_cadences=args.min_duration_cadences)
    print(format_flare_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
