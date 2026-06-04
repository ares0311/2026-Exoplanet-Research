"""Detect stellar flares in a TESS light curve using sigma-clipping."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FlareEvent:
    start_index: int
    peak_index: int
    end_index: int
    peak_flux_excess: float
    duration_cadences: int
    energy_proxy: float


@dataclass(frozen=True)
class FlareDetectionResult:
    n_flares: int
    flares: tuple[FlareEvent, ...]
    baseline_rms: float
    sigma_threshold: float
    flag: str


def detect_flares(
    flux: list[float],
    sigma_threshold: float = 3.0,
    min_duration_cadences: int = 2,
) -> FlareDetectionResult:
    """
    Detect stellar flares from normalised flux time series.

    Algorithm:
    1. Compute median and robust RMS (1.4826 x MAD).
    2. Flag cadences where flux > median + sigma_threshold x RMS.
    3. Group consecutive flagged cadences; keep runs >= min_duration_cadences.
    """
    n = len(flux)
    if n < 10:
        return FlareDetectionResult(
            n_flares=0, flares=(), baseline_rms=float("nan"),
            sigma_threshold=sigma_threshold, flag="INSUFFICIENT_DATA",
        )

    finite = [f for f in flux if math.isfinite(f)]
    if len(finite) < 5:
        return FlareDetectionResult(
            n_flares=0, flares=(), baseline_rms=float("nan"),
            sigma_threshold=sigma_threshold, flag="INSUFFICIENT_FINITE",
        )

    sorted_f = sorted(finite)
    mid = len(sorted_f) // 2
    median = sorted_f[mid] if len(sorted_f) % 2 else (sorted_f[mid - 1] + sorted_f[mid]) / 2.0
    mad = sorted([abs(f - median) for f in finite])[len(finite) // 2]
    rms = 1.4826 * mad if mad > 0 else 1e-9

    threshold = median + sigma_threshold * rms
    flagged = [i for i, f in enumerate(flux) if math.isfinite(f) and f > threshold]

    events: list[FlareEvent] = []
    if flagged:
        groups: list[list[int]] = [[flagged[0]]]
        for idx in flagged[1:]:
            if idx == groups[-1][-1] + 1:
                groups[-1].append(idx)
            else:
                groups.append([idx])

        for grp in groups:
            if len(grp) < min_duration_cadences:
                continue
            peak_idx = max(grp, key=lambda i: flux[i])
            peak_excess = flux[peak_idx] - median
            energy = sum(flux[i] - median for i in grp if math.isfinite(flux[i]))
            events.append(FlareEvent(
                start_index=grp[0],
                peak_index=peak_idx,
                end_index=grp[-1],
                peak_flux_excess=round(peak_excess, 6),
                duration_cadences=len(grp),
                energy_proxy=round(energy, 6),
            ))

    return FlareDetectionResult(
        n_flares=len(events),
        flares=tuple(events),
        baseline_rms=round(rms, 6),
        sigma_threshold=sigma_threshold,
        flag="OK",
    )


def format_flare_result(r: FlareDetectionResult) -> str:
    def _f(v: float) -> str:
        return f"{v:.6f}" if math.isfinite(v) else "N/A"

    header = (
        f"**Flare Detection** -- {r.n_flares} flare(s), "
        f"baseline RMS={_f(r.baseline_rms)}, threshold={r.sigma_threshold}sigma\n\n"
    )
    if not r.flares:
        return header + "No flares detected.\n"
    lines = [
        header,
        "| # | Start | Peak | End | Peak excess | Duration | Energy proxy |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, ev in enumerate(r.flares, 1):
        lines.append(
            f"| {i} | {ev.start_index} | {ev.peak_index} | {ev.end_index} | "
            f"{ev.peak_flux_excess:.6f} | {ev.duration_cadences} | {ev.energy_proxy:.6f} |"
        )
    return "\n".join(lines)


def _cli() -> int:
    p = argparse.ArgumentParser(description="Detect stellar flares in normalised flux.")
    p.add_argument("flux_json", help="JSON array of flux values")
    p.add_argument("--sigma-threshold", type=float, default=3.0)
    p.add_argument("--min-duration-cadences", type=int, default=2)
    args = p.parse_args()
    import json
    flux = json.loads(args.flux_json)
    r = detect_flares(flux, args.sigma_threshold, args.min_duration_cadences)
    print(format_flare_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
