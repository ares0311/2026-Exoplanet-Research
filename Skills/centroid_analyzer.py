"""Analyze centroid motion during transit as a background EB discriminator.

During a genuine planet transit the photometric centroid should not shift.
A significant shift (arcsec) toward or away from the target during transit
is strong evidence for a background eclipsing binary (bgEB) contaminating
the aperture.

Public API
----------
CentroidResult(in_transit_centroid_arcsec, out_transit_centroid_arcsec,
               delta_arcsec, delta_significance, is_shifted, note)
analyze_centroid(time, flux, centroid_ra, centroid_dec,
                 period, epoch, *, duration_days, threshold_arcsec) -> CentroidResult
format_centroid_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CentroidResult:
    in_transit_centroid_arcsec: float   # mean centroid offset during transit (arcsec)
    out_transit_centroid_arcsec: float  # mean centroid offset out-of-transit
    delta_arcsec: float                 # |in − out|
    delta_significance: float           # delta / noise (sigma)
    is_shifted: bool
    note: str = ""


def _phase_fold(
    time: list[float],
    period: float,
    epoch: float,
) -> list[float]:
    phases = []
    for t in time:
        ph = (t - epoch) % period
        if ph > period / 2:
            ph -= period
        phases.append(ph)
    return phases


def analyze_centroid(
    time: list[float],
    flux: list[float],
    centroid_ra: list[float],
    centroid_dec: list[float],
    period: float,
    epoch: float,
    *,
    duration_days: float = 0.1,
    threshold_arcsec: float = 1.0,
    plate_scale_arcsec_per_pixel: float = 21.0,  # TESS: 21"/pixel
) -> CentroidResult:
    """Compute centroid shift in/out of transit.

    Args:
        time: BJD time array.
        flux: Normalised flux (used only for phasing).
        centroid_ra: Per-cadence RA centroid (degrees or pixels).
        centroid_dec: Per-cadence Dec centroid (degrees or pixels).
        period: Transit period in days.
        epoch: Mid-transit BJD epoch.
        duration_days: Transit duration (half-width in days used for masking).
        threshold_arcsec: Delta above which is_shifted=True.
        plate_scale_arcsec_per_pixel: Scale for converting pixels → arcsec
            when centroid arrays are in pixels.

    Returns:
        :class:`CentroidResult`.
    """
    if len(time) < 4:
        return CentroidResult(0.0, 0.0, 0.0, 0.0, False, "insufficient data")

    phases = _phase_fold(time, period, epoch)
    half = duration_days / 2

    in_ra, in_dec, out_ra, out_dec = [], [], [], []
    for ph, ra, dec in zip(phases, centroid_ra, centroid_dec, strict=False):
        if abs(ph) <= half:
            in_ra.append(ra)
            in_dec.append(dec)
        else:
            out_ra.append(ra)
            out_dec.append(dec)

    def _mean(lst: list[float]) -> float:
        return sum(lst) / len(lst) if lst else 0.0

    def _std(lst: list[float]) -> float:
        if len(lst) < 2:
            return 0.0
        m = _mean(lst)
        var = sum((x - m) ** 2 for x in lst) / (len(lst) - 1)
        return math.sqrt(var)

    in_ra_m  = _mean(in_ra)
    in_dec_m = _mean(in_dec)
    out_ra_m  = _mean(out_ra)
    out_dec_m = _mean(out_dec)

    # Compute angular offset (same units as centroid inputs)
    delta_raw = math.hypot(in_ra_m - out_ra_m, in_dec_m - out_dec_m)

    # Estimate per-axis noise from out-of-transit scatter
    noise_ra  = _std(out_ra)
    noise_dec = _std(out_dec)
    noise = math.hypot(noise_ra, noise_dec) / max(math.sqrt(max(len(out_ra), 1)), 1)

    # Convert to arcsec if inputs look like pixels (> 1 is typical for arcsec already)
    # Heuristic: if values are small (< 0.01) assume arcsec × 3600 needed; else pixels
    if delta_raw < 0.01 and plate_scale_arcsec_per_pixel > 0:
        # likely degrees — convert to arcsec
        delta_arcsec = delta_raw * 3600.0
        noise_arcsec = noise * 3600.0
        in_arcsec    = math.hypot(in_ra_m, in_dec_m) * 3600.0
        out_arcsec   = math.hypot(out_ra_m, out_dec_m) * 3600.0
    elif delta_raw >= 1.0:
        # likely pixels — convert via plate scale
        delta_arcsec = delta_raw * plate_scale_arcsec_per_pixel
        noise_arcsec = noise * plate_scale_arcsec_per_pixel
        in_arcsec    = math.hypot(in_ra_m, in_dec_m) * plate_scale_arcsec_per_pixel
        out_arcsec   = math.hypot(out_ra_m, out_dec_m) * plate_scale_arcsec_per_pixel
    else:
        delta_arcsec = delta_raw
        noise_arcsec = noise
        in_arcsec    = math.hypot(in_ra_m, in_dec_m)
        out_arcsec   = math.hypot(out_ra_m, out_dec_m)

    significance = delta_arcsec / max(noise_arcsec, 1e-9)
    is_shifted   = delta_arcsec > threshold_arcsec

    note = ""
    if not in_ra:
        note = "no in-transit cadences"
    elif is_shifted:
        note = f"centroid shift {delta_arcsec:.2f}\" suggests background source"

    return CentroidResult(
        in_transit_centroid_arcsec=in_arcsec,
        out_transit_centroid_arcsec=out_arcsec,
        delta_arcsec=delta_arcsec,
        delta_significance=significance,
        is_shifted=is_shifted,
        note=note,
    )


def format_centroid_result(result: CentroidResult) -> str:
    """Format centroid analysis as a short Markdown block."""
    flag = "SHIFTED" if result.is_shifted else "OK"
    lines = [
        "## Centroid Analysis",
        "",
        f"- In-transit centroid offset: {result.in_transit_centroid_arcsec:.3f}\"",
        f"- Out-of-transit centroid:    {result.out_transit_centroid_arcsec:.3f}\"",
        f"- Delta: {result.delta_arcsec:.3f}\" ({result.delta_significance:.1f}σ) — {flag}",
    ]
    if result.note:
        lines.append(f"- Note: {result.note}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="centroid_analyzer",
        description="Analyze centroid shift during transit.",
    )
    parser.add_argument("--lc", required=True, metavar="JSON",
                        help="JSON with keys: time, flux, centroid_ra, centroid_dec.")
    parser.add_argument("--period", type=float, required=True)
    parser.add_argument("--epoch", type=float, required=True)
    parser.add_argument("--duration", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=1.0, metavar="ARCSEC")
    args = parser.parse_args(argv)

    lc = json.loads(Path(args.lc).read_text())
    result = analyze_centroid(
        lc["time"], lc["flux"],
        lc["centroid_ra"], lc["centroid_dec"],
        args.period, args.epoch,
        duration_days=args.duration,
        threshold_arcsec=args.threshold,
    )
    print(format_centroid_result(result))
    return 1 if result.is_shifted else 0


if __name__ == "__main__":
    raise SystemExit(_cli())
