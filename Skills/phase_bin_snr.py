"""Bin a phase-folded light curve and compute per-bin SNR.

After phase-folding, bins the flux into equal phase intervals and computes
the signal-to-noise ratio of each bin relative to the out-of-transit scatter.
Identifies which bins contain the transit signal.

Public API
----------
PhaseBin(phase_center, mean_flux, flux_err, snr, is_in_transit)
PhaseBinSNRResult(n_bins, bins, transit_bin_indices, peak_snr,
                  transit_depth_ppm, flag)
compute_phase_bin_snr(phase, flux, *, flux_err, n_bins,
                      transit_half_width) -> PhaseBinSNRResult
format_phase_bin_snr_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PhaseBin:
    phase_center: float
    mean_flux: float
    flux_err: float | None   # standard error of mean
    snr: float | None        # |bin_mean - oot_mean| / oot_scatter
    is_in_transit: bool


@dataclass(frozen=True)
class PhaseBinSNRResult:
    n_bins: int
    bins: tuple[PhaseBin, ...]
    transit_bin_indices: tuple[int, ...]   # indices of bins inside transit_half_width
    peak_snr: float | None
    transit_depth_ppm: float | None
    flag: str  # "OK" | "NO_TRANSIT_BINS" | "INSUFFICIENT" | "INVALID"


def compute_phase_bin_snr(
    phase: list[float],
    flux: list[float],
    *,
    flux_err: list[float] | None = None,
    n_bins: int = 50,
    transit_half_width: float = 0.05,
) -> PhaseBinSNRResult:
    """Bin a phase-folded LC and compute per-bin SNR.

    Args:
        phase: Phase values in [-0.5, 0.5).
        flux: Normalised flux array (same length as phase).
        flux_err: Per-point uncertainties (optional).
        n_bins: Number of equal-width phase bins.
        transit_half_width: Half-width in phase units defining the transit
            window (used to label bins as in-transit).

    Returns:
        :class:`PhaseBinSNRResult`.
    """
    n = len(flux)
    if n < 5 or len(phase) != n or n_bins < 2:
        return PhaseBinSNRResult(0, (), (), None, None, "INVALID")

    bin_width = 1.0 / n_bins
    bins_flux: list[list[float]] = [[] for _ in range(n_bins)]

    for ph, f in zip(phase, flux, strict=False):
        idx = min(int((ph + 0.5) * n_bins), n_bins - 1)
        bins_flux[idx].append(f)

    # Build bin objects (skip empty bins for OOT mean calc)
    bin_means: list[float | None] = []
    bin_errs: list[float | None] = []
    for bf in bins_flux:
        if not bf:
            bin_means.append(None)
            bin_errs.append(None)
        else:
            m = sum(bf) / len(bf)
            variance = sum((f - m) ** 2 for f in bf) / len(bf)
            se = math.sqrt(variance) / math.sqrt(len(bf)) if len(bf) > 1 else None
            bin_means.append(m)
            bin_errs.append(se)

    # OOT mean and scatter from bins outside transit window
    oot_vals: list[float] = []
    for i, m in enumerate(bin_means):
        if m is None:
            continue
        ph_center = (i + 0.5) * bin_width - 0.5
        if abs(ph_center) > transit_half_width:
            oot_vals.append(m)

    if not oot_vals:
        return PhaseBinSNRResult(0, (), (), None, None, "INSUFFICIENT")

    oot_mean = sum(oot_vals) / len(oot_vals)
    oot_rms = math.sqrt(sum((v - oot_mean) ** 2 for v in oot_vals) / len(oot_vals))

    result_bins: list[PhaseBin] = []
    transit_indices: list[int] = []
    peak_snr: float | None = None

    for i in range(n_bins):
        ph_center = (i + 0.5) * bin_width - 0.5
        in_transit = abs(ph_center) <= transit_half_width
        m = bin_means[i]
        se = bin_errs[i]

        snr: float | None = None
        if m is not None and oot_rms > 1e-12:
            snr = round(abs(m - oot_mean) / oot_rms, 3)
            if in_transit and (peak_snr is None or snr > peak_snr):
                peak_snr = snr

        if in_transit:
            transit_indices.append(i)

        result_bins.append(PhaseBin(
            phase_center=round(ph_center, 5),
            mean_flux=round(m, 7) if m is not None else float("nan"),
            flux_err=round(se, 7) if se is not None else None,
            snr=snr,
            is_in_transit=in_transit,
        ))

    if not transit_indices:
        return PhaseBinSNRResult(n_bins, tuple(result_bins), (), None, None, "NO_TRANSIT_BINS")

    # Transit depth: OOT mean minus mean of in-transit bins
    in_means = [result_bins[i].mean_flux for i in transit_indices
                if not math.isnan(result_bins[i].mean_flux)]
    depth_ppm: float | None = None
    if in_means:
        in_mean = sum(in_means) / len(in_means)
        depth_ppm = round((oot_mean - in_mean) * 1e6, 2)

    return PhaseBinSNRResult(
        n_bins=n_bins,
        bins=tuple(result_bins),
        transit_bin_indices=tuple(transit_indices),
        peak_snr=peak_snr,
        transit_depth_ppm=depth_ppm,
        flag="OK",
    )


def format_phase_bin_snr_result(result: PhaseBinSNRResult) -> str:
    """Format phase bin SNR result as Markdown."""
    lines = [
        "## Phase-Bin SNR",
        "",
        f"- Bins: {result.n_bins}",
        f"- Transit bins: {len(result.transit_bin_indices)}",
        f"- Peak SNR: {result.peak_snr}",
        f"- Transit depth: {result.transit_depth_ppm} ppm",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="phase_bin_snr",
        description="Compute per-bin SNR for a phase-folded light curve.",
    )
    parser.add_argument("--n-bins", type=int, default=50)
    parser.add_argument("--half-width", type=float, default=0.05)
    args = parser.parse_args(argv)

    result = compute_phase_bin_snr([], [], n_bins=args.n_bins,
                                   transit_half_width=args.half_width)
    print(format_phase_bin_snr_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
