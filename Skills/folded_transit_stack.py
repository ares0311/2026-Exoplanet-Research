"""Stack individual transit windows after phase-alignment to improve SNR.

Extracts a window around each known transit mid-time, phase-aligns them to a
common time axis, and combines them into a single stacked profile.  The
stacking improves the effective SNR by approximately √N_transits.

Public API
----------
StackedTransit(phase_bins, flux_mean, flux_std, flux_sem, n_transits,
               snr_estimate, flag)
stack_transit_windows(time, flux, midpoints, *,
                      half_width_days, n_bins) -> StackedTransit
format_stack_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class StackedTransit:
    phase_bins: tuple[float, ...]    # bin centres in days relative to mid-time
    flux_mean: tuple[float, ...]     # mean flux per bin
    flux_std: tuple[float, ...]      # standard deviation per bin (0 if 1 point)
    flux_sem: tuple[float, ...]      # standard error of the mean per bin
    n_transits: int                  # number of windows stacked
    snr_estimate: float | None       # depth / RMS(OOT), None if OOT empty
    flag: str  # "OK" | "INSUFFICIENT" | "INVALID"


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def stack_transit_windows(
    time: list[float],
    flux: list[float],
    midpoints: list[float],
    *,
    half_width_days: float = 0.5,
    n_bins: int = 50,
) -> StackedTransit:
    """Stack individual transit windows into a combined profile.

    Args:
        time: Time array (days), sorted ascending.
        flux: Flux array, same length as time.
        midpoints: Transit mid-times (BJD).
        half_width_days: Half-width of extraction window around each mid-time.
        n_bins: Number of phase bins in the stacked profile.

    Returns:
        :class:`StackedTransit`.
    """
    if len(time) != len(flux) or len(time) < 2:
        return StackedTransit((), (), (), (), 0, None, "INVALID")
    if not midpoints:
        return StackedTransit((), (), (), (), 0, None, "INSUFFICIENT")
    if half_width_days <= 0 or n_bins < 2:
        return StackedTransit((), (), (), (), 0, None, "INVALID")

    hw = half_width_days
    bin_width = 2.0 * hw / n_bins
    # bin centres: symmetric around 0
    bin_centres = [-hw + (i + 0.5) * bin_width for i in range(n_bins)]

    # Accumulate flux values per bin
    bins: list[list[float]] = [[] for _ in range(n_bins)]
    n_used = 0

    for mid in midpoints:
        lo = mid - hw
        hi = mid + hw
        # collect time indices in window
        added = False
        for t, f in zip(time, flux, strict=False):
            if lo <= t <= hi:
                phase = t - mid  # relative to mid-time
                bi = int((phase + hw) / bin_width)
                bi = max(0, min(n_bins - 1, bi))
                bins[bi].append(f)
                added = True
        if added:
            n_used += 1

    if n_used == 0:
        return StackedTransit((), (), (), (), 0, None, "INSUFFICIENT")

    # Compute statistics per bin
    means: list[float] = []
    stds: list[float] = []
    sems: list[float] = []
    valid_centres: list[float] = []

    for i, bvals in enumerate(bins):
        if not bvals:
            # empty bin — use NaN-equivalent (we'll just skip in SNR)
            means.append(float("nan"))
            stds.append(float("nan"))
            sems.append(float("nan"))
            valid_centres.append(bin_centres[i])
            continue
        n = len(bvals)
        mean = sum(bvals) / n
        variance = sum((v - mean) ** 2 for v in bvals) / n if n > 1 else 0.0
        std = math.sqrt(variance)
        sem = std / math.sqrt(n) if n > 1 else std
        means.append(mean)
        stds.append(std)
        sems.append(sem)
        valid_centres.append(bin_centres[i])

    # SNR estimate: depth / OOT RMS
    # OOT = outer 25% of bins on each side
    oot_n = max(1, n_bins // 8)
    oot_vals: list[float] = []
    for i in range(oot_n):
        v = means[i]
        if not math.isnan(v):
            oot_vals.append(v)
    for i in range(n_bins - oot_n, n_bins):
        v = means[i]
        if not math.isnan(v):
            oot_vals.append(v)

    snr_estimate: float | None = None
    if oot_vals:
        oot_mean = sum(oot_vals) / len(oot_vals)
        oot_rms = math.sqrt(sum((v - oot_mean) ** 2 for v in oot_vals) / len(oot_vals))
        # depth = 1 - min(mean) relative to OOT level
        finite_means = [v for v in means if not math.isnan(v)]
        if finite_means and oot_rms > 0:
            min_flux = min(finite_means)
            depth = oot_mean - min_flux
            snr_estimate = round(depth / oot_rms, 4) if depth > 0 else 0.0

    return StackedTransit(
        phase_bins=tuple(round(c, 8) for c in valid_centres),
        flux_mean=tuple(round(v, 8) if not math.isnan(v) else float("nan") for v in means),
        flux_std=tuple(round(v, 8) if not math.isnan(v) else float("nan") for v in stds),
        flux_sem=tuple(round(v, 8) if not math.isnan(v) else float("nan") for v in sems),
        n_transits=n_used,
        snr_estimate=snr_estimate,
        flag="OK",
    )


def format_stack_result(result: StackedTransit) -> str:
    """Format stacked transit result as Markdown."""
    lines = [
        "## Folded Transit Stack",
        "",
        f"- Transits stacked: {result.n_transits}",
        f"- Phase bins: {len(result.phase_bins)}",
        f"- SNR estimate (depth/OOT RMS): {result.snr_estimate}",
        f"- **Flag: {result.flag}**",
    ]
    if result.phase_bins:
        lines += [
            "",
            "| Phase (d) | Mean flux | SEM |",
            "|-----------|-----------|-----|",
        ]
        step = max(1, len(result.phase_bins) // 10)
        for i in range(0, len(result.phase_bins), step):
            ph = result.phase_bins[i]
            mn = result.flux_mean[i]
            se = result.flux_sem[i]
            mn_s = f"{mn:.6f}" if not math.isnan(mn) else "—"
            se_s = f"{se:.6f}" if not math.isnan(se) else "—"
            lines.append(f"| {ph:.4f} | {mn_s} | {se_s} |")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="folded_transit_stack",
        description="Stack phase-aligned transit windows to improve SNR.",
    )
    parser.add_argument("--half-width", type=float, default=0.5)
    parser.add_argument("--n-bins", type=int, default=50)
    args = parser.parse_args(argv)

    result = stack_transit_windows([], [], [], half_width_days=args.half_width, n_bins=args.n_bins)
    print(format_stack_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
