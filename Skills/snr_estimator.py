"""Estimate the per-transit and combined transit signal-to-noise ratio.

Uses the box-signal approximation:
  SNR_single = depth / (σ_OOT / sqrt(n_in))
  SNR_combined = SNR_single × sqrt(n_transits)

Public API
----------
SNRResult(depth_ppm, rms_ppm, n_in_transit, n_transits,
          snr_single, snr_combined, flag)
estimate_snr(time, flux, period_days, epoch_bjd, *,
             duration_days, flux_err) -> SNRResult
format_snr_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SNRResult:
    depth_ppm: float
    rms_ppm: float
    n_in_transit: int
    n_transits: int
    snr_single: float | None
    snr_combined: float | None
    flag: str  # "OK", "INSUFFICIENT"


def estimate_snr(
    time: list[float],
    flux: list[float],
    period_days: float,
    epoch_bjd: float,
    *,
    duration_days: float = 0.1,
    flux_err: list[float] | None = None,
) -> SNRResult:
    """Estimate single-transit and combined SNR.

    Args:
        time: Time array (BJD).
        flux: Normalised flux array.
        period_days: Orbital period in days.
        epoch_bjd: Epoch of primary transit in BJD.
        duration_days: Assumed transit duration in days.
        flux_err: Optional per-point flux uncertainties (normalised units).

    Returns:
        :class:`SNRResult`.
    """
    if not time or not flux or period_days <= 0:
        return SNRResult(0.0, 0.0, 0, 0, None, None, "INSUFFICIENT")

    half = duration_days / 2.0
    in_transit: list[float] = []
    oot: list[float] = []
    transit_numbers: set[int] = set()

    for t, f in zip(time, flux, strict=False):
        ph_abs = (t - epoch_bjd) % period_days
        if ph_abs > period_days / 2:
            ph_abs -= period_days
        if abs(ph_abs) <= half:
            in_transit.append(f)
            n = round((t - epoch_bjd) / period_days)
            transit_numbers.add(n)
        elif abs(ph_abs) > half * 3:
            oot.append(f)

    if not in_transit or not oot:
        return SNRResult(0.0, 0.0, len(in_transit), 0, None, None, "INSUFFICIENT")

    baseline = sum(oot) / len(oot)
    mean_in = sum(in_transit) / len(in_transit)
    depth_ppm = (baseline - mean_in) * 1e6

    # RMS from OOT scatter or from flux_err if provided
    if flux_err is not None and len(flux_err) == len(flux):
        oot_errs = []
        for i, t in enumerate(time):
            ph_abs = (t - epoch_bjd) % period_days
            if ph_abs > period_days / 2:
                ph_abs -= period_days
            if abs(ph_abs) > half * 3:
                oot_errs.append(flux_err[i])
        rms_ppm = (sum(e ** 2 for e in oot_errs) / len(oot_errs)) ** 0.5 * 1e6 if oot_errs else 0.0
    else:
        sq_dev = sum((f - baseline) ** 2 for f in oot)
        rms_ppm = math.sqrt(sq_dev / len(oot)) * 1e6

    n_in = len(in_transit)
    n_transits = len(transit_numbers)

    if rms_ppm <= 0 or n_in == 0:
        return SNRResult(
            round(depth_ppm, 2), round(rms_ppm, 2),
            n_in, n_transits, None, None, "INSUFFICIENT",
        )

    snr_single = depth_ppm / (rms_ppm / math.sqrt(n_in))
    snr_combined = snr_single * math.sqrt(max(n_transits, 1))

    return SNRResult(
        depth_ppm=round(depth_ppm, 2),
        rms_ppm=round(rms_ppm, 2),
        n_in_transit=n_in,
        n_transits=n_transits,
        snr_single=round(snr_single, 3),
        snr_combined=round(snr_combined, 3),
        flag="OK",
    )


def format_snr_result(result: SNRResult) -> str:
    """Format SNR estimate result as Markdown."""
    lines = [
        "## SNR Estimate",
        "",
        f"- Depth: {result.depth_ppm:.2f} ppm",
        f"- OOT RMS: {result.rms_ppm:.2f} ppm",
        f"- In-transit points: {result.n_in_transit}",
        f"- Transits found: {result.n_transits}",
    ]
    if result.flag == "INSUFFICIENT":
        lines.append("- **Flag: INSUFFICIENT** — not enough data")
    else:
        lines += [
            f"- SNR (single transit): {result.snr_single:.3f}",
            f"- SNR (combined): {result.snr_combined:.3f}",
            f"- **Flag: {result.flag}**",
        ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="snr_estimator",
        description="Estimate transit SNR.",
    )
    parser.add_argument("period_days", type=float)
    parser.add_argument("epoch_bjd", type=float)
    parser.add_argument("--duration-days", type=float, default=0.1)
    args = parser.parse_args(argv)

    result = estimate_snr([], [], args.period_days, args.epoch_bjd,
                          duration_days=args.duration_days)
    print(format_snr_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
