"""Map secondary eclipse depth and phase from a phase-folded light curve.

Searches the out-of-transit light curve at phase 0.5 for a secondary eclipse
signal and estimates its depth and significance.

Public API
----------
SecondaryEclipseResult(depth_ppm, err_ppm, phase, snr, is_detected, flag)
map_secondary_eclipse(time, flux, period, epoch, *, flux_err, duration_days,
                      search_phase_min, search_phase_max) -> SecondaryEclipseResult
format_secondary_eclipse_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SecondaryEclipseResult:
    depth_ppm: float | None
    err_ppm: float | None
    phase: float | None        # best-fit phase of eclipse (0.0–1.0)
    snr: float | None
    is_detected: bool
    flag: str                  # "DETECTED", "NOT_DETECTED", "INSUFFICIENT"


def map_secondary_eclipse(
    time: list[float],
    flux: list[float],
    period: float,
    epoch: float,
    *,
    flux_err: list[float] | None = None,
    duration_days: float = 0.1,
    search_phase_min: float = 0.4,
    search_phase_max: float = 0.6,
    detection_snr_threshold: float = 3.0,
) -> SecondaryEclipseResult:
    """Search for a secondary eclipse near phase 0.5.

    Args:
        time: BJD time array.
        flux: Normalised flux array.
        period: Orbital period in days.
        epoch: Primary transit mid-time (BJD).
        flux_err: Per-cadence flux uncertainties.
        duration_days: Window half-width for eclipse search.
        search_phase_min: Start of search phase range.
        search_phase_max: End of search phase range.
        detection_snr_threshold: SNR threshold for DETECTED flag.

    Returns:
        :class:`SecondaryEclipseResult`.
    """
    if not time or period <= 0:
        return SecondaryEclipseResult(None, None, None, None, False, "INSUFFICIENT")

    t_arr = [float(t) for t in time]
    f_arr = [float(f) for f in flux]
    e_arr = [float(e) for e in flux_err] if flux_err is not None else None

    # Compute phase for each cadence
    phases = [((t - epoch) % period) / period for t in t_arr]

    # OOT baseline (phase < 0.1 or > 0.9)
    oot_f = [f for ph, f in zip(phases, f_arr, strict=False) if ph < 0.1 or ph > 0.9]
    if not oot_f:
        return SecondaryEclipseResult(None, None, None, None, False, "INSUFFICIENT")
    baseline = sum(oot_f) / len(oot_f)

    # Grid search across secondary eclipse phase candidates
    half_phase = duration_days / period / 2.0
    n_steps = max(10, int((search_phase_max - search_phase_min) / half_phase))
    step = (search_phase_max - search_phase_min) / n_steps

    best_depth = 0.0
    best_phase = 0.5
    best_snr = 0.0
    best_err = None

    for i in range(n_steps + 1):
        test_phase = search_phase_min + i * step
        lo = test_phase - half_phase
        hi = test_phase + half_phase

        in_f = [f for ph, f in zip(phases, f_arr, strict=False) if lo <= ph <= hi]
        in_e = (
            [e for ph, e in zip(phases, e_arr, strict=False) if lo <= ph <= hi]
            if e_arr is not None else None
        )
        if len(in_f) < 2:
            continue

        mean_f = sum(in_f) / len(in_f)
        depth = (baseline - mean_f) * 1e6
        if depth <= 0:
            continue

        if in_e and len(in_e) == len(in_f):
            err = (sum(e ** 2 for e in in_e) ** 0.5 / len(in_e)) * 1e6
        else:
            # Use OOT scatter as photon-noise proxy; fall back to in-eclipse scatter
            if len(oot_f) > 1:
                oot_mean = sum(oot_f) / len(oot_f)
                oot_sq = sum((f - oot_mean) ** 2 for f in oot_f)
                oot_rms = math.sqrt(oot_sq / (len(oot_f) - 1)) * 1e6
                err = oot_rms / math.sqrt(len(in_f)) if len(in_f) > 0 else oot_rms
            elif len(in_f) > 1:
                sq = sum((f - mean_f) ** 2 for f in in_f)
                err = math.sqrt(sq / (len(in_f) - 1)) / math.sqrt(len(in_f)) * 1e6
            else:
                err = 0.0
            if err == 0.0:
                err = abs(depth) * 0.1 or 1.0

        snr = depth / err if err > 0 else 0.0
        if snr > best_snr:
            best_snr = snr
            best_depth = depth
            best_phase = test_phase
            best_err = err

    if best_snr == 0.0:
        return SecondaryEclipseResult(None, None, None, None, False, "NOT_DETECTED")

    is_detected = best_snr >= detection_snr_threshold
    flag = "DETECTED" if is_detected else "NOT_DETECTED"

    return SecondaryEclipseResult(
        depth_ppm=round(best_depth, 2),
        err_ppm=round(best_err, 2) if best_err is not None else None,
        phase=round(best_phase, 4),
        snr=round(best_snr, 3),
        is_detected=is_detected,
        flag=flag,
    )


def format_secondary_eclipse_result(result: SecondaryEclipseResult) -> str:
    """Format secondary eclipse result as Markdown."""
    lines = ["## Secondary Eclipse Search", ""]
    if result.flag == "INSUFFICIENT":
        lines.append("- Insufficient data for secondary eclipse search.")
    elif result.flag == "NOT_DETECTED":
        lines.append("- No secondary eclipse detected above threshold.")
        if result.snr is not None:
            lines.append(f"- Best SNR found: {result.snr:.2f}")
    else:
        err_str = f" ± {result.err_ppm:.1f}" if result.err_ppm is not None else ""
        lines += [
            "- **Secondary eclipse DETECTED**",
            f"- Depth: {result.depth_ppm:.1f}{err_str} ppm",
            f"- Phase: {result.phase:.4f}",
            f"- SNR: {result.snr:.2f}",
            f"- Flag: **{result.flag}**",
        ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="secondary_eclipse_mapper",
        description="Search for secondary eclipse in phase-folded light curve.",
    )
    parser.add_argument("--lc", required=True, metavar="JSON")
    parser.add_argument("--period", type=float, required=True)
    parser.add_argument("--epoch", type=float, required=True)
    parser.add_argument("--duration", type=float, default=0.1)
    args = parser.parse_args(argv)

    lc = json.loads(Path(args.lc).read_text())
    result = map_secondary_eclipse(
        lc["time"], lc["flux"], args.period, args.epoch,
        duration_days=args.duration,
    )
    print(format_secondary_eclipse_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
