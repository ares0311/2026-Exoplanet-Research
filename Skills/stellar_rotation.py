"""Detect stellar rotation period and flag flares in TESS/Kepler light curves.

Uses a Lomb-Scargle periodogram on out-of-transit flux to measure the rotation
period, and a simple sigma-clip flare detector.

Public API
----------
RotationResult(rotation_period_days, rotation_power, is_significant,
               n_flares, flare_times, flare_amplitudes)
detect_rotation(time, flux, *, flux_err, period_range, fap_threshold) -> RotationResult
format_rotation_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RotationResult:
    rotation_period_days: float | None  # best LS period; None if not significant
    rotation_power: float               # LS peak power (0–1)
    false_alarm_probability: float      # FAP at peak
    is_significant: bool                # FAP < threshold
    n_flares: int
    flare_times: tuple[float, ...]      # BJD of flare peaks
    flare_amplitudes: tuple[float, ...] # amplitude above quiescent level


def _lomb_scargle(
    time: list[float],
    flux: list[float],
    flux_err: list[float] | None,
    period_min: float,
    period_max: float,
    n_freqs: int = 5000,
) -> tuple[float, float, float]:
    """Return (best_period, peak_power, fap)."""
    import numpy as np
    from astropy.timeseries import LombScargle

    t = np.asarray(time, dtype=float)
    raw_f = np.asarray(flux, dtype=float)
    finite_mask = np.isfinite(t) & np.isfinite(raw_f)
    e = None
    if flux_err is not None:
        e = np.asarray(flux_err, dtype=float)
        finite_mask &= np.isfinite(e) & (e > 0.0)

    t = t[finite_mask]
    raw_f = raw_f[finite_mask]
    if e is not None:
        e = e[finite_mask]
    if len(t) < 10:
        return 0.0, 0.0, 1.0

    f = raw_f - float(np.median(raw_f))
    if float(np.var(f)) <= 0.0:
        return 0.0, 0.0, 1.0

    freq_min = 1.0 / period_max
    freq_max = 1.0 / period_min
    freqs = np.linspace(freq_min, freq_max, n_freqs)

    ls = LombScargle(t, f, e)
    power = np.nan_to_num(ls.power(freqs), nan=0.0, posinf=0.0, neginf=0.0)
    best_idx = int(np.argmax(power))
    best_freq = float(freqs[best_idx])
    best_power = min(max(float(power[best_idx]), 0.0), 1.0)
    fap_power = min(best_power, float(np.nextafter(1.0, 0.0)))
    fap = float(ls.false_alarm_probability(fap_power))
    if not np.isfinite(fap):
        fap = 1.0
    return 1.0 / best_freq, best_power, fap


def _detect_flares(
    time: list[float],
    flux: list[float],
    sigma_threshold: float = 4.0,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Flag cadences above median + threshold * MAD as flares."""
    if not flux:
        return (), ()
    sorted_f = sorted(flux)
    med = sorted_f[len(sorted_f) // 2]
    mad = sorted(abs(f - med) for f in flux)[len(flux) // 2]
    mad_scaled = max(mad * 1.4826, 1e-9)

    flare_t, flare_a = [], []
    for t, f in zip(time, flux, strict=False):
        if f > med + sigma_threshold * mad_scaled:
            flare_t.append(t)
            flare_a.append(f - med)

    # Merge consecutive flare cadences (group within 0.1 d)
    merged_t: list[float] = []
    merged_a: list[float] = []
    for t, a in zip(flare_t, flare_a, strict=False):
        if merged_t and t - merged_t[-1] < 0.1:
            if a > merged_a[-1]:
                merged_t[-1] = t
                merged_a[-1] = a
        else:
            merged_t.append(t)
            merged_a.append(a)

    return tuple(merged_t), tuple(merged_a)


def detect_rotation(
    time: list[float],
    flux: list[float],
    *,
    flux_err: list[float] | None = None,
    period_range: tuple[float, float] = (0.5, 30.0),
    fap_threshold: float = 0.01,
    flare_sigma: float = 4.0,
) -> RotationResult:
    """Detect stellar rotation period and flares.

    Args:
        time: BJD time array.
        flux: Normalised flux.
        flux_err: Optional per-cadence uncertainties.
        period_range: (min_days, max_days) search range for rotation period.
        fap_threshold: FAP below which the rotation is considered significant.
        flare_sigma: Sigma threshold for flare detection.

    Returns:
        :class:`RotationResult`.
    """
    if len(time) < 10:
        return RotationResult(
            rotation_period_days=None,
            rotation_power=0.0,
            false_alarm_probability=1.0,
            is_significant=False,
            n_flares=0,
            flare_times=(),
            flare_amplitudes=(),
        )

    p_min, p_max = period_range
    span = float(time[-1] - time[0]) if len(time) > 1 else 1.0
    p_max = min(p_max, span / 2.0)

    if p_min >= p_max or len(time) < 10:
        best_p, best_power, fap = None, 0.0, 1.0
    else:
        best_p, best_power, fap = _lomb_scargle(
            time, flux, flux_err, p_min, p_max
        )
        import math as _math
        if _math.isnan(best_power):
            best_power = 0.0
        if _math.isnan(fap):
            fap = 1.0

    is_sig = fap < fap_threshold
    flare_t, flare_a = _detect_flares(time, flux, flare_sigma)

    return RotationResult(
        rotation_period_days=best_p if is_sig else None,
        rotation_power=best_power,
        false_alarm_probability=fap,
        is_significant=is_sig,
        n_flares=len(flare_t),
        flare_times=flare_t,
        flare_amplitudes=flare_a,
    )


def format_rotation_result(result: RotationResult) -> str:
    """Format rotation result as a Markdown block."""
    lines = [
        "## Stellar Rotation Analysis",
        "",
    ]
    if result.is_significant and result.rotation_period_days is not None:
        lines.append(f"- Rotation period: {result.rotation_period_days:.3f} d")
        lines.append(f"- LS power: {result.rotation_power:.4f}")
        lines.append(f"- FAP: {result.false_alarm_probability:.2e}")
    else:
        fap_str = f"{result.false_alarm_probability:.2e}"
        lines.append(f"- No significant rotation detected (FAP={fap_str})")
    lines.append(f"- Flares detected: {result.n_flares}")
    if result.flare_times:
        lines.append(f"  - Times (BJD): {', '.join(f'{t:.2f}' for t in result.flare_times[:5])}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="stellar_rotation",
        description="Detect stellar rotation period and flares.",
    )
    parser.add_argument("--lc", required=True, metavar="JSON")
    parser.add_argument("--period-min", type=float, default=0.5)
    parser.add_argument("--period-max", type=float, default=30.0)
    parser.add_argument("--fap", type=float, default=0.01)
    args = parser.parse_args(argv)

    lc = json.loads(Path(args.lc).read_text())
    result = detect_rotation(
        lc["time"], lc["flux"],
        period_range=(args.period_min, args.period_max),
        fap_threshold=args.fap,
    )
    print(format_rotation_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
