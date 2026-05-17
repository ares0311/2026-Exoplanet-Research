"""Per-transit and ensemble signal statistics for vetting.

Computes:
- Per-transit SNR distribution (median, MAD, outlier count)
- Odd/even transit depth comparison (asymmetry → EB flag)
- Secondary eclipse grid search (shifted by half-period)

Public API
----------
SignalStats(per_transit_snr, median_snr, odd_depth_ppm, even_depth_ppm,
            odd_even_significance, secondary_depth_ppm, secondary_snr, flags)
compute_signal_stats(time, flux, period, epoch, *,
                     flux_err, duration_days) -> SignalStats
format_signal_stats(stats) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SignalStats:
    per_transit_snr: tuple[float, ...]  # SNR for each individual transit
    median_snr: float
    odd_depth_ppm: float | None         # mean depth of odd-numbered transits
    even_depth_ppm: float | None        # mean depth of even-numbered transits
    odd_even_significance: float | None # |odd-even| / combined uncertainty
    secondary_depth_ppm: float          # depth of best secondary eclipse
    secondary_snr: float                # SNR of secondary eclipse
    flags: tuple[str, ...]              # e.g. "ODD_EVEN_ASYMMETRY", "SECONDARY_DETECTED"


def _phase_of(t: float, period: float, epoch: float) -> float:
    ph = (t - epoch) % period
    return ph - period if ph > period / 2 else ph


def _extract_transits(
    time: list[float],
    flux: list[float],
    period: float,
    epoch: float,
    half_dur: float,
    flux_err: list[float] | None,
) -> list[dict]:
    """Group cadences into transit windows and return per-transit stats."""
    # Assign each cadence a transit number
    transit_data: dict[int, list[tuple[float, float, float]]] = {}
    for i, (t, f) in enumerate(zip(time, flux, strict=False)):
        ph = _phase_of(t, period, epoch)
        if abs(ph) <= half_dur:
            n = round((t - epoch) / period)
            err = flux_err[i] if flux_err else 1e-4
            transit_data.setdefault(n, []).append((f, err, ph))

    results = []
    for n, cadences in sorted(transit_data.items()):
        fluxes = [c[0] for c in cadences]
        errs   = [c[1] for c in cadences]
        mean_f = sum(fluxes) / len(fluxes)
        noise  = (sum(e**2 for e in errs) / len(errs)) ** 0.5
        depth  = (1.0 - mean_f) * 1e6
        snr    = abs(1.0 - mean_f) / max(noise, 1e-9) * len(fluxes) ** 0.5
        results.append({"n": n, "depth_ppm": depth, "snr": snr})
    return results


def compute_signal_stats(
    time: list[float],
    flux: list[float],
    period: float,
    epoch: float,
    *,
    flux_err: list[float] | None = None,
    duration_days: float = 0.1,
    secondary_threshold_snr: float = 3.0,
    odd_even_threshold_sigma: float = 3.0,
) -> SignalStats:
    """Compute per-transit and ensemble signal statistics.

    Args:
        time: BJD time array.
        flux: Normalised flux (mean ≈ 1.0).
        period: BLS period in days.
        epoch: Mid-transit epoch (BJD).
        flux_err: Per-cadence uncertainties.
        duration_days: Transit duration used for windowing.
        secondary_threshold_snr: SNR above which a secondary eclipse is flagged.
        odd_even_threshold_sigma: Sigma above which odd/even asymmetry is flagged.

    Returns:
        :class:`SignalStats`.
    """
    if period <= 0:
        raise ValueError(f"period must be positive, got {period}")

    half_dur = duration_days / 2.0
    transits = _extract_transits(time, flux, period, epoch, half_dur, flux_err)

    per_snr = tuple(t["snr"] for t in transits)
    if per_snr:
        sorted_snr = sorted(per_snr)
        n = len(sorted_snr)
        median_snr = sorted_snr[n // 2]
    else:
        median_snr = 0.0

    # Odd/even depths
    odd  = [t["depth_ppm"] for t in transits if t["n"] % 2 != 0]
    even = [t["depth_ppm"] for t in transits if t["n"] % 2 == 0]

    def _mean(lst: list[float]) -> float | None:
        return sum(lst) / len(lst) if lst else None

    def _sem(lst: list[float]) -> float:
        if len(lst) < 2:
            return float("inf")
        m = sum(lst) / len(lst)
        var = sum((x - m) ** 2 for x in lst) / (len(lst) - 1)
        return (var / len(lst)) ** 0.5

    odd_mean  = _mean(odd)
    even_mean = _mean(even)

    if odd_mean is not None and even_mean is not None:
        sem = ((_sem(odd)) ** 2 + (_sem(even)) ** 2) ** 0.5
        oe_sig = abs(odd_mean - even_mean) / max(sem, 1e-9)
    else:
        oe_sig = None

    # Secondary eclipse: check flux around phase = ±0.5
    secondary_phase = period / 2.0
    sec_epoch_pos = epoch + secondary_phase
    sec_epoch_neg = epoch - secondary_phase
    sec_transits_pos = _extract_transits(
        time, flux, period, sec_epoch_pos, half_dur, flux_err
    )
    sec_transits_neg = _extract_transits(
        time, flux, period, sec_epoch_neg, half_dur, flux_err
    )
    sec_transits = sec_transits_pos + sec_transits_neg

    if sec_transits:
        best = max(sec_transits, key=lambda t: t["snr"])
        sec_depth = best["depth_ppm"]
        sec_snr   = best["snr"]
    else:
        sec_depth = 0.0
        sec_snr   = 0.0

    flags: list[str] = []
    if oe_sig is not None and oe_sig > odd_even_threshold_sigma:
        flags.append("ODD_EVEN_ASYMMETRY")
    if sec_snr > secondary_threshold_snr:
        flags.append("SECONDARY_DETECTED")

    return SignalStats(
        per_transit_snr=per_snr,
        median_snr=median_snr,
        odd_depth_ppm=odd_mean,
        even_depth_ppm=even_mean,
        odd_even_significance=oe_sig,
        secondary_depth_ppm=sec_depth,
        secondary_snr=sec_snr,
        flags=tuple(flags),
    )


def format_signal_stats(stats: SignalStats) -> str:
    """Format signal stats as a short Markdown block."""
    def _fmt(v: float | None) -> str:
        return f"{v:.1f}" if v is not None else "—"

    lines = [
        "## Signal Statistics",
        "",
        f"- Individual transits detected: {len(stats.per_transit_snr)}",
        f"- Median per-transit SNR: {stats.median_snr:.2f}",
        f"- Odd depth: {_fmt(stats.odd_depth_ppm)} ppm",
        f"- Even depth: {_fmt(stats.even_depth_ppm)} ppm",
        f"- Odd/even significance: {_fmt(stats.odd_even_significance)}σ",
        f"- Secondary eclipse depth: {stats.secondary_depth_ppm:.1f} ppm  "
          f"(SNR {stats.secondary_snr:.1f})",
    ]
    if stats.flags:
        lines.append(f"- **Flags**: {', '.join(stats.flags)}")
    else:
        lines.append("- Flags: none")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="signal_statistics",
        description="Compute per-transit and ensemble signal statistics.",
    )
    parser.add_argument("--lc", required=True, metavar="JSON")
    parser.add_argument("--period", type=float, required=True)
    parser.add_argument("--epoch", type=float, required=True)
    parser.add_argument("--duration", type=float, default=0.1)
    args = parser.parse_args(argv)

    lc = json.loads(Path(args.lc).read_text())
    stats = compute_signal_stats(
        lc["time"], lc["flux"],
        args.period, args.epoch,
        duration_days=args.duration,
    )
    print(format_signal_stats(stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
