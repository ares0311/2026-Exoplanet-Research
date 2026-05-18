"""Check for period doubling (half-period alias) in a transit signal.

Compares the primary transit depth at the nominal period with the signal
at half that period. If a significant transit exists at P/2, the true
period may be P/2 (eclipsing binary with equal eclipses, or systematic).

Public API
----------
PeriodDoublingResult(period_days, half_period_days, depth_primary_ppm,
                     depth_half_ppm, ratio, flag)
check_period_doubling(time, flux, period_days, epoch_bjd, *,
                      duration_days, snr_threshold) -> PeriodDoublingResult
format_period_doubling_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PeriodDoublingResult:
    period_days: float
    half_period_days: float
    depth_primary_ppm: float | None
    depth_half_ppm: float | None
    ratio: float | None          # depth_half / depth_primary; None if undetermined
    flag: str                    # "OK", "POSSIBLE_DOUBLING", "INSUFFICIENT"


def _phase_fold_depth(
    time: list[float],
    flux: list[float],
    period: float,
    epoch: float,
    duration_days: float,
) -> float | None:
    """Return mean in-transit depth (1 - mean_flux) in ppm; None if no data."""
    in_transit = []
    oot = []
    half = duration_days / 2.0
    for t, f in zip(time, flux, strict=False):
        ph = ((t - epoch) % period) / period
        if ph > 0.5:
            ph -= 1.0
        if abs(ph) * period <= half:
            in_transit.append(f)
        elif abs(ph) * period > half * 3:
            oot.append(f)

    if not in_transit or not oot:
        return None
    baseline = sum(oot) / len(oot)
    depth = (baseline - sum(in_transit) / len(in_transit)) * 1e6
    return depth


def check_period_doubling(
    time: list[float],
    flux: list[float],
    period_days: float,
    epoch_bjd: float,
    *,
    duration_days: float = 0.1,
    snr_threshold: float = 3.0,
) -> PeriodDoublingResult:
    """Check whether a signal at P/2 suggests period doubling.

    Args:
        time: Time array (BJD).
        flux: Normalised flux array.
        period_days: Nominal period in days.
        epoch_bjd: Epoch of primary transit in BJD.
        duration_days: Assumed transit duration in days.
        snr_threshold: Ratio above which period doubling is flagged.

    Returns:
        :class:`PeriodDoublingResult`.
    """
    half_period = period_days / 2.0

    if not time or not flux or period_days <= 0:
        return PeriodDoublingResult(
            period_days, half_period, None, None, None, "INSUFFICIENT",
        )

    depth_primary = _phase_fold_depth(time, flux, period_days, epoch_bjd, duration_days)
    depth_half = _phase_fold_depth(time, flux, half_period, epoch_bjd, duration_days)

    if depth_primary is None or depth_half is None:
        return PeriodDoublingResult(
            period_days, half_period, depth_primary, depth_half, None, "INSUFFICIENT",
        )

    if depth_primary <= 0:
        return PeriodDoublingResult(
            period_days, half_period,
            round(depth_primary, 2), round(depth_half, 2), None, "INSUFFICIENT",
        )

    ratio = depth_half / depth_primary if depth_primary > 0 else None
    flag = "POSSIBLE_DOUBLING" if (ratio is not None and ratio >= snr_threshold) else "OK"

    return PeriodDoublingResult(
        period_days=period_days,
        half_period_days=half_period,
        depth_primary_ppm=round(depth_primary, 2),
        depth_half_ppm=round(depth_half, 2),
        ratio=round(ratio, 4) if ratio is not None else None,
        flag=flag,
    )


def format_period_doubling_result(result: PeriodDoublingResult) -> str:
    """Format period doubling check result as Markdown."""
    lines = [
        "## Period Doubling Check",
        "",
        f"- Nominal period: {result.period_days:.4f} days",
        f"- Half period: {result.half_period_days:.4f} days",
    ]
    if result.flag == "INSUFFICIENT":
        lines.append("- **Flag: INSUFFICIENT** — not enough in-transit data")
    else:
        lines += [
            f"- Depth at P: {result.depth_primary_ppm:.1f} ppm",
            f"- Depth at P/2: {result.depth_half_ppm:.1f} ppm",
            (f"- Ratio (half/primary): {result.ratio:.4f}"
             if result.ratio is not None else "- Ratio: N/A"),
            f"- **Flag: {result.flag}**",
        ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="period_doubling_checker",
        description="Check for period doubling in a transit signal.",
    )
    parser.add_argument("period_days", type=float)
    parser.add_argument("epoch_bjd", type=float)
    parser.add_argument("--duration-days", type=float, default=0.1)
    parser.add_argument("--snr-threshold", type=float, default=3.0)
    args = parser.parse_args(argv)

    result = check_period_doubling(
        [], [], args.period_days, args.epoch_bjd,
        duration_days=args.duration_days, snr_threshold=args.snr_threshold,
    )
    print(format_period_doubling_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
