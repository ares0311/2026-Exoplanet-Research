"""Detect harmonic or sub-harmonic periods in a light curve.

Checks whether the flux at integer multiples or fractions of the nominal
period shows transit-like dips with significant amplitude, which would
indicate that the true period is a harmonic of the reported period.

Public API
----------
HarmonicResult(nominal_period_days, harmonics_tested, best_harmonic,
               best_depth_ppm, ratio, flag)
analyze_harmonics(time, flux, period_days, epoch_bjd, *,
                  duration_days, max_harmonic, depth_threshold_ppm) -> HarmonicResult
format_harmonic_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HarmonicResult:
    nominal_period_days: float
    harmonics_tested: tuple[float, ...]   # periods tested (P/2, P/3, 2P, 3P…)
    best_harmonic: float | None           # period with strongest signal
    best_depth_ppm: float | None
    ratio: float | None                   # best_period / nominal_period
    flag: str                             # "OK", "HARMONIC_FOUND", "INSUFFICIENT"


def _mean_depth_at_period(
    time: list[float],
    flux: list[float],
    period: float,
    epoch: float,
    duration_days: float,
) -> float:
    """Return mean transit depth in ppm at given period (0 if no in-transit data)."""
    half = duration_days / 2.0
    in_transit: list[float] = []
    oot: list[float] = []
    for t, f in zip(time, flux, strict=False):
        ph = abs(((t - epoch) % period) / period)
        if ph > 0.5:
            ph = 1.0 - ph
        t_from_mid = ph * period
        if t_from_mid <= half:
            in_transit.append(f)
        elif t_from_mid > half * 3:
            oot.append(f)
    if not in_transit or not oot:
        return 0.0
    baseline = sum(oot) / len(oot)
    depth = (baseline - sum(in_transit) / len(in_transit)) * 1e6
    return max(depth, 0.0)


def analyze_harmonics(
    time: list[float],
    flux: list[float],
    period_days: float,
    epoch_bjd: float,
    *,
    duration_days: float = 0.1,
    max_harmonic: int = 3,
    depth_threshold_ppm: float = 200.0,
) -> HarmonicResult:
    """Test harmonic and sub-harmonic periods for transit signals.

    Args:
        time: Time array (BJD).
        flux: Normalised flux array.
        period_days: Nominal period in days.
        epoch_bjd: Reference epoch in BJD.
        duration_days: Assumed transit duration in days.
        max_harmonic: Maximum integer harmonic to test (2 = P/2, P/3; 2P, 3P).
        depth_threshold_ppm: Minimum depth to flag a harmonic as significant.

    Returns:
        :class:`HarmonicResult`.
    """
    if not time or not flux or period_days <= 0:
        return HarmonicResult(period_days, (), None, None, None, "INSUFFICIENT")

    candidates: list[float] = []
    for n in range(2, max_harmonic + 1):
        candidates.append(period_days / n)   # sub-harmonics P/2, P/3...
        candidates.append(period_days * n)   # harmonics 2P, 3P...

    harmonics_tested = tuple(round(p, 6) for p in candidates)

    best_period: float | None = None
    best_depth = 0.0

    for period in candidates:
        if period <= 0:
            continue
        depth = _mean_depth_at_period(time, flux, period, epoch_bjd, duration_days)
        if depth > best_depth:
            best_depth = depth
            best_period = period

    if best_period is None or best_depth < depth_threshold_ppm:
        return HarmonicResult(
            period_days, harmonics_tested, None, None, None, "OK",
        )

    ratio = best_period / period_days
    return HarmonicResult(
        nominal_period_days=period_days,
        harmonics_tested=harmonics_tested,
        best_harmonic=round(best_period, 6),
        best_depth_ppm=round(best_depth, 2),
        ratio=round(ratio, 6),
        flag="HARMONIC_FOUND",
    )


def format_harmonic_result(result: HarmonicResult) -> str:
    """Format harmonic analysis result as Markdown."""
    lines = [
        "## Harmonic Period Analysis",
        "",
        f"- Nominal period: {result.nominal_period_days:.4f} days",
        f"- Harmonics tested: {len(result.harmonics_tested)}",
    ]
    if result.flag == "INSUFFICIENT":
        lines.append("- **Flag: INSUFFICIENT** — not enough data")
    elif result.flag == "HARMONIC_FOUND":
        lines += [
            f"- Best harmonic period: {result.best_harmonic:.4f} days",
            f"- Depth at harmonic: {result.best_depth_ppm:.2f} ppm",
            f"- Ratio (harmonic/nominal): {result.ratio:.4f}",
            "- **Flag: HARMONIC_FOUND**",
        ]
    else:
        lines.append("- No significant harmonic signal found")
        lines.append("- **Flag: OK**")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="harmonic_period_analyzer",
        description="Detect harmonic periods in a light curve.",
    )
    parser.add_argument("period_days", type=float)
    parser.add_argument("epoch_bjd", type=float)
    parser.add_argument("--duration-days", type=float, default=0.1)
    parser.add_argument("--max-harmonic", type=int, default=3)
    parser.add_argument("--depth-threshold-ppm", type=float, default=200.0)
    args = parser.parse_args(argv)

    result = analyze_harmonics(
        [], [], args.period_days, args.epoch_bjd,
        duration_days=args.duration_days,
        max_harmonic=args.max_harmonic,
        depth_threshold_ppm=args.depth_threshold_ppm,
    )
    print(format_harmonic_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
