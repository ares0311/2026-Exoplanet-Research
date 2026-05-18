"""Score the ingress/egress asymmetry of a phase-folded transit.

Splits the in-transit region into ingress and egress halves around the
midpoint and compares their mean flux levels.  A genuine planet transit
should be symmetric; a significant asymmetry suggests a blended EB,
spot-crossing, or instrumental artifact.

Public API
----------
TransitAsymmetryResult(period_days, epoch_bjd, duration_hours,
                        ingress_mean, egress_mean, asymmetry_ratio,
                        asymmetry_score, is_asymmetric, flag)
score_transit_asymmetry(time, flux, period_days, epoch_bjd, *,
                        duration_hours, flux_err, n_bins,
                        asymmetry_threshold) -> TransitAsymmetryResult
format_transit_asymmetry_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransitAsymmetryResult:
    period_days: float
    epoch_bjd: float
    duration_hours: float
    ingress_mean: float | None
    egress_mean: float | None
    asymmetry_ratio: float | None  # |ingress - egress| / |transit_depth|
    asymmetry_score: float  # [0, 1]; high = more asymmetric
    is_asymmetric: bool
    flag: str  # "OK" | "ASYMMETRIC" | "INSUFFICIENT" | "INVALID"


def _phase_fold(time: list[float], epoch: float, period: float) -> list[float]:
    """Return phases in [-0.5, 0.5)."""
    phases = []
    for t in time:
        ph = ((t - epoch) % period) / period
        if ph >= 0.5:
            ph -= 1.0
        phases.append(ph)
    return phases


def score_transit_asymmetry(
    time: list[float],
    flux: list[float],
    period_days: float,
    epoch_bjd: float,
    *,
    duration_hours: float = 2.0,
    flux_err: list[float] | None = None,
    n_bins: int = 10,
    asymmetry_threshold: float = 0.20,
) -> TransitAsymmetryResult:
    """Score ingress/egress asymmetry of a phase-folded transit.

    Args:
        time: Time array (BJD).
        flux: Normalised flux array.
        period_days: Orbital period in days.
        epoch_bjd: Reference epoch (mid-transit).
        duration_hours: Transit duration used to define the in-transit window.
        flux_err: Per-point uncertainties (uniform 1.0 if None).
        n_bins: Number of phase bins inside the transit half-width.
        asymmetry_threshold: Asymmetry ratio above which ``is_asymmetric``
            is set True.

    Returns:
        :class:`TransitAsymmetryResult`.
    """
    n = len(flux)
    if n < 10 or period_days <= 0 or duration_hours <= 0:
        return TransitAsymmetryResult(
            period_days, epoch_bjd, duration_hours,
            None, None, None, 0.0, False, "INVALID",
        )

    errs = flux_err if (flux_err is not None and len(flux_err) == n) else [1.0] * n
    phases = _phase_fold(time, epoch_bjd, period_days)

    half_width = (duration_hours / 24.0) / period_days / 2.0

    ingress_flux: list[tuple[float, float]] = []  # (flux, weight)
    egress_flux: list[tuple[float, float]] = []

    for ph, f, e in zip(phases, flux, errs, strict=False):
        if abs(ph) > half_width:
            continue
        w = 1.0 / max(e ** 2, 1e-30)
        if ph < 0:
            ingress_flux.append((f, w))
        else:
            egress_flux.append((f, w))

    if len(ingress_flux) < 3 or len(egress_flux) < 3:
        return TransitAsymmetryResult(
            period_days, epoch_bjd, duration_hours,
            None, None, None, 0.0, False, "INSUFFICIENT",
        )

    def _weighted_mean(pairs: list[tuple[float, float]]) -> float:
        sw = sum(w for _, w in pairs)
        if sw <= 0:
            return 0.0
        return sum(f * w for f, w in pairs) / sw

    ingress_mean = _weighted_mean(ingress_flux)
    egress_mean = _weighted_mean(egress_flux)

    # Estimate transit depth from the out-of-transit baseline vs in-transit mean
    in_transit_all = ingress_flux + egress_flux
    in_mean = _weighted_mean(in_transit_all)

    oot_flux: list[tuple[float, float]] = []
    oot_half = min(3 * half_width, 0.4)
    for ph, f, e in zip(phases, flux, errs, strict=False):
        if half_width < abs(ph) <= oot_half:
            w = 1.0 / max(e ** 2, 1e-30)
            oot_flux.append((f, w))

    baseline = _weighted_mean(oot_flux) if oot_flux else 1.0
    transit_depth = abs(baseline - in_mean)

    if transit_depth < 1e-9:
        return TransitAsymmetryResult(
            period_days, epoch_bjd, duration_hours,
            round(ingress_mean, 6), round(egress_mean, 6),
            None, 0.0, False, "INSUFFICIENT",
        )

    asymmetry_ratio = abs(ingress_mean - egress_mean) / transit_depth
    asymmetry_score = min(1.0, asymmetry_ratio / max(asymmetry_threshold, 1e-9))
    is_asymmetric = asymmetry_ratio >= asymmetry_threshold

    flag = "ASYMMETRIC" if is_asymmetric else "OK"

    return TransitAsymmetryResult(
        period_days=period_days,
        epoch_bjd=epoch_bjd,
        duration_hours=duration_hours,
        ingress_mean=round(ingress_mean, 6),
        egress_mean=round(egress_mean, 6),
        asymmetry_ratio=round(asymmetry_ratio, 4),
        asymmetry_score=round(asymmetry_score, 4),
        is_asymmetric=is_asymmetric,
        flag=flag,
    )


def format_transit_asymmetry_result(result: TransitAsymmetryResult) -> str:
    """Format transit asymmetry result as Markdown."""
    lines = [
        "## Transit Asymmetry Score",
        "",
        f"- Period: {result.period_days:.4f} days",
        f"- Duration: {result.duration_hours:.2f} hours",
        f"- Ingress mean flux: {result.ingress_mean}",
        f"- Egress mean flux: {result.egress_mean}",
        f"- Asymmetry ratio: {result.asymmetry_ratio}",
        f"- Asymmetry score: {result.asymmetry_score:.4f}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="transit_asymmetry_scorer",
        description="Score ingress/egress asymmetry of a transit.",
    )
    parser.add_argument("period_days", type=float)
    parser.add_argument("epoch_bjd", type=float)
    parser.add_argument("--duration-hours", type=float, default=2.0)
    parser.add_argument("--asymmetry-threshold", type=float, default=0.20)
    args = parser.parse_args(argv)

    result = score_transit_asymmetry(
        [], [], args.period_days, args.epoch_bjd,
        duration_hours=args.duration_hours,
        asymmetry_threshold=args.asymmetry_threshold,
    )
    print(format_transit_asymmetry_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
