"""Evaluate a custom period grid via a lightweight BLS-like power statistic.

Useful when you have external period hints (e.g., from a published catalogue
or alias analysis) and want to score each period candidate without running a
full BLS grid.  Each period is evaluated by phase-folding the light curve and
computing the ratio of in-transit variance to total variance.

Public API
----------
PeriodGridResult(best_period_days, best_power, periods_days, powers,
                 n_periods_tested, flag)
search_period_grid(time, flux, periods_days, *, flux_err, duration_hours,
                   n_phase_bins) -> PeriodGridResult
format_period_grid_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PeriodGridResult:
    best_period_days: float | None
    best_power: float | None
    periods_days: tuple[float, ...]
    powers: tuple[float, ...]
    n_periods_tested: int
    flag: str  # "OK" | "INSUFFICIENT" | "INVALID"


def _phase_fold(time: list[float], period: float) -> list[float]:
    return [((t / period) % 1.0) for t in time]


def _bls_power(
    phase: list[float],
    flux: list[float],
    weights: list[float],
    duration_frac: float,
    n_phase_bins: int,
) -> float:
    """Best-box BLS-like power over phase offsets."""
    sw = sum(weights)
    if sw <= 0:
        return 0.0
    wf_sum = sum(weights[i] * flux[i] for i in range(len(flux)))
    mean_wf = wf_sum / sw

    best = 0.0
    for b in range(n_phase_bins):
        ph0 = b / n_phase_bins
        in_w = 0.0
        in_wf = 0.0
        for i in range(len(flux)):
            ph = phase[i]
            diff = min(abs(ph - ph0), 1.0 - abs(ph - ph0))
            if diff <= duration_frac / 2.0:
                in_w += weights[i]
                in_wf += weights[i] * flux[i]
        if in_w <= 0 or sw - in_w <= 0:
            continue
        in_mean = in_wf / in_w
        depth = mean_wf - in_mean
        r = in_w / sw
        power = depth ** 2 * r * (1.0 - r)
        if power > best:
            best = power
    return best


def search_period_grid(
    time: list[float],
    flux: list[float],
    periods_days: list[float],
    *,
    flux_err: list[float] | None = None,
    duration_hours: float = 2.0,
    n_phase_bins: int = 30,
) -> PeriodGridResult:
    """Score each period in a custom grid.

    Args:
        time: Time array (days).
        flux: Normalised flux array.
        periods_days: List of period candidates to evaluate.
        flux_err: Per-point uncertainties (uniform if None).
        duration_hours: Transit duration in hours (used for box width).
        n_phase_bins: Number of phase offset bins to search.

    Returns:
        :class:`PeriodGridResult`.
    """
    n = len(flux)
    if n < 5 or len(time) != n:
        return PeriodGridResult(None, None, (), (), 0, "INVALID")

    valid_periods = [p for p in periods_days if p > 0]
    if not valid_periods:
        return PeriodGridResult(None, None, (), (), 0, "INVALID")

    errs = flux_err if (flux_err is not None and len(flux_err) == n) else [1.0] * n
    weights = [1.0 / max(e ** 2, 1e-30) for e in errs]

    powers: list[float] = []
    for p in valid_periods:
        duration_frac = (duration_hours / 24.0) / p
        phase = _phase_fold(time, p)
        pw = _bls_power(phase, flux, weights, duration_frac, n_phase_bins)
        powers.append(round(pw, 8))

    best_idx = powers.index(max(powers))
    best_p = valid_periods[best_idx]
    best_pw = powers[best_idx]

    flag = "OK" if len(valid_periods) >= 1 else "INSUFFICIENT"

    return PeriodGridResult(
        best_period_days=best_p,
        best_power=round(best_pw, 8),
        periods_days=tuple(valid_periods),
        powers=tuple(powers),
        n_periods_tested=len(valid_periods),
        flag=flag,
    )


def format_period_grid_result(result: PeriodGridResult) -> str:
    """Format period grid search result as Markdown."""
    lines = [
        "## Period Grid Search",
        "",
        f"- Periods tested: {result.n_periods_tested}",
        f"- Best period: {result.best_period_days} days",
        f"- Best power: {result.best_power}",
        f"- **Flag: {result.flag}**",
    ]
    if result.periods_days:
        lines += ["", "| Period (days) | Power |", "|---|---|"]
        for p, pw in zip(result.periods_days, result.powers, strict=False):
            lines.append(f"| {p:.4f} | {pw:.6f} |")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="period_grid_search",
        description="Score a custom period grid against a light curve.",
    )
    parser.add_argument("--periods", nargs="+", type=float, default=[])
    parser.add_argument("--duration-hours", type=float, default=2.0)
    args = parser.parse_args(argv)

    result = search_period_grid([], [], args.periods, duration_hours=args.duration_hours)
    print(format_period_grid_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
