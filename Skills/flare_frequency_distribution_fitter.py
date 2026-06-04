"""Fit a power-law flare frequency distribution (FFD) to a list of flare energies."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FfdFitResult:
    n_flares: int
    observing_time_days: float
    completeness_threshold_log_energy: float
    alpha: float           # FFD power-law slope (negative; typically -0.5 to -2)
    log_nu0: float         # log10 of normalisation rate (flares/day above E0)
    e0_log_energy: float   # reference energy log10(E0/erg)
    rate_at_e0: float      # ν(>E0) in flares/day
    r_squared: float       # goodness of fit on log-log scale
    flag: str


def fit_flare_frequency_distribution(
    log_energies: list[float],
    observing_time_days: float,
    completeness_log_energy: float | None = None,
) -> FfdFitResult:
    """
    Fit a power-law to the cumulative flare frequency distribution.

    Cumulative FFD: log10(ν(>E)) = α * log10(E) + β
    where ν is in flares/day/star, E is flare energy (any consistent unit).

    Parameters
    ----------
    log_energies:             List of log10(flare energies) for each detected flare.
    observing_time_days:      Total baseline in days.
    completeness_log_energy:  Lower completeness threshold; flares below this are excluded.
                              If None, uses the minimum log energy in the list.
    """
    if len(log_energies) < 3:
        return FfdFitResult(len(log_energies), observing_time_days,
                            float("nan"), float("nan"), float("nan"),
                            float("nan"), float("nan"), float("nan"), "INSUFFICIENT_FLARES")
    if not math.isfinite(observing_time_days) or observing_time_days <= 0:
        return FfdFitResult(len(log_energies), observing_time_days,
                            float("nan"), float("nan"), float("nan"),
                            float("nan"), float("nan"), float("nan"), "INVALID_BASELINE")

    thresh = completeness_log_energy if completeness_log_energy is not None else min(log_energies)
    filtered = sorted([e for e in log_energies if math.isfinite(e) and e >= thresh])
    n = len(filtered)
    if n < 3:
        return FfdFitResult(len(log_energies), observing_time_days,
                            thresh, float("nan"), float("nan"),
                            thresh, float("nan"), float("nan"), "INSUFFICIENT_FLARES")

    x_vals = filtered
    y_vals = [math.log10((n - i) / observing_time_days) for i in range(n)]

    sx = sum(x_vals)
    sy = sum(y_vals)
    sxx = sum(xi ** 2 for xi in x_vals)
    sxy = sum(xi * yi for xi, yi in zip(x_vals, y_vals, strict=True))
    nn = float(n)
    denom = nn * sxx - sx ** 2
    if abs(denom) < 1e-12:
        return FfdFitResult(n, observing_time_days, thresh, float("nan"), float("nan"),
                            thresh, float("nan"), float("nan"), "DEGENERATE_FIT")

    alpha = (nn * sxy - sx * sy) / denom
    beta = (sy - alpha * sx) / nn

    y_mean = sy / nn
    ss_tot = sum((yi - y_mean) ** 2 for yi in y_vals)
    y_pred = [alpha * xi + beta for xi in x_vals]
    ss_res = sum((yi - yp) ** 2 for yi, yp in zip(y_vals, y_pred, strict=True))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    e0 = thresh
    rate_at_e0 = 10.0 ** (alpha * e0 + beta)

    return FfdFitResult(
        n_flares=n,
        observing_time_days=observing_time_days,
        completeness_threshold_log_energy=round(thresh, 4),
        alpha=round(alpha, 4),
        log_nu0=round(beta, 4),
        e0_log_energy=round(e0, 4),
        rate_at_e0=round(rate_at_e0, 6),
        r_squared=round(r2, 4) if math.isfinite(r2) else float("nan"),
        flag="OK",
    )


def format_ffd_fit_result(r: FfdFitResult) -> str:
    def _f(v: float, fmt: str = ".4f") -> str:
        return format(v, fmt) if math.isfinite(v) else "N/A"

    return (
        f"| Parameter | Value |\n|---|---|\n"
        f"| N flares (above threshold) | {r.n_flares} |\n"
        f"| Observing time (days) | {_f(r.observing_time_days)} |\n"
        f"| Completeness threshold (log E) | {_f(r.completeness_threshold_log_energy)} |\n"
        f"| Power-law slope α | {_f(r.alpha)} |\n"
        f"| log ν₀ | {_f(r.log_nu0)} |\n"
        f"| Rate at E₀ (flares/day) | {_f(r.rate_at_e0)} |\n"
        f"| R² | {_f(r.r_squared)} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Fit flare frequency distribution.")
    p.add_argument("log_energies", help="Comma-separated log10(energy) values")
    p.add_argument("observing_time_days", type=float)
    p.add_argument("--completeness", type=float, default=None)
    args = p.parse_args()
    log_e = [float(x) for x in args.log_energies.split(",")]
    r = fit_flare_frequency_distribution(log_e, args.observing_time_days, args.completeness)
    print(format_ffd_fit_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
