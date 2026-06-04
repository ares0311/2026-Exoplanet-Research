"""Chi-square test on per-transit duration scatter as a TTV/noise diagnostic."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DurationScatterResult:
    n_transits: int
    mean_duration_hours: float
    rms_scatter_hours: float
    chi2_reduced: float
    p_value_approx: float
    scatter_significant: bool
    flag: str


def _chi2_survival(chi2: float, dof: int) -> float:
    """Approximate chi-square survival function P(X > chi2) via regularised gamma."""
    if dof <= 0 or not math.isfinite(chi2) or chi2 < 0:
        return float("nan")
    # Use Wilson-Hilferty normal approximation for large dof
    k = float(dof)
    if chi2 == 0.0:
        return 1.0
    x = (chi2 / k) ** (1.0 / 3.0)
    mu = 1.0 - 2.0 / (9.0 * k)
    sigma = math.sqrt(2.0 / (9.0 * k))
    z = (x - mu) / sigma if sigma > 0 else 0.0
    # Erfc approximation for normal survival function
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def test_duration_scatter(
    durations_hours: list[float],
    duration_errors_hours: list[float] | None = None,
    significance_threshold: float = 3.0,
) -> DurationScatterResult:
    """
    Chi-square test on per-transit T14 duration scatter.

    If duration_errors_hours provided: reduced chi2 = sum((T_i - T_mean)^2 / sigma_i^2) / (N-1)
    Otherwise: uses RMS scatter vs mean as a dimensionless scatter metric.

    scatter_significant = True when chi2_reduced > significance_threshold.
    High scatter may indicate TTVs, measurement noise, or instrumental systematics.
    """
    n = len(durations_hours)
    if n < 3:
        return DurationScatterResult(
            n_transits=n, mean_duration_hours=float("nan"),
            rms_scatter_hours=float("nan"), chi2_reduced=float("nan"),
            p_value_approx=float("nan"), scatter_significant=False,
            flag="INSUFFICIENT_TRANSITS",
        )

    valid = [(d, (e if duration_errors_hours else None))
             for d, e in zip(durations_hours,
                             duration_errors_hours if duration_errors_hours else [None] * n,
                             strict=False)
             if math.isfinite(d)]

    if len(valid) < 3:
        return DurationScatterResult(
            n_transits=n, mean_duration_hours=float("nan"),
            rms_scatter_hours=float("nan"), chi2_reduced=float("nan"),
            p_value_approx=float("nan"), scatter_significant=False,
            flag="INSUFFICIENT_FINITE_VALUES",
        )

    durs = [v[0] for v in valid]
    errs = [v[1] for v in valid]

    if all(e is not None and e > 0 for e in errs):
        weights = [1.0 / e**2 for e in errs]  # type: ignore[operator]
        w_sum = sum(weights)
        mean = sum(w * d for w, d in zip(weights, durs, strict=False)) / w_sum
        chi2 = sum(w * (d - mean) ** 2 for w, d in zip(weights, durs, strict=False))
        chi2_red = chi2 / (len(durs) - 1)
    else:
        mean = sum(durs) / len(durs)
        var = sum((d - mean) ** 2 for d in durs) / (len(durs) - 1)
        rms = math.sqrt(var)
        # Normalise: chi2_reduced = (rms / mean)^2 * N as proxy
        chi2_red = (rms / mean) ** 2 * len(durs) if mean > 0 else 0.0

    rms_scatter = math.sqrt(sum((d - mean) ** 2 for d in durs) / (len(durs) - 1))
    p_val = _chi2_survival(chi2_red * (len(durs) - 1), len(durs) - 1)
    significant = chi2_red > significance_threshold

    return DurationScatterResult(
        n_transits=len(durs),
        mean_duration_hours=round(mean, 4),
        rms_scatter_hours=round(rms_scatter, 5),
        chi2_reduced=round(chi2_red, 4),
        p_value_approx=round(p_val, 5) if math.isfinite(p_val) else float("nan"),
        scatter_significant=significant,
        flag="SCATTER_DETECTED" if significant else "OK",
    )


def format_duration_scatter_result(r: DurationScatterResult) -> str:
    def _f(v: float, fmt: str = ".4f") -> str:
        return format(v, fmt) if math.isfinite(v) else "N/A"

    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| N transits | {r.n_transits} |\n"
        f"| Mean T14 (h) | {_f(r.mean_duration_hours)} |\n"
        f"| RMS scatter (h) | {_f(r.rms_scatter_hours, '.5f')} |\n"
        f"| Chi2 reduced | {_f(r.chi2_reduced)} |\n"
        f"| P-value (approx) | {_f(r.p_value_approx, '.5f')} |\n"
        f"| Scatter significant | {r.scatter_significant} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Chi-square test on per-transit duration scatter.")
    p.add_argument("durations_json", help="JSON array of T14 values in hours")
    p.add_argument("--errors-json", default=None, help="JSON array of T14 errors in hours")
    args = p.parse_args()
    import json
    durations = json.loads(args.durations_json)
    errors = json.loads(args.errors_json) if args.errors_json else None
    r = test_duration_scatter(durations, errors)
    print(format_duration_scatter_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
