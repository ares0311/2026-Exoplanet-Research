"""Estimate planet occurrence rate from detections and survey efficiency."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class OccurrenceRateResult:
    n_detections: int
    n_stars: int
    detection_efficiency: float
    transit_probability: float
    occurrence_rate: float
    rate_lower_1sigma: float
    rate_upper_1sigma: float
    flag: str


def estimate_occurrence_rate(
    n_detections: int,
    n_stars: int,
    detection_efficiency: float,
    transit_probability: float,
) -> OccurrenceRateResult:
    """
    Estimate η = N_det / (N★ × ε × p_tr) with Poisson 1-sigma confidence interval.

    Uses the Gehrels (1986) approximation for small counts.
    """
    if n_detections < 0:
        return OccurrenceRateResult(
            n_detections=n_detections, n_stars=n_stars,
            detection_efficiency=detection_efficiency,
            transit_probability=transit_probability,
            occurrence_rate=float("nan"),
            rate_lower_1sigma=float("nan"),
            rate_upper_1sigma=float("nan"),
            flag="INVALID_DETECTIONS",
        )
    if n_stars <= 0:
        return OccurrenceRateResult(
            n_detections=n_detections, n_stars=n_stars,
            detection_efficiency=detection_efficiency,
            transit_probability=transit_probability,
            occurrence_rate=float("nan"),
            rate_lower_1sigma=float("nan"),
            rate_upper_1sigma=float("nan"),
            flag="INVALID_N_STARS",
        )
    if not (0.0 < detection_efficiency <= 1.0):
        return OccurrenceRateResult(
            n_detections=n_detections, n_stars=n_stars,
            detection_efficiency=detection_efficiency,
            transit_probability=transit_probability,
            occurrence_rate=float("nan"),
            rate_lower_1sigma=float("nan"),
            rate_upper_1sigma=float("nan"),
            flag="INVALID_EFFICIENCY",
        )
    if not (0.0 < transit_probability <= 1.0):
        return OccurrenceRateResult(
            n_detections=n_detections, n_stars=n_stars,
            detection_efficiency=detection_efficiency,
            transit_probability=transit_probability,
            occurrence_rate=float("nan"),
            rate_lower_1sigma=float("nan"),
            rate_upper_1sigma=float("nan"),
            flag="INVALID_TRANSIT_PROB",
        )

    denominator = n_stars * detection_efficiency * transit_probability
    eta = n_detections / denominator

    # Gehrels (1986) Poisson upper/lower 1-sigma bounds on N_det
    n = float(n_detections)
    # Upper: N + 1 + sqrt(N + 0.75)
    n_upper = n + 1.0 + math.sqrt(n + 0.75)
    # Lower: max(0, N - sqrt(N + 0.25))
    n_lower = max(0.0, n - math.sqrt(n + 0.25))

    eta_upper = n_upper / denominator
    eta_lower = n_lower / denominator

    return OccurrenceRateResult(
        n_detections=n_detections,
        n_stars=n_stars,
        detection_efficiency=detection_efficiency,
        transit_probability=transit_probability,
        occurrence_rate=round(eta, 6),
        rate_lower_1sigma=round(eta_lower, 6),
        rate_upper_1sigma=round(eta_upper, 6),
        flag="OK",
    )


def format_occurrence_rate_result(r: OccurrenceRateResult) -> str:
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Detections | {r.n_detections} |\n"
        f"| Stars surveyed | {r.n_stars} |\n"
        f"| Detection efficiency | {r.detection_efficiency:.4f} |\n"
        f"| Transit probability | {r.transit_probability:.4f} |\n"
        f"| Occurrence rate η | {r.occurrence_rate:.4f} |\n"
        f"| η lower 1σ | {r.rate_lower_1sigma:.4f} |\n"
        f"| η upper 1σ | {r.rate_upper_1sigma:.4f} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Estimate planet occurrence rate.")
    p.add_argument("n_detections", type=int)
    p.add_argument("n_stars", type=int)
    p.add_argument("detection_efficiency", type=float)
    p.add_argument("transit_probability", type=float)
    args = p.parse_args()
    r = estimate_occurrence_rate(
        args.n_detections, args.n_stars,
        args.detection_efficiency, args.transit_probability,
    )
    print(format_occurrence_rate_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
