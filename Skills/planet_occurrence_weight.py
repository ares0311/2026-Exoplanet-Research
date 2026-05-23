"""Compute transit survey completeness weight for occurrence-rate statistics.

The occurrence weight w = 1 / (p_det × p_tr) corrects for the fact that
not all planets would have been detected (p_det) and not all would be
observed to transit (p_tr).

Public API
----------
OccurrenceWeightResult(snr, transit_probability, p_det, p_transit,
                       weight, is_reliable, flag)
compute_occurrence_weight(snr, transit_prob, *,
                          snr_threshold, completeness_model) -> OccurrenceWeightResult
format_occurrence_weight_result(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class OccurrenceWeightResult:
    snr: float | None
    transit_probability: float | None
    p_det: float | None           # detection completeness P(detected | transiting)
    p_transit: float | None       # geometric transit probability
    weight: float | None          # 1 / (p_det * p_transit)
    is_reliable: bool             # weight < max_weight cap
    flag: str  # "OK" | "INVALID"


_MAX_WEIGHT = 1000.0  # cap to prevent extreme weights


def _gaussian_cdf(x: float) -> float:
    """Gaussian CDF Φ(x) using math.erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _step_completeness(snr: float, snr_threshold: float) -> float:
    """Step function: 1 if SNR >= threshold, 0 otherwise."""
    return 1.0 if snr >= snr_threshold else 0.0


def _sigmoid_completeness(snr: float, snr_threshold: float) -> float:
    """Sigmoid completeness: Φ((SNR - threshold) / (0.5 * threshold))."""
    sigma = 0.5 * snr_threshold
    return _gaussian_cdf((snr - snr_threshold) / sigma)


def compute_occurrence_weight(
    snr: float | None,
    transit_prob: float | None,
    *,
    snr_threshold: float = 7.1,
    completeness_model: str = "sigmoid",
) -> OccurrenceWeightResult:
    """Compute occurrence-rate completeness weight.

    Args:
        snr: Measured or expected transit detection SNR.
        transit_prob: Geometric transit probability P_tr in [0, 1].
        snr_threshold: SNR above which a transit is detectable.
        completeness_model: ``"sigmoid"`` (default, smooth ramp) or
            ``"step"`` (hard threshold).

    Returns:
        :class:`OccurrenceWeightResult`.
    """
    if snr is not None and snr < 0:
        return OccurrenceWeightResult(snr, transit_prob, None, None, None, False, "INVALID")
    if transit_prob is not None and not (0 < transit_prob <= 1):
        return OccurrenceWeightResult(snr, transit_prob, None, None, None, False, "INVALID")

    p_det: float | None = None
    if snr is not None:
        if completeness_model == "step":
            p_det = _step_completeness(snr, snr_threshold)
        else:
            p_det = _sigmoid_completeness(snr, snr_threshold)

    weight: float | None = None
    is_reliable = False
    if p_det is not None and transit_prob is not None:
        denom = p_det * transit_prob
        if denom > 0:
            weight = min(_MAX_WEIGHT, 1.0 / denom)
            is_reliable = weight < _MAX_WEIGHT
        else:
            weight = _MAX_WEIGHT
            is_reliable = False

    return OccurrenceWeightResult(
        snr=snr,
        transit_probability=transit_prob,
        p_det=round(p_det, 6) if p_det is not None else None,
        p_transit=transit_prob,
        weight=round(weight, 4) if weight is not None else None,
        is_reliable=is_reliable,
        flag="OK",
    )


def format_occurrence_weight_result(result: OccurrenceWeightResult) -> str:
    """Format occurrence weight result as Markdown."""
    lines = [
        "## Planet Occurrence Weight",
        "",
        f"- SNR: {result.snr}",
        f"- Transit probability: {result.transit_probability}",
        f"- Detection completeness p_det: {result.p_det}",
        f"- **Occurrence weight: {result.weight}**",
        f"- Reliable (weight < 1000): {'Yes' if result.is_reliable else 'No'}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="planet_occurrence_weight",
        description="Compute transit survey completeness weight for occurrence statistics.",
    )
    parser.add_argument("--snr", type=float, default=None)
    parser.add_argument("--transit-prob", type=float, default=None)
    parser.add_argument("--snr-threshold", type=float, default=7.1)
    parser.add_argument("--model", default="sigmoid", choices=["sigmoid", "step"])
    args = parser.parse_args(argv)

    result = compute_occurrence_weight(
        args.snr, args.transit_prob,
        snr_threshold=args.snr_threshold,
        completeness_model=args.model,
    )
    print(format_occurrence_weight_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
