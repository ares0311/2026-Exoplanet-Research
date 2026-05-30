"""Apply configurable thresholds to candidate scores and return a GO/NO-GO decision.

Public API:
    ThresholdDecision  -- frozen dataclass
    make_threshold_decision(fpp, snr, detection_confidence, n_transits,
                             *, fpp_max, snr_min, dc_min, n_transits_min) -> ThresholdDecision
    format_decision_result(result) -> str
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class ThresholdDecision:
    fpp: float
    snr: float
    detection_confidence: float
    n_transits: int
    go: bool
    failed_criteria: list[str]
    flag: str


def make_threshold_decision(
    fpp: float,
    snr: float,
    detection_confidence: float,
    n_transits: int,
    *,
    fpp_max: float = 0.10,
    snr_min: float = 7.0,
    dc_min: float = 0.80,
    n_transits_min: int = 2,
) -> ThresholdDecision:
    if not (0.0 <= fpp <= 1.0):
        return ThresholdDecision(
            fpp=fpp, snr=snr, detection_confidence=detection_confidence,
            n_transits=n_transits, go=False, failed_criteria=["invalid_fpp"],
            flag="INVALID_INPUT",
        )
    if not (0.0 <= detection_confidence <= 1.0):
        return ThresholdDecision(
            fpp=fpp, snr=snr, detection_confidence=detection_confidence,
            n_transits=n_transits, go=False, failed_criteria=["invalid_dc"],
            flag="INVALID_INPUT",
        )
    failed: list[str] = []
    if fpp > fpp_max:
        failed.append(f"fpp>{fpp_max}")
    if snr < snr_min:
        failed.append(f"snr<{snr_min}")
    if detection_confidence < dc_min:
        failed.append(f"dc<{dc_min}")
    if n_transits < n_transits_min:
        failed.append(f"n_transits<{n_transits_min}")
    go = len(failed) == 0
    flag = "GO" if go else "NO_GO"
    return ThresholdDecision(
        fpp=fpp,
        snr=snr,
        detection_confidence=detection_confidence,
        n_transits=n_transits,
        go=go,
        failed_criteria=failed,
        flag=flag,
    )


def format_decision_result(result: ThresholdDecision) -> str:
    criteria = ", ".join(result.failed_criteria) if result.failed_criteria else "none"
    lines = [
        "## Threshold Decision",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| FPP | {result.fpp:.4f} |",
        f"| SNR | {result.snr:.2f} |",
        f"| Detection Confidence | {result.detection_confidence:.4f} |",
        f"| N Transits | {result.n_transits} |",
        f"| GO | {result.go} |",
        f"| Failed Criteria | {criteria} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Apply thresholds to candidate scores.")
    parser.add_argument("fpp", type=float)
    parser.add_argument("snr", type=float)
    parser.add_argument("detection_confidence", type=float)
    parser.add_argument("n_transits", type=int)
    parser.add_argument("--fpp-max", type=float, default=0.10)
    parser.add_argument("--snr-min", type=float, default=7.0)
    parser.add_argument("--dc-min", type=float, default=0.80)
    parser.add_argument("--n-transits-min", type=int, default=2)
    args = parser.parse_args()
    result = make_threshold_decision(
        args.fpp, args.snr, args.detection_confidence, args.n_transits,
        fpp_max=args.fpp_max, snr_min=args.snr_min, dc_min=args.dc_min,
        n_transits_min=args.n_transits_min,
    )
    print(format_decision_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
