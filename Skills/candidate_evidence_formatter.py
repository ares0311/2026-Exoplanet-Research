"""Format structured evidence reports for transit candidates.

Public API:
    EvidenceReport       -- frozen dataclass
    format_evidence_report(tic_id, period_days, fpp, scores) -> EvidenceReport
    format_evidence_text(report) -> str
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceReport:
    tic_id: str
    period_days: float
    disposition: str
    supporting_evidence: list[str]
    counter_evidence: list[str]
    n_supporting: int
    n_counter: int
    confidence_label: str
    flag: str


def format_evidence_report(
    tic_id: str,
    period_days: float,
    fpp: float,
    scores: dict[str, float],
) -> EvidenceReport:
    if fpp < 0.0 or fpp > 1.0:
        return EvidenceReport(
            tic_id=tic_id,
            period_days=period_days,
            disposition="UNKNOWN",
            supporting_evidence=[],
            counter_evidence=[],
            n_supporting=0,
            n_counter=0,
            confidence_label="LOW",
            flag="INVALID_FPP",
        )

    supporting: list[str] = []
    counter: list[str] = []

    for key, value in scores.items():
        if "centroid" in key and value > 0.5:
            counter.append("Centroid motion detected")
        elif "odd_even" in key and value > 0.5:
            counter.append("Odd-even depth asymmetry")
        elif "secondary" in key and value > 0.5:
            counter.append("Secondary eclipse detected")
        elif "depth_consistency" in key and value > 0.5:
            supporting.append("Consistent transit depths")
        elif "stellar_density" in key and value > 0.5:
            supporting.append("Stellar density consistent with planet")

    if fpp < 0.1:
        supporting.append("Low false positive probability")
    if fpp > 0.5:
        counter.append("High false positive probability")

    n_counter = len(counter)
    if fpp < 0.3 and n_counter == 0:
        disposition = "PC"
    elif fpp > 0.7:
        disposition = "FP"
    else:
        disposition = "APC"

    if fpp < 0.1 and n_counter == 0:
        confidence_label = "HIGH"
    elif fpp < 0.5:
        confidence_label = "MEDIUM"
    else:
        confidence_label = "LOW"

    return EvidenceReport(
        tic_id=tic_id,
        period_days=period_days,
        disposition=disposition,
        supporting_evidence=supporting,
        counter_evidence=counter,
        n_supporting=len(supporting),
        n_counter=n_counter,
        confidence_label=confidence_label,
        flag="OK",
    )


def format_evidence_text(report: EvidenceReport) -> str:
    lines = [
        f"## Evidence Report: TIC {report.tic_id}",
        "",
        f"**Period:** {report.period_days:.4f} days",
        f"**Disposition:** {report.disposition}",
        f"**Confidence:** {report.confidence_label}",
        "",
        "### Supporting Evidence",
    ]
    if report.supporting_evidence:
        for item in report.supporting_evidence:
            lines.append(f"- {item}")
    else:
        lines.append("- None")
    lines.append("")
    lines.append("### Counter Evidence")
    if report.counter_evidence:
        for item in report.counter_evidence:
            lines.append(f"- {item}")
    else:
        lines.append("- None")
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Format a candidate evidence report.")
    parser.add_argument("tic_id", type=str)
    parser.add_argument("period_days", type=float)
    parser.add_argument("fpp", type=float)
    parser.add_argument(
        "--scores",
        type=str,
        default="{}",
        help="JSON object of diagnostic scores.",
    )
    args = parser.parse_args()
    scores = json.loads(args.scores)
    report = format_evidence_report(args.tic_id, args.period_days, args.fpp, scores)
    print(format_evidence_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
