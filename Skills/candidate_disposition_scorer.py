"""Candidate disposition scorer.

Combines FPP, odd/even significance, secondary SNR, and centroid offset into
per-disposition probability scores and assigns a preliminary disposition label.

Public API
----------
DispositionResult(disposition, pc_score, fp_score, eb_score, neb_score, flag)
score_disposition(fpp, odd_even_significance, secondary_snr,
                  centroid_offset_arcsec) -> DispositionResult
format_disposition_result(result) -> str
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass


def _clip(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


@dataclass(frozen=True)
class DispositionResult:
    disposition: str
    pc_score: float
    fp_score: float
    eb_score: float
    neb_score: float
    flag: str


def score_disposition(
    fpp: float,
    odd_even_significance: float,
    secondary_snr: float,
    centroid_offset_arcsec: float,
) -> DispositionResult:
    """Derive per-disposition scores and assign a disposition label.

    Parameters
    ----------
    fpp:                    false-positive probability in [0, 1]
    odd_even_significance:  odd/even depth difference in sigma
    secondary_snr:          secondary-eclipse SNR
    centroid_offset_arcsec: in-transit centroid offset in arcseconds

    Returns
    -------
    DispositionResult with disposition one of PC, FP, EB, NEB, UNKNOWN and
    flag one of INVALID_FPP, OK.
    """
    if fpp < 0.0 or fpp > 1.0:
        return DispositionResult(
            disposition="UNKNOWN",
            pc_score=0.0,
            fp_score=0.0,
            eb_score=0.0,
            neb_score=0.0,
            flag="INVALID_FPP",
        )

    # Planet-candidate score
    pc_score = 1.0
    if fpp > 0.5:
        pc_score -= 0.4
    if odd_even_significance > 3.0:
        pc_score -= 0.3
    if secondary_snr > 3.0:
        pc_score -= 0.2
    if centroid_offset_arcsec > 3.0:
        pc_score -= 0.3
    pc_score = _clip(pc_score)

    # Eclipsing-binary score
    oe_contrib = odd_even_significance / 5.0 if odd_even_significance < 5.0 else 1.0
    sec_contrib = secondary_snr / 5.0 if secondary_snr < 5.0 else 1.0
    eb_score = _clip(0.2 + 0.5 * oe_contrib + 0.3 * sec_contrib)

    # Nearby/background eclipsing-binary score
    cen_contrib = centroid_offset_arcsec / 5.0 if centroid_offset_arcsec < 5.0 else 1.0
    neb_score = _clip(0.1 + 0.8 * cen_contrib)

    # False-positive score
    fp_score = _clip(0.3 * (1.0 - pc_score) + 0.3 * eb_score + 0.3 * neb_score)

    # Disposition: argmax of (pc, fp, eb, neb)
    scores = {
        "PC": pc_score,
        "FP": fp_score,
        "EB": eb_score,
        "NEB": neb_score,
    }
    max_val = max(scores.values())
    winners = [k for k, v in scores.items() if v == max_val]
    disposition = "UNKNOWN" if len(winners) == len(scores) else winners[0]

    return DispositionResult(
        disposition=disposition,
        pc_score=pc_score,
        fp_score=fp_score,
        eb_score=eb_score,
        neb_score=neb_score,
        flag="OK",
    )


def format_disposition_result(result: DispositionResult) -> str:
    """Return a Markdown table summarising the disposition scores."""
    lines = [
        "| Metric | Value |",
        "| --- | --- |",
        f"| Disposition | {result.disposition} |",
        f"| PC Score | {result.pc_score:.4f} |",
        f"| FP Score | {result.fp_score:.4f} |",
        f"| EB Score | {result.eb_score:.4f} |",
        f"| NEB Score | {result.neb_score:.4f} |",
        f"| Flag | {result.flag} |",
    ]
    return "\n".join(lines)


def _cli() -> int:
    parser = argparse.ArgumentParser(
        description="Score a transit candidate disposition from vetting metrics."
    )
    parser.add_argument(
        "--fpp",
        type=float,
        required=True,
        metavar="PROB",
        help="False-positive probability [0, 1].",
    )
    parser.add_argument(
        "--odd-even-significance",
        type=float,
        default=0.0,
        metavar="SIGMA",
        help="Odd/even depth difference in sigma (default 0).",
    )
    parser.add_argument(
        "--secondary-snr",
        type=float,
        default=0.0,
        metavar="SNR",
        help="Secondary eclipse SNR (default 0).",
    )
    parser.add_argument(
        "--centroid-offset",
        type=float,
        default=0.0,
        metavar="ARCSEC",
        help="In-transit centroid offset in arcsec (default 0).",
    )
    args = parser.parse_args()
    result = score_disposition(
        args.fpp,
        args.odd_even_significance,
        args.secondary_snr,
        args.centroid_offset,
    )
    print(format_disposition_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
