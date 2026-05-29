"""Assign a letter grade to signal quality based on diagnostic scores.

Combines SNR, depth consistency, centroid stability, and false-positive
probability into a single A–F grade with a breakdown of contributing factors.

Public API
----------
GradeFactor(name, score, weight, contribution)
SignalGrade(tic_id, period_days, grade, numeric_score, factors,
            summary, flag)
grade_signal_quality(row, *, weights) -> SignalGrade
format_signal_grade(grade) -> str
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass


@dataclass(frozen=True)
class GradeFactor:
    name: str
    score: float     # [0, 1]
    weight: float    # contribution weight
    contribution: float  # score * weight


@dataclass(frozen=True)
class SignalGrade:
    tic_id: int | None
    period_days: float | None
    grade: str          # "A" | "B" | "C" | "D" | "F"
    numeric_score: float  # [0, 1]
    factors: tuple[GradeFactor, ...]
    summary: str
    flag: str  # "OK" | "INCOMPLETE"


_GRADE_THRESHOLDS = [
    (0.85, "A"),
    (0.70, "B"),
    (0.55, "C"),
    (0.40, "D"),
    (0.00, "F"),
]


def _safe_float(v: object) -> float | None:
    with contextlib.suppress(TypeError, ValueError):
        return float(v)  # type: ignore[arg-type]
    return None


def _scores_dict(row: dict) -> dict:
    return row.get("scores") or {}


def _snr_score(row: dict) -> float | None:
    snr = _safe_float(row.get("snr") or row.get("detection_snr"))
    if snr is None:
        return None
    return min(snr / 20.0, 1.0)


def _fpp_score(row: dict) -> float | None:
    fpp = _safe_float(
        row.get("false_positive_probability")
        or _scores_dict(row).get("false_positive_probability")
    )
    if fpp is None:
        return None
    return 1.0 - fpp


def _detection_confidence_score(row: dict) -> float | None:
    dc = _safe_float(
        row.get("detection_confidence")
        or _scores_dict(row).get("detection_confidence")
    )
    return dc


def _novelty_score(row: dict) -> float | None:
    return _safe_float(
        row.get("novelty_score") or _scores_dict(row).get("novelty_score")
    )


_DEFAULT_WEIGHTS = {
    "snr": 0.30,
    "fpp": 0.35,
    "detection_confidence": 0.25,
    "novelty": 0.10,
}


def grade_signal_quality(
    row: dict,
    *,
    weights: dict[str, float] | None = None,
) -> SignalGrade:
    """Assign a letter grade to a candidate signal.

    Args:
        row: Pipeline output dict.
        weights: Optional override for factor weights. Keys: snr, fpp,
                 detection_confidence, novelty.

    Returns:
        SignalGrade with letter grade, numeric score, and factor breakdown.
    """
    w = {**_DEFAULT_WEIGHTS, **(weights or {})}

    extractors = {
        "snr": _snr_score,
        "fpp": _fpp_score,
        "detection_confidence": _detection_confidence_score,
        "novelty": _novelty_score,
    }

    factors: list[GradeFactor] = []
    total_weight = 0.0
    weighted_sum = 0.0
    n_missing = 0

    for name, extractor in extractors.items():
        score = extractor(row)
        weight = w.get(name, 0.0)
        if score is None:
            n_missing += 1
            score = 0.5  # neutral fallback for missing diagnostics
        contribution = score * weight
        weighted_sum += contribution
        total_weight += weight
        factors.append(GradeFactor(
            name=name,
            score=round(score, 4),
            weight=weight,
            contribution=round(contribution, 4),
        ))

    numeric_score = weighted_sum / total_weight if total_weight > 0 else 0.0
    numeric_score = round(numeric_score, 4)

    grade = "F"
    for threshold, letter in _GRADE_THRESHOLDS:
        if numeric_score >= threshold:
            grade = letter
            break

    tic_id: int | None = None
    with contextlib.suppress(TypeError, ValueError):
        raw = row.get("tic_id")
        if raw is not None:
            tic_id = int(raw)

    period = _safe_float(row.get("period_days"))
    flag = "INCOMPLETE" if n_missing >= 3 else "OK"

    grade_desc = {
        "A": "Excellent — strong candidate for follow-up",
        "B": "Good — warrants further investigation",
        "C": "Marginal — additional vetting recommended",
        "D": "Poor — likely false positive",
        "F": "Fail — insufficient evidence or high FPP",
    }
    summary = grade_desc.get(grade, "")

    return SignalGrade(
        tic_id=tic_id,
        period_days=period,
        grade=grade,
        numeric_score=numeric_score,
        factors=tuple(factors),
        summary=summary,
        flag=flag,
    )


def format_signal_grade(sg: SignalGrade) -> str:
    """Format signal grade as Markdown.

    Args:
        sg: SignalGrade to format.

    Returns:
        Markdown string.
    """
    tic_str = str(sg.tic_id) if sg.tic_id is not None else "Unknown"
    period_str = f"{sg.period_days:.4f} d" if sg.period_days is not None else "—"
    lines = [
        f"## Signal Quality Grade — TIC {tic_str}\n",
        f"**Grade**: **{sg.grade}** (score: {sg.numeric_score:.3f}) | "
        f"Period: {period_str} | Status: `{sg.flag}`\n",
        f"_{sg.summary}_\n",
        "",
        "| Factor | Score | Weight | Contribution |",
        "|---|---|---|---|",
    ]
    for f in sg.factors:
        lines.append(
            f"| {f.name} | {f.score:.3f} | {f.weight:.2f} | {f.contribution:.4f} |"
        )
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Grade signal quality.")
    parser.add_argument("input", help="Candidate JSON file.")
    args = parser.parse_args(argv)

    from pathlib import Path
    row = json.loads(Path(args.input).read_text())
    if isinstance(row, list):
        row = row[0]
    sg = grade_signal_quality(row)
    print(format_signal_grade(sg))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
