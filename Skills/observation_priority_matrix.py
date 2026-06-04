"""Build a prioritised follow-up observation matrix."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass


@dataclass(frozen=True)
class ObservationPriorityEntry:
    tic_id: str
    science_score: float
    feasibility_score: float
    urgency_score: float
    composite_score: float
    rank: int
    recommendation: str


@dataclass(frozen=True)
class ObservationPriorityMatrix:
    n_targets: int
    entries: tuple[ObservationPriorityEntry, ...]
    flag: str


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def build_priority_matrix(
    targets: list[dict],
    science_weight: float = 0.50,
    feasibility_weight: float = 0.30,
    urgency_weight: float = 0.20,
) -> ObservationPriorityMatrix:
    """
    Build a ranked follow-up observation matrix.

    Each target dict may contain:
    - tic_id
    - false_positive_probability (or scores.false_positive_probability)
    - detection_confidence
    - snr / best_snr
    - n_transits_remaining (urgency: more upcoming → higher)
    - visibility_fraction (0–1, fraction of night target is up)
    - tmag (brightness, easier → higher feasibility)

    Scores are normalised to [0, 1]; composite = weighted sum.
    """
    if not targets:
        return ObservationPriorityMatrix(n_targets=0, entries=(), flag="NO_TARGETS")

    def _fpp(t: dict) -> float:
        fpp = t.get("false_positive_probability")
        if fpp is None:
            fpp = t.get("scores", {}).get("false_positive_probability", 0.5)
        return float(fpp)

    entries: list[ObservationPriorityEntry] = []
    for target in targets:
        tic_id = str(target.get("tic_id", "unknown"))
        fpp = _fpp(target)
        dc = float(target.get("detection_confidence", target.get("scores", {}).get(
            "detection_confidence", 0.5)))
        snr = float(target.get("snr", target.get("best_snr", 5.0)))
        n_tr = float(target.get("n_transits_remaining", 1))
        vis = float(target.get("visibility_fraction", 0.5))
        tmag = float(target.get("tmag", 12.0))

        # Science score: high DC, low FPP, reasonable SNR
        sci = _clamp(0.5 * (1.0 - fpp) + 0.3 * dc + 0.2 * min(snr / 20.0, 1.0))

        # Feasibility: visible, bright enough, multiple transits available
        tmag_score = _clamp((15.0 - tmag) / 8.0)  # 7–15 mag → 1–0
        feas = _clamp(0.5 * vis + 0.5 * tmag_score)

        # Urgency: transit coming up soon, few remaining
        urg = _clamp(min(n_tr / 5.0, 1.0))

        composite = (
            science_weight * sci
            + feasibility_weight * feas
            + urgency_weight * urg
        )

        recommendation = (
            "OBSERVE_NOW" if composite >= 0.70
            else "SCHEDULE" if composite >= 0.40
            else "LOW_PRIORITY"
        )
        entries.append(ObservationPriorityEntry(
            tic_id=tic_id,
            science_score=round(sci, 3),
            feasibility_score=round(feas, 3),
            urgency_score=round(urg, 3),
            composite_score=round(composite, 3),
            rank=0,
            recommendation=recommendation,
        ))

    entries.sort(key=lambda e: e.composite_score, reverse=True)
    ranked = tuple(
        ObservationPriorityEntry(
            tic_id=e.tic_id, science_score=e.science_score,
            feasibility_score=e.feasibility_score, urgency_score=e.urgency_score,
            composite_score=e.composite_score, rank=i + 1,
            recommendation=e.recommendation,
        )
        for i, e in enumerate(entries)
    )

    return ObservationPriorityMatrix(n_targets=len(ranked), entries=ranked, flag="OK")


def format_priority_matrix(r: ObservationPriorityMatrix) -> str:
    if r.flag != "OK":
        return f"No targets (flag: {r.flag}).\n"
    lines = [
        f"**Observation Priority Matrix** — {r.n_targets} targets\n",
        "| Rank | TIC ID | Science | Feasibility | Urgency | Composite | Action |",
        "|---|---|---|---|---|---|---|",
    ]
    for e in r.entries:
        lines.append(
            f"| {e.rank} | {e.tic_id} | {e.science_score:.3f} | "
            f"{e.feasibility_score:.3f} | {e.urgency_score:.3f} | "
            f"{e.composite_score:.3f} | {e.recommendation} |"
        )
    return "\n".join(lines)


def _cli() -> int:
    p = argparse.ArgumentParser(description="Build observation priority matrix.")
    p.add_argument("targets_json", help="JSON array of target dicts or @file")
    args = p.parse_args()
    raw = args.targets_json
    if raw.startswith("@"):
        with open(raw[1:]) as f:
            targets = json.load(f)
    else:
        targets = json.loads(raw)
    r = build_priority_matrix(targets)
    print(format_priority_matrix(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
