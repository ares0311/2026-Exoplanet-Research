"""Rank unlabelled candidates by prediction uncertainty for active labelling.

Lower absolute distance from 0.5 = higher uncertainty = higher priority for
a human annotator.  Shannon entropy is provided as an additional uncertainty
metric (maximised at p=0.5).

Public API
----------
ActiveLearningEntry(tic_id, score, uncertainty, entropy, rank)
ActiveLearningResult(entries, n_candidates, n_returned, flag)
rank_by_uncertainty(candidates, *, top_n) -> ActiveLearningResult
format_active_learning(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ActiveLearningEntry:
    tic_id: str
    score: float          # model probability
    uncertainty: float    # abs(score - 0.5) — lower = more uncertain
    entropy: float        # -p*log2(p) - (1-p)*log2(1-p)
    rank: int             # 1 = most uncertain


@dataclass(frozen=True)
class ActiveLearningResult:
    entries: tuple[ActiveLearningEntry, ...]
    n_candidates: int
    n_returned: int
    flag: str   # "OK" | "EMPTY" | "INVALID"


def _binary_entropy(p: float) -> float:
    """Compute binary Shannon entropy H(p) = -p log2(p) - (1-p) log2(1-p)."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


def rank_by_uncertainty(
    candidates: list[dict],
    *,
    top_n: int | None = None,
) -> ActiveLearningResult:
    """Rank candidates by model prediction uncertainty.

    Candidates closest to a score of 0.5 are ranked first (most uncertain,
    most informative for active learning).

    Args:
        candidates: List of dicts, each with ``"tic_id"`` (str) and
            ``"score"`` (float in [0, 1]).
        top_n: If given, return only the top N most uncertain candidates.

    Returns:
        :class:`ActiveLearningResult`.
    """
    if len(candidates) == 0:
        return ActiveLearningResult(entries=(), n_candidates=0, n_returned=0, flag="EMPTY")

    # Validate inputs
    entries_raw: list[tuple[str, float]] = []
    for item in candidates:
        if not isinstance(item, dict):
            return ActiveLearningResult(
                entries=(), n_candidates=len(candidates), n_returned=0, flag="INVALID"
            )
        try:
            tic_id = str(item["tic_id"])
            score = float(item["score"])
        except (KeyError, TypeError, ValueError):
            return ActiveLearningResult(
                entries=(), n_candidates=len(candidates), n_returned=0, flag="INVALID"
            )
        entries_raw.append((tic_id, score))

    # Sort by ascending uncertainty (= ascending abs(score - 0.5))
    entries_raw.sort(key=lambda x: abs(x[1] - 0.5))

    n_candidates = len(entries_raw)
    if top_n is not None:
        entries_raw = entries_raw[:top_n]

    entries = tuple(
        ActiveLearningEntry(
            tic_id=tic_id,
            score=score,
            uncertainty=abs(score - 0.5),
            entropy=_binary_entropy(score),
            rank=rank,
        )
        for rank, (tic_id, score) in enumerate(entries_raw, start=1)
    )

    return ActiveLearningResult(
        entries=entries,
        n_candidates=n_candidates,
        n_returned=len(entries),
        flag="OK",
    )


def format_active_learning(result: ActiveLearningResult) -> str:
    """Format an :class:`ActiveLearningResult` as a Markdown string."""
    lines = [
        "## Active Learning Uncertainty Ranker",
        "",
        f"- **Total candidates:** {result.n_candidates}",
        f"- **Returned:** {result.n_returned}",
        f"- **Flag:** {result.flag}",
        "",
        "| Rank | TIC ID | Score | Uncertainty | Entropy |",
        "|------|--------|-------|-------------|---------|",
    ]
    for entry in result.entries:
        lines.append(
            f"| {entry.rank} | {entry.tic_id} | {entry.score:.4f} "
            f"| {entry.uncertainty:.4f} | {entry.entropy:.4f} |"
        )
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="active_learning_scorer",
        description="Rank unlabelled candidates by prediction uncertainty.",
    )
    parser.add_argument(
        "input",
        help='JSON file with list of {"tic_id": str, "score": float} dicts.',
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=None,
        help="Return only the top N most uncertain candidates.",
    )
    args = parser.parse_args(argv)

    with open(args.input) as fh:  # noqa: PTH123
        candidates = json.load(fh)

    result = rank_by_uncertainty(candidates, top_n=args.top_n)
    print(format_active_learning(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
