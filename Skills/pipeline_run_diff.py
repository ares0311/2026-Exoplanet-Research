"""Compute a structured diff between two pipeline runs.

Classifies each candidate as ADDED, REMOVED, IMPROVED, DEGRADED, or STABLE
by comparing FPP and rank_score between a before and after run.

Public API
----------
CandidateChange(tic_id, change_type, fpp_before, fpp_after,
                rank_score_before, rank_score_after, delta_fpp,
                delta_rank_score)
PipelineRunDiffResult(n_before, n_after, n_added, n_removed, n_changed,
                      changes, summary_markdown, flag)
diff_pipeline_runs(before, after, *, fpp_change_threshold,
                   rank_change_threshold, key) -> PipelineRunDiffResult
format_run_diff(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateChange:
    tic_id: int
    change_type: str          # "ADDED" | "REMOVED" | "IMPROVED" | "DEGRADED" | "STABLE"
    fpp_before: float | None
    fpp_after: float | None
    rank_score_before: float | None
    rank_score_after: float | None
    delta_fpp: float | None
    delta_rank_score: float | None


@dataclass(frozen=True)
class PipelineRunDiffResult:
    n_before: int
    n_after: int
    n_added: int
    n_removed: int
    n_changed: int
    changes: tuple[CandidateChange, ...]
    summary_markdown: str
    flag: str  # "OK" | "NO_CHANGE" | "EMPTY"


def _extract_fpp(row: dict) -> float | None:
    for k in ("false_positive_probability", "best_fpp", "fpp"):
        if k in row:
            try:
                return float(row[k])
            except (TypeError, ValueError):
                pass
    scores = row.get("scores", {})
    if isinstance(scores, dict):
        v = scores.get("false_positive_probability")
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    return None


def _extract_rank(row: dict) -> float | None:
    for k in ("rank_score", "score", "planet_posterior"):
        if k in row:
            try:
                return float(row[k])
            except (TypeError, ValueError):
                pass
    return None


def diff_pipeline_runs(
    before: list[dict],
    after: list[dict],
    *,
    fpp_change_threshold: float = 0.05,
    rank_change_threshold: float = 0.05,
    key: str = "tic_id",
) -> PipelineRunDiffResult:
    """Diff two pipeline runs.

    Args:
        before: Candidate list from the earlier run.
        after: Candidate list from the later run.
        fpp_change_threshold: Minimum |ΔFPP| to count as changed.
        rank_change_threshold: Minimum |Δrank_score| to count as changed.
        key: Dict key used to identify candidates (default "tic_id").

    Returns:
        :class:`PipelineRunDiffResult`.
    """
    if not before and not after:
        return PipelineRunDiffResult(0, 0, 0, 0, 0, (), "", "EMPTY")

    before_map = {str(r.get(key, i)): r for i, r in enumerate(before)}
    after_map = {str(r.get(key, i)): r for i, r in enumerate(after)}

    all_keys = set(before_map) | set(after_map)
    changes: list[CandidateChange] = []

    for k in sorted(all_keys):
        try:
            tic_id = int(k)
        except ValueError:
            tic_id = 0

        in_before = k in before_map
        in_after = k in after_map

        if in_after and not in_before:
            fpp_a = _extract_fpp(after_map[k])
            rank_a = _extract_rank(after_map[k])
            changes.append(CandidateChange(
                tic_id=tic_id, change_type="ADDED",
                fpp_before=None, fpp_after=fpp_a,
                rank_score_before=None, rank_score_after=rank_a,
                delta_fpp=None, delta_rank_score=None,
            ))
        elif in_before and not in_after:
            fpp_b = _extract_fpp(before_map[k])
            rank_b = _extract_rank(before_map[k])
            changes.append(CandidateChange(
                tic_id=tic_id, change_type="REMOVED",
                fpp_before=fpp_b, fpp_after=None,
                rank_score_before=rank_b, rank_score_after=None,
                delta_fpp=None, delta_rank_score=None,
            ))
        else:
            fpp_b = _extract_fpp(before_map[k])
            fpp_a = _extract_fpp(after_map[k])
            rank_b = _extract_rank(before_map[k])
            rank_a = _extract_rank(after_map[k])

            d_fpp = (fpp_a - fpp_b) if (fpp_a is not None and fpp_b is not None) else None
            d_rank = (rank_a - rank_b) if (rank_a is not None and rank_b is not None) else None

            fpp_changed = d_fpp is not None and abs(d_fpp) >= fpp_change_threshold
            rank_changed = d_rank is not None and abs(d_rank) >= rank_change_threshold

            if not fpp_changed and not rank_changed:
                change_type = "STABLE"
            elif (d_fpp is not None and d_fpp < -fpp_change_threshold) or (
                d_rank is not None and d_rank > rank_change_threshold
            ):
                change_type = "IMPROVED"
            else:
                change_type = "DEGRADED"

            changes.append(CandidateChange(
                tic_id=tic_id, change_type=change_type,
                fpp_before=fpp_b, fpp_after=fpp_a,
                rank_score_before=rank_b, rank_score_after=rank_a,
                delta_fpp=round(d_fpp, 4) if d_fpp is not None else None,
                delta_rank_score=round(d_rank, 4) if d_rank is not None else None,
            ))

    n_added = sum(1 for c in changes if c.change_type == "ADDED")
    n_removed = sum(1 for c in changes if c.change_type == "REMOVED")
    n_changed = sum(1 for c in changes if c.change_type in {"IMPROVED", "DEGRADED"})

    flag = "NO_CHANGE" if (n_added + n_removed + n_changed == 0) else "OK"

    summary_lines = [
        "| Change | Count |",
        "|---|---|",
        f"| Added | {n_added} |",
        f"| Removed | {n_removed} |",
        f"| Improved | {sum(1 for c in changes if c.change_type == 'IMPROVED')} |",
        f"| Degraded | {sum(1 for c in changes if c.change_type == 'DEGRADED')} |",
        f"| Stable | {sum(1 for c in changes if c.change_type == 'STABLE')} |",
    ]

    return PipelineRunDiffResult(
        n_before=len(before),
        n_after=len(after),
        n_added=n_added,
        n_removed=n_removed,
        n_changed=n_changed,
        changes=tuple(changes),
        summary_markdown="\n".join(summary_lines),
        flag=flag,
    )


def format_run_diff(result: PipelineRunDiffResult) -> str:
    """Format run diff result as Markdown."""
    lines = [
        "## Pipeline Run Diff",
        "",
        f"- Before: {result.n_before} candidates",
        f"- After: {result.n_after} candidates",
        f"- Added: {result.n_added}",
        f"- Removed: {result.n_removed}",
        f"- Changed: {result.n_changed}",
        f"- **Flag: {result.flag}**",
        "",
        result.summary_markdown,
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="pipeline_run_diff",
        description="Diff two pipeline run JSON files.",
    )
    parser.add_argument("before_json", type=str)
    parser.add_argument("after_json", type=str)
    parser.add_argument("--fpp-change-threshold", type=float, default=0.05)
    parser.add_argument("--rank-change-threshold", type=float, default=0.05)
    args = parser.parse_args(argv)

    with open(args.before_json) as f:
        before = json.load(f)
    with open(args.after_json) as f:
        after = json.load(f)

    result = diff_pipeline_runs(
        before if isinstance(before, list) else [before],
        after if isinstance(after, list) else [after],
        fpp_change_threshold=args.fpp_change_threshold,
        rank_change_threshold=args.rank_change_threshold,
    )
    print(format_run_diff(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
