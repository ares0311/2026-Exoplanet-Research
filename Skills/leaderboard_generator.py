"""Generate a ranked leaderboard from scan-log records.

Reads a scan_log JSON (produced by ``star_scanner.py``) or a list of
per-target result dicts and ranks contributors (or targets) by configurable
metrics: candidates found, scanned count, mean FPP improvement, etc.

In the contributor mode the log must contain an ``"author"`` field per entry
(populated by the caller); otherwise targets are ranked directly.

Public API
----------
LeaderboardEntry(rank, name, n_scanned, n_candidates, n_clear,
                 n_errors, best_fpp, mean_fpp, score)
LeaderboardResult(mode, metric, n_entries, entries, generated_at, flag)
generate_leaderboard(records, *, mode, metric, top_n) -> LeaderboardResult
format_leaderboard(result) -> str
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass


@dataclass(frozen=True)
class LeaderboardEntry:
    rank: int
    name: str                     # tic_id (str) or author handle
    n_scanned: int
    n_candidates: int
    n_clear: int
    n_errors: int
    best_fpp: float | None
    mean_fpp: float | None
    score: float                  # composite sort key


@dataclass(frozen=True)
class LeaderboardResult:
    mode: str          # "target" | "contributor"
    metric: str        # "candidates" | "score" | "scanned"
    n_entries: int
    entries: tuple[LeaderboardEntry, ...]
    generated_at: str  # ISO-8601
    flag: str          # "OK" | "EMPTY"


def _safe_float(v: object) -> float | None:
    try:
        f = float(v)  # type: ignore[arg-type]
        return f
    except (TypeError, ValueError):
        return None


def _composite_score(n_candidates: int, best_fpp: float | None, n_scanned: int) -> float:
    """Higher is better: rewards candidates found, low FPP, large scan count."""
    cand_score = n_candidates * 10.0
    fpp_score = (1.0 - best_fpp) * 5.0 if best_fpp is not None else 0.0
    scan_bonus = min(n_scanned / 10.0, 5.0)
    return cand_score + fpp_score + scan_bonus


def generate_leaderboard(
    records: list[dict],
    *,
    mode: str = "target",
    metric: str = "score",
    top_n: int = 20,
) -> LeaderboardResult:
    """Generate a ranked leaderboard from scan records.

    Args:
        records: List of result dicts from star_scanner / batch_scan.
            Each dict should contain ``status``, ``tic_id``,
            and optionally ``best_fpp``, ``author``.
        mode: ``"target"`` ranks TIC IDs; ``"contributor"`` groups by ``author``.
        metric: Sort key — ``"score"`` (composite), ``"candidates"``,
            or ``"scanned"``.
        top_n: Maximum number of entries to return.

    Returns:
        :class:`LeaderboardResult`.
    """
    if not records:
        return LeaderboardResult(mode, metric, 0, (), _now(), "EMPTY")

    groups: dict[str, dict] = {}

    for rec in records:
        if mode == "contributor":
            key = str(rec.get("author", "unknown"))
        else:
            key = str(rec.get("tic_id", rec.get("TIC_ID", "unknown")))

        if key not in groups:
            groups[key] = {
                "n_scanned": 0,
                "n_candidates": 0,
                "n_clear": 0,
                "n_errors": 0,
                "fpps": [],
            }

        g = groups[key]
        g["n_scanned"] += 1
        status = rec.get("status", "")
        if status == "candidate_found":
            g["n_candidates"] += 1
        elif status == "scanned_clear":
            g["n_clear"] += 1
        elif status == "error":
            g["n_errors"] += 1

        fpp = _safe_float(rec.get("best_fpp"))
        if fpp is not None:
            g["fpps"].append(fpp)

    entries_unsorted: list[LeaderboardEntry] = []
    for name, g in groups.items():
        fpps = g["fpps"]
        best_fpp = min(fpps) if fpps else None
        mean_fpp = sum(fpps) / len(fpps) if fpps else None
        score = _composite_score(g["n_candidates"], best_fpp, g["n_scanned"])
        entries_unsorted.append(LeaderboardEntry(
            rank=0,
            name=name,
            n_scanned=g["n_scanned"],
            n_candidates=g["n_candidates"],
            n_clear=g["n_clear"],
            n_errors=g["n_errors"],
            best_fpp=round(best_fpp, 4) if best_fpp is not None else None,
            mean_fpp=round(mean_fpp, 4) if mean_fpp is not None else None,
            score=round(score, 3),
        ))

    if metric == "candidates":
        sort_key = lambda e: (-e.n_candidates, -e.score)  # noqa: E731
    elif metric == "scanned":
        sort_key = lambda e: (-e.n_scanned, -e.score)  # noqa: E731
    else:
        sort_key = lambda e: (-e.score, -e.n_candidates)  # noqa: E731

    sorted_entries = sorted(entries_unsorted, key=sort_key)[:top_n]
    ranked = tuple(
        LeaderboardEntry(
            rank=i + 1,
            name=e.name,
            n_scanned=e.n_scanned,
            n_candidates=e.n_candidates,
            n_clear=e.n_clear,
            n_errors=e.n_errors,
            best_fpp=e.best_fpp,
            mean_fpp=e.mean_fpp,
            score=e.score,
        )
        for i, e in enumerate(sorted_entries)
    )

    return LeaderboardResult(
        mode=mode,
        metric=metric,
        n_entries=len(ranked),
        entries=ranked,
        generated_at=_now(),
        flag="OK",
    )


def _now() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def format_leaderboard(result: LeaderboardResult) -> str:
    """Format leaderboard as a GitHub-flavored Markdown table."""
    lines = [
        f"## Leaderboard ({result.mode.title()}, sorted by {result.metric})",
        "",
        f"Generated: {result.generated_at}",
        "",
        "| Rank | Name | Scanned | Candidates | Clear | Errors | Best FPP | Score |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in result.entries:
        fpp_str = f"{e.best_fpp:.3f}" if e.best_fpp is not None else "—"
        lines.append(
            f"| {e.rank} | {e.name} | {e.n_scanned} | {e.n_candidates} "
            f"| {e.n_clear} | {e.n_errors} | {fpp_str} | {e.score:.1f} |"
        )
    lines.append("")
    lines.append(f"*Total entries: {result.n_entries} — Flag: **{result.flag}***")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="leaderboard_generator",
        description="Generate a ranked leaderboard from scan records.",
    )
    parser.add_argument("scan_log", type=str)
    parser.add_argument("--mode", choices=["target", "contributor"], default="target")
    parser.add_argument("--metric", choices=["score", "candidates", "scanned"],
                        default="score")
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args(argv)

    with open(args.scan_log) as f:
        data = json.load(f)

    if isinstance(data, dict) and "entries" in data:
        records = list(data["entries"].values())
    elif isinstance(data, list):
        records = data
    else:
        records = [data]

    result = generate_leaderboard(records, mode=args.mode, metric=args.metric,
                                  top_n=args.top_n)
    print(format_leaderboard(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
