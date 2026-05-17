"""Merge and deduplicate pipeline results from multiple runs.

When the same TIC ID appears in multiple result files, the highest-ranked
(lowest FPP) entry is kept.  Entries from different TIC IDs are all retained.

Public API
----------
merge_candidates(paths, *, prefer) -> list[dict]
write_merged(rows, output_path) -> Path
format_merge_summary(rows, n_sources) -> str
"""
from __future__ import annotations

import json
from pathlib import Path


def _load_file(path: Path) -> list[dict]:
    """Load a single result file (list or single dict)."""
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        # May be {tic_id: row} or a wrapped {"results": [...]} shape
        if "results" in raw:
            return list(raw["results"])
        return [raw]
    return []


def _fpp(row: dict) -> float:
    """Extract FPP from a candidate row."""
    for key in ("best_fpp", "false_positive_probability"):
        val = row.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    scores = row.get("scores") or {}
    val = scores.get("false_positive_probability")
    if val is not None:
        try:
            return float(val)
        except (TypeError, ValueError):
            pass
    return 1.0


def _rank_score(row: dict) -> float:
    """Extract rank_score if present; fallback to 1 - FPP."""
    val = row.get("rank_score")
    if val is not None:
        try:
            return float(val)
        except (TypeError, ValueError):
            pass
    return 1.0 - _fpp(row)


def merge_candidates(
    paths: list[Path | str],
    *,
    prefer: str = "best_fpp",
) -> list[dict]:
    """Merge candidate rows from multiple files, deduplicating by TIC ID.

    Args:
        paths: List of JSON result files.
        prefer: Which row to keep when duplicates exist.
            ``"best_fpp"`` keeps lowest FPP; ``"rank_score"`` keeps highest.

    Returns:
        Deduplicated list of candidate dicts, sorted by FPP ascending.
    """
    all_rows: list[dict] = []
    for p in paths:
        rows = _load_file(Path(p))
        for row in rows:
            if isinstance(row, dict):
                all_rows.append(row)

    # Deduplicate by tic_id
    best: dict[int, dict] = {}
    for row in all_rows:
        tic_id = row.get("tic_id")
        if tic_id is None:
            continue
        try:
            tid = int(tic_id)
        except (TypeError, ValueError):
            continue

        if tid not in best:
            best[tid] = row
        else:
            existing = best[tid]
            if prefer == "rank_score":
                if _rank_score(row) > _rank_score(existing):
                    best[tid] = row
            else:
                if _fpp(row) < _fpp(existing):
                    best[tid] = row

    merged = list(best.values())
    merged.sort(key=_fpp)
    return merged


def write_merged(rows: list[dict], output_path: Path | str) -> Path:
    """Write merged candidate list to a JSON file."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(rows, indent=2))
    return p


def format_merge_summary(rows: list[dict], n_sources: int) -> str:
    """Format a summary of merged candidates as Markdown."""
    candidate_found = [r for r in rows if r.get("status") == "candidate_found"
                       or _fpp(r) < 0.5]
    lines = [
        "## Candidate Merge Summary",
        "",
        f"- Source files: {n_sources}",
        f"- Unique targets: {len(rows)}",
        f"- Candidate signals (FPP < 0.5): {len(candidate_found)}",
    ]
    if candidate_found:
        lines += ["", "| TIC ID | FPP | Rank Score |",
                  "|---|---|---|"]
        for r in candidate_found[:10]:
            tid = r.get("tic_id", "?")
            fpp = _fpp(r)
            rs = _rank_score(r)
            lines.append(f"| {tid} | {fpp:.3f} | {rs:.3f} |")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="candidate_merger",
        description="Merge and deduplicate pipeline results from multiple files.",
    )
    parser.add_argument("inputs", nargs="+", metavar="JSON")
    parser.add_argument("--output", required=True, metavar="JSON")
    parser.add_argument("--prefer", choices=["best_fpp", "rank_score"],
                        default="best_fpp")
    args = parser.parse_args(argv)

    rows = merge_candidates(args.inputs, prefer=args.prefer)
    write_merged(rows, args.output)
    print(format_merge_summary(rows, n_sources=len(args.inputs)))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
