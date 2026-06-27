"""Rank and compare exoplanet candidate outputs from the exo CLI.

Reads one or more JSON files produced by ``exo <TIC-ID> --output results.json``
or a ScanLog produced by ``star_scanner.py --log <path>`` and prints a ranked
table sorted by a composite score.

Composite ranking score
-----------------------
``rank_score = (1 - FPP) * 0.45 + detection_confidence * 0.30 + novelty_score * 0.15
              + provenance_score * 0.10``

Pathway bonus: +0.10 for ``tfop_ready``, +0.05 for ``kepler_archive_candidate``,
+0.03 for ``planet_hunters_discussion``.  Score is clipped to [0, 1].

Public API
----------
load_candidates(paths)            -> list[dict]
compute_rank_score(row)           -> float
rank_candidates(rows, top_n=None) -> list[dict]  (sorted by rank_score desc)
print_rank_table(rows)            -> None         (Rich table to stdout)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ranking weights and pathway bonuses
# ---------------------------------------------------------------------------

_W_FPP: float = 0.45
_W_DC: float = 0.30
_W_NOVELTY: float = 0.15
_W_PROV: float = 0.10

_PATHWAY_BONUS: dict[str, float] = {
    "tfop_ready": 0.10,
    "kepler_archive_candidate": 0.05,
    "planet_hunters_discussion": 0.03,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _scan_log_to_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert a star_scanner ScanLog dict to rank_candidates row format.

    Only entries with status ``candidate_found`` are returned.
    """
    rows: list[dict[str, Any]] = []
    for entry in data.get("entries", {}).values():
        if entry.get("status") != "candidate_found":
            continue
        fpp = entry.get("best_fpp")
        rows.append({
            "tic_id": entry.get("tic_id"),
            "target_id": f"TIC {entry.get('tic_id', '?')}",
            "candidate_id": f"TIC {entry.get('tic_id', '?')}",
            "period_days": entry.get("best_period_days") or 0.0,
            "pathway": entry.get("best_pathway") or "",
            "provenance_score": 0.0,
            "snr": 0.0,
            "scores": {
                "false_positive_probability": fpp if fpp is not None else 1.0,
                "detection_confidence": 0.0,
                "novelty_score": 0.0,
            },
        })
    return rows


def load_candidates(paths: list[Path]) -> list[dict[str, Any]]:
    """Load candidate rows from JSON output files.

    Accepts ``exo --output`` / ``batch_scan`` format (list of dicts or single
    dict) and ``star_scanner`` ScanLog format (dict with ``entries`` +
    ``last_updated`` keys).  Only ``candidate_found`` entries are returned
    from ScanLog files.  Returns a flat list with ``_source_file`` added.
    """
    rows: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "entries" in data and "last_updated" in data:
            converted: list[dict[str, Any]] = _scan_log_to_rows(data)
        elif isinstance(data, dict):
            converted = [data]
        else:
            converted = list(data)
        for row in converted:
            row = dict(row)
            row.setdefault("_source_file", str(path))
            rows.append(row)
    return rows


def compute_rank_score(row: dict[str, Any]) -> float:
    """Compute a composite ranking score in [0, 1] for one candidate row."""
    scores = row.get("scores", {})
    fpp = float(scores.get("false_positive_probability", 1.0))
    dc = float(scores.get("detection_confidence", 0.0))
    novelty = float(scores.get("novelty_score", 0.0))
    prov = float(row.get("provenance_score", 0.0))

    base = (
        _W_FPP * (1.0 - fpp)
        + _W_DC * dc
        + _W_NOVELTY * novelty
        + _W_PROV * prov
    )
    pathway = row.get("pathway", "")
    bonus = _PATHWAY_BONUS.get(pathway, 0.0)
    return min(1.0, base + bonus)


def rank_candidates(
    rows: list[dict[str, Any]],
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    """Return rows sorted by rank_score descending, optionally limited to top_n."""
    scored = [dict(row, rank_score=compute_rank_score(row)) for row in rows]
    scored.sort(key=lambda r: r["rank_score"], reverse=True)
    if top_n is not None:
        scored = scored[:top_n]
    return scored


def print_rank_table(rows: list[dict[str, Any]]) -> None:
    """Print a Rich-formatted ranking table to stdout."""
    try:
        from rich.console import Console  # noqa: PLC0415
        from rich.table import Table  # noqa: PLC0415
    except ImportError:
        _print_plain_table(rows)
        return

    c = Console()
    t = Table(title=f"Ranked candidates ({len(rows)} total)", show_header=True)
    t.add_column("#", style="dim", justify="right")
    t.add_column("Candidate ID")
    t.add_column("Target")
    t.add_column("Period (d)", justify="right")
    t.add_column("SNR", justify="right")
    t.add_column("FPP", justify="right")
    t.add_column("DC", justify="right")
    t.add_column("Prov", justify="right")
    t.add_column("Rank", justify="right")
    t.add_column("Pathway")

    for i, row in enumerate(rows, 1):
        scores = row.get("scores", {})
        fpp = scores.get("false_positive_probability", float("nan"))
        dc = scores.get("detection_confidence", float("nan"))
        t.add_row(
            str(i),
            str(row.get("candidate_id", "?")),
            str(row.get("target_id", "?")),
            f"{row.get('period_days', 0.0):.4f}",
            f"{row.get('snr', 0.0):.1f}",
            f"{fpp:.3f}",
            f"{dc:.3f}",
            f"{row.get('provenance_score', 0.0):.3f}",
            f"{row.get('rank_score', 0.0):.3f}",
            str(row.get("pathway", "?")),
        )
    c.print(t)


def _print_plain_table(rows: list[dict[str, Any]]) -> None:
    header = f"{'#':>3}  {'Candidate':30}  {'FPP':>6}  {'Rank':>6}  Pathway"
    print(header)
    print("-" * len(header))
    for i, row in enumerate(rows, 1):
        fpp = row.get("scores", {}).get("false_positive_probability", float("nan"))
        print(
            f"{i:>3}  {str(row.get('candidate_id','?')):30}  "
            f"{fpp:>6.3f}  {row.get('rank_score', 0.0):>6.3f}  "
            f"{row.get('pathway', '?')}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        prog="rank_candidates",
        description="Rank exo-toolkit candidate JSON outputs by composite score.",
    )
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        metavar="FILE",
        help="JSON file(s) produced by `exo --output`.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=None,
        metavar="N",
        help="Show only the top N candidates.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print ranked rows as JSON instead of a table.",
    )
    args = parser.parse_args(argv)

    rows = load_candidates(args.files)
    if not rows:
        print("No candidates found.", file=sys.stderr)
        return 1
    ranked = rank_candidates(rows, top_n=args.top)

    if args.json:
        print(json.dumps(ranked, indent=2))
    else:
        print_rank_table(ranked)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
