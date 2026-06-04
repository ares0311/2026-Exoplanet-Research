"""Build a side-by-side comparison matrix for multiple exoplanet candidates."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

_COLUMNS = [
    ("tic_id", "TIC ID"),
    ("period_days", "Period (d)"),
    ("depth_ppm", "Depth (ppm)"),
    ("false_positive_probability", "FPP"),
    ("detection_confidence", "DC"),
    ("pathway", "Pathway"),
    ("n_transits", "N transits"),
]


@dataclass(frozen=True)
class ComparisonMatrixResult:
    n_candidates: int
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    flag: str


def _extract(candidate: dict, key: str) -> str:
    val = candidate.get(key)
    if val is None:
        scores = candidate.get("scores", {})
        val = scores.get(key)
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.4f}" if abs(val) < 10000 else f"{val:.1f}"
    return str(val)


def build_comparison_matrix(
    candidates: list[dict],
    extra_columns: list[str] | None = None,
) -> ComparisonMatrixResult:
    """
    Build a comparison matrix from a list of candidate dicts.

    Standard columns: TIC ID, period, depth, FPP, DC, pathway, n_transits.
    Optional extra_columns adds additional keys from the candidate dicts.
    """
    if not candidates:
        return ComparisonMatrixResult(
            n_candidates=0, columns=(), rows=(), flag="NO_CANDIDATES",
        )

    col_defs = list(_COLUMNS)
    if extra_columns:
        for ec in extra_columns:
            col_defs.append((ec, ec))

    col_labels = tuple(label for _, label in col_defs)
    rows: list[tuple[str, ...]] = []
    for cand in candidates:
        row = tuple(_extract(cand, key) for key, _ in col_defs)
        rows.append(row)

    return ComparisonMatrixResult(
        n_candidates=len(candidates),
        columns=col_labels,
        rows=tuple(rows),
        flag="OK",
    )


def format_comparison_matrix(r: ComparisonMatrixResult) -> str:
    if r.flag != "OK":
        return f"No candidates to compare (flag: {r.flag}).\n"
    header = "| " + " | ".join(r.columns) + " |"
    sep = "|" + "|".join("---" for _ in r.columns) + "|"
    lines = [header, sep]
    for row in r.rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _cli() -> int:
    p = argparse.ArgumentParser(description="Build candidate comparison matrix.")
    p.add_argument("candidates_json", help="JSON array string or @file")
    p.add_argument("--extra-columns", nargs="*", default=None)
    args = p.parse_args()
    raw = args.candidates_json
    if raw.startswith("@"):
        with open(raw[1:]) as f:
            candidates = json.load(f)
    else:
        candidates = json.loads(raw)
    r = build_comparison_matrix(candidates, args.extra_columns)
    print(format_comparison_matrix(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
