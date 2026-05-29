"""Export pipeline candidates for human labeling or ML dataset construction.

Converts pipeline output rows into a labeling-ready format with fields
relevant for human reviewers: TIC ID, period, FPP, pathway, and plots.

Public API
----------
LabelExportRow(tic_id, period_days, depth_ppm, duration_hours,
               fpp, pathway, suggested_label, review_notes, label)
export_for_labeling(rows, output_path, *, overwrite,
                    fpp_threshold) -> LabelExportResult
load_labeled(path) -> list[LabelExportRow]
format_export_result(result) -> str
"""
from __future__ import annotations

import contextlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LabelExportRow:
    tic_id: int | None
    period_days: float | None
    depth_ppm: float | None
    duration_hours: float | None
    fpp: float | None
    pathway: str
    suggested_label: str   # "planet_candidate" | "false_positive" | "unknown"
    review_notes: str
    label: str             # human-assigned; "" if not yet labeled


@dataclass(frozen=True)
class LabelExportResult:
    n_exported: int
    n_suggested_pc: int
    n_suggested_fp: int
    n_suggested_unknown: int
    output_path: str
    flag: str  # "OK" | "EMPTY"


def _safe_float(v: object) -> float | None:
    with contextlib.suppress(TypeError, ValueError):
        return float(v)  # type: ignore[arg-type]
    return None


def _suggest_label(fpp: float | None, pathway: str) -> str:
    if fpp is None:
        return "unknown"
    if fpp < 0.15:
        return "planet_candidate"
    if fpp > 0.60:
        return "false_positive"
    return "unknown"


def _review_notes(row: dict) -> str:
    notes: list[str] = []
    pathway = str(row.get("pathway") or "")
    fpp = _safe_float(
        row.get("false_positive_probability")
        or (row.get("scores") or {}).get("false_positive_probability")
    )
    if fpp is not None and fpp < 0.10:
        notes.append("Low FPP — likely real")
    if "tfop" in pathway:
        notes.append("TFOP-ready pathway")
    if "github_only" in pathway:
        notes.append("Low-confidence pathway")
    return "; ".join(notes) if notes else ""


def export_for_labeling(
    rows: list[dict],
    output_path: str | Path,
    *,
    overwrite: bool = False,
    fpp_threshold: float | None = None,
) -> LabelExportResult:
    """Export pipeline rows as a labeling-ready JSON file.

    Args:
        rows: Pipeline output dicts.
        output_path: Path for the output JSON.
        overwrite: If False, raise if file already exists.
        fpp_threshold: Only export rows with FPP below this threshold.

    Returns:
        LabelExportResult with counts and path.
    """
    p = Path(output_path)
    if p.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {p}")

    export_rows: list[LabelExportRow] = []
    for row in rows:
        fpp = _safe_float(
            row.get("false_positive_probability")
            or (row.get("scores") or {}).get("false_positive_probability")
        )
        if fpp_threshold is not None and fpp is not None and fpp > fpp_threshold:
            continue

        tic_id: int | None = None
        with contextlib.suppress(TypeError, ValueError):
            raw = row.get("tic_id")
            if raw is not None:
                tic_id = int(raw)

        suggested = _suggest_label(fpp, str(row.get("pathway") or ""))
        export_rows.append(LabelExportRow(
            tic_id=tic_id,
            period_days=_safe_float(row.get("period_days")),
            depth_ppm=_safe_float(row.get("depth_ppm")),
            duration_hours=_safe_float(row.get("duration_hours")),
            fpp=fpp,
            pathway=str(row.get("pathway") or ""),
            suggested_label=suggested,
            review_notes=_review_notes(row),
            label="",
        ))

    if not export_rows:
        return LabelExportResult(
            n_exported=0, n_suggested_pc=0, n_suggested_fp=0,
            n_suggested_unknown=0, output_path=str(p), flag="EMPTY"
        )

    records = [
        {
            "tic_id": r.tic_id,
            "period_days": r.period_days,
            "depth_ppm": r.depth_ppm,
            "duration_hours": r.duration_hours,
            "fpp": r.fpp,
            "pathway": r.pathway,
            "suggested_label": r.suggested_label,
            "review_notes": r.review_notes,
            "label": r.label,
        }
        for r in export_rows
    ]

    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
        with os.fdopen(fd, "w") as fh:
            json.dump(records, fh, indent=2)
        os.replace(tmp, p)
    except Exception:
        with contextlib.suppress(OSError):
            if tmp:
                os.unlink(tmp)
        raise

    n_pc = sum(1 for r in export_rows if r.suggested_label == "planet_candidate")
    n_fp = sum(1 for r in export_rows if r.suggested_label == "false_positive")
    n_unk = sum(1 for r in export_rows if r.suggested_label == "unknown")

    return LabelExportResult(
        n_exported=len(export_rows),
        n_suggested_pc=n_pc,
        n_suggested_fp=n_fp,
        n_suggested_unknown=n_unk,
        output_path=str(p),
        flag="OK",
    )


def load_labeled(path: str | Path) -> list[LabelExportRow]:
    """Load a labeled export file.

    Args:
        path: Path to the JSON export file.

    Returns:
        List of LabelExportRow with human-assigned labels.
    """
    records = json.loads(Path(path).read_text())
    rows = []
    for r in records:
        with contextlib.suppress(Exception):
            rows.append(LabelExportRow(
                tic_id=r.get("tic_id"),
                period_days=_safe_float(r.get("period_days")),
                depth_ppm=_safe_float(r.get("depth_ppm")),
                duration_hours=_safe_float(r.get("duration_hours")),
                fpp=_safe_float(r.get("fpp")),
                pathway=str(r.get("pathway") or ""),
                suggested_label=str(r.get("suggested_label") or "unknown"),
                review_notes=str(r.get("review_notes") or ""),
                label=str(r.get("label") or ""),
            ))
    return rows


def format_export_result(result: LabelExportResult) -> str:
    """Format export result as Markdown.

    Args:
        result: LabelExportResult to format.

    Returns:
        Markdown string.
    """
    lines = [
        "## Candidate Label Exporter\n",
        f"**Status**: `{result.flag}` | Exported: {result.n_exported}\n",
        "",
        "| Category | Count |",
        "|---|---|",
        f"| Suggested planet_candidate | {result.n_suggested_pc} |",
        f"| Suggested false_positive | {result.n_suggested_fp} |",
        f"| Unknown / needs review | {result.n_suggested_unknown} |",
        f"| Output path | `{result.output_path}` |",
    ]
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Export candidates for labeling.")
    parser.add_argument("input", help="Pipeline output JSON.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fpp-threshold", type=float, default=None)
    args = parser.parse_args(argv)

    rows = json.loads(Path(args.input).read_text())
    if not isinstance(rows, list):
        rows = [rows]
    result = export_for_labeling(rows, args.output, overwrite=args.overwrite,
                                 fpp_threshold=args.fpp_threshold)
    print(format_export_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
