"""Flatten nested pipeline JSON output to analysis-ready CSV rows.

Unpacks posterior.*, scores.*, and meta.* sub-dicts into flat column names
and writes them as CSV — either to a file or as a string.

Public API
----------
CsvExportResult
flatten_candidate(row) -> dict
export_to_csv(rows, output_path) -> CsvExportResult
load_and_export(json_path, output_path) -> CsvExportResult
format_export_result(result) -> str
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CsvExportResult:
    n_rows: int
    n_columns: int
    output_path: str | None
    columns: tuple[str, ...]
    flag: str  # "OK" | "EMPTY" | "INVALID"


def flatten_candidate(row: dict) -> dict:
    """Flatten one pipeline output row to a flat dict.

    Extracts top-level scalars and unpacks nested posterior/scores/meta.

    Args:
        row: Pipeline output candidate dict.

    Returns:
        Flat dict ready for CSV writing.
    """
    flat: dict[str, object] = {}

    # Top-level scalar fields
    for key in (
        "tic_id", "target_id", "period_days", "depth_ppm",
        "snr", "pathway", "scorer", "provenance_score",
        "transit_count", "duration_hours", "epoch_bjd",
    ):
        if key in row:
            flat[key] = row[key]

    # Posterior sub-dict
    posterior = row.get("posterior", {}) or {}
    if isinstance(posterior, dict):
        for k, v in posterior.items():
            flat[f"posterior_{k}"] = v

    # Scores sub-dict
    scores = row.get("scores", {}) or {}
    if isinstance(scores, dict):
        for k, v in scores.items():
            flat[f"scores_{k}"] = v

    # Meta sub-dict
    meta = row.get("meta", {}) or {}
    if isinstance(meta, dict):
        for key in ("toolkit_version", "run_at", "scorer", "git_commit", "features_available"):
            if key in meta:
                flat[f"meta_{key}"] = meta[key]

    return flat


def export_to_csv(
    rows: list[dict],
    output_path: str | Path | None = None,
) -> CsvExportResult:
    """Flatten and export pipeline rows to CSV.

    Args:
        rows: List of pipeline output dicts.
        output_path: Write CSV here; None returns result with no file written.

    Returns:
        :class:`CsvExportResult`.
    """
    if not isinstance(rows, list):
        return CsvExportResult(
            n_rows=0,
            n_columns=0,
            output_path=None,
            columns=(),
            flag="INVALID",
        )

    if len(rows) == 0:
        return CsvExportResult(
            n_rows=0,
            n_columns=0,
            output_path=str(output_path) if output_path is not None else None,
            columns=(),
            flag="EMPTY",
        )

    # Flatten all rows
    flat_rows = [flatten_candidate(r) for r in rows]

    # Collect all columns (stable order: union of all keys)
    seen: dict[str, None] = {}
    for fr in flat_rows:
        for k in fr:
            seen[k] = None
    columns = list(seen.keys())

    # Write CSV
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for fr in flat_rows:
        writer.writerow(fr)

    csv_str = buf.getvalue()

    out_str: str | None = None
    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(csv_str, encoding="utf-8")
        out_str = str(p)

    return CsvExportResult(
        n_rows=len(flat_rows),
        n_columns=len(columns),
        output_path=out_str,
        columns=tuple(columns),
        flag="OK",
    )


def load_and_export(
    json_path: str | Path,
    output_path: str | Path | None = None,
) -> CsvExportResult:
    """Load a pipeline JSON file and export it to CSV.

    Args:
        json_path: Path to JSON file (list or single dict).
        output_path: CSV output path; None returns result without writing.

    Returns:
        :class:`CsvExportResult`.
    """
    try:
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    except Exception:
        return CsvExportResult(
            n_rows=0,
            n_columns=0,
            output_path=None,
            columns=(),
            flag="INVALID",
        )

    if isinstance(data, dict):
        rows = [data]
    elif isinstance(data, list):
        rows = data
    else:
        return CsvExportResult(
            n_rows=0,
            n_columns=0,
            output_path=None,
            columns=(),
            flag="INVALID",
        )

    return export_to_csv(rows, output_path)


def format_export_result(result: CsvExportResult) -> str:
    """Format CSV export result as Markdown."""
    out_str = result.output_path or "(no file written)"
    col_list = ", ".join(result.columns[:8])
    if len(result.columns) > 8:
        col_list += f", … ({len(result.columns)} total)"
    lines = [
        "## JSON to CSV Exporter",
        "",
        f"- Rows exported: {result.n_rows}",
        f"- Columns: {result.n_columns}",
        f"- Output path: {out_str}",
        f"- Sample columns: {col_list}",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="json_to_csv_exporter",
        description="Flatten pipeline JSON output to CSV.",
    )
    parser.add_argument("json_path", help="Input pipeline JSON file.")
    parser.add_argument("--output", default=None, help="Output CSV path.")
    args = parser.parse_args(argv)

    result = load_and_export(args.json_path, args.output)
    print(format_export_result(result))
    return 0 if result.flag in ("OK", "EMPTY") else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
