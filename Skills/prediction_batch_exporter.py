"""Export batch model predictions to CSV.

Writes tic_id, model, score, label, and threshold_decision columns.
Supports round-trip loading and atomic write to avoid partial files.

Public API
----------
PredictionRow(tic_id, model, score, label, threshold_decision)
ExportResult(n_rows, output_path, n_positive_decisions,
             n_negative_decisions, flag)
make_row(tic_id, model, score, *, label, threshold) -> PredictionRow
export_predictions(rows, output_path) -> ExportResult
load_predictions(path) -> list[PredictionRow]
format_export_result(result) -> str
"""
from __future__ import annotations

import contextlib
import csv
import io
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

_HEADER = ("tic_id", "model", "score", "label", "threshold_decision")


@dataclass(frozen=True)
class PredictionRow:
    tic_id: str
    model: str
    score: float
    label: int | None          # None if unknown
    threshold_decision: str    # "POSITIVE" | "NEGATIVE"


@dataclass(frozen=True)
class ExportResult:
    n_rows: int
    output_path: str
    n_positive_decisions: int
    n_negative_decisions: int
    flag: str  # "OK" | "EMPTY" | "INVALID"


def make_row(
    tic_id: str,
    model: str,
    score: float,
    *,
    label: int | None = None,
    threshold: float = 0.5,
) -> PredictionRow:
    """Create a PredictionRow with a threshold decision.

    Args:
        tic_id: Target TIC identifier.
        model: Model name/identifier.
        score: Predicted probability in [0, 1].
        label: Ground-truth label (None if unknown).
        threshold: Decision boundary; score >= threshold → POSITIVE.

    Returns:
        PredictionRow with threshold_decision set.
    """
    decision = "POSITIVE" if score >= threshold else "NEGATIVE"
    return PredictionRow(
        tic_id=tic_id,
        model=model,
        score=score,
        label=label,
        threshold_decision=decision,
    )


def export_predictions(rows: list[PredictionRow], output_path: Path) -> ExportResult:
    """Export prediction rows to a CSV file.

    Args:
        rows: List of PredictionRow objects.
        output_path: Destination CSV path (written atomically).

    Returns:
        ExportResult describing the export.
    """
    if not rows:
        return ExportResult(
            n_rows=0,
            output_path=str(output_path),
            n_positive_decisions=0,
            n_negative_decisions=0,
            flag="EMPTY",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(_HEADER)
    for r in rows:
        label_str = "" if r.label is None else str(r.label)
        writer.writerow([r.tic_id, r.model, r.score, label_str, r.threshold_decision])

    fd, tmp_path = tempfile.mkstemp(dir=output_path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(buf.getvalue())
        os.replace(tmp_path, output_path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise

    n_pos = sum(1 for r in rows if r.threshold_decision == "POSITIVE")
    n_neg = len(rows) - n_pos

    return ExportResult(
        n_rows=len(rows),
        output_path=str(output_path),
        n_positive_decisions=n_pos,
        n_negative_decisions=n_neg,
        flag="OK",
    )


def load_predictions(path: Path) -> list[PredictionRow]:
    """Load prediction rows from a CSV file.

    Args:
        path: Path to the CSV file created by export_predictions.

    Returns:
        List of PredictionRow objects.

    Raises:
        FileNotFoundError: If path does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Predictions file not found: {path}")

    rows: list[PredictionRow] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            label_raw = row.get("label", "")
            label = int(label_raw) if label_raw.strip() else None
            rows.append(
                PredictionRow(
                    tic_id=row["tic_id"],
                    model=row["model"],
                    score=float(row["score"]),
                    label=label,
                    threshold_decision=row["threshold_decision"],
                )
            )
    return rows


def format_export_result(result: ExportResult) -> str:
    """Format export result as a Markdown summary.

    Args:
        result: ExportResult to format.

    Returns:
        Markdown string.
    """
    lines: list[str] = [
        "## Prediction Batch Export\n",
        f"Flag: `{result.flag}` | Rows: {result.n_rows}\n",
        f"Output: `{result.output_path}`\n",
    ]

    if result.flag == "EMPTY":
        lines.append("\n_No predictions to export._\n")
        return "\n".join(lines)

    lines.append("")
    lines.append(f"- Positive decisions: {result.n_positive_decisions}")
    lines.append(f"- Negative decisions: {result.n_negative_decisions}")

    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Export or inspect batch model predictions CSV."
    )
    parser.add_argument("csv_path", help="Path to predictions CSV file.")
    parser.add_argument("--inspect", action="store_true", help="Print summary.")
    args = parser.parse_args(argv)

    path = Path(args.csv_path)
    if args.inspect:
        rows = load_predictions(path)
        n_pos = sum(1 for r in rows if r.threshold_decision == "POSITIVE")
        result = ExportResult(
            n_rows=len(rows),
            output_path=str(path),
            n_positive_decisions=n_pos,
            n_negative_decisions=len(rows) - n_pos,
            flag="OK",
        )
        print(format_export_result(result))
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
