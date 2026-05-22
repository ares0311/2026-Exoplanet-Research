"""Import candidate transit signals from a CSV file into pipeline-compatible dicts.

Reads a CSV file where each row represents a candidate signal (TIC ID, period,
epoch, depth, duration, SNR) and normalises column names and types.  Validates
required fields and skips malformed rows with a warning.

Public API
----------
ImportedCandidate(tic_id, period_days, epoch_bjd, depth_ppm,
                  duration_hours, snr, source_file, row_index, notes)
CandidateImportResult(n_rows_read, n_imported, n_skipped,
                      candidates, skip_reasons, flag)
import_candidates_csv(path, *, required_cols, col_map) -> CandidateImportResult
format_import_result(result) -> str
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImportedCandidate:
    tic_id: int
    period_days: float
    epoch_bjd: float
    depth_ppm: float | None
    duration_hours: float | None
    snr: float | None
    source_file: str
    row_index: int
    notes: str


@dataclass(frozen=True)
class CandidateImportResult:
    n_rows_read: int
    n_imported: int
    n_skipped: int
    candidates: tuple[ImportedCandidate, ...]
    skip_reasons: tuple[str, ...]
    flag: str  # "OK" | "EMPTY" | "INVALID"


_DEFAULT_COL_MAP: dict[str, list[str]] = {
    "tic_id": ["tic_id", "ticid", "tic", "TIC_ID", "TICID"],
    "period_days": ["period_days", "period", "period_d", "PERIOD", "Period"],
    "epoch_bjd": ["epoch_bjd", "epoch", "t0", "T0", "Epoch"],
    "depth_ppm": ["depth_ppm", "depth", "transit_depth", "DEPTH"],
    "duration_hours": ["duration_hours", "duration", "dur", "DURATION"],
    "snr": ["snr", "SNR", "signal_to_noise"],
    "notes": ["notes", "comment", "comment_text", "flag"],
}


def _find_col(header: list[str], candidates: list[str]) -> str | None:
    h_lower = {h.lower(): h for h in header}
    for c in candidates:
        if c in header:
            return c
        if c.lower() in h_lower:
            return h_lower[c.lower()]
    return None


def import_candidates_csv(
    path: Path | str,
    *,
    required_cols: list[str] | None = None,
    col_map: dict[str, list[str]] | None = None,
) -> CandidateImportResult:
    """Import candidates from a CSV file.

    Args:
        path: Path to the CSV file.
        required_cols: Column names that must be present (default: tic_id, period_days, epoch_bjd).
        col_map: Override mapping from logical name → list of accepted CSV column names.

    Returns:
        :class:`CandidateImportResult`.
    """
    path = Path(path)
    if required_cols is None:
        required_cols = ["tic_id", "period_days", "epoch_bjd"]
    if col_map is None:
        col_map = _DEFAULT_COL_MAP

    if not path.exists():
        return CandidateImportResult(0, 0, 0, (), (f"File not found: {path}",), "INVALID")

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return CandidateImportResult(0, 0, 0, (), (str(exc),), "INVALID")

    reader = csv.DictReader(text.splitlines())
    if reader.fieldnames is None:
        return CandidateImportResult(0, 0, 0, (), ("No header row",), "EMPTY")

    header = list(reader.fieldnames)
    # Resolve column names
    resolved: dict[str, str | None] = {}
    for logical, aliases in col_map.items():
        resolved[logical] = _find_col(header, aliases)

    # Check required columns
    missing = [r for r in required_cols if resolved.get(r) is None]
    if missing:
        return CandidateImportResult(
            0, 0, 0, (), (f"Missing required columns: {missing}",), "INVALID"
        )

    candidates: list[ImportedCandidate] = []
    skip_reasons: list[str] = []
    n_read = 0

    for row_idx, row in enumerate(reader):
        n_read += 1

        def _get_col(logical: str, _row: dict = row) -> str | None:  # type: ignore[assignment]
            col = resolved.get(logical)
            if col is None:
                return None
            return _row.get(col, "").strip() or None

        try:
            tic_str = _get_col("tic_id", row)
            if tic_str is None:
                raise ValueError("tic_id missing")
            tic_id = int(float(tic_str))

            period_str = _get_col("period_days", row)
            if period_str is None:
                raise ValueError("period_days missing")
            period_days = float(period_str)
            if period_days <= 0:
                raise ValueError(f"period_days={period_days} <= 0")

            epoch_str = _get_col("epoch_bjd", row)
            if epoch_str is None:
                raise ValueError("epoch_bjd missing")
            epoch_bjd = float(epoch_str)

        except (ValueError, TypeError) as exc:
            skip_reasons.append(f"Row {row_idx}: {exc}")
            continue

        def _opt_float(logical: str, _row: dict = row) -> float | None:  # type: ignore[assignment]
            s = _get_col(logical, _row)
            if s is None:
                return None
            try:
                return float(s)
            except (ValueError, TypeError):
                return None

        candidates.append(ImportedCandidate(
            tic_id=tic_id,
            period_days=period_days,
            epoch_bjd=epoch_bjd,
            depth_ppm=_opt_float("depth_ppm", row),
            duration_hours=_opt_float("duration_hours", row),
            snr=_opt_float("snr", row),
            source_file=str(path),
            row_index=row_idx,
            notes=_get_col("notes", row) or "",
        ))

    if n_read == 0:
        return CandidateImportResult(0, 0, 0, (), (), "EMPTY")

    return CandidateImportResult(
        n_rows_read=n_read,
        n_imported=len(candidates),
        n_skipped=len(skip_reasons),
        candidates=tuple(candidates),
        skip_reasons=tuple(skip_reasons),
        flag="OK",
    )


def format_import_result(result: CandidateImportResult) -> str:
    """Format import result as Markdown."""
    lines = [
        "## Candidate CSV Import",
        "",
        f"- Rows read: {result.n_rows_read}",
        f"- Imported: {result.n_imported}",
        f"- Skipped: {result.n_skipped}",
        f"- **Flag: {result.flag}**",
    ]
    if result.skip_reasons:
        lines.append("")
        lines.append("**Skip reasons:**")
        for r in result.skip_reasons[:5]:
            lines.append(f"  - {r}")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="candidate_csv_importer",
        description="Import candidates from a CSV file.",
    )
    parser.add_argument("path")
    args = parser.parse_args(argv)

    result = import_candidates_csv(Path(args.path))
    print(format_import_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
