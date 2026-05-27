"""Fetch the ExoFOP Community TOI (CTOI) table.

Downloads community-submitted transit candidates for TESS from the ExoFOP
CTOI table.  Can be used as an additional label source alongside the
official TFOPWG TOI list.  All network I/O is injectable for offline tests.

Public API
----------
CtoisResult(rows, n_cp, n_fp, n_pc, fetched_at, flag)
fetch_ctoi_table(*, ctoi_url, min_ratings, fetch_fn) -> CtoisResult
ctoi_rows_to_label_rows(rows) -> tuple[dict, ...]
write_json(path, payload) -> Path
format_ctoi_result(result) -> str
"""
from __future__ import annotations

import csv
import io
import sys
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeAlias

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CTOI_URL = (
    "https://exofop.ipac.caltech.edu/tess/download_ctoi.php?sort=ctoi&output=csv"
)

# Possible column names across ExoFOP CTOI CSV versions
_COL_CTOI = ("CTOI", "ctoi")
_COL_TIC = ("TIC", "TIC ID", "tic_id", "tic id")
_COL_DISP = ("User Disposition", "user_disposition", "Disposition")
_COL_PERIOD = ("Period (days)", "period_days", "Period")
_COL_DURATION = ("Duration (hours)", "duration_hours", "Duration (hrs)")
_COL_EPOCH = ("Epoch (BJD)", "epoch_bjd", "Epoch")
_COL_NREPORTS = ("Num Reports", "n_ratings", "Num Ratings", "num_reports")

FetchFn: TypeAlias = Callable[[str], str]
CtoiRow: TypeAlias = dict[str, object | None]
LabelRow: TypeAlias = dict[str, object | None]


def _find_col(header: list[str], candidates: tuple[str, ...]) -> str | None:
    """Return the first candidate that exists in *header*, else None."""
    for c in candidates:
        if c in header:
            return c
    return None


def _disposition_to_class(disp: str) -> str:
    """Normalise raw CTOI disposition to 'cp' | 'fp' | 'pc'."""
    d = disp.strip().upper()
    if d == "CP":
        return "cp"
    if d in ("FP", "EB"):
        return "fp"
    return "pc"


def _safe_float(row: dict, col: str | None) -> float | None:
    """Extract a float from *row[col]*, returning None on failure."""
    if col is None:
        return None
    try:
        return float(row.get(col, ""))
    except (ValueError, TypeError):
        return None


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: object, *, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _label_confidence(n_ratings: int) -> float:
    """Return a deterministic conflict-resolution weight, not a probability."""
    return min(0.90, 0.65 + 0.05 * max(n_ratings, 0))


def _default_fetch(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
        return resp.read().decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CtoisResult:
    """Result of fetching the ExoFOP CTOI table.

    Attributes:
        rows: Tuple of dicts, each with keys: tic_id, toi, disposition,
              period_days, epoch_bjd, duration_hours, n_ratings.
        n_cp: Number of confirmed-planet (CP) rows.
        n_fp: Number of false-positive (FP/EB) rows.
        n_pc: Number of planet-candidate (PC) rows.
        fetched_at: ISO 8601 UTC timestamp of the fetch.
        flag: "OK" | "EMPTY" | "FETCH_ERROR"
    """

    rows: tuple[CtoiRow, ...]
    n_cp: int
    n_fp: int
    n_pc: int
    fetched_at: str
    flag: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_ctoi_table(
    *,
    ctoi_url: str = _CTOI_URL,
    min_ratings: int = 1,
    fetch_fn: FetchFn | None = None,
) -> CtoisResult:
    """Download and parse the ExoFOP CTOI table.

    Args:
        ctoi_url: URL of the CTOI CSV download endpoint.
        min_ratings: Minimum number of community ratings to include a row.
        fetch_fn: Injectable; accepts a URL string, returns raw CSV as str.
                  Defaults to urllib.request.urlopen.

    Returns:
        :class:`CtoisResult` with parsed rows and flag.
    """
    now = datetime.now(UTC).isoformat()
    _fn = fetch_fn if fetch_fn is not None else _default_fetch

    try:
        raw = _fn(ctoi_url)
    except Exception:
        return CtoisResult(rows=(), n_cp=0, n_fp=0, n_pc=0, fetched_at=now, flag="FETCH_ERROR")

    # Strip comment lines (lines beginning with '#')
    clean_lines = [ln for ln in raw.splitlines() if not ln.startswith("#")]
    if not clean_lines:
        return CtoisResult(rows=(), n_cp=0, n_fp=0, n_pc=0, fetched_at=now, flag="EMPTY")

    reader = csv.DictReader(io.StringIO("\n".join(clean_lines)))
    header = reader.fieldnames or []

    col_ctoi = _find_col(list(header), _COL_CTOI)
    col_tic = _find_col(list(header), _COL_TIC)
    col_disp = _find_col(list(header), _COL_DISP)
    col_period = _find_col(list(header), _COL_PERIOD)
    col_dur = _find_col(list(header), _COL_DURATION)
    col_epoch = _find_col(list(header), _COL_EPOCH)
    col_nrep = _find_col(list(header), _COL_NREPORTS)

    rows: list[CtoiRow] = []
    n_cp = n_fp = n_pc = 0

    for raw_row in reader:
        # Extract n_ratings; apply min_ratings filter
        n_ratings_raw = raw_row.get(col_nrep, "0") if col_nrep else "0"
        try:
            n_ratings = int(float(n_ratings_raw))
        except (ValueError, TypeError):
            n_ratings = 0
        if n_ratings < min_ratings:
            continue

        disp_raw = raw_row.get(col_disp, "") if col_disp else ""
        disposition = _disposition_to_class(disp_raw)

        tic_raw = raw_row.get(col_tic, "") if col_tic else ""
        try:
            tic_id = str(int(float(tic_raw))) if tic_raw.strip() else ""
        except (ValueError, TypeError):
            tic_id = tic_raw.strip()

        row = {
            "tic_id": tic_id,
            "toi": raw_row.get(col_ctoi, "") if col_ctoi else "",
            "disposition": disposition,
            "period_days": _safe_float(raw_row, col_period),
            "epoch_bjd": _safe_float(raw_row, col_epoch),
            "duration_hours": _safe_float(raw_row, col_dur),
            "n_ratings": n_ratings,
        }
        rows.append(row)

        if disposition == "cp":
            n_cp += 1
        elif disposition == "fp":
            n_fp += 1
        else:
            n_pc += 1

    if not rows:
        return CtoisResult(rows=(), n_cp=0, n_fp=0, n_pc=0, fetched_at=now, flag="EMPTY")

    return CtoisResult(
        rows=tuple(rows),
        n_cp=n_cp,
        n_fp=n_fp,
        n_pc=n_pc,
        fetched_at=now,
        flag="OK",
    )


def ctoi_rows_to_label_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    source: str = "ctoi",
) -> tuple[LabelRow, ...]:
    """Convert eligible CTOI rows into opt-in training label rows.

    Only externally dispositioned `CP`, `FP`, and `EB` rows become supervised
    labels. `PC` and other uncertain/community-only candidate rows are skipped
    so the default training flow does not learn from unresolved candidates.
    """
    label_rows: list[LabelRow] = []
    for row in rows:
        disposition = str(row.get("disposition", "")).strip().lower()
        if disposition == "cp":
            label = 1
        elif disposition == "fp":
            label = 0
        else:
            continue

        tic_id = str(row.get("tic_id", "")).strip()
        if not tic_id:
            continue

        n_ratings = _coerce_int(row.get("n_ratings"))
        label_rows.append(
            {
                "tic_id": tic_id,
                "label": label,
                "source": source,
                "confidence": _label_confidence(n_ratings),
                "period_days": _coerce_float(row.get("period_days")),
                "epoch": _coerce_float(row.get("epoch_bjd")),
                "duration_hours": _coerce_float(row.get("duration_hours")),
                "ctoi": str(row.get("toi", "")).strip(),
                "disposition": disposition,
                "n_ratings": n_ratings,
            }
        )
    return tuple(label_rows)


def write_json(path: str | Path, payload: object) -> Path:
    """Write JSON payload to path and return the resolved Path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json_dumps(payload))
    return out


def json_dumps(payload: object) -> str:
    import json

    return json.dumps(payload, indent=2)


def format_ctoi_result(result: CtoisResult) -> str:
    """Return a Markdown summary of a :class:`CtoisResult`."""
    lines = [
        "## ExoFOP CTOI Fetch Result",
        "",
        f"**Flag**: {result.flag}",
        f"**Fetched at**: {result.fetched_at}",
        f"**Total rows**: {len(result.rows)}",
        f"- CP (confirmed planet): {result.n_cp}",
        f"- FP (false positive): {result.n_fp}",
        f"- PC (planet candidate): {result.n_pc}",
    ]
    if result.rows:
        lines += [
            "",
            "### Sample rows (first 3)",
            "| TIC ID | TOI | Disposition | Period (d) | N Ratings |",
            "|--------|-----|-------------|------------|-----------|",
        ]
        for row in result.rows[:3]:
            period = (
                f"{row['period_days']:.4f}"
                if row["period_days"] is not None
                else "N/A"
            )
            lines.append(
                f"| {row['tic_id']} | {row['toi']} | {row['disposition']} "
                f"| {period} | {row['n_ratings']} |"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="fetch_exofop_ctoi",
        description="Download and display the ExoFOP CTOI table.",
    )
    parser.add_argument(
        "--min-ratings", type=int, default=1,
        help="Minimum community ratings to include (default: 1)",
    )
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to save JSON output")
    parser.add_argument(
        "--labels-output",
        type=str,
        default=None,
        help="Optional path to save assembler-compatible CP/FP label rows",
    )
    args = parser.parse_args(argv)

    result = fetch_ctoi_table(min_ratings=args.min_ratings)
    print(format_ctoi_result(result))

    if args.output:
        out = write_json(
            args.output,
            {
                "flag": result.flag,
                "fetched_at": result.fetched_at,
                "n_cp": result.n_cp,
                "n_fp": result.n_fp,
                "n_pc": result.n_pc,
                "rows": list(result.rows),
            },
        )
        print(f"\nSaved to {out}")
    if args.labels_output:
        labels_out = write_json(
            args.labels_output,
            list(ctoi_rows_to_label_rows(result.rows)),
        )
        print(f"\nSaved label rows to {labels_out}")
    return 0 if result.flag == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
