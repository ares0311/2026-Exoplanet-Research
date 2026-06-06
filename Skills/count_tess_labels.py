"""Check the TESS TOI label counts against the CNN Tier-2 gate.

Queries ExoFOP-TESS for the current CP/FP/EB counts and prints whether
the training-data threshold for building the 1D CNN (Tier 2) is met.

Gate logic (as of 2026-06-06):
  - TESS has ~550 confirmed planets total — a CP-only threshold of 5,000
    is unreachable. Gate is now on TOTAL quality labels (CP + FP + EB).
  - Default gate: total >= 2,000 AND cp >= 400.
  - These minimums are sufficient for a calibrated 1D CNN with class
    weighting; the Shallue & Vanderburg (2018) architecture used ~15,000
    Kepler examples, but TESS transfer-learning requires far fewer.

Each CLI run writes a top-level SQLite audit log by default so live
network checks remain traceable without committing runtime artifacts.

Usage
-----
    python Skills/count_tess_labels.py
    python Skills/count_tess_labels.py --min-total 2000 --min-cp 400
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

_CNN_MIN_TOTAL = 2_000   # CP + FP + EB combined
_CNN_MIN_CP = 400        # confirmed planets minimum for positive class
_CNN_THRESHOLD = _CNN_MIN_TOTAL  # kept for backwards-compatible log column
_EXOFOP_URL = (
    "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LOG_DB = _PROJECT_ROOT / "logs" / "tess_label_check.sqlite3"
_LOG_SCHEMA_VERSION = 1
_EXIT_GATE_OPEN = 0
_EXIT_GATE_CLOSED = 1
_EXIT_ERROR = 2


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _read_exofop_table(source_url: str, timeout_seconds: int) -> Any:
    """Fetch the ExoFOP TOI table with an explicit timeout."""
    import pandas as pd

    with urlopen(source_url, timeout=timeout_seconds) as response:
        return pd.read_csv(BytesIO(response.read()), comment="#")


def count_labels(
    threshold: int = _CNN_THRESHOLD,
    *,
    min_total: int = _CNN_MIN_TOTAL,
    min_cp: int = _CNN_MIN_CP,
    source_url: str = _EXOFOP_URL,
    timeout_seconds: int = 30,
    read_table_fn: Callable[[str, int], Any] | None = None,
) -> dict[str, int | bool]:
    """Fetch the TESS TOI table and count CP/FP/EB dispositions.

    Gate is open when BOTH conditions hold:
      - total (CP + FP + EB) >= min_total  (default 2,000)
      - cp >= min_cp                        (default 400)

    Args:
        threshold: Legacy arg, ignored when min_total/min_cp are set.
        min_total: Minimum total labeled examples required.
        min_cp: Minimum confirmed-planet examples required.
        source_url: ExoFOP CSV URL.
        timeout_seconds: Network timeout for the live CSV fetch.
        read_table_fn: Injectable table reader for offline tests.

    Returns:
        Dict with keys: ``cp``, ``fp``, ``eb``, ``total``, ``gate_open``.
    """
    table_reader = read_table_fn or _read_exofop_table
    df = table_reader(source_url, timeout_seconds)
    col = next(
        (c for c in df.columns if "disposition" in c.lower() and "tfop" in c.lower()),
        None,
    )
    if col is None:
        raise ValueError("Could not locate TFOPWG Disposition column in TOI table")

    counts = df[col].value_counts().to_dict()
    cp = int(counts.get("CP", 0))
    fp = int(counts.get("FP", 0))
    eb = int(counts.get("EB", 0))
    total = cp + fp + eb

    return {
        "cp": cp,
        "fp": fp,
        "eb": eb,
        "total": total,
        "gate_open": total >= min_total and cp >= min_cp,
    }


def initialize_log_db(db_path: Path) -> Path:
    """Create the SQLite audit-log schema if needed."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tess_label_checks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                schema_version INTEGER NOT NULL,
                started_at TEXT NOT NULL,
                finished_at TEXT NOT NULL,
                elapsed_ms INTEGER NOT NULL,
                source_url TEXT NOT NULL,
                threshold INTEGER NOT NULL,
                cp INTEGER,
                fp INTEGER,
                eb INTEGER,
                total INTEGER,
                gate_open INTEGER,
                exit_code INTEGER NOT NULL,
                status TEXT NOT NULL,
                error_message TEXT
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tess_label_checks_started "
            "ON tess_label_checks(started_at)"
        )
    return db_path


def write_log_entry(
    db_path: Path,
    *,
    started_at: str,
    finished_at: str,
    elapsed_ms: int,
    source_url: str,
    threshold: int,
    exit_code: int,
    status: str,
    result: dict[str, int | bool] | None = None,
    error_message: str | None = None,
) -> Path:
    """Append one live label-check result to the SQLite audit log."""
    initialize_log_db(db_path)
    result = result or {}
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO tess_label_checks (
                schema_version,
                started_at,
                finished_at,
                elapsed_ms,
                source_url,
                threshold,
                cp,
                fp,
                eb,
                total,
                gate_open,
                exit_code,
                status,
                error_message
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _LOG_SCHEMA_VERSION,
                started_at,
                finished_at,
                elapsed_ms,
                source_url,
                threshold,
                int(result["cp"]) if "cp" in result else None,
                int(result["fp"]) if "fp" in result else None,
                int(result["eb"]) if "eb" in result else None,
                int(result["total"]) if "total" in result else None,
                int(bool(result["gate_open"])) if "gate_open" in result else None,
                exit_code,
                status,
                error_message,
            ),
        )
    return db_path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--min-total",
        type=int,
        default=_CNN_MIN_TOTAL,
        help=f"Minimum total CP+FP+EB labels to open gate (default: {_CNN_MIN_TOTAL})",
    )
    p.add_argument(
        "--min-cp",
        type=int,
        default=_CNN_MIN_CP,
        help=f"Minimum confirmed-planet labels to open gate (default: {_CNN_MIN_CP})",
    )
    p.add_argument(
        "--timeout-seconds",
        type=int,
        default=30,
        help="Timeout for the ExoFOP CSV request (default: 30).",
    )
    p.add_argument(
        "--log-db",
        type=Path,
        default=_DEFAULT_LOG_DB,
        help=f"SQLite audit log path (default: {_DEFAULT_LOG_DB}).",
    )
    p.add_argument(
        "--no-log",
        action="store_true",
        help="Do not write the SQLite audit log for this run.",
    )
    return p.parse_args(argv)


def _cli(
    argv: list[str] | None = None,
    *,
    read_table_fn: Callable[[str, int], Any] | None = None,
) -> int:
    args = _parse_args(argv)
    started_at = _utc_now()
    start_time = time.monotonic()
    result: dict[str, int | bool] | None = None
    exit_code = _EXIT_ERROR
    status = "error"
    error_message: str | None = None
    try:
        result = count_labels(
            min_total=args.min_total,
            min_cp=args.min_cp,
            timeout_seconds=args.timeout_seconds,
            read_table_fn=read_table_fn,
        )
        print("TESS TOI label counts:")
        print(f"  Confirmed planets (CP): {result['cp']:,}")
        print(f"  False positives   (FP): {result['fp']:,}")
        print(f"  Eclipsing binaries(EB): {result['eb']:,}")
        print(f"  Total labeled         : {result['total']:,}")
        print(f"  Gate thresholds       : total >= {args.min_total:,} AND cp >= {args.min_cp:,}")
        print()
        if result["gate_open"]:
            exit_code = _EXIT_GATE_OPEN
            print("Gate OPEN — label counts meet training thresholds")
            print("  Continue with docs/CNN_SPEC.md")
        else:
            exit_code = _EXIT_GATE_CLOSED
            need_total = max(0, args.min_total - result["total"])
            need_cp = max(0, args.min_cp - result["cp"])
            print("Gate CLOSED")
            if need_total:
                print(f"  Need {need_total:,} more total labels (have {result['total']:,})")
            if need_cp:
                print(f"  Need {need_cp:,} more CP labels (have {result['cp']:,})")
        status = "success"
    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        print("TESS TOI label count failed:", file=sys.stderr)
        print(f"  {error_message}", file=sys.stderr)
    finally:
        finished_at = _utc_now()
        elapsed_ms = int((time.monotonic() - start_time) * 1000)
        if not args.no_log:
            log_path = write_log_entry(
                args.log_db,
                started_at=started_at,
                finished_at=finished_at,
                elapsed_ms=elapsed_ms,
                source_url=_EXOFOP_URL,
                threshold=args.min_total,
                exit_code=exit_code,
                status=status,
                result=result,
                error_message=error_message,
            )
            print(f"SQLite audit log: {log_path}")
    return exit_code


if __name__ == "__main__":
    sys.exit(_cli())
