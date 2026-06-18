"""Check the TESS TOI label counts against the CNN Tier-2 gate.

Queries ExoFOP-TESS for the current CP/KP/FP/FA counts and prints whether
the training-data threshold for building the 1D CNN (Tier 2) is met.

ExoFOP TFOPWG disposition codes and their CNN training class:
  CP  Confirmed Planet         → positive class
  KP  Known Planet (TESS obs)  → positive class (confirmed, same quality as CP)
  FP  False Positive           → negative class
  FA  False Alarm (artifact)   → negative class
  PC  Planet Candidate         → NOT used (unvetted)
  APC Ambiguous PC             → NOT used (unvetted)

Gate logic (as of 2026-06-06):
  - positive (CP + KP) >= 400   — enough confirmed-planet examples
  - total (positive + negative) >= 2,000  — minimum corpus for class weighting
  ExoFOP does not use an "EB" disposition code; eclipsing binaries are
  classified as FP. The old "EB" label was incorrect and has been removed.

Each CLI run writes a top-level SQLite audit log by default so live
network checks remain traceable without committing runtime artifacts.

Usage
-----
    .venv/bin/python Skills/count_tess_labels.py
    .venv/bin/python Skills/count_tess_labels.py --min-total 2000 --min-positive 400
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

_CNN_MIN_TOTAL = 2_000
_CNN_MIN_POSITIVE = 400
_EXOFOP_URL = (
    "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LOG_DB = _PROJECT_ROOT / "logs" / "tess_label_check.sqlite3"
_LOG_SCHEMA_VERSION = 2
_EXIT_GATE_OPEN = 0
_EXIT_GATE_CLOSED = 1
_EXIT_ERROR = 2

_LOG_COLUMNS = (
    "id",
    "schema_version",
    "started_at",
    "finished_at",
    "elapsed_ms",
    "source_url",
    "min_total",
    "min_positive",
    "cp",
    "kp",
    "fp",
    "fa",
    "positive",
    "negative",
    "total",
    "gate_open",
    "exit_code",
    "status",
    "error_message",
)
_LOG_COLUMN_SET = set(_LOG_COLUMNS)

_LOG_COLUMN_MIGRATIONS: dict[str, str] = {
    "schema_version": f"INTEGER NOT NULL DEFAULT {_LOG_SCHEMA_VERSION}",
    "started_at": "TEXT NOT NULL DEFAULT ''",
    "finished_at": "TEXT NOT NULL DEFAULT ''",
    "elapsed_ms": "INTEGER NOT NULL DEFAULT 0",
    "source_url": f"TEXT NOT NULL DEFAULT '{_EXOFOP_URL}'",
    "min_total": f"INTEGER NOT NULL DEFAULT {_CNN_MIN_TOTAL}",
    "min_positive": f"INTEGER NOT NULL DEFAULT {_CNN_MIN_POSITIVE}",
    "cp": "INTEGER",
    "kp": "INTEGER",
    "fp": "INTEGER",
    "fa": "INTEGER",
    "positive": "INTEGER",
    "negative": "INTEGER",
    "total": "INTEGER",
    "gate_open": "INTEGER",
    "exit_code": f"INTEGER NOT NULL DEFAULT {_EXIT_ERROR}",
    "status": "TEXT NOT NULL DEFAULT 'unknown'",
    "error_message": "TEXT",
}


def _create_log_table(conn: sqlite3.Connection, table_name: str) -> None:
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            schema_version INTEGER NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT NOT NULL,
            elapsed_ms INTEGER NOT NULL,
            source_url TEXT NOT NULL,
            min_total INTEGER NOT NULL,
            min_positive INTEGER NOT NULL,
            cp INTEGER,
            kp INTEGER,
            fp INTEGER,
            fa INTEGER,
            positive INTEGER,
            negative INTEGER,
            total INTEGER,
            gate_open INTEGER,
            exit_code INTEGER NOT NULL,
            status TEXT NOT NULL,
            error_message TEXT
        )
        """
    )


def _rebuild_legacy_log_table(conn: sqlite3.Connection) -> None:
    """Replace a legacy audit-log table with the canonical schema."""
    existing_columns = {
        str(row[1])
        for row in conn.execute("PRAGMA table_info(tess_label_checks)").fetchall()
    }
    copy_columns = [column for column in _LOG_COLUMNS if column in existing_columns]
    conn.execute("DROP TABLE IF EXISTS tess_label_checks__new")
    _create_log_table(conn, "tess_label_checks__new")
    if copy_columns:
        column_csv = ", ".join(copy_columns)
        conn.execute(
            f"""
            INSERT INTO tess_label_checks__new ({column_csv})
            SELECT {column_csv}
            FROM tess_label_checks
            """
        )
    conn.execute("DROP TABLE tess_label_checks")
    conn.execute("ALTER TABLE tess_label_checks__new RENAME TO tess_label_checks")


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _read_exofop_table(source_url: str, timeout_seconds: int) -> Any:
    """Fetch the ExoFOP TOI table with an explicit timeout."""
    import ssl

    import pandas as pd

    try:
        import certifi
        ssl_ctx: ssl.SSLContext | None = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ssl_ctx = None

    with urlopen(source_url, timeout=timeout_seconds, context=ssl_ctx) as response:
        return pd.read_csv(BytesIO(response.read()), comment="#")


def count_labels(
    *,
    min_total: int = _CNN_MIN_TOTAL,
    min_positive: int = _CNN_MIN_POSITIVE,
    source_url: str = _EXOFOP_URL,
    timeout_seconds: int = 30,
    read_table_fn: Callable[[str, int], Any] | None = None,
) -> dict[str, int | bool]:
    """Fetch the TESS TOI table and count quality training labels.

    Positive class: CP (Confirmed Planet) + KP (Known Planet).
    Negative class: FP (False Positive) + FA (False Alarm).

    Gate is open when BOTH conditions hold:
      - positive (CP + KP) >= min_positive  (default 400)
      - total (positive + negative) >= min_total  (default 2,000)

    Args:
        min_total: Minimum total quality labels required.
        min_positive: Minimum positive-class (confirmed planet) labels required.
        source_url: ExoFOP CSV URL.
        timeout_seconds: Network timeout for the live CSV fetch.
        read_table_fn: Injectable table reader for offline tests.

    Returns:
        Dict with keys: ``cp``, ``kp``, ``fp``, ``fa``,
        ``positive``, ``negative``, ``total``, ``gate_open``.
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
    kp = int(counts.get("KP", 0))
    fp = int(counts.get("FP", 0))
    fa = int(counts.get("FA", 0))
    positive = cp + kp
    negative = fp + fa
    total = positive + negative

    return {
        "cp": cp,
        "kp": kp,
        "fp": fp,
        "fa": fa,
        "positive": positive,
        "negative": negative,
        "total": total,
        "gate_open": total >= min_total and positive >= min_positive,
    }


def initialize_log_db(db_path: Path) -> Path:
    """Create or migrate the SQLite audit-log schema if needed."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        _create_log_table(conn, "tess_label_checks")
        existing_columns = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(tess_label_checks)").fetchall()
        }
        for column, definition in _LOG_COLUMN_MIGRATIONS.items():
            if column not in existing_columns:
                conn.execute(f"ALTER TABLE tess_label_checks ADD COLUMN {column} {definition}")
        migrated_columns = {
            str(row[1])
            for row in conn.execute("PRAGMA table_info(tess_label_checks)").fetchall()
        }
        if any(column not in _LOG_COLUMN_SET for column in migrated_columns):
            _rebuild_legacy_log_table(conn)
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
    min_total: int,
    min_positive: int,
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
                schema_version, started_at, finished_at, elapsed_ms,
                source_url, min_total, min_positive,
                cp, kp, fp, fa, positive, negative, total, gate_open,
                exit_code, status, error_message
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _LOG_SCHEMA_VERSION,
                started_at,
                finished_at,
                elapsed_ms,
                source_url,
                min_total,
                min_positive,
                result.get("cp"),
                result.get("kp"),
                result.get("fp"),
                result.get("fa"),
                result.get("positive"),
                result.get("negative"),
                result.get("total"),
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
        help=f"Minimum total quality labels to open gate (default: {_CNN_MIN_TOTAL})",
    )
    p.add_argument(
        "--min-positive",
        type=int,
        default=_CNN_MIN_POSITIVE,
        help=f"Minimum positive-class labels to open gate (default: {_CNN_MIN_POSITIVE})",
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
            min_positive=args.min_positive,
            timeout_seconds=args.timeout_seconds,
            read_table_fn=read_table_fn,
        )
        print("TESS TOI label counts:")
        print(f"  Confirmed planets  (CP): {result['cp']:,}")
        print(f"  Known planets      (KP): {result['kp']:,}  ← positive class")
        print(f"  False positives    (FP): {result['fp']:,}")
        print(f"  False alarms       (FA): {result['fa']:,}  ← negative class")
        print(f"  {'─' * 36}")
        print(f"  Positive class (CP+KP) : {result['positive']:,}")
        print(f"  Negative class (FP+FA) : {result['negative']:,}")
        print(f"  Total quality labels   : {result['total']:,}")
        print(
            f"  Gate thresholds        : total >= {args.min_total:,}"
            f" AND positive >= {args.min_positive:,}"
        )
        print()
        if result["gate_open"]:
            exit_code = _EXIT_GATE_OPEN
            status = "success"
            print("Gate OPEN — label counts meet training thresholds")
            print("  Continue with docs/CNN_SPEC.md")
        else:
            exit_code = _EXIT_GATE_CLOSED
            status = "success"
            need_total = max(0, args.min_total - int(result["total"]))
            need_pos = max(0, args.min_positive - int(result["positive"]))
            print("Gate CLOSED")
            if need_total:
                print(f"  Need {need_total:,} more total labels (have {result['total']:,})")
            if need_pos:
                print(f"  Need {need_pos:,} more positive labels (have {result['positive']:,})")
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
                min_total=args.min_total,
                min_positive=args.min_positive,
                exit_code=exit_code,
                status=status,
                result=result,
                error_message=error_message,
            )
            print(f"SQLite audit log: {log_path}")
    return exit_code


if __name__ == "__main__":
    sys.exit(_cli())
