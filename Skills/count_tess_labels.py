"""Check the TESS TOI confirmed-planet count against the CNN Tier-2 gate.

Queries ExoFOP-TESS for the current number of CP (confirmed planet)
dispositions and prints whether the 5,000-label threshold for building
the 1D CNN (Tier 2) is met. Each CLI run writes a top-level SQLite audit log
by default so live network checks remain traceable without committing runtime
artifacts.

Usage
-----
    python Skills/count_tess_labels.py
    python Skills/count_tess_labels.py --threshold 5000
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

_CNN_THRESHOLD = 5_000
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
    threshold: int = _CNN_THRESHOLD,
    *,
    source_url: str = _EXOFOP_URL,
    timeout_seconds: int = 30,
    read_table_fn: Callable[[str, int], Any] | None = None,
) -> dict[str, int | bool]:
    """Fetch the TESS TOI table and count CP/FP/EB dispositions.

    Args:
        threshold: Minimum CP count to unlock Tier-2 CNN training.
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
        "gate_open": cp >= threshold,
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
        "--threshold",
        type=int,
        default=_CNN_THRESHOLD,
        help=f"CP count needed to unlock CNN (default: {_CNN_THRESHOLD})",
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
            threshold=args.threshold,
            timeout_seconds=args.timeout_seconds,
            read_table_fn=read_table_fn,
        )
        print("TESS TOI label counts:")
        print(f"  Confirmed planets (CP): {result['cp']:,}")
        print(f"  False positives   (FP): {result['fp']:,}")
        print(f"  Eclipsing binaries(EB): {result['eb']:,}")
        print(f"  Total labeled         : {result['total']:,}")
        print()
        if result["gate_open"]:
            exit_code = _EXIT_GATE_OPEN
            print(
                f"Gate OPEN - CP count ({result['cp']:,}) "
                f">= threshold ({args.threshold:,})"
            )
            print("  Label-count gate open; continue with docs/CNN_SPEC.md checks")
        else:
            exit_code = _EXIT_GATE_CLOSED
            remaining = args.threshold - result["cp"]
            print(f"Gate CLOSED - need {remaining:,} more CP labels")
            print("  Continue collecting TESS data; re-check with this script")
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
                threshold=args.threshold,
                exit_code=exit_code,
                status=status,
                result=result,
                error_message=error_message,
            )
            print(f"SQLite audit log: {log_path}")
    return exit_code


if __name__ == "__main__":
    sys.exit(_cli())
