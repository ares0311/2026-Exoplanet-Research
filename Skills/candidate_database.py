"""SQLite-backed versioned candidate store.

Each pipeline run appends a new row.  The same (tic_id, period_days) pair may
appear multiple times; the latest row is the "current" result.  Supports
simple queries: list all candidates, get history for a TIC ID, export to CSV.

Public API
----------
CandidateDatabase(db_path) — open/create the DB
  .insert(row: dict) -> int          row_id
  .latest(tic_id) -> dict | None
  .history(tic_id) -> list[dict]
  .all_latest() -> list[dict]
  .delete(tic_id) -> int             rows deleted
  .export_csv(path) -> Path
"""
from __future__ import annotations

import csv
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_DDL = """
CREATE TABLE IF NOT EXISTS candidates (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    tic_id          INTEGER  NOT NULL,
    period_days     REAL,
    epoch_bjd       REAL,
    depth_ppm       REAL,
    duration_hours  REAL,
    fpp             REAL,
    snr             REAL,
    pathway         TEXT,
    scorer          TEXT,
    run_at          TEXT NOT NULL,
    meta            TEXT
);
CREATE INDEX IF NOT EXISTS idx_tic ON candidates(tic_id);
"""

_COLUMNS = (
    "id", "tic_id", "period_days", "epoch_bjd", "depth_ppm",
    "duration_hours", "fpp", "snr", "pathway", "scorer", "run_at", "meta",
)


class CandidateDatabase:
    """Thin SQLite wrapper for candidate results.

    Args:
        db_path: Path to SQLite database file.  Created if absent.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_DDL)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _row_to_dict(self, row: sqlite3.Row) -> dict[str, Any]:
        return dict(row)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert(self, row: dict[str, Any]) -> int:
        """Insert a candidate result row.

        Required keys: ``tic_id``.  All other columns are optional.

        Returns:
            The auto-assigned row ID.
        """
        import json
        run_at = row.get("run_at") or datetime.now(UTC).isoformat()
        meta_raw = row.get("meta")
        meta = (json.dumps(meta_raw)
                if meta_raw is not None and not isinstance(meta_raw, str)
                else meta_raw)
        cur = self._conn.execute(
            """INSERT INTO candidates
               (tic_id, period_days, epoch_bjd, depth_ppm, duration_hours,
                fpp, snr, pathway, scorer, run_at, meta)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                int(row["tic_id"]),
                row.get("period_days"),
                row.get("epoch_bjd"),
                row.get("depth_ppm"),
                row.get("duration_hours"),
                row.get("fpp"),
                row.get("snr"),
                row.get("pathway"),
                row.get("scorer"),
                run_at,
                meta,
            ),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def latest(self, tic_id: int) -> dict[str, Any] | None:
        """Return the most recent row for *tic_id*, or None."""
        cur = self._conn.execute(
            "SELECT * FROM candidates WHERE tic_id=? ORDER BY id DESC LIMIT 1",
            (tic_id,),
        )
        row = cur.fetchone()
        return self._row_to_dict(row) if row else None

    def history(self, tic_id: int) -> list[dict[str, Any]]:
        """Return all rows for *tic_id* ordered oldest-first."""
        cur = self._conn.execute(
            "SELECT * FROM candidates WHERE tic_id=? ORDER BY id ASC",
            (tic_id,),
        )
        return [self._row_to_dict(r) for r in cur.fetchall()]

    def all_latest(self) -> list[dict[str, Any]]:
        """Return the most recent row per TIC ID, sorted by FPP ascending."""
        cur = self._conn.execute(
            """SELECT * FROM candidates
               WHERE id IN (
                   SELECT MAX(id) FROM candidates GROUP BY tic_id
               )
               ORDER BY fpp ASC NULLS LAST""",
        )
        return [self._row_to_dict(r) for r in cur.fetchall()]

    def delete(self, tic_id: int) -> int:
        """Delete all rows for *tic_id*; return count removed."""
        cur = self._conn.execute(
            "DELETE FROM candidates WHERE tic_id=?", (tic_id,)
        )
        self._conn.commit()
        return cur.rowcount

    def count(self) -> int:
        """Total number of rows in the database."""
        cur = self._conn.execute("SELECT COUNT(*) FROM candidates")
        return cur.fetchone()[0]

    def export_csv(self, path: Path | str) -> Path:
        """Export all rows to a CSV file; return the path written."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        rows = self._conn.execute(
            "SELECT * FROM candidates ORDER BY id"
        ).fetchall()
        with p.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(_COLUMNS))
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))
        return p

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> CandidateDatabase:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="candidate_database",
        description="Query or export the candidate SQLite database.",
    )
    parser.add_argument("--db", default="data/candidates.sqlite3")
    sub = parser.add_subparsers(dest="cmd")

    lp = sub.add_parser("list", help="List latest candidate per TIC ID.")
    _ = lp

    hp = sub.add_parser("history", help="Show all runs for a TIC ID.")
    hp.add_argument("tic_id", type=int)

    ep = sub.add_parser("export", help="Export to CSV.")
    ep.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    db = CandidateDatabase(args.db)

    if args.cmd == "list":
        rows = db.all_latest()
        print(f"{len(rows)} candidates:")
        for r in rows:
            fpp = f"{r['fpp']:.3f}" if r["fpp"] is not None else "—"
            print(f"  TIC {r['tic_id']}  P={r['period_days']}d  FPP={fpp}  {r['pathway']}")
    elif args.cmd == "history":
        rows = db.history(args.tic_id)
        print(json.dumps(rows, indent=2))
    elif args.cmd == "export":
        out = db.export_csv(args.out)
        print(f"Exported {db.count()} rows to {out}")
    else:
        print(f"Total rows: {db.count()}")

    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
