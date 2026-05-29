"""Record and query observation metadata for pipeline runs.

Stores per-observation context (instrument, conditions, data quality notes)
alongside pipeline run identifiers for audit and reproducibility.

Public API
----------
ObservationMetadata(obs_id, tic_id, run_at, instrument, cadence_min,
                    sector, quality_flags, notes)
ObservationRecord(obs_id, tic_id, run_at, instrument, cadence_min,
                  sector, quality_flags, notes)
MetadataStore(path)
    .record(tic_id, *, instrument, cadence_min, sector, quality_flags,
            notes, obs_id) -> ObservationRecord
    .get(obs_id) -> ObservationRecord | None
    .list_by_tic(tic_id) -> list[ObservationRecord]
    .all_records() -> list[ObservationRecord]
format_metadata_record(record) -> str
"""
from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ObservationRecord:
    obs_id: str
    tic_id: int
    run_at: float
    instrument: str
    cadence_min: float | None
    sector: int | None
    quality_flags: tuple[str, ...]
    notes: str
    flag: str  # "OK" | "FLAGGED" | "INCOMPLETE"


class MetadataStore:
    """Persistent observation metadata store backed by JSON."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._data: dict[str, dict] = self._load()

    def _load(self) -> dict[str, dict]:
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text())
        except Exception:
            return {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = None
        try:
            fd, tmp = tempfile.mkstemp(dir=self._path.parent, suffix=".tmp")
            with os.fdopen(fd, "w") as fh:
                json.dump(self._data, fh, indent=2)
            os.replace(tmp, self._path)
        except Exception:
            with contextlib.suppress(OSError):
                if tmp:
                    os.unlink(tmp)
            raise

    def record(
        self,
        tic_id: int,
        *,
        instrument: str = "",
        cadence_min: float | None = None,
        sector: int | None = None,
        quality_flags: list[str] | None = None,
        notes: str = "",
        obs_id: str | None = None,
    ) -> ObservationRecord:
        """Record observation metadata."""
        if obs_id is None:
            obs_id = str(uuid.uuid4())[:8]
        flags = list(quality_flags or [])
        flag = "FLAGGED" if flags else ("INCOMPLETE" if not instrument else "OK")
        entry = {
            "obs_id": obs_id,
            "tic_id": tic_id,
            "run_at": time.time(),
            "instrument": instrument,
            "cadence_min": cadence_min,
            "sector": sector,
            "quality_flags": flags,
            "notes": notes,
            "flag": flag,
        }
        self._data[obs_id] = entry
        self._save()
        return self._to_record(entry)

    def get(self, obs_id: str) -> ObservationRecord | None:
        entry = self._data.get(obs_id)
        return self._to_record(entry) if entry else None

    def list_by_tic(self, tic_id: int) -> list[ObservationRecord]:
        return [
            self._to_record(e)
            for e in self._data.values()
            if e.get("tic_id") == tic_id
        ]

    def all_records(self) -> list[ObservationRecord]:
        return [self._to_record(e) for e in self._data.values()]

    @staticmethod
    def _to_record(entry: dict) -> ObservationRecord:
        return ObservationRecord(
            obs_id=entry.get("obs_id", ""),
            tic_id=entry.get("tic_id", 0),
            run_at=entry.get("run_at", 0.0),
            instrument=entry.get("instrument", ""),
            cadence_min=entry.get("cadence_min"),
            sector=entry.get("sector"),
            quality_flags=tuple(entry.get("quality_flags", [])),
            notes=entry.get("notes", ""),
            flag=entry.get("flag", "OK"),
        )


def format_metadata_record(record: ObservationRecord) -> str:
    """Format observation metadata as Markdown.

    Args:
        record: ObservationRecord to format.

    Returns:
        Markdown string.
    """
    flags_str = ", ".join(record.quality_flags) if record.quality_flags else "—"
    cadence_str = f"{record.cadence_min:.1f} min" if record.cadence_min is not None else "—"
    sector_str = str(record.sector) if record.sector is not None else "—"
    lines = [
        f"## Observation Metadata — TIC {record.tic_id}\n",
        f"**ID**: `{record.obs_id}` | **Status**: `{record.flag}` | "
        f"Instrument: {record.instrument or '—'}\n",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Cadence | {cadence_str} |",
        f"| Sector | {sector_str} |",
        f"| Quality flags | {flags_str} |",
        f"| Notes | {record.notes or '—'} |",
    ]
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Record observation metadata.")
    sub = parser.add_subparsers(dest="cmd")

    rec_p = sub.add_parser("record")
    rec_p.add_argument("--db", required=True)
    rec_p.add_argument("--tic-id", type=int, required=True)
    rec_p.add_argument("--instrument", default="")
    rec_p.add_argument("--cadence", type=float, default=None)
    rec_p.add_argument("--sector", type=int, default=None)
    rec_p.add_argument("--notes", default="")

    list_p = sub.add_parser("list")
    list_p.add_argument("--db", required=True)
    list_p.add_argument("--tic-id", type=int, required=True)

    args = parser.parse_args(argv)
    if args.cmd == "record":
        store = MetadataStore(args.db)
        rec = store.record(
            args.tic_id,
            instrument=args.instrument,
            cadence_min=args.cadence,
            sector=args.sector,
            notes=args.notes,
        )
        print(format_metadata_record(rec))
    elif args.cmd == "list":
        store = MetadataStore(args.db)
        records = store.list_by_tic(args.tic_id)
        if not records:
            print("No records found.")
        for r in records:
            print(format_metadata_record(r))
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
