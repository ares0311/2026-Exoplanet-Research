"""Archive and retrieve scored pipeline candidates in a persistent JSON store.

Supports insert, lookup by TIC ID, history retrieval, export, and search
by FPP threshold or pathway.

Public API
----------
ArchiveRecord(tic_id, period_days, fpp, pathway, planet_posterior,
              detection_confidence, run_at, scorer, mission, note)
CandidateArchive(path)
    .insert(row, *, note) -> ArchiveRecord
    .latest(tic_id, period_days) -> ArchiveRecord | None
    .history(tic_id, period_days) -> list[ArchiveRecord]
    .search(*, fpp_max, pathway, mission) -> list[ArchiveRecord]
    .all_latest() -> list[ArchiveRecord]
    .export_csv(path) -> Path
format_archive_record(record) -> str
"""
from __future__ import annotations

import contextlib
import csv
import json
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArchiveRecord:
    tic_id: int
    period_days: float
    fpp: float | None
    pathway: str
    planet_posterior: float | None
    detection_confidence: float | None
    run_at: float
    scorer: str
    mission: str
    note: str


def _safe_float(v: object) -> float | None:
    with contextlib.suppress(TypeError, ValueError):
        return float(v)  # type: ignore[arg-type]
    return None


def _candidate_key(tic_id: int, period_days: float) -> str:
    return f"{tic_id}_{period_days:.6f}"


class CandidateArchive:
    """Persistent JSON archive for scored candidates."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        self._data: dict[str, list[dict]] = self._load()

    def _load(self) -> dict[str, list[dict]]:
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

    def insert(self, row: dict, *, note: str = "") -> ArchiveRecord:
        """Insert a pipeline result row into the archive."""
        tic_id = int(row.get("tic_id", 0))
        period_days = float(row.get("period_days", 0.0))

        scores = row.get("scores") or {}
        posterior = row.get("posterior") or {}

        fpp = _safe_float(
            row.get("false_positive_probability")
            or scores.get("false_positive_probability")
        )
        dc = _safe_float(
            row.get("detection_confidence")
            or scores.get("detection_confidence")
        )
        pp = _safe_float(
            row.get("planet_posterior")
            or posterior.get("planet_candidate")
        )
        meta = row.get("meta") or {}
        entry = {
            "tic_id": tic_id,
            "period_days": period_days,
            "fpp": fpp,
            "pathway": str(row.get("pathway") or ""),
            "planet_posterior": pp,
            "detection_confidence": dc,
            "run_at": time.time(),
            "scorer": str(meta.get("scorer") or ""),
            "mission": str(row.get("mission") or ""),
            "note": note,
        }
        key = _candidate_key(tic_id, period_days)
        if key not in self._data:
            self._data[key] = []
        self._data[key].append(entry)
        self._save()
        return self._to_record(entry)

    def latest(self, tic_id: int, period_days: float) -> ArchiveRecord | None:
        key = _candidate_key(tic_id, period_days)
        entries = self._data.get(key)
        if not entries:
            return None
        return self._to_record(entries[-1])

    def history(self, tic_id: int, period_days: float) -> list[ArchiveRecord]:
        key = _candidate_key(tic_id, period_days)
        return [self._to_record(e) for e in self._data.get(key, [])]

    def search(
        self,
        *,
        fpp_max: float | None = None,
        pathway: str | None = None,
        mission: str | None = None,
    ) -> list[ArchiveRecord]:
        """Return the latest record for each candidate matching filters."""
        results = []
        for entries in self._data.values():
            if not entries:
                continue
            rec = self._to_record(entries[-1])
            if fpp_max is not None and (rec.fpp is None or rec.fpp > fpp_max):
                continue
            if pathway is not None and rec.pathway != pathway:
                continue
            if mission is not None and rec.mission != mission:
                continue
            results.append(rec)
        return results

    def all_latest(self) -> list[ArchiveRecord]:
        return self.search()

    def export_csv(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        records = self.all_latest()
        with open(p, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=[
                "tic_id", "period_days", "fpp", "pathway", "planet_posterior",
                "detection_confidence", "run_at", "scorer", "mission", "note",
            ])
            writer.writeheader()
            for r in records:
                writer.writerow({
                    "tic_id": r.tic_id, "period_days": r.period_days,
                    "fpp": r.fpp, "pathway": r.pathway,
                    "planet_posterior": r.planet_posterior,
                    "detection_confidence": r.detection_confidence,
                    "run_at": r.run_at, "scorer": r.scorer,
                    "mission": r.mission, "note": r.note,
                })
        return p

    @staticmethod
    def _to_record(entry: dict) -> ArchiveRecord:
        return ArchiveRecord(
            tic_id=entry.get("tic_id", 0),
            period_days=entry.get("period_days", 0.0),
            fpp=entry.get("fpp"),
            pathway=entry.get("pathway", ""),
            planet_posterior=entry.get("planet_posterior"),
            detection_confidence=entry.get("detection_confidence"),
            run_at=entry.get("run_at", 0.0),
            scorer=entry.get("scorer", ""),
            mission=entry.get("mission", ""),
            note=entry.get("note", ""),
        )


def format_archive_record(record: ArchiveRecord) -> str:
    """Format an archive record as Markdown.

    Args:
        record: ArchiveRecord to format.

    Returns:
        Markdown string.
    """
    fpp_str = f"{record.fpp:.3f}" if record.fpp is not None else "—"
    pp_str = f"{record.planet_posterior:.3f}" if record.planet_posterior is not None else "—"
    dc_str = (f"{record.detection_confidence:.3f}"
              if record.detection_confidence is not None else "—")
    lines = [
        f"## Candidate Archive — TIC {record.tic_id}\n",
        f"**Period**: {record.period_days:.4f} d | "
        f"**Pathway**: `{record.pathway or '—'}` | "
        f"FPP: {fpp_str}\n",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Planet posterior | {pp_str} |",
        f"| Detection confidence | {dc_str} |",
        f"| Scorer | {record.scorer or '—'} |",
        f"| Mission | {record.mission or '—'} |",
        f"| Note | {record.note or '—'} |",
    ]
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Candidate archive.")
    sub = parser.add_subparsers(dest="cmd")

    ins_p = sub.add_parser("insert")
    ins_p.add_argument("--db", required=True)
    ins_p.add_argument("--row", required=True, help="JSON row file.")
    ins_p.add_argument("--note", default="")

    get_p = sub.add_parser("latest")
    get_p.add_argument("--db", required=True)
    get_p.add_argument("--tic-id", type=int, required=True)
    get_p.add_argument("--period", type=float, required=True)

    sub.add_parser("list").add_argument("--db", required=True)

    args = parser.parse_args(argv)
    arch = CandidateArchive(args.db)

    if args.cmd == "insert":
        row = json.loads(Path(args.row).read_text())
        rec = arch.insert(row, note=args.note)
        print(format_archive_record(rec))
    elif args.cmd == "latest":
        rec = arch.latest(args.tic_id, args.period)
        print(format_archive_record(rec) if rec else "Not found.")
    elif args.cmd == "list":
        for rec in arch.all_latest():
            print(format_archive_record(rec))
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
