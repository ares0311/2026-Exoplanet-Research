"""Track per-target, per-sector pipeline stage completion.

Persistent JSON log recording which pipeline stages (fetch / clean / search /
vet / score) have been completed for each (TIC-ID, sector) combination.
Supports resumable batch runs.

Public API
----------
SectorCompletionLog(path)
    mark_complete(tic_id, sector, stage) -> None
    is_complete(tic_id, sector, stage) -> bool
    completed_stages(tic_id, sector) -> set[str]
    completion_summary() -> dict
    export_incomplete(required_stages) -> list[dict]
format_completion_report(log) -> str
"""
from __future__ import annotations

import contextlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

PIPELINE_STAGES: tuple[str, ...] = ("fetch", "clean", "search", "vet", "score")


@dataclass(frozen=True)
class _Entry:
    tic_id: int
    sector: int
    stages: frozenset[str]


class SectorCompletionLog:
    """Atomic-write JSON log for per-(TIC, sector) stage completion."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._data: dict[str, dict[str, list[str]]] = {}
        if self._path.exists():
            self._load()

    # ------------------------------------------------------------------
    def _load(self) -> None:
        try:
            raw = json.loads(self._path.read_text())
            if isinstance(raw, dict):
                self._data = raw
        except (json.JSONDecodeError, OSError):
            self._data = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self._path.parent, suffix=".tmp"
        )
        try:
            with os.fdopen(tmp_fd, "w") as fh:
                json.dump(self._data, fh, indent=2)
            os.replace(tmp_path, self._path)
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

    # ------------------------------------------------------------------
    def _key(self, tic_id: int, sector: int) -> str:
        return f"{tic_id}:{sector}"

    def mark_complete(self, tic_id: int, sector: int, stage: str) -> None:
        """Record that *stage* has been completed for (tic_id, sector)."""
        k = self._key(tic_id, sector)
        if k not in self._data:
            self._data[k] = {"stages": []}
        if stage not in self._data[k]["stages"]:
            self._data[k]["stages"].append(stage)
        self._save()

    def is_complete(self, tic_id: int, sector: int, stage: str) -> bool:
        """Return True if *stage* is recorded for (tic_id, sector)."""
        k = self._key(tic_id, sector)
        return stage in self._data.get(k, {}).get("stages", [])

    def completed_stages(self, tic_id: int, sector: int) -> set[str]:
        """Return the set of completed stages for (tic_id, sector)."""
        k = self._key(tic_id, sector)
        return set(self._data.get(k, {}).get("stages", []))

    # ------------------------------------------------------------------
    def completion_summary(self) -> dict:
        """Summary counts: total entries, fully complete, per-stage counts."""
        total = len(self._data)
        fully_done = 0
        stage_counts: dict[str, int] = dict.fromkeys(PIPELINE_STAGES, 0)

        for v in self._data.values():
            stages = set(v.get("stages", []))
            if set(PIPELINE_STAGES).issubset(stages):
                fully_done += 1
            for s in PIPELINE_STAGES:
                if s in stages:
                    stage_counts[s] += 1

        return {
            "total_entries": total,
            "fully_complete": fully_done,
            "stage_counts": stage_counts,
        }

    def export_incomplete(
        self,
        required_stages: list[str] | tuple[str, ...] | None = None,
    ) -> list[dict]:
        """Return entries that have not completed all *required_stages*.

        Defaults to all :data:`PIPELINE_STAGES`.
        """
        req = set(required_stages) if required_stages is not None else set(PIPELINE_STAGES)
        result: list[dict] = []
        for k, v in self._data.items():
            tic_str, sector_str = k.split(":", 1)
            done = set(v.get("stages", []))
            missing = req - done
            if missing:
                result.append(
                    {
                        "tic_id": int(tic_str),
                        "sector": int(sector_str),
                        "completed": sorted(done),
                        "missing": sorted(missing),
                    }
                )
        return result


def format_completion_report(log: SectorCompletionLog) -> str:
    """Format a Markdown completion report for a :class:`SectorCompletionLog`."""
    summary = log.completion_summary()
    incomplete = log.export_incomplete()
    lines = [
        "## Sector Completion Tracker",
        "",
        f"- Total (TIC, sector) entries: {summary['total_entries']}",
        f"- Fully complete: {summary['fully_complete']}",
        "- Stage counts:",
    ]
    for stage in PIPELINE_STAGES:
        lines.append(f"  - {stage}: {summary['stage_counts'][stage]}")
    lines += [
        "",
        f"**Incomplete entries: {len(incomplete)}**",
    ]
    if incomplete:
        lines.append("")
        lines.append("| TIC | Sector | Completed | Missing |")
        lines.append("|-----|--------|-----------|---------|")
        for e in incomplete[:20]:
            lines.append(
                f"| {e['tic_id']} | {e['sector']} | "
                f"{', '.join(e['completed']) or '—'} | "
                f"{', '.join(e['missing'])} |"
            )
        if len(incomplete) > 20:
            lines.append(f"| … | … | … | +{len(incomplete)-20} more |")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="sector_completion_tracker",
        description="Track per-target, per-sector pipeline stage completion.",
    )
    parser.add_argument("--log", required=True, help="Path to JSON log file")
    parser.add_argument("--mark", nargs=3, metavar=("TIC_ID", "SECTOR", "STAGE"))
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args(argv)

    log = SectorCompletionLog(args.log)

    if args.mark:
        tic_id, sector, stage = args.mark
        log.mark_complete(int(tic_id), int(sector), stage)
        print(f"Marked {stage} complete for TIC {tic_id} sector {sector}.")

    if args.summary or not args.mark:
        print(format_completion_report(log))

    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
