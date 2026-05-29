"""Archive pipeline batch results into a dated directory structure.

Copies or moves result files into an archive with metadata, a manifest,
and optional compression.

Public API
----------
ArchiveRecord(source, destination, size_bytes, archived_at)
ArchiveResult(n_archived, n_failed, total_bytes, manifest_path,
              archive_dir, flag)
archive_batch_results(source_paths, archive_dir, *, session_label,
                      move_files, write_manifest) -> ArchiveResult
format_archive_result(result) -> str
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArchiveRecord:
    source: str
    destination: str
    size_bytes: int
    archived_at: float  # Unix timestamp


@dataclass(frozen=True)
class ArchiveResult:
    n_archived: int
    n_failed: int
    total_bytes: int
    manifest_path: str | None
    archive_dir: str
    flag: str  # "OK" | "PARTIAL" | "FAILED" | "EMPTY"


def archive_batch_results(
    source_paths: list[str | Path],
    archive_dir: str | Path,
    *,
    session_label: str | None = None,
    move_files: bool = False,
    write_manifest: bool = True,
) -> ArchiveResult:
    """Archive a list of result files into a dated archive directory.

    Args:
        source_paths: List of files to archive.
        archive_dir: Root archive directory.
        session_label: Optional label suffix for the archive subdirectory.
        move_files: If True, move files instead of copying.
        write_manifest: If True, write a JSON manifest file.

    Returns:
        ArchiveResult with counts and manifest path.
    """
    if not source_paths:
        return ArchiveResult(
            n_archived=0, n_failed=0, total_bytes=0,
            manifest_path=None, archive_dir=str(archive_dir), flag="EMPTY"
        )

    ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    label = f"_{session_label}" if session_label else ""
    dest_dir = Path(archive_dir) / f"{ts}{label}"
    dest_dir.mkdir(parents=True, exist_ok=True)

    records: list[ArchiveRecord] = []
    n_failed = 0

    for src_raw in source_paths:
        src = Path(src_raw)
        if not src.exists():
            n_failed += 1
            continue
        dest = dest_dir / src.name
        try:
            if move_files:
                shutil.move(str(src), dest)
            else:
                shutil.copy2(src, dest)
            size = dest.stat().st_size
            records.append(ArchiveRecord(
                source=str(src),
                destination=str(dest),
                size_bytes=size,
                archived_at=time.time(),
            ))
        except Exception:
            n_failed += 1

    total_bytes = sum(r.size_bytes for r in records)
    manifest_path: str | None = None

    if write_manifest and records:
        manifest_data = {
            "session_label": session_label,
            "archived_at": ts,
            "n_files": len(records),
            "total_bytes": total_bytes,
            "files": [
                {"source": r.source, "destination": r.destination,
                 "size_bytes": r.size_bytes}
                for r in records
            ],
        }
        manifest_file = dest_dir / "manifest.json"
        tmp = None
        try:
            fd, tmp = tempfile.mkstemp(dir=dest_dir, suffix=".tmp")
            with os.fdopen(fd, "w") as fh:
                json.dump(manifest_data, fh, indent=2)
            os.replace(tmp, manifest_file)
            manifest_path = str(manifest_file)
        except Exception:
            with contextlib.suppress(OSError):
                if tmp:
                    os.unlink(tmp)

    n_archived = len(records)
    if n_archived == 0:
        flag = "FAILED"
    elif n_failed > 0:
        flag = "PARTIAL"
    else:
        flag = "OK"

    return ArchiveResult(
        n_archived=n_archived,
        n_failed=n_failed,
        total_bytes=total_bytes,
        manifest_path=manifest_path,
        archive_dir=str(dest_dir),
        flag=flag,
    )


def format_archive_result(result: ArchiveResult) -> str:
    """Format archive result as Markdown.

    Args:
        result: ArchiveResult to format.

    Returns:
        Markdown string.
    """
    size_kb = result.total_bytes / 1024
    manifest_str = result.manifest_path if result.manifest_path else "—"
    lines = [
        "## Batch Result Archiver\n",
        f"**Status**: `{result.flag}`\n",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Files archived | {result.n_archived} |",
        f"| Files failed | {result.n_failed} |",
        f"| Total size | {size_kb:.1f} KB |",
        f"| Archive directory | `{result.archive_dir}` |",
        f"| Manifest | `{manifest_str}` |",
    ]
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Archive batch pipeline results.")
    parser.add_argument("files", nargs="+", help="Files to archive.")
    parser.add_argument("--archive-dir", required=True, help="Archive root directory.")
    parser.add_argument("--label", default=None, help="Session label.")
    parser.add_argument("--move", action="store_true", help="Move instead of copy.")
    args = parser.parse_args(argv)

    result = archive_batch_results(
        args.files, args.archive_dir,
        session_label=args.label,
        move_files=args.move,
    )
    print(format_archive_result(result))
    return 0 if result.flag in ("OK", "EMPTY") else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
