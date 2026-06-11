"""Orchestrate large-scale CNN snippet extraction with checkpointing.

Iterates over a LabelManifest, calls a user-supplied snippet_fn per record,
and writes results incrementally to a JSON file.  Checkpointing lets long runs
resume without re-processing completed TIC IDs.

Public API
----------
SnippetEntry(tic_id, label, source, phase, flux, period_days, snr)
BatchBuildResult(n_attempted, n_succeeded, n_failed, n_snippets,
                 label_counts, output_path, checkpoint_path, flag)
build_snippet_batch(manifest, *, snippet_fn, output_path,
                    checkpoint_path, n_bins, resume, workers) -> BatchBuildResult
load_batch_output(path) -> list[SnippetEntry]
save_batch_output(entries, path) -> None
format_batch_result(result) -> str
"""
from __future__ import annotations

import contextlib
import json
import os
import tempfile
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multi_source_label_assembler import LabelManifest  # noqa: F401


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SnippetEntry:
    """One phase-folded snippet extracted from a light curve."""

    tic_id: str
    label: int
    source: str
    phase: tuple[float, ...]   # length n_bins
    flux: tuple[float, ...]    # length n_bins
    period_days: float | None
    snr: float | None


@dataclass(frozen=True)
class BatchBuildResult:
    """Summary of a batch snippet-extraction run."""

    n_attempted: int
    n_succeeded: int
    n_failed: int
    n_snippets: int
    label_counts: dict          # {0: n_fp, 1: n_planet}
    output_path: str
    checkpoint_path: str
    flag: str  # "OK" | "PARTIAL" | "EMPTY" | "INVALID"


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _load_checkpoint(path: Path) -> dict:
    with contextlib.suppress(OSError, json.JSONDecodeError):
        data = json.loads(path.read_text())
        return {
            "completed": set(data.get("completed_tic_ids", [])),
            "failed": set(data.get("failed_tic_ids", [])),
        }
    return {"completed": set(), "failed": set()}


def _save_checkpoint(path: Path, completed: set, failed: set) -> None:
    data = {
        "completed_tic_ids": sorted(completed),
        "failed_tic_ids": sorted(failed),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, str(path))
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


# ---------------------------------------------------------------------------
# Batch I/O
# ---------------------------------------------------------------------------


def save_batch_output(entries: list[SnippetEntry], path: Path) -> None:
    """Atomically write *entries* to *path* as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serialised = [
        {
            "tic_id": e.tic_id,
            "label": e.label,
            "source": e.source,
            "phase": list(e.phase),
            "flux": list(e.flux),
            "period_days": e.period_days,
            "snr": e.snr,
        }
        for e in entries
    ]
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(serialised, fh)
        os.replace(tmp, str(path))
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def load_batch_output(path: Path) -> list[SnippetEntry]:
    """Load :class:`SnippetEntry` objects from a JSON file written by :func:`save_batch_output`."""
    raw = json.loads(path.read_text())
    result = []
    for item in raw:
        result.append(
            SnippetEntry(
                tic_id=item["tic_id"],
                label=item["label"],
                source=item["source"],
                phase=tuple(item["phase"]),
                flux=tuple(item["flux"]),
                period_days=item.get("period_days"),
                snr=item.get("snr"),
            )
        )
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_snippet_batch(
    manifest: LabelManifest,
    *,
    snippet_fn: Callable,
    output_path: Path,
    checkpoint_path: Path,
    n_bins: int = 201,
    resume: bool = True,
    workers: int = 1,
) -> BatchBuildResult:
    """Extract CNN snippets for all records in *manifest*.

    Args:
        manifest:         :class:`LabelManifest` from multi_source_label_assembler.
        snippet_fn:       ``callable(tic_id, period, epoch, duration)``
                          returning ``(phase_list, flux_list)`` or ``None`` on failure.
        output_path:      Path to write the accumulated snippet JSON.
        checkpoint_path:  Path for the checkpoint JSON (completed/failed TIC IDs).
        n_bins:           Expected number of phase bins (for validation).
        resume:           If True, skip TIC IDs already in checkpoint.
        workers:          Concurrent threads for snippet_fn calls (default 1).
                          Values > 1 use ThreadPoolExecutor; state mutations
                          remain in the main thread via as_completed().

    Returns:
        :class:`BatchBuildResult`
    """
    if not hasattr(manifest, "records"):
        return BatchBuildResult(
            n_attempted=0, n_succeeded=0, n_failed=0, n_snippets=0,
            label_counts={}, output_path=str(output_path),
            checkpoint_path=str(checkpoint_path), flag="INVALID",
        )

    empty_cp: dict[str, set[str]] = {"completed": set(), "failed": set()}
    checkpoint = _load_checkpoint(checkpoint_path) if resume else empty_cp
    completed: set[str] = checkpoint["completed"]
    failed: set[str] = checkpoint["failed"]

    # Load existing snippets if resuming
    existing: list[SnippetEntry] = []
    if resume and output_path.exists():
        with contextlib.suppress(Exception):
            existing = load_batch_output(output_path)

    entries: list[SnippetEntry] = list(existing)
    n_attempted = 0
    n_succeeded = 0
    n_failed = 0

    pending = [r for r in manifest.records if not (resume and r.tic_id in completed)]

    def _process_record(record: object) -> tuple[object, object]:
        try:
            result = snippet_fn(
                record.tic_id,
                record.period_days,
                record.epoch,
                record.duration_hours,
            )
        except Exception:  # noqa: BLE001
            result = None
        return record, result

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process_record, r): r for r in pending}
            for future in as_completed(futures):
                record, result = future.result()
                tic_id = record.tic_id
                n_attempted += 1

                if result is None:
                    failed.add(tic_id)
                    n_failed += 1
                else:
                    phase_list, flux_list = result
                    if len(phase_list) != n_bins or len(flux_list) != n_bins:
                        failed.add(tic_id)
                        n_failed += 1
                    else:
                        entries.append(SnippetEntry(
                            tic_id=tic_id, label=record.label, source=record.source,
                            phase=tuple(float(x) for x in phase_list),
                            flux=tuple(float(x) for x in flux_list),
                            period_days=record.period_days, snr=None,
                        ))
                        completed.add(tic_id)
                        n_succeeded += 1

                _save_checkpoint(checkpoint_path, completed, failed)
    else:
        for record in pending:
            tic_id = record.tic_id
            n_attempted += 1
            _, result = _process_record(record)

            if result is None:
                failed.add(tic_id)
                n_failed += 1
            else:
                phase_list, flux_list = result
                if len(phase_list) != n_bins or len(flux_list) != n_bins:
                    failed.add(tic_id)
                    n_failed += 1
                    _save_checkpoint(checkpoint_path, completed, failed)
                    continue

                entries.append(SnippetEntry(
                    tic_id=tic_id, label=record.label, source=record.source,
                    phase=tuple(float(x) for x in phase_list),
                    flux=tuple(float(x) for x in flux_list),
                    period_days=record.period_days, snr=None,
                ))
                completed.add(tic_id)
                n_succeeded += 1

            # Save checkpoint after each record
            _save_checkpoint(checkpoint_path, completed, failed)

    # Save final output
    save_batch_output(entries, output_path)

    n_snippets = len(entries)
    label_counts: dict[int, int] = {0: 0, 1: 0}
    for e in entries:
        label_counts[e.label] = label_counts.get(e.label, 0) + 1

    if n_snippets == 0 or n_failed > 0 and n_succeeded == 0:
        flag = "EMPTY"
    elif n_failed > 0:
        flag = "PARTIAL"
    else:
        flag = "OK"

    return BatchBuildResult(
        n_attempted=n_attempted,
        n_succeeded=n_succeeded,
        n_failed=n_failed,
        n_snippets=n_snippets,
        label_counts=label_counts,
        output_path=str(output_path),
        checkpoint_path=str(checkpoint_path),
        flag=flag,
    )


def format_batch_result(result: BatchBuildResult) -> str:
    """Return a Markdown summary of a :class:`BatchBuildResult`."""
    lines = [
        "## Snippet Batch Build Result",
        "",
        f"**Flag**: {result.flag}",
        f"**Attempted**: {result.n_attempted}",
        f"**Succeeded**: {result.n_succeeded}",
        f"**Failed**: {result.n_failed}",
        f"**Total snippets**: {result.n_snippets}",
        f"- Planet (label=1): {result.label_counts.get(1, 0)}",
        f"- FP (label=0): {result.label_counts.get(0, 0)}",
        f"**Output**: {result.output_path}",
        f"**Checkpoint**: {result.checkpoint_path}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    parser = argparse.ArgumentParser(
        prog="lc_snippet_batch_builder",
        description="Build CNN snippet batch from a label manifest JSON.",
    )
    parser.add_argument("manifest", help="Path to label manifest JSON.")
    parser.add_argument("--output", required=True, help="Output snippet JSON path.")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint JSON path.")
    parser.add_argument("--n-bins", type=int, default=201)
    parser.add_argument("--workers", type=int, default=1, help="Concurrent threads (default 1)")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args(argv)

    # Import lazily so CLI works even without the assembler in path
    from multi_source_label_assembler import LabelManifest, LabelRecord  # noqa: PLC0415

    data = json.loads(Path(args.manifest).read_text())
    records = tuple(
        LabelRecord(
            tic_id=r["tic_id"], label=r["label"], source=r["source"],
            confidence=r.get("confidence", 1.0),
            period_days=r.get("period_days"), epoch=r.get("epoch"),
            duration_hours=r.get("duration_hours"), conflict=r.get("conflict", False),
        )
        for r in data.get("records", [])
    )
    manifest = LabelManifest(
        records=records,
        n_positive=data.get("n_positive", 0),
        n_negative=data.get("n_negative", 0),
        n_conflicts=data.get("n_conflicts", 0),
        sources=tuple(data.get("sources", [])),
        created_at=data.get("created_at", ""),
        flag=data.get("flag", "OK"),
    )

    def _dummy_snippet_fn(tic_id, period, epoch, duration):
        return None

    result = build_snippet_batch(
        manifest,
        snippet_fn=_dummy_snippet_fn,
        output_path=Path(args.output),
        checkpoint_path=Path(args.checkpoint),
        n_bins=args.n_bins,
        resume=not args.no_resume,
        workers=args.workers,
    )
    print(format_batch_result(result))
    return 0 if result.flag in ("OK", "PARTIAL") else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
