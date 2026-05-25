"""Parse and report progress of an LC snippet batch build job.

Reads the checkpoint JSON written by lc_snippet_batch_builder and an optional
output JSON with snippet records, then summarises how far along the batch run is.

Public API
----------
BatchProgressResult(n_completed, n_failed, n_total_manifest, pct_done,
                    n_snippets, label_counts, flag)
parse_batch_progress(checkpoint_path, *, output_path, total_manifest_size)
    -> BatchProgressResult
format_batch_progress(result) -> str
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BatchProgressResult:
    n_completed: int
    n_failed: int
    n_total_manifest: int | None   # from optional arg
    pct_done: float | None         # 100 * n_completed / n_total_manifest
    n_snippets: int                # from output JSON; 0 if not provided
    label_counts: dict             # {0: n, 1: n}
    flag: str                      # "OK" | "IN_PROGRESS" | "EMPTY" | "INVALID"


def parse_batch_progress(
    checkpoint_path: Path,
    *,
    output_path: Path | None = None,
    total_manifest_size: int | None = None,
) -> BatchProgressResult:
    """Parse a batch build checkpoint and optional output file.

    Args:
        checkpoint_path: Path to checkpoint JSON with keys
            ``completed_tic_ids`` and ``failed_tic_ids``.
        output_path: Optional path to output JSON list of snippet dicts
            (each should have a "label" key).
        total_manifest_size: Total number of TIC IDs in the job manifest,
            used to calculate ``pct_done``.

    Returns:
        :class:`BatchProgressResult`.
    """
    # --- load checkpoint ---
    try:
        with open(checkpoint_path) as fh:  # noqa: PTH123
            data = json.load(fh)
        completed = list(data.get("completed_tic_ids", []))
        failed = list(data.get("failed_tic_ids", []))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return BatchProgressResult(
            n_completed=0,
            n_failed=0,
            n_total_manifest=total_manifest_size,
            pct_done=None,
            n_snippets=0,
            label_counts={},
            flag="INVALID",
        )

    n_completed = len(completed)
    n_failed = len(failed)

    # --- load output snippets if provided ---
    label_counts: dict[int, int] = {}
    n_snippets = 0
    if output_path is not None:
        try:
            with open(output_path) as fh:  # noqa: PTH123
                snippets = json.load(fh)
            if isinstance(snippets, list):
                n_snippets = len(snippets)
                for snip in snippets:
                    if isinstance(snip, dict) and "label" in snip:
                        lbl = int(snip["label"])
                        label_counts[lbl] = label_counts.get(lbl, 0) + 1
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            pass  # leave empty

    # --- pct_done ---
    pct_done: float | None = None
    if total_manifest_size is not None and total_manifest_size > 0:
        raw = 100.0 * n_completed / total_manifest_size
        pct_done = min(raw, 100.0)

    # --- flag ---
    if n_completed + n_failed == 0:
        flag = "EMPTY"
    elif total_manifest_size is not None and n_completed < total_manifest_size:
        flag = "IN_PROGRESS"
    else:
        flag = "OK"

    return BatchProgressResult(
        n_completed=n_completed,
        n_failed=n_failed,
        n_total_manifest=total_manifest_size,
        pct_done=pct_done,
        n_snippets=n_snippets,
        label_counts=label_counts,
        flag=flag,
    )


def format_batch_progress(result: BatchProgressResult) -> str:
    """Format batch progress result as Markdown."""
    pct_str = f"{result.pct_done:.1f}%" if result.pct_done is not None else "N/A"
    total_str = str(result.n_total_manifest) if result.n_total_manifest is not None else "N/A"
    lines = [
        "## Snippet Batch Progress",
        "",
        f"- **completed:** {result.n_completed}",
        f"- **Failed:** {result.n_failed}",
        f"- **Total manifest:** {total_str}",
        f"- **% Done:** {pct_str}",
        f"- **Snippets in output:** {result.n_snippets}",
        f"- **Label counts:** {result.label_counts}",
        f"- **Flag:** {result.flag}",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="snippet_batch_progress",
        description="Report progress of an LC snippet batch build.",
    )
    parser.add_argument("checkpoint", type=Path, help="Checkpoint JSON file.")
    parser.add_argument("--output", type=Path, default=None, help="Output snippets JSON.")
    parser.add_argument(
        "--total",
        type=int,
        default=None,
        help="Total manifest size.",
    )
    args = parser.parse_args(argv)

    result = parse_batch_progress(
        args.checkpoint,
        output_path=args.output,
        total_manifest_size=args.total,
    )
    print(format_batch_progress(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
