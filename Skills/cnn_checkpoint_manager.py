"""Manage CNN checkpoint directories: list, select best, prune old.

Reads the ``metrics.json`` file written by ``train_cnn.py`` and provides
helpers for selecting the best checkpoint and optionally pruning stale ones.

Public API
----------
CheckpointRecord(path, epoch, val_loss, val_auc, created_at)
CheckpointSummary(checkpoint_dir, records, best_by_loss, best_by_auc, flag)
list_checkpoints(checkpoint_dir) -> CheckpointSummary
select_best(summary, *, criterion) -> CheckpointRecord | None
prune_checkpoints(summary, *, keep_top_k, criterion, dry_run) -> tuple[CheckpointRecord, ...]
format_checkpoint_summary(summary) -> str
"""
from __future__ import annotations

import json
import os
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckpointRecord:
    """Metadata for a single saved model checkpoint."""

    path: str
    epoch: int
    val_loss: float
    val_auc: float
    created_at: str


@dataclass(frozen=True)
class CheckpointSummary:
    """Summary of all checkpoints in a directory."""

    checkpoint_dir: str
    records: tuple[CheckpointRecord, ...]
    best_by_loss: CheckpointRecord | None
    best_by_auc: CheckpointRecord | None
    flag: str  # "OK" | "EMPTY" | "INVALID"


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def list_checkpoints(checkpoint_dir: Path) -> CheckpointSummary:
    """Scan *checkpoint_dir* and return a :class:`CheckpointSummary`.

    Reads ``checkpoint_dir/metrics.json`` if present (written by
    ``train_cnn.py``); otherwise falls back to scanning for ``*.pt`` files
    and constructing minimal records.

    Args:
        checkpoint_dir: Directory containing checkpoint files.

    Returns:
        :class:`CheckpointSummary` with flag ``"OK"``, ``"EMPTY"``, or
        ``"INVALID"`` (directory does not exist).
    """
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return CheckpointSummary(
            checkpoint_dir=str(checkpoint_dir),
            records=(),
            best_by_loss=None,
            best_by_auc=None,
            flag="INVALID",
        )

    metrics_path = ckpt_dir / "metrics.json"
    records: list[CheckpointRecord] = []

    if metrics_path.exists():
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            for entry in payload.get("checkpoints", []):
                records.append(
                    CheckpointRecord(
                        path=str(entry.get("path", "")),
                        epoch=int(entry.get("epoch", 0)),
                        val_loss=float(entry.get("val_loss", float("inf"))),
                        val_auc=float(entry.get("val_auc", 0.0)),
                        created_at=str(entry.get("created_at", "")),
                    )
                )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            records = []

    if not records:
        # Fall back: scan for epoch_*.pt files
        for pt_path in sorted(ckpt_dir.glob("epoch_*.pt")):
            # Try to extract epoch number from filename e.g. "epoch_0003.pt"
            stem = pt_path.stem  # "epoch_0003"
            try:
                epoch = int(stem.split("_")[-1])
            except ValueError:
                epoch = 0
            records.append(
                CheckpointRecord(
                    path=str(pt_path),
                    epoch=epoch,
                    val_loss=float("inf"),
                    val_auc=0.0,
                    created_at="",
                )
            )

    if not records:
        return CheckpointSummary(
            checkpoint_dir=str(checkpoint_dir),
            records=(),
            best_by_loss=None,
            best_by_auc=None,
            flag="EMPTY",
        )

    records_tuple = tuple(records)
    best_by_loss = min(records, key=lambda r: r.val_loss)
    best_by_auc = max(records, key=lambda r: r.val_auc)

    return CheckpointSummary(
        checkpoint_dir=str(checkpoint_dir),
        records=records_tuple,
        best_by_loss=best_by_loss,
        best_by_auc=best_by_auc,
        flag="OK",
    )


def select_best(
    summary: CheckpointSummary,
    *,
    criterion: str = "val_auc",
) -> CheckpointRecord | None:
    """Select the best checkpoint record by *criterion*.

    Args:
        summary: Result of :func:`list_checkpoints`.
        criterion: Either ``"val_auc"`` (higher is better) or ``"val_loss"``
            (lower is better).

    Returns:
        The best :class:`CheckpointRecord`, or ``None`` if *summary* is empty.

    Raises:
        ValueError: If *criterion* is not recognized.
    """
    if criterion not in {"val_auc", "val_loss"}:
        raise ValueError(
            f"criterion must be 'val_auc' or 'val_loss', got '{criterion}'"
        )
    if summary.flag != "OK" or not summary.records:
        return None
    if criterion == "val_auc":
        return summary.best_by_auc
    return summary.best_by_loss


def prune_checkpoints(
    summary: CheckpointSummary,
    *,
    keep_top_k: int = 3,
    criterion: str = "val_auc",
    dry_run: bool = True,
) -> tuple[CheckpointRecord, ...]:
    """Identify (and optionally delete) low-quality checkpoints.

    Keeps the *keep_top_k* best checkpoints (by *criterion*) and returns the
    records that would be or were deleted.

    Args:
        summary: Checkpoint summary from :func:`list_checkpoints`.
        keep_top_k: Number of best checkpoints to keep.
        criterion: Ranking criterion — ``"val_auc"`` or ``"val_loss"``.
        dry_run: If ``True`` (default), do not delete anything; just return
            what *would* be deleted.

    Returns:
        Tuple of :class:`CheckpointRecord` objects that were (or would be)
        deleted.

    Raises:
        ValueError: If *criterion* is not recognized or *keep_top_k* < 0.
    """
    if criterion not in {"val_auc", "val_loss"}:
        raise ValueError(
            f"criterion must be 'val_auc' or 'val_loss', got '{criterion}'"
        )
    if keep_top_k < 0:
        raise ValueError(f"keep_top_k must be >= 0, got {keep_top_k}")
    if not summary.records:
        return ()

    reverse = criterion == "val_auc"  # higher AUC is better
    sorted_records = sorted(
        summary.records,
        key=lambda r: r.val_auc if criterion == "val_auc" else r.val_loss,
        reverse=reverse,
    )

    to_keep = {id(r) for r in sorted_records[:keep_top_k]}
    to_delete = tuple(r for r in sorted_records if id(r) not in to_keep)

    if not dry_run:
        for record in to_delete:
            with suppress(OSError):
                os.unlink(record.path)

    return to_delete


def format_checkpoint_summary(summary: CheckpointSummary) -> str:
    """Format a :class:`CheckpointSummary` as Markdown.

    Args:
        summary: Checkpoint summary from :func:`list_checkpoints`.

    Returns:
        Markdown string.
    """
    lines = [
        "## CNN Checkpoint Summary",
        "",
        f"- Directory: `{summary.checkpoint_dir}`",
        f"- Status: {summary.flag}",
        f"- Total checkpoints: {len(summary.records)}",
    ]

    if summary.best_by_auc is not None:
        r = summary.best_by_auc
        lines.append(
            f"- Best by AUC: epoch {r.epoch}  val_auc={r.val_auc:.4f}  "
            f"val_loss={r.val_loss:.4f}"
        )
    if summary.best_by_loss is not None:
        r = summary.best_by_loss
        lines.append(
            f"- Best by loss: epoch {r.epoch}  val_loss={r.val_loss:.4f}  "
            f"val_auc={r.val_auc:.4f}"
        )

    if summary.records:
        lines += ["", "### All Checkpoints", ""]
        lines.append("| Epoch | val_loss | val_auc | Path |")
        lines.append("|---|---|---|---|")
        for rec in summary.records:
            lines.append(
                f"| {rec.epoch} | {rec.val_loss:.4f} | {rec.val_auc:.4f} | `{rec.path}` |"
            )

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="cnn_checkpoint_manager",
        description="List, select, or prune CNN checkpoints.",
    )
    sub = parser.add_subparsers(dest="cmd")

    p_list = sub.add_parser("list", help="List all checkpoints in a directory.")
    p_list.add_argument("checkpoint_dir", type=Path)

    p_best = sub.add_parser("best", help="Print the best checkpoint path.")
    p_best.add_argument("checkpoint_dir", type=Path)
    p_best.add_argument(
        "--criterion",
        choices=["val_auc", "val_loss"],
        default="val_auc",
    )

    p_prune = sub.add_parser("prune", help="Prune low-quality checkpoints.")
    p_prune.add_argument("checkpoint_dir", type=Path)
    p_prune.add_argument("--keep-top-k", type=int, default=3)
    p_prune.add_argument(
        "--criterion",
        choices=["val_auc", "val_loss"],
        default="val_auc",
    )
    p_prune.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete (default is dry_run).",
    )

    args = parser.parse_args(argv)

    if args.cmd == "list":
        summary = list_checkpoints(args.checkpoint_dir)
        print(format_checkpoint_summary(summary))
        return 0

    if args.cmd == "best":
        summary = list_checkpoints(args.checkpoint_dir)
        best = select_best(summary, criterion=args.criterion)
        if best is None:
            print("No checkpoints found.")
            return 1
        print(best.path)
        return 0

    if args.cmd == "prune":
        summary = list_checkpoints(args.checkpoint_dir)
        deleted = prune_checkpoints(
            summary,
            keep_top_k=args.keep_top_k,
            criterion=args.criterion,
            dry_run=not args.execute,
        )
        action = "Would delete" if not args.execute else "Deleted"
        for rec in deleted:
            print(f"{action}: {rec.path}")
        if not deleted:
            print("Nothing to prune.")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
