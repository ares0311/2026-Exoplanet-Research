"""Resume an interrupted CNN training run from the latest valid checkpoint.

Scans a checkpoint directory, identifies the most recent checkpoint by epoch,
validates it, and returns the resumption state (epoch, path). Supports
writing a resumption config that training scripts can read at startup.

Public API
----------
CheckpointInfo(path, epoch, val_auc, created_at, is_valid)
ResumptionState(checkpoint, next_epoch, n_epochs_remaining, flag)
find_latest_checkpoint(checkpoint_dir, *, validate_fn) -> CheckpointInfo | None
plan_resumption(checkpoint_dir, *, total_epochs, validate_fn) -> ResumptionState
format_resumption_state(state) -> str
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    epoch: int
    val_auc: float | None
    created_at: str
    is_valid: bool


@dataclass(frozen=True)
class ResumptionState:
    checkpoint: CheckpointInfo | None
    next_epoch: int
    n_epochs_remaining: int
    flag: str  # "RESUME" | "START_FRESH" | "COMPLETE" | "INVALID"


def _default_validate(path: Path) -> bool:
    """Check that a checkpoint file is a readable JSON or binary stub."""
    try:
        if path.suffix == ".json":
            json.loads(path.read_text())
        return path.stat().st_size > 0
    except Exception:
        return False


def _parse_epoch_from_name(name: str) -> int | None:
    """Extract epoch number from filenames like 'epoch_007.json' or 'ckpt_7.pt'."""
    m = re.search(r"(\d+)", name)
    return int(m.group(1)) if m else None


def find_latest_checkpoint(
    checkpoint_dir: Path,
    *,
    validate_fn: Callable[[Path], bool] | None = None,
) -> CheckpointInfo | None:
    """Scan a checkpoint directory and return the latest valid checkpoint.

    Args:
        checkpoint_dir: Directory containing checkpoint files.
        validate_fn: Optional callable that returns True if a checkpoint is
            readable. Defaults to checking file size > 0.

    Returns:
        CheckpointInfo for the latest valid checkpoint, or None.
    """
    if validate_fn is None:
        validate_fn = _default_validate

    d = Path(checkpoint_dir)
    if not d.exists():
        return None

    candidates: list[tuple[int, Path]] = []
    for f in d.iterdir():
        if not f.is_file():
            continue
        epoch = _parse_epoch_from_name(f.name)
        if epoch is not None:
            candidates.append((epoch, f))

    if not candidates:
        return None

    candidates.sort(key=lambda t: t[0], reverse=True)

    for epoch, path in candidates:
        valid = False
        import contextlib
        with contextlib.suppress(Exception):
            valid = validate_fn(path)

        val_auc: float | None = None
        if path.suffix == ".json" and valid:
            try:
                meta = json.loads(path.read_text())
                val_auc = float(meta.get("val_auc") or 0.0) or None
            except Exception:
                pass

        try:
            import os
            from datetime import UTC, datetime
            mtime = os.path.getmtime(path)
            created_at = datetime.fromtimestamp(mtime, tz=UTC).isoformat()
        except Exception:
            created_at = ""

        return CheckpointInfo(
            path=path,
            epoch=epoch,
            val_auc=val_auc,
            created_at=created_at,
            is_valid=valid,
        )

    return None


def plan_resumption(
    checkpoint_dir: Path,
    *,
    total_epochs: int = 20,
    validate_fn: Callable[[Path], bool] | None = None,
) -> ResumptionState:
    """Determine whether to resume or start fresh, and from which epoch.

    Args:
        checkpoint_dir: Directory containing checkpoint files.
        total_epochs: Total number of training epochs planned.
        validate_fn: Optional checkpoint validator.

    Returns:
        ResumptionState describing what to do next.
    """
    if total_epochs < 1:
        return ResumptionState(checkpoint=None, next_epoch=0,
                               n_epochs_remaining=0, flag="INVALID")

    ckpt = find_latest_checkpoint(Path(checkpoint_dir), validate_fn=validate_fn)

    if ckpt is None or not ckpt.is_valid:
        return ResumptionState(
            checkpoint=None,
            next_epoch=0,
            n_epochs_remaining=total_epochs,
            flag="START_FRESH",
        )

    if ckpt.epoch >= total_epochs:
        return ResumptionState(
            checkpoint=ckpt,
            next_epoch=ckpt.epoch,
            n_epochs_remaining=0,
            flag="COMPLETE",
        )

    next_epoch = ckpt.epoch + 1
    remaining = total_epochs - next_epoch
    return ResumptionState(
        checkpoint=ckpt,
        next_epoch=next_epoch,
        n_epochs_remaining=remaining,
        flag="RESUME",
    )


def format_resumption_state(state: ResumptionState) -> str:
    """Format a Markdown resumption state report.

    Args:
        state: ResumptionState to format.

    Returns:
        Markdown string.
    """
    lines = [
        "## Training Resumption Manager\n",
        f"Flag: `{state.flag}`\n",
    ]
    if state.flag == "INVALID":
        lines.append("\n_INVALID: cannot plan resumption._\n")
        return "\n".join(lines)

    lines += [
        f"**Next epoch**: {state.next_epoch}\n",
        f"**Epochs remaining**: {state.n_epochs_remaining}\n",
    ]

    if state.checkpoint:
        ckpt = state.checkpoint
        auc_str = f"{ckpt.val_auc:.4f}" if ckpt.val_auc is not None else "—"
        lines += [
            "",
            "### Latest Checkpoint",
            "",
            f"- Path: `{ckpt.path}`",
            f"- Epoch: {ckpt.epoch}",
            f"- Val AUC: {auc_str}",
            f"- Valid: {'Yes' if ckpt.is_valid else 'No'}",
            f"- Created: {ckpt.created_at}",
        ]
    else:
        lines.append("\n_No valid checkpoint found — starting from epoch 0._\n")

    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Plan CNN training resumption.")
    parser.add_argument("checkpoint_dir", help="Path to checkpoint directory.")
    parser.add_argument("--total-epochs", type=int, default=20)
    args = parser.parse_args(argv)

    state = plan_resumption(Path(args.checkpoint_dir), total_epochs=args.total_epochs)
    print(format_resumption_state(state))
    return 0 if state.flag in ("RESUME", "START_FRESH", "COMPLETE") else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
