"""Append-only JSONL training log for CNN / XGBoost training runs.

Each epoch is stored as one JSON line.  Multiple run IDs can coexist in the
same file.  A summary function derives best epoch, best AUC, and trend from
already-loaded records.

Public API
----------
TrainingRunRecord(run_id, epoch, train_loss, val_loss, val_auc, timestamp)
TrainingSummary(run_id, n_epochs, best_epoch, best_val_auc, best_val_loss,
                final_train_loss, flag)
append_epoch(path, record) -> None
load_training_log(path, *, run_id) -> list[TrainingRunRecord]
summarize_training_log(records, run_id) -> TrainingSummary
format_training_summary(result) -> str
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingRunRecord:
    run_id: str
    epoch: int
    train_loss: float
    val_loss: float
    val_auc: float
    timestamp: str     # ISO-8601


@dataclass(frozen=True)
class TrainingSummary:
    run_id: str
    n_epochs: int
    best_epoch: int
    best_val_auc: float
    best_val_loss: float
    final_train_loss: float
    flag: str   # "OK" | "EMPTY" | "INVALID"


def append_epoch(path: Path, record: TrainingRunRecord) -> None:
    """Append one epoch record to a JSONL log file.

    Creates the file (and parent dirs) if it does not yet exist.

    Args:
        path: Destination ``.jsonl`` file path.
        record: Epoch record to append.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:  # noqa: PTH123
        fh.write(json.dumps(asdict(record)) + "\n")


def load_training_log(
    path: Path,
    *,
    run_id: str | None = None,
) -> list[TrainingRunRecord]:
    """Load records from a JSONL training log.

    Args:
        path: Path to the ``.jsonl`` log file.
        run_id: If given, only return records for this run ID.

    Returns:
        List of :class:`TrainingRunRecord`; empty list if file not found.
    """
    if not path.exists():
        return []

    records: list[TrainingRunRecord] = []
    with open(path) as fh:  # noqa: PTH123
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                rec = TrainingRunRecord(
                    run_id=str(d["run_id"]),
                    epoch=int(d["epoch"]),
                    train_loss=float(d["train_loss"]),
                    val_loss=float(d["val_loss"]),
                    val_auc=float(d["val_auc"]),
                    timestamp=str(d["timestamp"]),
                )
                if run_id is None or rec.run_id == run_id:
                    records.append(rec)
            except (KeyError, ValueError, TypeError):
                # malformed line — skip
                continue

    return records


def summarize_training_log(
    records: list[TrainingRunRecord],
    run_id: str,
) -> TrainingSummary:
    """Summarise a list of already-loaded training records for one run.

    Args:
        records: Records to summarise (may contain other run IDs; they
            are ignored).
        run_id: Run whose records should be summarised.

    Returns:
        :class:`TrainingSummary`.
    """
    filtered = [r for r in records if r.run_id == run_id]
    if not filtered:
        return TrainingSummary(
            run_id=run_id,
            n_epochs=0,
            best_epoch=0,
            best_val_auc=0.0,
            best_val_loss=0.0,
            final_train_loss=0.0,
            flag="EMPTY",
        )

    best_rec = max(filtered, key=lambda r: r.val_auc)
    last_rec = filtered[-1]

    return TrainingSummary(
        run_id=run_id,
        n_epochs=len(filtered),
        best_epoch=best_rec.epoch,
        best_val_auc=best_rec.val_auc,
        best_val_loss=best_rec.val_loss,
        final_train_loss=last_rec.train_loss,
        flag="OK",
    )


def format_training_summary(result: TrainingSummary) -> str:
    """Format a :class:`TrainingSummary` as a Markdown string."""
    lines = [
        "## Training Curve Summary",
        "",
        f"- **Run ID:** {result.run_id}",
        f"- **Epochs logged:** {result.n_epochs}",
        f"- **Best epoch:** {result.best_epoch}",
        f"- **best_val_auc:** {result.best_val_auc:.6f}",
        f"- **Best val loss:** {result.best_val_loss:.6f}",
        f"- **Final train loss:** {result.final_train_loss:.6f}",
        f"- **Flag:** {result.flag}",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="training_curve_logger",
        description="Summarise a JSONL training log.",
    )
    parser.add_argument("log", type=Path, help="Path to .jsonl training log.")
    parser.add_argument("run_id", help="Run ID to summarise.")
    args = parser.parse_args(argv)

    records = load_training_log(args.log, run_id=args.run_id)
    summary = summarize_training_log(records, args.run_id)
    print(format_training_summary(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
