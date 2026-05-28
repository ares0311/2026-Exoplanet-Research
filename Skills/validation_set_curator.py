"""Carve out a clean held-out validation set from a labeled training corpus.

Ensures the validation set has:
- A configurable number of rows per class
- No TIC IDs that appear in the training split
- Balanced classes (or as close as the data allows)

Public API
----------
CurationResult(val_rows, train_rows, n_val_pos, n_val_neg, n_train,
               excluded_tic_ids, flag)
curate_validation_set(rows, *, n_val_per_class, pos_label, seed) -> CurationResult
format_curation_report(result) -> str
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class CurationResult:
    val_rows: tuple[dict, ...]
    train_rows: tuple[dict, ...]
    n_val_pos: int
    n_val_neg: int
    n_train: int
    excluded_tic_ids: tuple[int, ...]   # TIC IDs reserved for validation only
    flag: str  # "OK" | "PARTIAL" | "EMPTY" | "INVALID"


def curate_validation_set(
    rows: list[dict],
    *,
    n_val_per_class: int = 250,
    pos_label: str = "planet_candidate",
    seed: int = 42,
) -> CurationResult:
    """Split a label corpus into validation + training sets.

    Validation set is drawn first (stratified); remaining rows form training.
    TIC IDs in the validation set are excluded from training to prevent leakage.

    Args:
        rows: List of label dicts with ``tic_id`` and ``label`` keys.
        n_val_per_class: Target number of validation rows per class.
        pos_label: String value for the positive class label.
        seed: Random seed for reproducibility.

    Returns:
        CurationResult with split rows and metadata.
    """
    if not isinstance(rows, list):
        return CurationResult(val_rows=(), train_rows=(), n_val_pos=0, n_val_neg=0,
                              n_train=0, excluded_tic_ids=(), flag="INVALID")
    if not rows:
        return CurationResult(val_rows=(), train_rows=(), n_val_pos=0, n_val_neg=0,
                              n_train=0, excluded_tic_ids=(), flag="EMPTY")

    rng = random.Random(seed)
    pos_rows = [r for r in rows if r.get("label") == pos_label]
    neg_rows = [r for r in rows if r.get("label") != pos_label]

    rng.shuffle(pos_rows)
    rng.shuffle(neg_rows)

    val_pos = pos_rows[:n_val_per_class]
    val_neg = neg_rows[:n_val_per_class]
    val_all = val_pos + val_neg

    # Collect TIC IDs reserved for validation (exclude from train)
    import contextlib

    val_tic_ids: set[int] = set()
    for r in val_all:
        with contextlib.suppress(KeyError, TypeError, ValueError):
            val_tic_ids.add(int(r["tic_id"]))

    train_rows = [
        r for r in rows
        if r not in val_all and _tic_id(r) not in val_tic_ids
    ]

    n_val_pos = len(val_pos)
    n_val_neg = len(val_neg)

    flag = "OK"
    if n_val_pos < n_val_per_class or n_val_neg < n_val_per_class:
        flag = "PARTIAL"

    return CurationResult(
        val_rows=tuple(val_all),
        train_rows=tuple(train_rows),
        n_val_pos=n_val_pos,
        n_val_neg=n_val_neg,
        n_train=len(train_rows),
        excluded_tic_ids=tuple(sorted(val_tic_ids)),
        flag=flag,
    )


def _tic_id(row: dict) -> int | None:
    try:
        return int(row["tic_id"])
    except (KeyError, TypeError, ValueError):
        return None


def format_curation_report(result: CurationResult) -> str:
    """Format a Markdown curation summary.

    Args:
        result: CurationResult to format.

    Returns:
        Markdown string.
    """
    lines = [
        "## Validation Set Curation\n",
        f"Flag: `{result.flag}`\n",
    ]
    if result.flag in ("EMPTY", "INVALID"):
        lines.append(f"\n_{result.flag}: cannot curate validation set._\n")
        return "\n".join(lines)

    lines.append("")
    lines.append("| Split | Positive | Negative | Total |")
    lines.append("|---|---|---|---|")
    n_val = result.n_val_pos + result.n_val_neg
    lines.append(f"| Validation | {result.n_val_pos} | {result.n_val_neg} | {n_val} |")
    n_train_pos = sum(1 for r in result.train_rows if r.get("label") == "planet_candidate")
    n_train_neg = result.n_train - n_train_pos
    lines.append(f"| Training | {n_train_pos} | {n_train_neg} | {result.n_train} |")
    lines.append("")
    lines.append(f"**Excluded TIC IDs** (leakage prevention): {len(result.excluded_tic_ids)}\n")
    if result.flag == "PARTIAL":
        lines.append("> **Warning**: Fewer rows than requested for one or both classes.\n")
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Carve out a clean validation set.")
    parser.add_argument("label_json", help="Input JSON label rows.")
    parser.add_argument("--n-val", type=int, default=250, dest="n_val")
    parser.add_argument("--pos-label", default="planet_candidate")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-output", help="Write validation rows to this file.")
    parser.add_argument("--train-output", help="Write training rows to this file.")
    args = parser.parse_args(argv)

    rows = json.loads(Path(args.label_json).read_text())
    result = curate_validation_set(rows, n_val_per_class=args.n_val,
                                   pos_label=args.pos_label, seed=args.seed)
    print(format_curation_report(result))

    if args.val_output:
        Path(args.val_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.val_output).write_text(json.dumps(list(result.val_rows), indent=2))
    if args.train_output:
        Path(args.train_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.train_output).write_text(json.dumps(list(result.train_rows), indent=2))

    return 0 if result.flag in ("OK", "PARTIAL") else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
