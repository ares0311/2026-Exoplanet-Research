"""In-memory stratified train/val/test splitter with balance reporting.

Uses a seeded ``random.Random`` instance for reproducibility without touching
global state.  Splits positives and negatives separately, then concatenates
and shuffles each split.

Public API
----------
SplitBalanceReport(n_total, n_positive, n_negative, balance_ratio)
StratifiedSplitResult(train, val, test, train_report, val_report, test_report,
                      seed, flag)
stratified_split(examples, *, train_frac, val_frac, seed) -> StratifiedSplitResult
format_split_result(result) -> str
"""
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class SplitBalanceReport:
    n_total: int
    n_positive: int
    n_negative: int
    balance_ratio: float | None  # n_positive / n_total; None if n_total==0


@dataclass(frozen=True)
class StratifiedSplitResult:
    train: tuple[dict, ...]
    val: tuple[dict, ...]
    test: tuple[dict, ...]
    train_report: SplitBalanceReport
    val_report: SplitBalanceReport
    test_report: SplitBalanceReport
    seed: int
    flag: str   # "OK" | "INSUFFICIENT" | "INVALID"


def _balance_report(items: list[dict]) -> SplitBalanceReport:
    n_total = len(items)
    n_positive = sum(1 for d in items if d.get("label") == 1)
    n_negative = sum(1 for d in items if d.get("label") == 0)
    balance_ratio = n_positive / n_total if n_total > 0 else None
    return SplitBalanceReport(
        n_total=n_total,
        n_positive=n_positive,
        n_negative=n_negative,
        balance_ratio=balance_ratio,
    )


def _split_class(items: list[dict], rng: random.Random, frac1: float, frac2: float
                 ) -> tuple[list[dict], list[dict], list[dict]]:
    """Split a single-class list into three parts by fraction."""
    n = len(items)
    shuffled = list(items)
    rng.shuffle(shuffled)
    n1 = round(n * frac1)
    n2 = round(n * frac2)
    return shuffled[:n1], shuffled[n1:n1 + n2], shuffled[n1 + n2:]


def stratified_split(
    examples: list[dict],
    *,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 42,
) -> StratifiedSplitResult:
    """Split examples into stratified train/val/test sets.

    Args:
        examples: List of dicts, each must have a ``"label"`` key (int 0 or 1).
        train_frac: Fraction of data for training (default 0.8).
        val_frac: Fraction of data for validation (default 0.1).
            The test fraction is ``1 - train_frac - val_frac``.
        seed: Random seed for reproducibility (default 42).

    Returns:
        :class:`StratifiedSplitResult`.

    Raises:
        ValueError: If ``train_frac + val_frac > 1.0``.
    """
    _INVALID_RESULT = StratifiedSplitResult(
        train=(), val=(), test=(),
        train_report=SplitBalanceReport(0, 0, 0, None),
        val_report=SplitBalanceReport(0, 0, 0, None),
        test_report=SplitBalanceReport(0, 0, 0, None),
        seed=seed,
        flag="INVALID",
    )

    if train_frac + val_frac > 1.0:
        return _INVALID_RESULT

    if not examples:
        return StratifiedSplitResult(
            train=(), val=(), test=(),
            train_report=SplitBalanceReport(0, 0, 0, None),
            val_report=SplitBalanceReport(0, 0, 0, None),
            test_report=SplitBalanceReport(0, 0, 0, None),
            seed=seed,
            flag="INSUFFICIENT",
        )

    positives = [d for d in examples if d.get("label") == 1]
    negatives = [d for d in examples if d.get("label") == 0]

    if len(positives) == 0 or len(negatives) == 0:
        return StratifiedSplitResult(
            train=(), val=(), test=(),
            train_report=SplitBalanceReport(0, 0, 0, None),
            val_report=SplitBalanceReport(0, 0, 0, None),
            test_report=SplitBalanceReport(0, 0, 0, None),
            seed=seed,
            flag="INSUFFICIENT",
        )

    rng = random.Random(seed)

    pos_train, pos_val, pos_test = _split_class(positives, rng, train_frac, val_frac)
    neg_train, neg_val, neg_test = _split_class(negatives, rng, train_frac, val_frac)

    train_list = pos_train + neg_train
    val_list = pos_val + neg_val
    test_list = pos_test + neg_test

    rng.shuffle(train_list)
    rng.shuffle(val_list)
    rng.shuffle(test_list)

    return StratifiedSplitResult(
        train=tuple(train_list),
        val=tuple(val_list),
        test=tuple(test_list),
        train_report=_balance_report(train_list),
        val_report=_balance_report(val_list),
        test_report=_balance_report(test_list),
        seed=seed,
        flag="OK",
    )


def format_split_result(result: StratifiedSplitResult) -> str:
    """Format a :class:`StratifiedSplitResult` as a Markdown string."""
    def _fmt_report(report: SplitBalanceReport) -> str:
        ratio_str = (
            f"{report.balance_ratio:.3f}" if report.balance_ratio is not None else "N/A"
        )
        return (
            f"{report.n_total} total, {report.n_positive} pos, "
            f"{report.n_negative} neg, ratio={ratio_str}"
        )

    lines = [
        "## Stratified Dataset Split",
        "",
        f"- **Seed:** {result.seed}",
        f"- **Flag:** {result.flag}",
        "",
        "| Split | Details |",
        "|-------|---------|",
        f"| Train | {_fmt_report(result.train_report)} |",
        f"| Val   | {_fmt_report(result.val_report)} |",
        f"| Test  | {_fmt_report(result.test_report)} |",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="stratified_dataset_splitter",
        description="Stratified train/val/test split of a JSON dataset.",
    )
    parser.add_argument("input", help='JSON file with list of {"label": 0|1, ...} dicts.')
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    with open(args.input) as fh:  # noqa: PTH123
        examples = json.load(fh)

    result = stratified_split(
        examples, train_frac=args.train_frac, val_frac=args.val_frac, seed=args.seed
    )
    print(format_split_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
