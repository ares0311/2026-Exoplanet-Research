"""Analyze positive/negative class balance in a training label set.

Reports class counts, imbalance ratio, and computes class weights for use in
CNN training loss functions. Flags when imbalance exceeds a configurable
threshold.

Public API
----------
LabelBalanceResult(n_positive, n_negative, n_total, ratio, class_weight_pos,
                   class_weight_neg, flag)
analyze_label_balance(rows, *, pos_label, imbalance_threshold) -> LabelBalanceResult
format_balance_report(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LabelBalanceResult:
    n_positive: int
    n_negative: int
    n_total: int
    ratio: float          # n_positive / n_negative; 0.0 if n_negative == 0
    class_weight_pos: float
    class_weight_neg: float
    flag: str  # "OK" | "IMBALANCED" | "EMPTY" | "INVALID"


def analyze_label_balance(
    rows: list[dict],
    *,
    pos_label: str = "planet_candidate",
    imbalance_threshold: float = 5.0,
) -> LabelBalanceResult:
    """Compute class balance statistics for a label set.

    Args:
        rows: List of dicts, each with a ``"label"`` key.
        pos_label: Label value that counts as positive class.
        imbalance_threshold: Flag as IMBALANCED if n_neg/n_pos > threshold
            (or n_pos/n_neg > threshold).

    Returns:
        LabelBalanceResult with counts, ratio, and balanced class weights.
    """
    if not isinstance(rows, list):
        return LabelBalanceResult(
            n_positive=0, n_negative=0, n_total=0, ratio=0.0,
            class_weight_pos=1.0, class_weight_neg=1.0, flag="INVALID",
        )
    if not rows:
        return LabelBalanceResult(
            n_positive=0, n_negative=0, n_total=0, ratio=0.0,
            class_weight_pos=1.0, class_weight_neg=1.0, flag="EMPTY",
        )

    n_pos = sum(1 for r in rows if r.get("label") == pos_label)
    n_neg = len(rows) - n_pos
    n_total = len(rows)

    ratio = n_pos / n_neg if n_neg > 0 else 0.0

    # Balanced class weights: w_i = n_total / (n_classes * n_i)
    # For binary: w_pos = n_total / (2 * n_pos), w_neg = n_total / (2 * n_neg)
    w_pos = (n_total / (2 * n_pos)) if n_pos > 0 else 1.0
    w_neg = (n_total / (2 * n_neg)) if n_neg > 0 else 1.0

    # Determine flag: imbalanced if either class outnumbers the other by threshold
    dominant = max(n_pos, n_neg)
    minority = min(n_pos, n_neg)
    imbalanced = (minority == 0) or (dominant / minority > imbalance_threshold)
    flag = "IMBALANCED" if imbalanced else "OK"

    return LabelBalanceResult(
        n_positive=n_pos,
        n_negative=n_neg,
        n_total=n_total,
        ratio=ratio,
        class_weight_pos=w_pos,
        class_weight_neg=w_neg,
        flag=flag,
    )


def format_balance_report(result: LabelBalanceResult) -> str:
    """Format a Markdown balance report.

    Args:
        result: LabelBalanceResult to format.

    Returns:
        Markdown string.
    """
    lines = [
        "## Label Balance Analysis\n",
        f"Flag: `{result.flag}` | Total: {result.n_total}\n",
    ]
    if result.flag in ("EMPTY", "INVALID"):
        lines.append(f"\n_{result.flag}: no label data._\n")
        return "\n".join(lines)

    lines.append("")
    lines.append("| Class | Count | Class Weight |")
    lines.append("|---|---|---|")
    pos_w = result.class_weight_pos
    neg_w = result.class_weight_neg
    lines.append(f"| Positive (planet_candidate) | {result.n_positive} | {pos_w:.4f} |")
    lines.append(f"| Negative (false_positive) | {result.n_negative} | {neg_w:.4f} |")
    lines.append("")
    lines.append(f"**Pos/Neg ratio**: {result.ratio:.3f}\n")

    if result.flag == "IMBALANCED":
        lines.append("> **Warning**: Imbalance exceeds threshold. Use class weights in loss.\n")
    else:
        lines.append("> Balance is acceptable for unweighted training.\n")

    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Analyze label class balance.")
    parser.add_argument("label_json", help="Path to JSON label rows file.")
    parser.add_argument("--pos-label", default="planet_candidate")
    parser.add_argument("--threshold", type=float, default=5.0)
    args = parser.parse_args(argv)

    rows = json.loads(Path(args.label_json).read_text())
    result = analyze_label_balance(rows, pos_label=args.pos_label,
                                   imbalance_threshold=args.threshold)
    print(format_balance_report(result))
    return 0 if result.flag in ("OK", "IMBALANCED") else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
