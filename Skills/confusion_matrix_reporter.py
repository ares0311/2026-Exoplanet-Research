"""Compute TP/FP/TN/FN + precision/recall/F1/accuracy at a threshold.

Public API
----------
ConfusionMatrixResult
compute_confusion_matrix(y_true, y_score, *, threshold) -> ConfusionMatrixResult
format_confusion_matrix(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfusionMatrixResult:
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float    # TP / (TP + FP); 0.0 when TP+FP==0
    recall: float       # TP / (TP + FN); 0.0 when TP+FN==0
    f1: float           # 0.0 when precision+recall==0
    accuracy: float     # (TP + TN) / n_total
    n_total: int
    n_positive: int
    n_negative: int
    flag: str           # "OK" | "DEGENERATE" | "INVALID"


def compute_confusion_matrix(
    y_true: list[int],
    y_score: list[float],
    *,
    threshold: float = 0.5,
) -> ConfusionMatrixResult:
    """Compute confusion matrix and derived metrics at a decision threshold."""
    if len(y_true) == 0 or len(y_true) != len(y_score):
        return ConfusionMatrixResult(
            threshold=threshold,
            tp=0, fp=0, tn=0, fn=0,
            precision=0.0, recall=0.0, f1=0.0, accuracy=0.0,
            n_total=0, n_positive=0, n_negative=0,
            flag="INVALID",
        )

    n_positive = sum(y_true)
    n_negative = len(y_true) - n_positive

    if n_positive == 0 or n_negative == 0:
        return ConfusionMatrixResult(
            threshold=threshold,
            tp=0, fp=0, tn=0, fn=0,
            precision=0.0, recall=0.0, f1=0.0, accuracy=0.0,
            n_total=len(y_true), n_positive=n_positive, n_negative=n_negative,
            flag="DEGENERATE",
        )

    tp = fp = tn = fn = 0
    for label, score in zip(y_true, y_score, strict=True):
        predicted = score >= threshold
        if predicted and label == 1:
            tp += 1
        elif predicted and label == 0:
            fp += 1
        elif not predicted and label == 0:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )
    accuracy = (tp + tn) / len(y_true)

    return ConfusionMatrixResult(
        threshold=threshold,
        tp=tp, fp=fp, tn=tn, fn=fn,
        precision=round(precision, 6),
        recall=round(recall, 6),
        f1=round(f1, 6),
        accuracy=round(accuracy, 6),
        n_total=len(y_true),
        n_positive=n_positive,
        n_negative=n_negative,
        flag="OK",
    )


def format_confusion_matrix(result: ConfusionMatrixResult) -> str:
    """Format confusion matrix result as Markdown."""
    lines = [
        "## Confusion Matrix Report",
        "",
        f"- Threshold: {result.threshold}",
        f"- Flag: **{result.flag}**",
        "",
    ]
    if result.flag != "OK":
        return "\n".join(lines) + "\n"

    lines += [
        "### 2x2 Matrix",
        "",
        "| | Predicted Positive | Predicted Negative |",
        "|---|---|---|",
        f"| **Actual Positive** | TP = {result.tp} | FN = {result.fn} |",
        f"| **Actual Negative** | FP = {result.fp} | TN = {result.tn} |",
        "",
        "### Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Precision | {result.precision:.4f} |",
        f"| Recall | {result.recall:.4f} |",
        f"| F1 | {result.f1:.4f} |",
        f"| Accuracy | {result.accuracy:.4f} |",
        f"| N total | {result.n_total} |",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415
    import json  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        prog="confusion_matrix_reporter",
        description="Compute confusion matrix at a threshold.",
    )
    parser.add_argument("input_json", help='JSON with {"y_true":[...],"y_score":[...]}')
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args(argv)

    try:
        data = json.loads(open(args.input_json).read())  # noqa: SIM115
        result = compute_confusion_matrix(
            data["y_true"], data["y_score"], threshold=args.threshold
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}")
        return 1

    print(format_confusion_matrix(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
