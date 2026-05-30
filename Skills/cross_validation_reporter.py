"""Report k-fold cross-validation metrics with 95% confidence intervals.

Computes mean, standard deviation, and 95% CI (mean ± 1.96 * std / sqrt(k))
for AUC, F1, precision, and recall from a list of per-fold metric dicts.

Public API
----------
CVReport(k_folds, mean_auc, std_auc, ci_auc_low, ci_auc_high,
         mean_f1, std_f1, mean_precision, mean_recall, flag)
report_cross_validation(fold_metrics) -> CVReport
format_cv_report(result) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CVReport:
    k_folds: int
    mean_auc: float
    std_auc: float
    ci_auc_low: float
    ci_auc_high: float
    mean_f1: float
    std_f1: float
    mean_precision: float
    mean_recall: float
    flag: str  # "OK", "HIGH_VARIANCE"(std_auc > 0.05)


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _std(vals: list[float], m: float) -> float:
    if len(vals) < 2:
        return 0.0
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def report_cross_validation(fold_metrics: list[dict]) -> CVReport:
    """Report k-fold CV metrics with 95% CI.

    Args:
        fold_metrics: List of dicts, each with keys: auc, f1, precision, recall.

    Returns:
        CVReport with means, stds, and CI for each metric.
    """
    if not fold_metrics:
        return CVReport(
            k_folds=0,
            mean_auc=0.0, std_auc=0.0, ci_auc_low=0.0, ci_auc_high=0.0,
            mean_f1=0.0, std_f1=0.0,
            mean_precision=0.0, mean_recall=0.0,
            flag="OK",
        )

    k = len(fold_metrics)
    aucs = [float(fm.get("auc", 0.0)) for fm in fold_metrics]
    f1s = [float(fm.get("f1", 0.0)) for fm in fold_metrics]
    precs = [float(fm.get("precision", 0.0)) for fm in fold_metrics]
    recs = [float(fm.get("recall", 0.0)) for fm in fold_metrics]

    mean_auc = _mean(aucs)
    std_auc = _std(aucs, mean_auc)
    mean_f1 = _mean(f1s)
    std_f1 = _std(f1s, mean_f1)
    mean_precision = _mean(precs)
    mean_recall = _mean(recs)

    # 95% CI: mean ± 1.96 * std / sqrt(k)
    half_width = 1.96 * std_auc / math.sqrt(k) if k > 0 else 0.0
    ci_low = max(0.0, mean_auc - half_width)
    ci_high = min(1.0, mean_auc + half_width)

    flag = "HIGH_VARIANCE" if std_auc > 0.05 else "OK"

    return CVReport(
        k_folds=k,
        mean_auc=round(mean_auc, 4),
        std_auc=round(std_auc, 4),
        ci_auc_low=round(ci_low, 4),
        ci_auc_high=round(ci_high, 4),
        mean_f1=round(mean_f1, 4),
        std_f1=round(std_f1, 4),
        mean_precision=round(mean_precision, 4),
        mean_recall=round(mean_recall, 4),
        flag=flag,
    )


def format_cv_report(result: CVReport) -> str:
    """Format cross-validation report as Markdown.

    Args:
        result: CVReport to format.

    Returns:
        Markdown string.
    """
    lines = [
        "## Cross-Validation Report",
        "",
        f"- Folds: {result.k_folds}",
        f"- AUC: {result.mean_auc:.4f} ± {result.std_auc:.4f}",
        f"  - 95% CI: [{result.ci_auc_low:.4f}, {result.ci_auc_high:.4f}]",
        f"- F1: {result.mean_f1:.4f} ± {result.std_f1:.4f}",
        f"- Precision: {result.mean_precision:.4f}",
        f"- Recall: {result.mean_recall:.4f}",
        f"- Status: `{result.flag}`",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Report k-fold CV metrics.")
    parser.add_argument("metrics_json", help="JSON file with list of fold metric dicts.")
    args = parser.parse_args(argv)

    fold_metrics = json.loads(Path(args.metrics_json).read_text())
    result = report_cross_validation(fold_metrics)
    print(format_cv_report(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
