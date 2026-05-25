"""Side-by-side Markdown comparison of multiple model metric dicts.

Accepts a list of metric dicts (one per model) and produces a
ComparisonResult identifying best models by AUC, F1, and Brier score.
Note: best_by_brier is the model with the LOWEST Brier score.

Public API
----------
ModelMetrics(model_id, auc, f1, brier, n_train, notes)
ComparisonResult(models, best_by_auc, best_by_f1, best_by_brier, n_models, flag)
compare_models(metrics_list) -> ComparisonResult
format_comparison(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelMetrics:
    model_id: str
    auc: float | None
    f1: float | None
    brier: float | None
    n_train: int | None
    notes: str


@dataclass(frozen=True)
class ComparisonResult:
    models: tuple[ModelMetrics, ...]
    best_by_auc: str | None   # model_id of highest auc
    best_by_f1: str | None    # model_id of highest f1
    best_by_brier: str | None  # model_id of LOWEST brier (lower=better)
    n_models: int
    flag: str  # "OK" | "EMPTY" | "INVALID"


def _parse_model(d: dict) -> ModelMetrics:
    """Parse a single metrics dict into a ModelMetrics object."""
    return ModelMetrics(
        model_id=str(d["model_id"]),
        auc=float(d["auc"]) if d.get("auc") is not None else None,
        f1=float(d["f1"]) if d.get("f1") is not None else None,
        brier=float(d["brier"]) if d.get("brier") is not None else None,
        n_train=int(d["n_train"]) if d.get("n_train") is not None else None,
        notes=str(d.get("notes", "")),
    )


def compare_models(metrics_list: list[dict]) -> ComparisonResult:
    """Compare multiple model metrics dicts.

    Args:
        metrics_list: List of dicts, each with at least "model_id"; optional
            "auc", "f1", "brier", "n_train", "notes" fields.

    Returns:
        ComparisonResult with best model IDs by metric.
    """
    if not metrics_list:
        return ComparisonResult(
            models=(),
            best_by_auc=None,
            best_by_f1=None,
            best_by_brier=None,
            n_models=0,
            flag="EMPTY",
        )

    models = tuple(_parse_model(d) for d in metrics_list)

    auc_candidates = [(m.model_id, m.auc) for m in models if m.auc is not None]
    f1_candidates = [(m.model_id, m.f1) for m in models if m.f1 is not None]
    brier_candidates = [(m.model_id, m.brier) for m in models if m.brier is not None]

    best_by_auc = max(auc_candidates, key=lambda t: t[1])[0] if auc_candidates else None
    best_by_f1 = max(f1_candidates, key=lambda t: t[1])[0] if f1_candidates else None
    # Lower brier is better
    best_by_brier = (
        min(brier_candidates, key=lambda t: t[1])[0] if brier_candidates else None
    )

    return ComparisonResult(
        models=models,
        best_by_auc=best_by_auc,
        best_by_f1=best_by_f1,
        best_by_brier=best_by_brier,
        n_models=len(models),
        flag="OK",
    )


def format_comparison(result: ComparisonResult) -> str:
    """Format comparison result as a Markdown table.

    Args:
        result: ComparisonResult to format.

    Returns:
        Markdown string.
    """
    lines: list[str] = [
        "## Model Performance Comparison\n",
        f"Flag: `{result.flag}` | Models: {result.n_models}\n",
    ]

    if result.flag == "EMPTY":
        lines.append("\n_No models to compare._\n")
        return "\n".join(lines)

    lines.append("")
    lines.append("| Model | AUC | F1 | Brier | N Train | Notes |")
    lines.append("|-------|-----|-----|-------|---------|-------|")
    for m in result.models:
        auc_str = f"{m.auc:.4f}" if m.auc is not None else "—"
        f1_str = f"{m.f1:.4f}" if m.f1 is not None else "—"
        brier_str = f"{m.brier:.4f}" if m.brier is not None else "—"
        n_str = str(m.n_train) if m.n_train is not None else "—"
        lines.append(
            f"| {m.model_id} | {auc_str} | {f1_str} | {brier_str} | {n_str} | {m.notes} |"
        )

    lines.append("")
    if result.best_by_auc:
        lines.append(f"**Best AUC**: {result.best_by_auc}")
    if result.best_by_f1:
        lines.append(f"**Best F1**: {result.best_by_f1}")
    if result.best_by_brier:
        lines.append(f"**Best Brier (lowest)**: {result.best_by_brier}")

    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Compare model performance metrics side-by-side."
    )
    parser.add_argument(
        "metrics_json",
        help="Path to JSON file containing a list of model metric dicts.",
    )
    args = parser.parse_args(argv)

    with open(args.metrics_json) as fh:
        metrics_list = json.load(fh)

    result = compare_models(metrics_list)
    print(format_comparison(result))
    return 0 if result.flag == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
