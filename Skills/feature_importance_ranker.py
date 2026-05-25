"""Rank XGBoost features by importance (gain/cover/weight).

Parses an XGBoost metadata JSON dict (as produced by train_xgboost.py)
and ranks features by the requested importance type.

Public API
----------
FeatureImportanceEntry(name, importance, rank)
FeatureImportanceResult(importance_type, entries, n_features, flag)
rank_features(metadata, *, importance_type) -> FeatureImportanceResult
load_and_rank(metadata_path, *, importance_type) -> FeatureImportanceResult
format_feature_importance(result) -> str
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FeatureImportanceEntry:
    name: str
    importance: float
    rank: int


@dataclass(frozen=True)
class FeatureImportanceResult:
    importance_type: str  # "gain" | "cover" | "weight"
    entries: tuple[FeatureImportanceEntry, ...]
    n_features: int
    flag: str  # "OK" | "EMPTY" | "INVALID"


def rank_features(
    metadata: dict,
    *,
    importance_type: str = "gain",
) -> FeatureImportanceResult:
    """Parse metadata dict and return ranked feature importances.

    Args:
        metadata: Loaded XGBoost metadata JSON dict.
        importance_type: One of "gain", "cover", or "weight".

    Returns:
        FeatureImportanceResult with entries sorted descending by importance.
    """
    if not isinstance(metadata, dict):
        return FeatureImportanceResult(
            importance_type=importance_type,
            entries=(),
            n_features=0,
            flag="INVALID",
        )

    training_result = metadata.get("training_result")
    if training_result is None:
        return FeatureImportanceResult(
            importance_type=importance_type,
            entries=(),
            n_features=0,
            flag="INVALID",
        )

    importance_dict: dict = training_result.get("feature_importance", {})

    if not importance_dict:
        return FeatureImportanceResult(
            importance_type=importance_type,
            entries=(),
            n_features=0,
            flag="EMPTY",
        )

    sorted_items = sorted(importance_dict.items(), key=lambda kv: kv[1], reverse=True)
    entries = tuple(
        FeatureImportanceEntry(name=name, importance=float(imp), rank=i + 1)
        for i, (name, imp) in enumerate(sorted_items)
    )

    return FeatureImportanceResult(
        importance_type=importance_type,
        entries=entries,
        n_features=len(entries),
        flag="OK",
    )


def load_and_rank(
    metadata_path: Path,
    *,
    importance_type: str = "gain",
) -> FeatureImportanceResult:
    """Load metadata JSON from file and return ranked features.

    Args:
        metadata_path: Path to the XGBoost metadata JSON file.
        importance_type: One of "gain", "cover", or "weight".

    Returns:
        FeatureImportanceResult.
    """
    with open(metadata_path) as fh:
        metadata = json.load(fh)
    return rank_features(metadata, importance_type=importance_type)


def format_feature_importance(result: FeatureImportanceResult) -> str:
    """Format feature importance result as a Markdown table.

    Args:
        result: FeatureImportanceResult to format.

    Returns:
        Markdown string with rank, feature name, and importance columns.
    """
    lines: list[str] = [
        f"## Feature Importance ({result.importance_type})\n",
        f"Flag: `{result.flag}` | Features: {result.n_features}\n",
    ]

    if result.flag in ("INVALID", "EMPTY"):
        lines.append(f"\n_{result.flag}: no feature importance data available._\n")
        return "\n".join(lines)

    lines.append("")
    lines.append("| Rank | Feature | Importance |")
    lines.append("|------|---------|------------|")
    for entry in result.entries:
        lines.append(f"| {entry.rank} | {entry.name} | {entry.importance:.6f} |")

    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Rank XGBoost features by importance."
    )
    parser.add_argument("metadata_path", help="Path to XGBoost metadata JSON file.")
    parser.add_argument(
        "--importance-type",
        default="gain",
        choices=["gain", "cover", "weight"],
        help="Importance type to rank by (default: gain).",
    )
    args = parser.parse_args(argv)

    result = load_and_rank(
        Path(args.metadata_path),
        importance_type=args.importance_type,
    )
    print(format_feature_importance(result))
    return 0 if result.flag == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
