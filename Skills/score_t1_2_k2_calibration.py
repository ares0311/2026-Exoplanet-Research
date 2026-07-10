"""Score the T1-2 K2 calibration manifest with the Bayesian and XGBoost tiers.

Reads ``metadata/t1_2_k2_calibration_manifest.jsonl`` (built by
``build_t1_2_k2_calibration_manifest.py``) and computes, per row:

- ``bayes_prob`` — ``P(planet_candidate)`` from the Bayesian log-score model
  (``compute_posterior``), computed purely from catalog-derived features.
- ``xgb_prob`` — ``P(planet_candidate)`` from the trained
  ``models/xgboost_koi.json`` scorer, using the same catalog-derived features.

Both scores require no light curve at all — they are computed directly from
k2pandc catalog columns mapped into ``CandidateFeatures`` using the exact same
transform logic (``Skills/build_training_data.py``'s ``row_to_features``) the
XGBoost KOI model was trained with, so K2 columns are substituted in place of
their KOI equivalents:

============  =============================================  ==============
KOI column    K2 substitute                                    Notes
============  =============================================  ==============
koi_model_snr None (no K2 equivalent)                          stays None
koi_count     ``system_planet_count`` (k2pandc ``sy_pnum``)    multiplicity proxy
koi_duration  ``duration_hours``                               already hours
koi_period    ``period_days``                                  direct match
koi_depth     ``depth_ppm``                                    already ppm
koi_prad      ``planet_radius_rearth``                         direct match
koi_dikco_msky None (no K2 equivalent)                          stays None
============  =============================================  ==============

The ``cnn_prob`` field is intentionally left ``None`` here — the CNN tier
needs native K2 light-curve snippets, which
``fetch_t1_2_k2_calibration_snippets.py`` fetches separately. This script's
output is a **partial** predictions file; once snippets exist, a follow-up
step fills in ``cnn_prob`` before ``calibrate_stacking_weights.py`` can run
(it requires all three scores per row).

Public API
----------
row_to_candidate_features(row: dict) -> CandidateFeatures
score_calibration_manifest(manifest_path, *, xgb_model_path) -> list[dict]
write_partial_predictions(rows, output_path) -> None
format_scoring_summary(rows) -> str
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from build_training_data import row_to_features  # noqa: E402

from exo_toolkit.schemas import CandidateFeatures  # noqa: E402
from exo_toolkit.scoring import compute_posterior  # noqa: E402


def row_to_candidate_features(row: dict[str, Any]) -> CandidateFeatures:
    """Map one calibration manifest row to ``CandidateFeatures``.

    Reuses ``build_training_data.row_to_features`` unchanged so the transform
    exactly matches what the XGBoost KOI model was trained on. K2 columns are
    substituted for their (already unit-converted) KOI equivalents.

    Args:
        row: One parsed row from ``t1_2_k2_calibration_manifest.jsonl``.

    Returns:
        A ``CandidateFeatures`` instance (many fields ``None`` — expected;
        XGBoost handles missing values natively and the Bayesian model
        treats ``None`` as neutral evidence).
    """
    koi_style_row = {
        "koi_model_snr": None,
        "koi_count": row.get("system_planet_count"),
        "koi_duration": row.get("duration_hours"),
        "koi_period": row.get("period_days"),
        "koi_depth": row.get("depth_ppm"),
        "koi_prad": row.get("planet_radius_rearth"),
        "koi_dikco_msky": None,
    }
    return row_to_features(koi_style_row)


def _load_manifest(manifest_path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(manifest_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def score_calibration_manifest(
    manifest_path: Path,
    *,
    xgb_model_path: Path = Path("models/xgboost_koi.json"),
) -> list[dict[str, Any]]:
    """Score every manifest row with the Bayesian and XGBoost tiers.

    Args:
        manifest_path: Path to ``t1_2_k2_calibration_manifest.jsonl``.
        xgb_model_path: Path to the trained XGBoost KOI scorer metadata JSON.

    Returns:
        List of dicts with ``epic_id``, ``label``, ``xgb_prob``,
        ``bayes_prob``, and ``cnn_prob`` (always ``None`` — see module
        docstring).
    """
    from exo_toolkit.ml.xgboost_scorer import XGBoostScorer

    xgb_scorer = XGBoostScorer.load(xgb_model_path)
    manifest_rows = _load_manifest(manifest_path)

    scored: list[dict[str, Any]] = []
    for row in manifest_rows:
        features = row_to_candidate_features(row)
        bayes_prob = compute_posterior(features).planet_candidate
        xgb_prob = xgb_scorer.predict_proba(features)
        scored.append(
            {
                "epic_id": row["epic_id"],
                "label": row["label"],
                "xgb_prob": float(xgb_prob),
                "bayes_prob": float(bayes_prob),
                "cnn_prob": None,
            }
        )
    return scored


def write_partial_predictions(rows: list[dict[str, Any]], output_path: Path) -> None:
    """Write scored rows as JSONL (``cnn_prob`` will be ``null`` until filled in)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def format_scoring_summary(rows: list[dict[str, Any]]) -> str:
    """Render a concise Markdown summary of the partial scoring pass."""
    n = len(rows)
    n_pos = sum(1 for r in rows if r["label"] == 1)
    n_missing_cnn = sum(1 for r in rows if r["cnn_prob"] is None)
    return (
        "# T1-2 K2 Calibration — Bayesian + XGBoost Scoring\n\n"
        f"**Rows scored**: {n}\n"
        f"**Positive (label=1)**: {n_pos}\n"
        f"**Negative (label=0)**: {n - n_pos}\n"
        f"**Rows still missing cnn_prob**: {n_missing_cnn}\n\n"
        "## Next Action\n"
        "Fetch native K2 light-curve snippets for these EPIC targets "
        "(fetch_t1_2_k2_calibration_snippets.py), fill in cnn_prob for each "
        "row, then run calibrate_stacking_weights.py on the completed "
        "predictions file.\n"
    )


def _cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Score the T1-2 K2 calibration manifest with the Bayesian and "
            "XGBoost tiers (catalog-only, no light curve required)."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("metadata/t1_2_k2_calibration_manifest.jsonl"),
    )
    parser.add_argument(
        "--xgb-model-path", type=Path, default=Path("models/xgboost_koi.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("metadata/t1_2_k2_calibration_partial_predictions.jsonl"),
    )
    args = parser.parse_args(argv)

    rows = score_calibration_manifest(args.manifest, xgb_model_path=args.xgb_model_path)
    write_partial_predictions(rows, args.output)
    print(format_scoring_summary(rows), flush=True)
    print(f"Saved partial predictions to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
