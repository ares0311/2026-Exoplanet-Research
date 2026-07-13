"""Build empirical production score context from held-out labeled predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from exo_toolkit.candidate_context import CandidateContextReference  # noqa: E402
from Skills.run_report import RunReport, report_path_for, run_and_commit_report  # noqa: E402

ReportFn = Callable[..., bool]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_candidate_context(
    predictions_path: Path,
    weights_path: Path,
    output_path: Path,
    *,
    report_fn: ReportFn = run_and_commit_report,
) -> CandidateContextReference:
    """Build, validate, write, and report one empirical score reference."""
    started_at = datetime.now(UTC)
    started = time.monotonic()
    weights: dict[str, Any] = json.loads(weights_path.read_text(encoding="utf-8"))
    w_xgb = float(weights["w_xgb"])
    w_cnn = float(weights["w_cnn"])
    w_bayes = float(weights["w_bayes"])
    if abs(w_xgb + w_cnn + w_bayes - 1.0) > 1e-9:
        raise ValueError("stacking weights must sum to one")

    rows = [
        json.loads(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("predictions file contains no rows")
    print(
        f"Candidate context startup: rows={len(rows)} score=full_ensemble "
        f"dataset=t1_2_k2pandc_calibration",
        flush=True,
    )
    scored: list[tuple[float, int]] = []
    for index, row in enumerate(rows, 1):
        label = int(row["label"])
        if label not in (0, 1):
            raise ValueError(f"row {index} label must be 0 or 1")
        score = (
            w_xgb * float(row["xgb_prob"])
            + w_cnn * float(row["cnn_prob"])
            + w_bayes * float(row["bayes_prob"])
        )
        if not 0.0 <= score <= 1.0:
            raise ValueError(f"row {index} blended score is outside [0, 1]")
        scored.append((score, label))
        if index % 100 == 0 or index == len(rows):
            elapsed = time.monotonic() - started
            rate = index / elapsed if elapsed else 0.0
            eta = (len(rows) - index) / rate if rate else 0.0
            print(
                f"  [{index}/{len(rows)}] elapsed={elapsed:.2f}s ETA={eta:.2f}s",
                flush=True,
            )
    scored.sort(key=lambda item: item[0])
    labels = tuple(label for _, label in scored)
    reference = CandidateContextReference(
        context_id="full_ensemble_k2_empirical_context_v1",
        calibration_dataset_id="t1_2_k2pandc_calibration",
        score_name="full_ensemble_planet_probability",
        threshold_version="no_decision_threshold_v1",
        scores_ascending=tuple(score for score, _ in scored),
        labels_aligned=labels,
        n_samples=len(scored),
        n_positive=sum(labels),
        n_negative=len(labels) - sum(labels),
        source_predictions_path=str(predictions_path),
        source_predictions_sha256=_sha256(predictions_path),
        stacking_weights_path=str(weights_path),
        stacking_weights_sha256=_sha256(weights_path),
        limitations=(
            "Reference domain is held-out K2 catalog candidates, not TESS live search.",
            "The stacking optimizer maximized AUC; its score is not probability-calibrated.",
            "False-discovery context is the observed negative fraction at or above "
            "the score, not a guaranteed FDR bound.",
            "No decision threshold is defined or implied.",
        ),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(reference.model_dump_json(indent=2) + "\n", encoding="utf-8")
    completed_at = datetime.now(UTC)
    elapsed = time.monotonic() - started
    report = RunReport(
        script="build_candidate_context",
        status="success",
        started_at=started_at.isoformat(),
        completed_at=completed_at.isoformat(),
        elapsed_seconds=elapsed,
        items_processed=len(scored),
        items_written=len(scored),
        output_paths=(str(output_path),),
        notes="empirical full-ensemble rank/FDR context; no probability calibration or threshold",
    )
    report_path = report_path_for("build_candidate_context")
    if not report_fn(report, report_path):
        print(f"WARNING: Run Report push failed for {report_path}", flush=True)
    print(
        f"Candidate context COMPLETE: rows={len(scored)} elapsed={elapsed:.2f}s "
        f"output={output_path}",
        flush=True,
    )
    return reference


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("metadata/t1_2_k2_calibration_completed_predictions.jsonl"),
    )
    parser.add_argument(
        "--weights", type=Path, default=Path("models/stacking_weights.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("models/candidate_context_v1.json")
    )
    args = parser.parse_args(argv)
    build_candidate_context(args.predictions, args.weights, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
