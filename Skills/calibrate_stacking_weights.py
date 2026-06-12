"""Calibrate stacking ensemble blend weights from held-out predictions.

Reads per-target predictions from three scorers (CNN, XGBoost, Bayesian) with
known labels, runs an AUC-maximising grid search, and saves the optimal blend
weights to ``models/stacking_weights.json``.

Two input modes
---------------
**Mode 1 — JSONL predictions file** (simplest)::

    python Skills/calibrate_stacking_weights.py \\
        --predictions predictions.jsonl \\
        --output models/stacking_weights.json

Each line of ``predictions.jsonl``::

    {"label": 1, "cnn_prob": 0.82, "xgb_prob": 0.75, "bayes_prob": 0.65}

**Mode 2 — pipeline JSON + labels CSV** (uses ``batch_scan`` output directly)::

    python Skills/calibrate_stacking_weights.py \\
        --pipeline-output pipeline_out.json \\
        --labels labels.csv \\
        --output models/stacking_weights.json

``labels.csv`` must have columns ``tic_id`` and ``label`` (0 or 1).
``pipeline_out.json`` must be from ``batch_scan.py --scorer full-ensemble``.

Public API
----------
StackingCalibResult(best_weights, best_auc, n_samples, n_positive, n_negative,
                    grid_step, flag, calibrated_at)
load_predictions_jsonl(path) -> tuple[list[int], list[float], list[float], list[float]]
extract_from_pipeline_output(pipeline_path, labels_path) -> Path
calibrate_stacking_weights(predictions_path, *, output_path, step) -> StackingCalibResult
format_calibration_result(result) -> str
"""
from __future__ import annotations

import contextlib
import csv
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StackingCalibResult:
    """Outcome of stacking weight calibration."""

    best_weights: tuple[float, float, float]  # (w_xgb, w_cnn, w_bayes)
    best_auc: float
    n_samples: int
    n_positive: int
    n_negative: int
    grid_step: float
    flag: str  # "OK" | "INSUFFICIENT" | "DEGENERATE" | "MISSING_FILE"
    calibrated_at: str
    output_path: str


# ---------------------------------------------------------------------------
# Input loaders
# ---------------------------------------------------------------------------


def load_predictions_jsonl(
    path: Path,
) -> tuple[list[int], list[float], list[float], list[float]]:
    """Load predictions from a JSONL file.

    Each line must be a JSON object with keys ``label``, ``cnn_prob``,
    ``xgb_prob``, and ``bayes_prob``.

    Args:
        path: Path to the JSONL file.

    Returns:
        Tuple of (labels, xgb_probs, cnn_probs, bayes_probs).

    Raises:
        ValueError: If required keys are missing from any record.
    """
    labels: list[int] = []
    xgb_probs: list[float] = []
    cnn_probs: list[float] = []
    bayes_probs: list[float] = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        for key in ("label", "cnn_prob", "xgb_prob", "bayes_prob"):
            if key not in rec:
                raise ValueError(f"Line {i+1}: missing key '{key}'")
        labels.append(int(rec["label"]))
        xgb_probs.append(float(rec["xgb_prob"]))
        cnn_probs.append(float(rec["cnn_prob"]))
        bayes_probs.append(float(rec["bayes_prob"]))
    return labels, xgb_probs, cnn_probs, bayes_probs


def extract_from_pipeline_output(
    pipeline_path: Path,
    labels_path: Path,
    *,
    output_path: Path | None = None,
) -> Path:
    """Extract per-target scorer predictions from batch_scan full-ensemble output.

    Joins the pipeline JSON (from ``batch_scan.py --scorer full-ensemble``) with
    a labels CSV (``tic_id``, ``label``) and writes a JSONL predictions file.

    Args:
        pipeline_path: Path to ``batch_scan`` JSON output.
        labels_path: Path to CSV with ``tic_id`` and ``label`` columns.
        output_path: Where to write the JSONL. Defaults to
            ``<pipeline_path.stem>_predictions.jsonl`` in the same directory.

    Returns:
        Path to the written JSONL file.

    Raises:
        ValueError: If no records could be matched between pipeline and labels.
    """
    pipeline_path = Path(pipeline_path)
    labels_path = Path(labels_path)
    if output_path is None:
        output_path = pipeline_path.parent / f"{pipeline_path.stem}_predictions.jsonl"
    output_path = Path(output_path)

    # Load labels
    label_map: dict[int, int] = {}
    with open(labels_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            tic_id = int(row["tic_id"])
            label_map[tic_id] = int(row["label"])

    # Load pipeline output (list or single dict)
    raw = json.loads(pipeline_path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        records = [raw]
    elif isinstance(raw, list):
        records = raw
    else:
        records = []

    matched = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for rec in records:
            tic_id = rec.get("tic_id")
            if tic_id is None or int(tic_id) not in label_map:
                continue
            label = label_map[int(tic_id)]

            # Extract individual scorer probabilities
            posterior = rec.get("posterior", {})
            bayes_prob = float(
                posterior.get("planet_candidate", rec.get("bayes_planet_probability", 0.5))
            )
            xgb_prob = float(rec.get("xgb_planet_probability", 0.5))
            cnn_prob = float(rec.get("cnn_planet_probability", 0.5))

            fh.write(json.dumps({
                "tic_id": int(tic_id),
                "label": label,
                "cnn_prob": cnn_prob,
                "xgb_prob": xgb_prob,
                "bayes_prob": bayes_prob,
            }) + "\n")
            matched += 1

    if matched == 0:
        raise ValueError(
            "No records matched between pipeline output and labels CSV. "
            "Check that tic_id values align."
        )
    return output_path


# ---------------------------------------------------------------------------
# Main calibration function
# ---------------------------------------------------------------------------


def calibrate_stacking_weights(
    predictions_path: Path,
    *,
    output_path: Path = Path("models/stacking_weights.json"),
    step: float = 0.05,
) -> StackingCalibResult:
    """Find optimal stacking blend weights that maximise AUC on held-out data.

    Loads predictions from a JSONL file (see :func:`load_predictions_jsonl`),
    runs a grid search over (w_xgb, w_cnn, w_bayes) constrained to sum to 1.0,
    and writes the optimal weights to ``output_path``.

    Args:
        predictions_path: Path to predictions JSONL file.
        output_path: Where to write ``stacking_weights.json``.
        step: Grid step for weight search (default 0.05 → 231 combinations).

    Returns:
        :class:`StackingCalibResult` with optimal weights and AUC.
    """
    predictions_path = Path(predictions_path)
    output_path = Path(output_path)
    calibrated_at = datetime.now(UTC).isoformat()

    if not predictions_path.exists():
        return StackingCalibResult(
            best_weights=(0.35, 0.35, 0.30),
            best_auc=0.0,
            n_samples=0,
            n_positive=0,
            n_negative=0,
            grid_step=step,
            flag="MISSING_FILE",
            calibrated_at=calibrated_at,
            output_path=str(output_path),
        )

    try:
        labels, xgb_probs, cnn_probs, bayes_probs = load_predictions_jsonl(
            predictions_path
        )
    except (ValueError, json.JSONDecodeError):
        return StackingCalibResult(
            best_weights=(0.35, 0.35, 0.30),
            best_auc=0.0,
            n_samples=0,
            n_positive=0,
            n_negative=0,
            grid_step=step,
            flag="MISSING_FILE",
            calibrated_at=calibrated_at,
            output_path=str(output_path),
        )

    n = len(labels)
    n_pos = sum(labels)
    n_neg = n - n_pos

    if n < 20 or n_pos == 0 or n_neg == 0:
        return StackingCalibResult(
            best_weights=(0.35, 0.35, 0.30),
            best_auc=0.0,
            n_samples=n,
            n_positive=n_pos,
            n_negative=n_neg,
            grid_step=step,
            flag="INSUFFICIENT",
            calibrated_at=calibrated_at,
            output_path=str(output_path),
        )

    # Import optimize_weights from ensemble_weight_optimizer
    _skills = str(Path(__file__).resolve().parent)
    if _skills not in sys.path:
        sys.path.insert(0, _skills)
    try:
        from Skills.ensemble_weight_optimizer import optimize_weights
    except ModuleNotFoundError:
        from ensemble_weight_optimizer import optimize_weights  # type: ignore[no-redef]

    opt = optimize_weights(
        labels, xgb_probs, cnn_probs, bayes_probs, step=step
    )

    if opt.flag not in ("OK",):
        return StackingCalibResult(
            best_weights=(0.35, 0.35, 0.30),
            best_auc=opt.best_auc,
            n_samples=n,
            n_positive=n_pos,
            n_negative=n_neg,
            grid_step=step,
            flag="DEGENERATE",
            calibrated_at=calibrated_at,
            output_path=str(output_path),
        )

    w_xgb, w_cnn, w_bayes = opt.best_weights
    payload = {
        "w_xgb": round(w_xgb, 4),
        "w_cnn": round(w_cnn, 4),
        "w_bayes": round(w_bayes, 4),
        "best_auc": round(opt.best_auc, 6),
        "n_samples": n,
        "n_positive": n_pos,
        "n_negative": n_neg,
        "grid_step": step,
        "calibrated_at": calibrated_at,
        "flag": "OK",
    }
    _atomic_write_json(output_path, payload)

    return StackingCalibResult(
        best_weights=opt.best_weights,
        best_auc=opt.best_auc,
        n_samples=n,
        n_positive=n_pos,
        n_negative=n_neg,
        grid_step=step,
        flag="OK",
        calibrated_at=calibrated_at,
        output_path=str(output_path),
    )


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------


def format_calibration_result(result: StackingCalibResult) -> str:
    """Format calibration result as a human-readable report."""
    w = result.best_weights
    lines = [
        "## Stacking Weight Calibration",
        f"- Flag: {result.flag}",
        f"- Best weights: XGBoost={w[0]:.3f}  CNN={w[1]:.3f}  Bayesian={w[2]:.3f}",
        f"- Best AUC: {result.best_auc:.4f}",
        f"- Samples: {result.n_samples}  "
        f"(pos={result.n_positive} neg={result.n_negative})",
        f"- Grid step: {result.grid_step}",
        f"- Output: {result.output_path}",
    ]
    if result.flag == "OK":
        lines += [
            "",
            "### stacking_scorer.py update",
            "Update the conservative fallback weights in",
            "``src/exo_toolkit/ml/stacking_scorer.py``:",
            "",
            "```python",
            f"# Calibrated {result.calibrated_at[:10]}",
            f"_DEFAULT_WEIGHTS = ({w[0]:.4f}, {w[1]:.4f}, {w[2]:.4f})  "
            "# (xgb, cnn, bayes)",
            "```",
            "",
            "### Git commit recipe",
            "```bash",
            "git pull origin main",
            f"git add {result.output_path}",
            'git commit -m "Calibrate stacking blend weights — T1-2 complete"',
            "git push -u origin HEAD",
            "```",
        ]
    elif result.flag == "INSUFFICIENT":
        lines.append(
            "\nNeed ≥ 20 samples with both positive and negative labels."
        )
    elif result.flag == "MISSING_FILE":
        lines.append("\nPredictions JSONL file not found.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Atomic write helper
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Calibrate stacking ensemble blend weights from held-out predictions."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--predictions", type=Path,
        help="JSONL file with {label, cnn_prob, xgb_prob, bayes_prob} per line.",
    )
    group.add_argument(
        "--pipeline-output", type=Path,
        help="batch_scan full-ensemble JSON output (use with --labels).",
    )
    parser.add_argument(
        "--labels", type=Path, default=None,
        help="CSV with tic_id,label columns (required with --pipeline-output).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("models/stacking_weights.json")
    )
    parser.add_argument(
        "--step", type=float, default=0.05,
        help="Weight grid step (default 0.05).",
    )
    args = parser.parse_args()

    if args.pipeline_output is not None:
        if args.labels is None:
            parser.error("--labels is required when using --pipeline-output")
        predictions_path = extract_from_pipeline_output(
            args.pipeline_output, args.labels
        )
    else:
        predictions_path = args.predictions

    result = calibrate_stacking_weights(
        predictions_path, output_path=args.output, step=args.step
    )
    print(format_calibration_result(result))
    sys.exit(0 if result.flag == "OK" else 1)


if __name__ == "__main__":
    _main()
