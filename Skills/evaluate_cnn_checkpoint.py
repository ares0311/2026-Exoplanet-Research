"""Evaluate a trained CNN checkpoint against held-out promotion gates.

Fits Platt calibration on the validation split, evaluates on the sealed
test split, and checks whether the checkpoint passes the production gates
(default: raw AUC ≥ 0.85, calibrated F1 ≥ 0.80, calibrated Brier/ECE no
worse than raw Brier/ECE).

Saves calibration JSON alongside the checkpoint when gates pass.

Usage
-----
    python Skills/evaluate_cnn_checkpoint.py \\
        --split-dir data/cnn_splits \\
        --checkpoint models/cnn/best.pt \\
        --output-calibration models/cnn/calibration.json

Exit codes: 0 = PASS, 1 = FAIL, 2 = NO_TORCH or data/prediction error.

Public API
----------
CnnEvalMetrics(n, auc, f1, threshold, brier, ece)
CnnEvalResult(val_metrics_raw, test_metrics_raw, test_metrics_cal,
              platt_a, platt_b, gate_auc, gate_f1, passed, flag, evaluated_at)
evaluate_cnn_checkpoint(split_dir, checkpoint_path, *, gate_auc, gate_f1,
                        output_calibration, model_fn) -> CnnEvalResult
format_eval_result(result) -> str
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CnnEvalMetrics:
    """Metrics for a single split."""

    n: int
    auc: float
    f1: float
    threshold: float
    brier: float
    ece: float


@dataclass(frozen=True)
class CnnEvalResult:
    """Full evaluation outcome for a CNN checkpoint."""

    val_metrics_raw: CnnEvalMetrics | None
    test_metrics_raw: CnnEvalMetrics | None
    test_metrics_cal: CnnEvalMetrics | None
    platt_a: float
    platt_b: float
    gate_auc: float
    gate_f1: float
    passed: bool
    flag: str  # "PASS" | "FAIL" | "NO_TORCH" | "MISSING_SPLIT" | "INVALID_SPLIT" | "LOAD_ERROR"
    evaluated_at: str


# ---------------------------------------------------------------------------
# Metric helpers (no sklearn, no scipy)
# ---------------------------------------------------------------------------


def _auc_roc(y_true: list[int], y_score: list[float]) -> float:
    """Tie-aware ROC-AUC using average ranks."""
    pos = sum(y_true)
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return 0.5
    pairs = sorted(zip(y_score, y_true, strict=True), key=lambda p: p[0])
    rank_sum_pos = 0.0
    i = 0
    while i < len(pairs):
        j = i + 1
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        rank_sum_pos += avg_rank * sum(label for _, label in pairs[i:j])
        i = j
    return (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


def _best_f1_threshold(
    y_true: list[int], y_score: list[float], n_steps: int = 100
) -> tuple[float, float]:
    """Return (threshold, f1) that maximises F1 by sweep."""
    best_t = 0.5
    best_f1 = 0.0
    for step in range(n_steps + 1):
        t = step / n_steps
        tp = fp = fn = 0
        for yt, yp in zip(y_true, y_score, strict=True):
            pred = 1 if yp >= t else 0
            if pred == 1 and yt == 1:
                tp += 1
            elif pred == 1 and yt == 0:
                fp += 1
            elif pred == 0 and yt == 1:
                fn += 1
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


def _brier(y_true: list[int], y_score: list[float]) -> float:
    return sum((p - y) ** 2 for y, p in zip(y_true, y_score, strict=True)) / len(y_true)


def _ece(y_true: list[int], y_score: list[float], n_bins: int = 10) -> float:
    """Expected calibration error (uniform-width bins)."""
    n = len(y_true)
    if n == 0:
        return 0.0
    bins: list[list[tuple[int, float]]] = [[] for _ in range(n_bins)]
    for yt, yp in zip(y_true, y_score, strict=True):
        idx = min(int(yp * n_bins), n_bins - 1)
        bins[idx].append((yt, yp))
    total_err = 0.0
    for b in bins:
        if not b:
            continue
        frac_pos = sum(yt for yt, _ in b) / len(b)
        mean_prob = sum(yp for _, yp in b) / len(b)
        total_err += abs(frac_pos - mean_prob) * len(b)
    return total_err / n


def _compute_metrics(y_true: list[int], y_score: list[float]) -> CnnEvalMetrics:
    t, f1 = _best_f1_threshold(y_true, y_score)
    return CnnEvalMetrics(
        n=len(y_true),
        auc=round(_auc_roc(y_true, y_score), 6),
        f1=round(f1, 6),
        threshold=round(t, 4),
        brier=round(_brier(y_true, y_score), 6),
        ece=round(_ece(y_true, y_score), 6),
    )


def _valid_probabilities(probs: list[float], expected_n: int) -> bool:
    """Return True when model outputs are finite probabilities of expected size."""
    if len(probs) != expected_n:
        return False
    return all(
        isinstance(p, int | float)
        and not isinstance(p, bool)
        and math.isfinite(float(p))
        and 0.0 <= float(p) <= 1.0
        for p in probs
    )


# ---------------------------------------------------------------------------
# Platt calibration (gradient descent, no scipy)
# ---------------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return ex / (1.0 + ex)


def _fit_platt(
    y_true: list[int], y_prob: list[float], max_iter: int = 2000, lr: float = 0.01
) -> tuple[float, float]:
    """Return (A, B) for p_cal = sigmoid(A*raw + B)."""
    a, b = 1.0, 0.0
    for _ in range(max_iter):
        da = db = 0.0
        for yt, yp in zip(y_true, y_prob, strict=True):
            p_cal = _sigmoid(a * yp + b)
            err = p_cal - yt
            da += err * yp
            db += err
        a -= lr * da / len(y_true)
        b -= lr * db / len(y_true)
    return a, b


def _apply_platt(raw: float, a: float, b: float) -> float:
    return max(1e-7, min(1.0 - 1e-7, _sigmoid(a * raw + b)))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_split_examples(path: Path) -> list[dict] | None:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        examples = raw.get("examples")
    elif isinstance(raw, list):
        examples = raw
    else:
        return None
    if not isinstance(examples, list) or not all(isinstance(ex, dict) for ex in examples):
        return None
    return examples


def _examples_to_arrays(
    examples: list[dict], n_bins: int
) -> tuple[list[list[float]], list[int]] | None:
    fluxes, labels = [], []
    for ex in examples:
        if "flux" not in ex or "label" not in ex:
            return None
        label = ex["label"]
        if not isinstance(label, int) or isinstance(label, bool) or label not in (0, 1):
            return None
        if not isinstance(ex["flux"], list) or not ex["flux"]:
            return None
        flux = []
        for value in ex["flux"]:
            if (
                not isinstance(value, int | float)
                or isinstance(value, bool)
                or not math.isfinite(float(value))
            ):
                return None
            flux.append(float(value))
        flux = flux + [0.0] * (n_bins - len(flux)) if len(flux) < n_bins else flux[:n_bins]
        fluxes.append(flux)
        labels.append(label)
    return fluxes, labels


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _torch_infer(
    fluxes: list[list[float]], checkpoint_path: Path, config_path: Path
) -> list[float]:
    """Run inference with PyTorch. Raises ImportError if torch absent."""
    import torch

    # Add Skills dir to sys.path for relative imports when called as a script
    _skills = str(Path(__file__).resolve().parent)
    if _skills not in sys.path:
        sys.path.insert(0, _skills)

    try:
        from Skills.cnn_training_config import load_config
        from Skills.train_cnn import _build_torch_model
    except ModuleNotFoundError:
        from cnn_training_config import load_config  # type: ignore[no-redef]
        from train_cnn import _build_torch_model  # type: ignore[no-redef]

    config = load_config(config_path)
    model = _build_torch_model(config)
    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()

    batch_size = 256
    probs: list[float] = []
    with torch.no_grad():
        for i in range(0, len(fluxes), batch_size):
            batch = fluxes[i : i + batch_size]
            x = torch.tensor(batch, dtype=torch.float32).unsqueeze(1)
            out = model(x).tolist()
            probs.extend(out if isinstance(out, list) else [out])
    return probs


def _ensemble_infer(
    fluxes: list[list[float]], checkpoint_paths: list[Path], config_path: Path
) -> list[float]:
    """Average predictions from multiple checkpoints (ensemble inference)."""
    all_probs = [_torch_infer(fluxes, ckpt, config_path) for ckpt in checkpoint_paths]
    n = len(fluxes)
    return [sum(all_probs[m][i] for m in range(len(checkpoint_paths))) / len(checkpoint_paths)
            for i in range(n)]


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------


def evaluate_cnn_checkpoint(
    split_dir: Path,
    checkpoint_path: Path,
    *,
    gate_auc: float = 0.85,
    gate_f1: float = 0.80,
    output_calibration: Path | None = None,
    model_fn: Callable[[list[list[float]]], list[float]] | None = None,
    checkpoint_paths: list[Path] | None = None,
) -> CnnEvalResult:
    """Evaluate a CNN checkpoint (or ensemble) against production gates.

    Fits Platt calibration on the validation split, evaluates on the sealed test
    split with and without calibration, and checks raw AUC, calibrated F1, and
    calibration non-regression gates.

    Args:
        split_dir: Directory containing ``val.json`` and ``test.json``.
        checkpoint_path: Path to the primary ``.pt`` file (used for config).
        gate_auc: Minimum raw AUC on the test split to pass (default 0.85).
        gate_f1: Minimum calibrated F1 on the test split to pass (default 0.80).
            Calibration must also avoid worsening test Brier score or ECE.
        output_calibration: If provided, write calibration JSON here on pass.
        model_fn: Injectable inference function for testing.
        checkpoint_paths: If provided, run ensemble inference by averaging
            predictions from all listed checkpoints. ``checkpoint_path``
            is still used to locate the shared ``config.json``.

    Returns:
        :class:`CnnEvalResult` with all metrics and pass/fail flag.
    """
    evaluated_at = datetime.now(UTC).isoformat()
    split_dir = Path(split_dir)
    checkpoint_path = Path(checkpoint_path)
    config_path = checkpoint_path.parent / "config.json"

    val_path = split_dir / "val.json"
    test_path = split_dir / "test.json"

    for p in (val_path, test_path):
        if not p.exists():
            return CnnEvalResult(
                val_metrics_raw=None,
                test_metrics_raw=None,
                test_metrics_cal=None,
                platt_a=1.0,
                platt_b=0.0,
                gate_auc=gate_auc,
                gate_f1=gate_f1,
                passed=False,
                flag="MISSING_SPLIT",
                evaluated_at=evaluated_at,
            )

    # Determine n_bins from config (default 201)
    n_bins = 201
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            n_bins = int(cfg.get("n_bins", 201))
        except Exception:
            pass

    try:
        val_examples = _load_split_examples(val_path)
        test_examples = _load_split_examples(test_path)
    except Exception:
        return CnnEvalResult(
            val_metrics_raw=None,
            test_metrics_raw=None,
            test_metrics_cal=None,
            platt_a=1.0,
            platt_b=0.0,
            gate_auc=gate_auc,
            gate_f1=gate_f1,
            passed=False,
            flag="INVALID_SPLIT",
            evaluated_at=evaluated_at,
        )

    if val_examples is None or test_examples is None:
        return CnnEvalResult(
            val_metrics_raw=None,
            test_metrics_raw=None,
            test_metrics_cal=None,
            platt_a=1.0,
            platt_b=0.0,
            gate_auc=gate_auc,
            gate_f1=gate_f1,
            passed=False,
            flag="INVALID_SPLIT",
            evaluated_at=evaluated_at,
        )

    val_arrays = _examples_to_arrays(val_examples, n_bins)
    test_arrays = _examples_to_arrays(test_examples, n_bins)

    if val_arrays is None or test_arrays is None:
        return CnnEvalResult(
            val_metrics_raw=None,
            test_metrics_raw=None,
            test_metrics_cal=None,
            platt_a=1.0,
            platt_b=0.0,
            gate_auc=gate_auc,
            gate_f1=gate_f1,
            passed=False,
            flag="INVALID_SPLIT",
            evaluated_at=evaluated_at,
        )

    val_fluxes, val_labels = val_arrays
    test_fluxes, test_labels = test_arrays

    if not val_fluxes or not test_fluxes:
        return CnnEvalResult(
            val_metrics_raw=None,
            test_metrics_raw=None,
            test_metrics_cal=None,
            platt_a=1.0,
            platt_b=0.0,
            gate_auc=gate_auc,
            gate_f1=gate_f1,
            passed=False,
            flag="MISSING_SPLIT",
            evaluated_at=evaluated_at,
        )

    # Run inference
    if model_fn is not None:
        try:
            val_probs = model_fn(val_fluxes)
            test_probs = model_fn(test_fluxes)
        except Exception:
            return CnnEvalResult(
                val_metrics_raw=None,
                test_metrics_raw=None,
                test_metrics_cal=None,
                platt_a=1.0,
                platt_b=0.0,
                gate_auc=gate_auc,
                gate_f1=gate_f1,
                passed=False,
                flag="LOAD_ERROR",
                evaluated_at=evaluated_at,
            )
    elif checkpoint_paths is not None and len(checkpoint_paths) > 1:
        try:
            val_probs = _ensemble_infer(val_fluxes, checkpoint_paths, config_path)
            test_probs = _ensemble_infer(test_fluxes, checkpoint_paths, config_path)
        except ImportError:
            return CnnEvalResult(
                val_metrics_raw=None,
                test_metrics_raw=None,
                test_metrics_cal=None,
                platt_a=1.0,
                platt_b=0.0,
                gate_auc=gate_auc,
                gate_f1=gate_f1,
                passed=False,
                flag="NO_TORCH",
                evaluated_at=evaluated_at,
            )
        except Exception:
            return CnnEvalResult(
                val_metrics_raw=None,
                test_metrics_raw=None,
                test_metrics_cal=None,
                platt_a=1.0,
                platt_b=0.0,
                gate_auc=gate_auc,
                gate_f1=gate_f1,
                passed=False,
                flag="LOAD_ERROR",
                evaluated_at=evaluated_at,
            )
    else:
        try:
            val_probs = _torch_infer(val_fluxes, checkpoint_path, config_path)
            test_probs = _torch_infer(test_fluxes, checkpoint_path, config_path)
        except ImportError:
            return CnnEvalResult(
                val_metrics_raw=None,
                test_metrics_raw=None,
                test_metrics_cal=None,
                platt_a=1.0,
                platt_b=0.0,
                gate_auc=gate_auc,
                gate_f1=gate_f1,
                passed=False,
                flag="NO_TORCH",
                evaluated_at=evaluated_at,
            )
        except Exception:
            return CnnEvalResult(
                val_metrics_raw=None,
                test_metrics_raw=None,
                test_metrics_cal=None,
                platt_a=1.0,
                platt_b=0.0,
                gate_auc=gate_auc,
                gate_f1=gate_f1,
                passed=False,
                flag="LOAD_ERROR",
                evaluated_at=evaluated_at,
            )

    if (
        not _valid_probabilities(val_probs, len(val_labels))
        or not _valid_probabilities(test_probs, len(test_labels))
    ):
        return CnnEvalResult(
            val_metrics_raw=None,
            test_metrics_raw=None,
            test_metrics_cal=None,
            platt_a=1.0,
            platt_b=0.0,
            gate_auc=gate_auc,
            gate_f1=gate_f1,
            passed=False,
            flag="INVALID_PREDICTIONS",
            evaluated_at=evaluated_at,
        )

    val_metrics_raw = _compute_metrics(val_labels, val_probs)
    test_metrics_raw = _compute_metrics(test_labels, test_probs)

    # Fit Platt calibration on val predictions
    platt_a, platt_b = _fit_platt(val_labels, val_probs)
    test_probs_cal = [_apply_platt(p, platt_a, platt_b) for p in test_probs]
    test_metrics_cal = _compute_metrics(test_labels, test_probs_cal)

    calibration_not_worse = (
        test_metrics_cal.brier <= test_metrics_raw.brier
        and test_metrics_cal.ece <= test_metrics_raw.ece
    )
    passed = (
        test_metrics_raw.auc >= gate_auc
        and test_metrics_cal.f1 >= gate_f1
        and calibration_not_worse
    )

    # Save calibration JSON if requested and gates pass
    if output_calibration is not None and passed:
        output_calibration = Path(output_calibration)
        _atomic_write_json(
            output_calibration,
            {
                "method": "platt",
                "platt_a": round(platt_a, 8),
                "platt_b": round(platt_b, 8),
                "n_val_samples": len(val_labels),
                "fitted_at": evaluated_at,
                "gate_auc": gate_auc,
                "gate_f1": gate_f1,
                "test_auc_raw": test_metrics_raw.auc,
                "test_f1_raw": test_metrics_raw.f1,
                "test_brier_raw": test_metrics_raw.brier,
                "test_ece_raw": test_metrics_raw.ece,
                "test_f1_cal": test_metrics_cal.f1,
                "test_brier_cal": test_metrics_cal.brier,
                "test_ece_cal": test_metrics_cal.ece,
                "flag": "OK",
            },
        )

    return CnnEvalResult(
        val_metrics_raw=val_metrics_raw,
        test_metrics_raw=test_metrics_raw,
        test_metrics_cal=test_metrics_cal,
        platt_a=round(platt_a, 8),
        platt_b=round(platt_b, 8),
        gate_auc=gate_auc,
        gate_f1=gate_f1,
        passed=passed,
        flag="PASS" if passed else "FAIL",
        evaluated_at=evaluated_at,
    )


def format_eval_result(result: CnnEvalResult) -> str:
    """Format evaluation result as a human-readable report."""
    lines = [
        "## CNN Checkpoint Evaluation",
        f"- Flag: {result.flag}",
        f"- Gates: raw AUC ≥ {result.gate_auc}, calibrated F1 ≥ {result.gate_f1}, "
        "calibrated Brier/ECE no worse than raw",
        f"- Platt calibration: A={result.platt_a}, B={result.platt_b}",
    ]
    if result.val_metrics_raw is not None:
        m = result.val_metrics_raw
        lines.append(
            f"- Val (raw):  n={m.n}  AUC={m.auc:.4f}  F1={m.f1:.4f}"
            f"  Brier={m.brier:.4f}  ECE={m.ece:.4f}"
        )
    if result.test_metrics_raw is not None:
        m = result.test_metrics_raw
        lines.append(
            f"- Test (raw): n={m.n}  AUC={m.auc:.4f}  F1={m.f1:.4f}"
            f"  Brier={m.brier:.4f}  ECE={m.ece:.4f}"
        )
    if result.test_metrics_cal is not None:
        m = result.test_metrics_cal
        lines.append(
            f"- Test (cal): n={m.n}  AUC={m.auc:.4f}  F1={m.f1:.4f}"
            f"  threshold={m.threshold:.2f}  Brier={m.brier:.4f}  ECE={m.ece:.4f}"
        )
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
        description="Evaluate a CNN checkpoint against production gates."
    )
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path, required=True, nargs="+",
        help="Checkpoint path(s). Pass multiple to enable ensemble averaging.",
    )
    parser.add_argument(
        "--output-calibration", type=Path, default=None,
        help="Write calibration JSON here (only written if gates pass).",
    )
    parser.add_argument("--gate-auc", type=float, default=0.85)
    parser.add_argument("--gate-f1", type=float, default=0.80)
    args = parser.parse_args()

    checkpoints: list[Path] = args.checkpoint
    result = evaluate_cnn_checkpoint(
        args.split_dir,
        checkpoints[0],
        gate_auc=args.gate_auc,
        gate_f1=args.gate_f1,
        output_calibration=args.output_calibration,
        checkpoint_paths=checkpoints if len(checkpoints) > 1 else None,
    )
    print(format_eval_result(result))

    if result.flag == "NO_TORCH":
        sys.exit(2)
    if result.flag in ("MISSING_SPLIT", "INVALID_SPLIT", "LOAD_ERROR", "INVALID_PREDICTIONS"):
        sys.exit(2)
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    _main()
