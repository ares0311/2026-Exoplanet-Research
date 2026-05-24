"""Platt-scaling calibration for CNN raw output probabilities.

Fits a sigmoid calibration function P_cal = 1/(1+exp(A*f+B)) using
gradient descent on binary cross-entropy loss (pure Python, no external deps).

Public API
----------
CnnCalibrationResult(method, platt_a, platt_b, n_samples,
                     brier_score_uncal, brier_score_cal, fitted_at, flag)
fit_cnn_calibration(y_true, y_prob) -> CnnCalibrationResult
apply_cnn_calibration(raw_prob, result) -> float
save_cnn_calibration(result, path) -> Path
load_cnn_calibration(path) -> CnnCalibrationResult
format_cnn_calibration(result) -> str
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CnnCalibrationResult:
    """Result of Platt-scaling calibration of CNN outputs."""

    method: str            # "platt"
    platt_a: float
    platt_b: float
    n_samples: int
    brier_score_uncal: float | None
    brier_score_cal: float | None
    fitted_at: str
    flag: str  # "OK" | "INSUFFICIENT" | "INVALID"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _brier_score(y_true: list[int], y_prob: list[float]) -> float:
    """Mean squared error between labels and probabilities."""
    if not y_true:
        return float("nan")
    return sum((p - y) ** 2 for y, p in zip(y_true, y_prob, strict=True)) / len(y_true)


def _log_loss_grad(
    y_true: list[int],
    y_prob: list[float],
    a: float,
    b: float,
) -> tuple[float, float, float]:
    """Return (loss, dL/dA, dL/dB) for Platt calibration.

    Calibrated probability: p_cal = sigmoid(A*f + B) where f = raw prob.
    """
    loss = 0.0
    grad_a = 0.0
    grad_b = 0.0
    n = len(y_true)
    for y, f in zip(y_true, y_prob, strict=True):
        logit = a * f + b
        p_cal = _sigmoid(logit)
        p_cal = max(1e-7, min(1.0 - 1e-7, p_cal))
        loss += -(y * math.log(p_cal) + (1 - y) * math.log(1.0 - p_cal))
        err = p_cal - y
        grad_a += err * f
        grad_b += err
    return loss / n, grad_a / n, grad_b / n


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fit_cnn_calibration(
    y_true: list[int],
    y_prob: list[float],
) -> CnnCalibrationResult:
    """Fit Platt scaling to CNN raw output probabilities.

    Uses gradient descent (pure Python) to minimise log-loss:
    ``L = -sum(y*log(p_cal) + (1-y)*log(1-p_cal))``
    where ``p_cal = sigmoid(A*raw_prob + B)``.

    Args:
        y_true: Binary ground-truth labels (0 or 1).
        y_prob: Raw predicted probabilities from CNN (in [0, 1]).

    Returns:
        :class:`CnnCalibrationResult` with fitted Platt parameters.
    """
    fitted_at = datetime.now(UTC).isoformat()

    if len(y_true) != len(y_prob):
        return CnnCalibrationResult(
            method="platt",
            platt_a=1.0,
            platt_b=0.0,
            n_samples=len(y_true),
            brier_score_uncal=None,
            brier_score_cal=None,
            fitted_at=fitted_at,
            flag="INVALID",
        )

    n = len(y_true)
    if n < 4:
        return CnnCalibrationResult(
            method="platt",
            platt_a=1.0,
            platt_b=0.0,
            n_samples=n,
            brier_score_uncal=None,
            brier_score_cal=None,
            fitted_at=fitted_at,
            flag="INSUFFICIENT",
        )

    # Check both classes are present
    labels_set = set(y_true)
    if not (0 in labels_set and 1 in labels_set):
        return CnnCalibrationResult(
            method="platt",
            platt_a=1.0,
            platt_b=0.0,
            n_samples=n,
            brier_score_uncal=_brier_score(y_true, y_prob),
            brier_score_cal=None,
            fitted_at=fitted_at,
            flag="INSUFFICIENT",
        )

    # Compute uncalibrated Brier score
    brier_uncal = _brier_score(y_true, y_prob)

    # Gradient descent for Platt A, B
    lr = 0.01
    max_iter = 1000
    a = 1.0
    b = 0.0
    for _ in range(max_iter):
        _loss, da, db = _log_loss_grad(y_true, y_prob, a, b)
        a -= lr * da
        b -= lr * db

    # Compute calibrated Brier score
    cal_probs = [apply_cnn_calibration(p, _make_temp_result(a, b)) for p in y_prob]
    brier_cal = _brier_score(y_true, cal_probs)

    return CnnCalibrationResult(
        method="platt",
        platt_a=round(a, 8),
        platt_b=round(b, 8),
        n_samples=n,
        brier_score_uncal=round(brier_uncal, 8),
        brier_score_cal=round(brier_cal, 8),
        fitted_at=fitted_at,
        flag="OK",
    )


def _make_temp_result(a: float, b: float) -> CnnCalibrationResult:
    """Construct a minimal CnnCalibrationResult for intermediate use."""
    return CnnCalibrationResult(
        method="platt",
        platt_a=a,
        platt_b=b,
        n_samples=0,
        brier_score_uncal=None,
        brier_score_cal=None,
        fitted_at="",
        flag="OK",
    )


def apply_cnn_calibration(raw_prob: float, result: CnnCalibrationResult) -> float:
    """Apply Platt scaling to a raw CNN output probability.

    Computes ``1 / (1 + exp(A*raw_prob + B))`` clipped to [1e-7, 1-1e-7].

    Args:
        raw_prob: Raw predicted probability from CNN (in [0, 1]).
        result: Fitted :class:`CnnCalibrationResult`.

    Returns:
        Calibrated probability in (0, 1).
    """
    logit = result.platt_a * raw_prob + result.platt_b
    p_cal = _sigmoid(logit)
    return max(1e-7, min(1.0 - 1e-7, p_cal))


def save_cnn_calibration(result: CnnCalibrationResult, path: Path) -> Path:
    """Serialize *result* to *path* as JSON (atomic write).

    Args:
        result: Calibration result to save.
        path: Destination file path.

    Returns:
        The resolved destination path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": result.method,
        "platt_a": result.platt_a,
        "platt_b": result.platt_b,
        "n_samples": result.n_samples,
        "brier_score_uncal": result.brier_score_uncal,
        "brier_score_cal": result.brier_score_cal,
        "fitted_at": result.fitted_at,
        "flag": result.flag,
    }
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.write("\n")
        os.replace(tmp, path)
    except Exception:
        with suppress(OSError):
            os.unlink(tmp)
        raise
    return path.resolve()


def load_cnn_calibration(path: Path) -> CnnCalibrationResult:
    """Load a :class:`CnnCalibrationResult` from a JSON file.

    Args:
        path: Path to the JSON file written by :func:`save_cnn_calibration`.

    Returns:
        Deserialized :class:`CnnCalibrationResult`.
    """
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return CnnCalibrationResult(
        method=str(d["method"]),
        platt_a=float(d["platt_a"]),
        platt_b=float(d["platt_b"]),
        n_samples=int(d["n_samples"]),
        brier_score_uncal=(
            float(d["brier_score_uncal"])
            if d.get("brier_score_uncal") is not None
            else None
        ),
        brier_score_cal=(
            float(d["brier_score_cal"])
            if d.get("brier_score_cal") is not None
            else None
        ),
        fitted_at=str(d.get("fitted_at", "")),
        flag=str(d["flag"]),
    )


def format_cnn_calibration(result: CnnCalibrationResult) -> str:
    """Format a :class:`CnnCalibrationResult` as Markdown.

    Args:
        result: Calibration result to format.

    Returns:
        Markdown summary string.
    """
    lines = [
        "## CNN Calibration Result",
        "",
        f"- Method: {result.method}",
        f"- Flag: {result.flag}",
        f"- Platt A: {result.platt_a:.6f}",
        f"- Platt B: {result.platt_b:.6f}",
        f"- n_samples: {result.n_samples}",
    ]
    if result.brier_score_uncal is not None:
        lines.append(f"- Brier (uncalibrated): {result.brier_score_uncal:.6f}")
    if result.brier_score_cal is not None:
        lines.append(f"- Brier (calibrated): {result.brier_score_cal:.6f}")
    lines.append(f"- Fitted at: {result.fitted_at}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="cnn_calibrator",
        description="Fit and apply Platt calibration to CNN output probabilities.",
    )
    sub = parser.add_subparsers(dest="cmd")

    p_fit = sub.add_parser("fit", help="Fit calibration from a JSON labels file.")
    p_fit.add_argument("labels_json", type=Path, metavar="LABELS_JSON",
                       help='JSON: {"y_true": [...], "y_prob": [...]}')
    p_fit.add_argument("--output", type=Path, default=None)

    p_apply = sub.add_parser("apply", help="Apply saved calibration to a probability.")
    p_apply.add_argument("calibration_json", type=Path)
    p_apply.add_argument("raw_prob", type=float)

    args = parser.parse_args(argv)

    if args.cmd == "fit":
        data = json.loads(Path(args.labels_json).read_text())
        y_true = [int(v) for v in data["y_true"]]
        y_prob = [float(v) for v in data["y_prob"]]
        result = fit_cnn_calibration(y_true, y_prob)
        print(format_cnn_calibration(result))
        if args.output:
            save_cnn_calibration(result, args.output)
            print(f"Saved to {args.output}")
        return 0 if result.flag == "OK" else 1

    if args.cmd == "apply":
        result = load_cnn_calibration(args.calibration_json)
        cal_prob = apply_cnn_calibration(args.raw_prob, result)
        print(f"raw_prob={args.raw_prob:.6f}  calibrated={cal_prob:.6f}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
