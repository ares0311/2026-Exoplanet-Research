"""Batch CNN inference with optional Platt calibration.

Runs a trained 1D CNN model over a list of phase-folded flux arrays.
Supports an injectable ``model_fn`` for testing without PyTorch.

Public API
----------
CnnInferenceResult(n_inputs, probabilities, calibration_applied,
                   model_path, inference_time_ms, flag)
run_cnn_inference(flux_arrays, *, model_fn, model_path, calibration_path,
                  batch_size, n_bins) -> CnnInferenceResult
format_cnn_inference(result) -> str
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CnnInferenceResult:
    """Output of a batch CNN inference pass."""

    n_inputs: int
    probabilities: tuple[float, ...]
    calibration_applied: bool
    model_path: str | None
    inference_time_ms: float
    flag: str  # "OK" | "NO_TORCH" | "INVALID" | "EMPTY"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pad_or_truncate(flux: list[float], n_bins: int) -> list[float]:
    """Pad normalized flux with 0.0 or truncate it to exactly *n_bins* elements."""
    if len(flux) < n_bins:
        return flux + [0.0] * (n_bins - len(flux))
    return flux[:n_bins]


def _load_torch_model(model_path: Path):  # noqa: ANN201
    """Attempt to load a PyTorch CNN model from *model_path*.

    Returns ``(model, config)`` or raises ImportError / FileNotFoundError.
    """
    import sys
    from pathlib import Path as _Path

    # Add src to path so exo_toolkit is importable
    src_root = _Path(__file__).resolve().parent.parent / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    import torch  # noqa: F401  # raises ImportError if torch absent

    config_path = Path(model_path).parent / "config.json"
    try:
        from Skills.cnn_training_config import load_config
    except ModuleNotFoundError:  # Direct script execution adds Skills/ to sys.path.
        from cnn_training_config import load_config

    config = load_config(config_path)

    # Rebuild the model architecture using train_cnn._build_torch_model
    try:
        from Skills.train_cnn import _build_torch_model
    except ModuleNotFoundError:  # Direct script execution adds Skills/ to sys.path.
        from train_cnn import _build_torch_model  # type: ignore[import]

    model = _build_torch_model(config)
    state = torch.load(str(model_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, config


def _run_torch_batch(
    model,
    flux_arrays: list[list[float]],
    batch_size: int,
    n_bins: int,
) -> list[float]:
    """Run batched inference with a PyTorch model."""
    import torch

    probabilities: list[float] = []
    for start in range(0, len(flux_arrays), batch_size):
        batch = flux_arrays[start : start + batch_size]
        padded = [_pad_or_truncate(f, n_bins) for f in batch]
        x = torch.tensor(padded, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            preds = model(x)
        probabilities.extend(preds.tolist())
    return probabilities


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_cnn_inference(
    flux_arrays: list[list[float]],
    *,
    model_fn: Callable[[list[float]], float] | None = None,
    model_path: Path | None = None,
    calibration_path: Path | None = None,
    batch_size: int = 64,
    n_bins: int = 201,
) -> CnnInferenceResult:
    """Run batch CNN inference over *flux_arrays*.

    Priority for inference backend:
    1. ``model_fn`` — callable taking a flux array and returning a float;
       used directly without PyTorch (enables testing).
    2. ``model_path`` — path to a ``best.pt`` checkpoint; loads PyTorch model.
    3. If neither is available and PyTorch cannot be imported, returns
       ``flag="NO_TORCH"``.

    Args:
        flux_arrays: List of phase-folded flux arrays (each ~201 elements).
        model_fn: Optional injectable scoring function for testing.
        model_path: Path to a ``*.pt`` checkpoint file.
        calibration_path: Optional path to calibration JSON written by
            :func:`~cnn_calibrator.save_cnn_calibration`.
        batch_size: Batch size for PyTorch inference (ignored when
            ``model_fn`` is provided).
        n_bins: Expected flux array length; shorter normalized arrays are padded
            with 0.0, longer arrays are truncated.

    Returns:
        :class:`CnnInferenceResult` with all predicted probabilities.
    """
    t0 = time.monotonic()

    if not flux_arrays:
        return CnnInferenceResult(
            n_inputs=0,
            probabilities=(),
            calibration_applied=False,
            model_path=str(model_path) if model_path else None,
            inference_time_ms=0.0,
            flag="EMPTY",
        )

    # Validate inputs
    for _i, arr in enumerate(flux_arrays):
        if not isinstance(arr, (list, tuple)):
            return CnnInferenceResult(
                n_inputs=len(flux_arrays),
                probabilities=(),
                calibration_applied=False,
                model_path=str(model_path) if model_path else None,
                inference_time_ms=0.0,
                flag="INVALID",
            )

    probabilities: list[float]

    if model_fn is not None:
        # Use injectable model function
        probabilities = []
        for arr in flux_arrays:
            padded = _pad_or_truncate(list(arr), n_bins)
            prob = float(model_fn(padded))
            prob = max(0.0, min(1.0, prob))
            probabilities.append(prob)
        resolved_path = None

    elif model_path is not None:
        # Load PyTorch model
        try:
            model, config = _load_torch_model(Path(model_path))
            _n_bins = config.n_bins
            probabilities = _run_torch_batch(
                model, [list(a) for a in flux_arrays], batch_size, _n_bins
            )
            probabilities = [max(0.0, min(1.0, p)) for p in probabilities]
            resolved_path = str(model_path)
        except ImportError:
            return CnnInferenceResult(
                n_inputs=len(flux_arrays),
                probabilities=(),
                calibration_applied=False,
                model_path=str(model_path),
                inference_time_ms=_elapsed_ms(t0),
                flag="NO_TORCH",
            )
        except (FileNotFoundError, OSError):
            return CnnInferenceResult(
                n_inputs=len(flux_arrays),
                probabilities=(),
                calibration_applied=False,
                model_path=str(model_path),
                inference_time_ms=_elapsed_ms(t0),
                flag="INVALID",
            )
    else:
        # No model_fn and no model_path — try torch as last resort
        try:
            import torch  # noqa: F401
            # No model provided
            return CnnInferenceResult(
                n_inputs=len(flux_arrays),
                probabilities=(),
                calibration_applied=False,
                model_path=None,
                inference_time_ms=_elapsed_ms(t0),
                flag="INVALID",
            )
        except ImportError:
            return CnnInferenceResult(
                n_inputs=len(flux_arrays),
                probabilities=(),
                calibration_applied=False,
                model_path=None,
                inference_time_ms=_elapsed_ms(t0),
                flag="NO_TORCH",
            )

    # Apply calibration if requested
    calibration_applied = False
    if calibration_path is not None:
        try:
            try:
                from Skills.cnn_calibrator import (
                    apply_cnn_calibration,
                    load_cnn_calibration,
                )
            except ModuleNotFoundError:  # Direct script execution adds Skills/ to sys.path.
                from cnn_calibrator import (  # type: ignore[no-redef]
                    apply_cnn_calibration,
                    load_cnn_calibration,
                )
            cal = load_cnn_calibration(Path(calibration_path))
            if cal.flag == "OK":
                probabilities = [
                    apply_cnn_calibration(p, cal) for p in probabilities
                ]
                calibration_applied = True
        except (FileNotFoundError, OSError, KeyError, ValueError):
            pass  # Calibration optional; skip on error

    return CnnInferenceResult(
        n_inputs=len(flux_arrays),
        probabilities=tuple(round(p, 8) for p in probabilities),
        calibration_applied=calibration_applied,
        model_path=resolved_path,
        inference_time_ms=_elapsed_ms(t0),
        flag="OK",
    )


def _elapsed_ms(t0: float) -> float:
    return round((time.monotonic() - t0) * 1000.0, 2)


def format_cnn_inference(result: CnnInferenceResult) -> str:
    """Format a :class:`CnnInferenceResult` as Markdown.

    Args:
        result: Inference result to format.

    Returns:
        Markdown summary string.
    """
    lines = [
        "## CNN Inference Result",
        "",
        f"- Flag: {result.flag}",
        f"- Inputs: {result.n_inputs}",
        f"- Calibration applied: {result.calibration_applied}",
        f"- Inference time: {result.inference_time_ms:.1f} ms",
        f"- Model: {result.model_path or '(none)'}",
    ]
    if result.probabilities:
        probs = result.probabilities
        mean_p = sum(probs) / len(probs)
        lines += [
            "",
            f"- Mean probability: {mean_p:.4f}",
            f"- Min probability: {min(probs):.4f}",
            f"- Max probability: {max(probs):.4f}",
        ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="cnn_inference_batcher",
        description="Run batch CNN inference on flux arrays.",
    )
    parser.add_argument(
        "flux_json",
        type=Path,
        metavar="FLUX_JSON",
        help='JSON file: list of flux arrays or {"flux_arrays": [...]}',
    )
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--calibration-path", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-bins", type=int, default=201)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    raw = json.loads(Path(args.flux_json).read_text())
    flux_arrays = raw.get("flux_arrays", []) if isinstance(raw, dict) else raw

    result = run_cnn_inference(
        flux_arrays,
        model_path=args.model_path,
        calibration_path=args.calibration_path,
        batch_size=args.batch_size,
        n_bins=args.n_bins,
    )
    print(format_cnn_inference(result))
    if args.output:
        out = {
            "flag": result.flag,
            "n_inputs": result.n_inputs,
            "probabilities": list(result.probabilities),
            "calibration_applied": result.calibration_applied,
            "model_path": result.model_path,
            "inference_time_ms": result.inference_time_ms,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(out, indent=2))
    return 0 if result.flag == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
