"""CNN training loop with early stopping for exoplanet transit classification.

Trains a 1D convolutional neural network on phase-folded light curve snippets.
Requires PyTorch if available; returns a NO_TORCH result gracefully if not.

Split files are produced by ``build_cnn_training_data.py``.  Each split JSON
has the structure::

    {"split": "train", "examples": [{"flux": [...], "label": 0|1}, ...]}

Public API
----------
EpochRecord(epoch, train_loss, val_loss, val_auc)
CnnTrainingResult(best_epoch, best_val_loss, best_val_auc, train_history,
                  checkpoint_path, config_path, n_train, n_val,
                  n_positive, n_negative, flag)
train_cnn(split_dir, config, *, checkpoint_dir) -> CnnTrainingResult
format_training_result(result) -> str
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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Skills.cnn_training_config import CnnTrainingConfig


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpochRecord:
    """Metrics for a single training epoch."""

    epoch: int
    train_loss: float
    val_loss: float
    val_auc: float
    learning_rate: float = 0.0


@dataclass(frozen=True)
class CnnTrainingResult:
    """Outcome of a CNN training run."""

    best_epoch: int
    best_val_loss: float
    best_val_auc: float
    train_history: tuple[EpochRecord, ...]
    checkpoint_path: str
    config_path: str
    n_train: int
    n_val: int
    n_positive: int
    n_negative: int
    flag: str  # "OK" | "NO_TORCH" | "INSUFFICIENT_DATA" | "INVALID"


# ---------------------------------------------------------------------------
# AUC helper (no sklearn)
# ---------------------------------------------------------------------------


def _compute_auc(y_true: list[int], y_pred: list[float]) -> float:
    """Compute ROC-AUC via the trapezoidal rule without sklearn.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted probabilities.

    Returns:
        AUC in [0, 1], or 0.5 if degenerate.
    """
    if not y_true:
        return 0.5
    total_pos = sum(y_true)
    total_neg = len(y_true) - total_pos
    if total_pos == 0 or total_neg == 0:
        return 0.5

    pairs = sorted(zip(y_pred, y_true, strict=True), reverse=True)
    tp = fp = 0
    tps: list[int] = []
    fps: list[int] = []
    for _prob, label in pairs:
        if label == 1:
            tp += 1
        else:
            fp += 1
        tps.append(tp)
        fps.append(fp)

    tprs = [t / total_pos for t in tps]
    fprs = [f / total_neg for f in fps]
    # prepend origin
    tprs = [0.0] + tprs
    fprs = [0.0] + fprs
    auc = sum(
        (fprs[i] - fprs[i - 1]) * (tprs[i] + tprs[i - 1]) / 2
        for i in range(1, len(fprs))
    )
    return max(0.0, min(1.0, auc))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_split(split_path: Path) -> list[dict]:
    """Load examples from a split JSON file."""
    raw = json.loads(split_path.read_text(encoding="utf-8"))
    # Support both {"examples": [...]} and flat list
    if isinstance(raw, dict):
        return raw.get("examples", [])
    if isinstance(raw, list):
        return raw
    return []


def _bce_loss(pred: float, label: int) -> float:
    """Binary cross-entropy for a single sample."""
    p = max(1e-7, min(1.0 - 1e-7, pred))
    if label == 1:
        return -math.log(p)
    return -math.log(1.0 - p)


def _augment_training_batch(x, config: CnnTrainingConfig):  # noqa: ANN001, ANN201
    """Apply deterministic, train-only perturbations in normalized flux space."""
    import torch

    if not config.augment:
        return x
    scale = torch.empty(
        (x.shape[0], 1, 1),
        dtype=x.dtype,
        device=x.device,
    ).uniform_(config.augmentation_scale_min, config.augmentation_scale_max)
    sample_std = x.std(dim=2, keepdim=True).clamp_min(1e-6)
    noise = torch.randn_like(x) * sample_std * config.augmentation_noise_fraction
    return x * scale + noise


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
        with suppress(OSError):
            os.unlink(tmp)
        raise


# ---------------------------------------------------------------------------
# PyTorch CNN model (defined only if torch available)
# ---------------------------------------------------------------------------


def _build_torch_model(config: CnnTrainingConfig):  # noqa: ANN201
    """Build a 1D CNN in PyTorch from *config*."""
    import torch
    import torch.nn as nn

    class CnnModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            in_ch = 1
            conv_blocks: list[nn.Module] = []
            current_len = config.n_bins
            for cl in config.conv_layers:
                pad = cl.kernel_size // 2
                conv_blocks.append(nn.Conv1d(in_ch, cl.out_channels, cl.kernel_size, padding=pad))
                conv_blocks.append(nn.ReLU())
                conv_blocks.append(nn.MaxPool1d(cl.pool_size))
                in_ch = cl.out_channels
                current_len = current_len // cl.pool_size
            self.conv = nn.Sequential(*conv_blocks)
            flat_size = in_ch * max(current_len, 1)
            dense_layers: list[nn.Module] = []
            prev = flat_size
            for units, dropout_rate in zip(
                config.dense_units,
                config.dense_dropout_rates,
                strict=True,
            ):
                dense_layers.append(nn.Linear(prev, units))
                dense_layers.append(nn.ReLU())
                dense_layers.append(nn.Dropout(dropout_rate))
                prev = units
            dense_layers.append(nn.Linear(prev, 1))
            dense_layers.append(nn.Sigmoid())
            self.fc = nn.Sequential(*dense_layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[name-defined]
            x = self.conv(x)
            x = x.view(x.size(0), -1)
            return self.fc(x).squeeze(1)

    torch.manual_seed(config.seed)
    return CnnModel()


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------


def train_cnn(
    split_dir: Path,
    config: CnnTrainingConfig,
    *,
    checkpoint_dir: Path,
) -> CnnTrainingResult:
    """Train a 1D CNN on phase-folded light curves with early stopping.

    Loads ``split_dir/train.json`` and ``split_dir/val.json`` produced by
    ``build_cnn_training_data.py``.  Saves epoch checkpoints and ``best.pt``
    under *checkpoint_dir*.

    Args:
        split_dir: Directory containing ``train.json`` and ``val.json``.
        config: Hyperparameter config (see :func:`~cnn_training_config.default_config`).
        checkpoint_dir: Directory to write checkpoint ``.pt`` files and
            ``config.json``.

    Returns:
        :class:`CnnTrainingResult` with training history and flags.
    """
    # Try importing torch
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:
        return CnnTrainingResult(
            best_epoch=0,
            best_val_loss=float("inf"),
            best_val_auc=0.0,
            train_history=(),
            checkpoint_path="",
            config_path="",
            n_train=0,
            n_val=0,
            n_positive=0,
            n_negative=0,
            flag="NO_TORCH",
        )

    try:
        from Skills.cnn_training_config import validate_config
    except ModuleNotFoundError:
        from cnn_training_config import validate_config
    if not validate_config(config).ok:
        return CnnTrainingResult(
            best_epoch=0,
            best_val_loss=float("inf"),
            best_val_auc=0.0,
            train_history=(),
            checkpoint_path="",
            config_path="",
            n_train=0,
            n_val=0,
            n_positive=0,
            n_negative=0,
            flag="INVALID",
        )

    # Validate split directory
    split_dir = Path(split_dir)
    train_path = split_dir / "train.json"
    val_path = split_dir / "val.json"
    if not split_dir.exists() or not train_path.exists() or not val_path.exists():
        return CnnTrainingResult(
            best_epoch=0,
            best_val_loss=float("inf"),
            best_val_auc=0.0,
            train_history=(),
            checkpoint_path="",
            config_path="",
            n_train=0,
            n_val=0,
            n_positive=0,
            n_negative=0,
            flag="INVALID",
        )

    train_examples = _load_split(train_path)
    val_examples = _load_split(val_path)

    # Require at least 2 examples in each split with both labels in train
    train_labels = [int(e["label"]) for e in train_examples if "flux" in e and "label" in e]
    val_labels = [int(e["label"]) for e in val_examples if "flux" in e and "label" in e]
    n_positive = train_labels.count(1)
    n_negative = train_labels.count(0)

    if len(train_labels) < 4 or len(set(train_labels)) < 2 or len(val_labels) < 2:
        return CnnTrainingResult(
            best_epoch=0,
            best_val_loss=float("inf"),
            best_val_auc=0.0,
            train_history=(),
            checkpoint_path="",
            config_path="",
            n_train=len(train_labels),
            n_val=len(val_labels),
            n_positive=n_positive,
            n_negative=n_negative,
            flag="INSUFFICIENT_DATA",
        )

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    try:
        from Skills.cnn_training_config import save_config as _save_cfg
    except ModuleNotFoundError:  # Direct script execution adds Skills/ to sys.path.
        from cnn_training_config import save_config as _save_cfg
    config_path = checkpoint_dir / "config.json"
    _save_cfg(config, config_path)

    # Build dataset tensors
    def _examples_to_tensors(
        examples: list[dict],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fluxes = []
        labels = []
        for e in examples:
            if "flux" not in e or "label" not in e:
                continue
            flux = list(e["flux"])
            # Pad or truncate to n_bins
            if len(flux) < config.n_bins:
                flux = flux + [0.0] * (config.n_bins - len(flux))
            else:
                flux = flux[: config.n_bins]
            fluxes.append(flux)
            labels.append(float(e["label"]))
        x = torch.tensor(fluxes, dtype=torch.float32).unsqueeze(1)  # (N, 1, n_bins)
        y = torch.tensor(labels, dtype=torch.float32)
        return x, y

    x_train, y_train = _examples_to_tensors(train_examples)
    x_val, y_val = _examples_to_tensors(val_examples)

    model = _build_torch_model(config)
    criterion = nn.BCELoss()
    if config.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    elif config.optimizer == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_scheduler_factor,
        patience=config.lr_scheduler_patience,
        min_lr=config.min_learning_rate,
    )

    best_val_loss = float("inf")
    best_val_auc = 0.0
    best_selection_value = float("-inf")
    best_epoch = 0
    patience_counter = 0
    history: list[EpochRecord] = []
    best_checkpoint = str(checkpoint_dir / "best.pt")
    metrics_records: list[dict] = []

    n_train = len(x_train)
    batch_size = config.batch_size

    print(
        f"Training: {n_train} train / {len(x_val)} val"
        f" | batch={batch_size} max_epochs={config.max_epochs}"
        f" | early_stop patience={config.early_stopping_patience}",
        flush=True,
    )

    for epoch in range(1, config.max_epochs + 1):
        # --- training ---
        model.train()
        perm = torch.randperm(n_train)
        x_shuf = x_train[perm]
        y_shuf = y_train[perm]
        total_train_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            xb = x_shuf[start : start + batch_size]
            yb = y_shuf[start : start + batch_size]
            xb = _augment_training_batch(xb, config)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            if config.gradient_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.gradient_clip_norm,
                )
            optimizer.step()
            total_train_loss += loss.item()
            n_batches += 1
        train_loss = total_train_loss / max(n_batches, 1)

        # --- validation ---
        model.eval()
        with torch.no_grad():
            val_pred = model(x_val)
            val_loss_t = criterion(val_pred, y_val)
            val_loss = val_loss_t.item()
            y_prob_list = val_pred.tolist()
            y_true_list = y_val.tolist()

        val_auc = _compute_auc(
            [int(v) for v in y_true_list],
            [float(v) for v in y_prob_list],
        )
        learning_rate = float(optimizer.param_groups[0]["lr"])

        rec = EpochRecord(
            epoch=epoch,
            train_loss=round(train_loss, 6),
            val_loss=round(val_loss, 6),
            val_auc=round(val_auc, 6),
            learning_rate=learning_rate,
        )
        history.append(rec)

        # Save per-epoch checkpoint
        epoch_ckpt = str(checkpoint_dir / f"epoch_{epoch:04d}.pt")
        torch.save(model.state_dict(), epoch_ckpt)

        # Track for metrics.json
        metrics_records.append(
            {
                "path": epoch_ckpt,
                "epoch": epoch,
                "val_loss": round(val_loss, 6),
                "val_auc": round(val_auc, 6),
                "learning_rate": learning_rate,
                "created_at": datetime.now(UTC).isoformat(),
            }
        )

        selection_value = val_auc if config.selection_metric == "val_auc" else -val_loss
        is_new_best = selection_value > best_selection_value
        if is_new_best:
            best_selection_value = selection_value
            best_val_loss = val_loss
            best_val_auc = val_auc
            best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), best_checkpoint)
        else:
            patience_counter += 1

        _pat = f"patience {patience_counter}/{config.early_stopping_patience}"
        print(
            f"Epoch {epoch:3d}/{config.max_epochs}"
            f"  train={train_loss:.4f}  val={val_loss:.4f}"
            f"  auc={val_auc:.4f}  lr={learning_rate:.2e}"
            + ("  ← best" if is_new_best else f"  ({_pat})"),
            flush=True,
        )

        if patience_counter >= config.early_stopping_patience:
            print(
                f"Early stopping at epoch {epoch}"
                f" — best epoch {best_epoch} val_auc={best_val_auc:.4f}",
                flush=True,
            )
            break
        scheduler.step(val_loss)

    # Write metrics.json for checkpoint manager
    _atomic_write_json(
        checkpoint_dir / "metrics.json",
        {"checkpoints": metrics_records},
    )

    return CnnTrainingResult(
        best_epoch=best_epoch,
        best_val_loss=round(best_val_loss, 6),
        best_val_auc=round(best_val_auc, 6),
        train_history=tuple(history),
        checkpoint_path=best_checkpoint,
        config_path=str(config_path),
        n_train=len(x_train),
        n_val=len(x_val),
        n_positive=n_positive,
        n_negative=n_negative,
        flag="OK",
    )


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------


def format_training_result(result: CnnTrainingResult) -> str:
    """Format a :class:`CnnTrainingResult` as Markdown.

    Args:
        result: Training result to format.

    Returns:
        Markdown summary string.
    """
    lines = [
        "## CNN Training Result",
        "",
        f"- Flag: {result.flag}",
        f"- Best epoch: {result.best_epoch}",
        f"- Best val loss: {result.best_val_loss:.4f}",
        f"- Best val AUC: {result.best_val_auc:.4f}",
        f"- Epochs run: {len(result.train_history)}",
        f"- n_train: {result.n_train}  (pos={result.n_positive} neg={result.n_negative})",
        f"- n_val: {result.n_val}",
        f"- Checkpoint: {result.checkpoint_path}",
        f"- Config: {result.config_path}",
    ]
    if result.train_history:
        last = result.train_history[-1]
        lines.append(
            f"- Final epoch {last.epoch}: "
            f"train_loss={last.train_loss:.4f}  "
            f"val_loss={last.val_loss:.4f}  "
            f"val_auc={last.val_auc:.4f}"
        )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="train_cnn",
        description="Train 1D CNN on CNN split files with early stopping.",
    )
    parser.add_argument("--split-dir", type=Path, required=True, metavar="DIR")
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints/cnn"), metavar="DIR"
    )
    parser.add_argument("--config", type=Path, default=None, metavar="JSON")
    args = parser.parse_args(argv)

    try:
        from Skills.cnn_training_config import default_config, load_config
    except ModuleNotFoundError:  # Direct script execution adds Skills/ to sys.path.
        from cnn_training_config import default_config, load_config

    cfg = load_config(args.config) if args.config else default_config()
    result = train_cnn(args.split_dir, cfg, checkpoint_dir=args.checkpoint_dir)
    print(format_training_result(result))
    return 0 if result.flag == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
