"""Configuration for Kepler→TESS transfer learning (layer freezing + LR schedule).

When only limited TESS-labeled examples are available, fine-tuning a
Kepler-pretrained CNN with frozen early layers is more effective than
training from scratch. This module manages that configuration.

Public API
----------
TransferConfig(frozen_layers, learning_rate, fine_tune_lr_multiplier,
               n_fine_tune_epochs, n_warmup_epochs, weight_decay,
               dropout_rate, notes)
default_transfer_config() -> TransferConfig
load_transfer_config(path) -> TransferConfig
save_transfer_config(config, path) -> None
validate_transfer_config(config) -> list[str]   # list of error messages
format_transfer_config(config) -> str
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class TransferConfig:
    frozen_layers: tuple[str, ...]   # layer name prefixes to freeze
    learning_rate: float             # initial LR for unfrozen layers
    fine_tune_lr_multiplier: float   # LR multiplier for frozen-then-unfrozen phase
    n_fine_tune_epochs: int          # epochs with frozen layers
    n_warmup_epochs: int             # epochs before unfreezing
    weight_decay: float
    dropout_rate: float
    notes: str


def default_transfer_config() -> TransferConfig:
    """Return the recommended Kepler→TESS transfer config.

    Freezes the first two convolutional blocks; fine-tunes the fully-connected
    head for n_fine_tune_epochs, then unfreezes everything at a lower LR.
    """
    return TransferConfig(
        frozen_layers=("conv1", "conv2"),
        learning_rate=1e-4,
        fine_tune_lr_multiplier=0.1,
        n_fine_tune_epochs=10,
        n_warmup_epochs=3,
        weight_decay=1e-5,
        dropout_rate=0.5,
        notes="Kepler->TESS transfer: freeze early conv, fine-tune head first.",
    )


def load_transfer_config(path: Path) -> TransferConfig:
    """Load a TransferConfig from a JSON file.

    Args:
        path: Path to the JSON config file.

    Returns:
        TransferConfig instance.
    """
    raw = json.loads(Path(path).read_text())
    return TransferConfig(
        frozen_layers=tuple(raw.get("frozen_layers", [])),
        learning_rate=float(raw["learning_rate"]),
        fine_tune_lr_multiplier=float(raw.get("fine_tune_lr_multiplier", 0.1)),
        n_fine_tune_epochs=int(raw.get("n_fine_tune_epochs", 10)),
        n_warmup_epochs=int(raw.get("n_warmup_epochs", 3)),
        weight_decay=float(raw.get("weight_decay", 1e-5)),
        dropout_rate=float(raw.get("dropout_rate", 0.5)),
        notes=str(raw.get("notes", "")),
    )


def save_transfer_config(config: TransferConfig, path: Path) -> None:
    """Save a TransferConfig to a JSON file (atomic write).

    Args:
        config: TransferConfig to save.
        path: Destination path.
    """
    import contextlib
    import os
    import tempfile

    d = asdict(config)
    d["frozen_layers"] = list(config.frozen_layers)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(d, fh, indent=2)
        os.replace(tmp, path)
    except Exception:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def validate_transfer_config(config: TransferConfig) -> list[str]:
    """Validate a TransferConfig and return a list of error messages.

    Args:
        config: TransferConfig to validate.

    Returns:
        Empty list if valid; list of error strings otherwise.
    """
    errors: list[str] = []
    if config.learning_rate <= 0:
        errors.append("learning_rate must be > 0.")
    if not (0.0 < config.fine_tune_lr_multiplier <= 1.0):
        errors.append("fine_tune_lr_multiplier must be in (0, 1].")
    if config.n_fine_tune_epochs < 0:
        errors.append("n_fine_tune_epochs must be >= 0.")
    if config.n_warmup_epochs < 0:
        errors.append("n_warmup_epochs must be >= 0.")
    if config.weight_decay < 0:
        errors.append("weight_decay must be >= 0.")
    if not (0.0 <= config.dropout_rate < 1.0):
        errors.append("dropout_rate must be in [0, 1).")
    return errors


def format_transfer_config(config: TransferConfig) -> str:
    """Format a TransferConfig as a Markdown summary.

    Args:
        config: TransferConfig to format.

    Returns:
        Markdown string.
    """
    errors = validate_transfer_config(config)
    flag = "VALID" if not errors else "INVALID"
    lines = [
        "## Transfer Learning Config\n",
        f"Flag: `{flag}`\n",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| Frozen layers | {', '.join(config.frozen_layers) or '(none)'} |",
        f"| Learning rate | {config.learning_rate} |",
        f"| Fine-tune LR multiplier | {config.fine_tune_lr_multiplier} |",
        f"| Fine-tune epochs | {config.n_fine_tune_epochs} |",
        f"| Warmup epochs | {config.n_warmup_epochs} |",
        f"| Weight decay | {config.weight_decay} |",
        f"| Dropout rate | {config.dropout_rate} |",
        "",
    ]
    if config.notes:
        lines.append(f"**Notes**: {config.notes}\n")
    if errors:
        lines.append("**Errors**:")
        for e in errors:
            lines.append(f"- {e}")
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Manage transfer learning config.")
    parser.add_argument("--config", help="Path to JSON config file.")
    parser.add_argument("--output", help="Write default config to this path.")
    args = parser.parse_args(argv)

    if args.output:
        cfg = default_transfer_config()
        save_transfer_config(cfg, Path(args.output))
        print(f"Default transfer config written to {args.output}")
        return 0

    cfg = load_transfer_config(Path(args.config)) if args.config else default_transfer_config()

    errors = validate_transfer_config(cfg)
    print(format_transfer_config(cfg))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
