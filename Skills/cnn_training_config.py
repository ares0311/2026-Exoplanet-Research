"""Define, validate, and serialize the CNN training hyperparameter config.

Provides a frozen dataclass representation of all tunable knobs for 1D-CNN
training (architecture, optimiser, regularisation, schedule).  Configs can be
round-tripped through JSON for reproducible experiment tracking.

Public API
----------
ConvLayerSpec(out_channels, kernel_size, pool_size)
CnnTrainingConfig(n_bins, conv_layers, dense_units, dropout_rate,
                  dense_dropout_rates, optimizer, learning_rate, batch_size, max_epochs,
                  early_stopping_patience, augment, seed, checkpoint_dir)
CnnConfigValidation(ok, errors)
default_config() -> CnnTrainingConfig
load_config(path) -> CnnTrainingConfig
save_config(config, path) -> Path
validate_config(config) -> CnnConfigValidation
"""
from __future__ import annotations

import json
import os
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConvLayerSpec:
    """Specification for a single 1D convolutional + max-pool block."""

    out_channels: int
    kernel_size: int
    pool_size: int


@dataclass(frozen=True)
class CnnTrainingConfig:
    """Full CNN training hyperparameter configuration."""

    n_bins: int
    conv_layers: tuple[ConvLayerSpec, ...]
    dense_units: tuple[int, ...]
    dropout_rate: float
    dense_dropout_rates: tuple[float, ...]
    optimizer: str
    learning_rate: float
    batch_size: int
    max_epochs: int
    early_stopping_patience: int
    augment: bool
    seed: int
    checkpoint_dir: str


@dataclass(frozen=True)
class CnnConfigValidation:
    """Result of validating a :class:`CnnTrainingConfig`."""

    ok: bool
    errors: tuple[str, ...]


# ---------------------------------------------------------------------------
# Defaults (per CNN_SPEC.md)
# ---------------------------------------------------------------------------


def default_config() -> CnnTrainingConfig:
    """Return the default CNN config per CNN_SPEC.md.

    Architecture: 3 conv layers (16/32/64 channels), dense layers 256/64,
    dropout rates 0.5/0.3, adam lr=1e-3, batch=64, max_epochs=50,
    patience=10.
    """
    return CnnTrainingConfig(
        n_bins=201,
        conv_layers=(
            ConvLayerSpec(out_channels=16, kernel_size=5, pool_size=2),
            ConvLayerSpec(out_channels=32, kernel_size=5, pool_size=2),
            ConvLayerSpec(out_channels=64, kernel_size=3, pool_size=2),
        ),
        dense_units=(256, 64),
        dropout_rate=0.5,
        dense_dropout_rates=(0.5, 0.3),
        optimizer="adam",
        learning_rate=1e-3,
        batch_size=64,
        max_epochs=50,
        early_stopping_patience=10,
        augment=True,
        seed=42,
        checkpoint_dir="checkpoints/cnn",
    )


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def _config_to_dict(config: CnnTrainingConfig) -> dict:
    """Convert a :class:`CnnTrainingConfig` to a JSON-serialisable dict."""
    return {
        "n_bins": config.n_bins,
        "conv_layers": [
            {
                "out_channels": cl.out_channels,
                "kernel_size": cl.kernel_size,
                "pool_size": cl.pool_size,
            }
            for cl in config.conv_layers
        ],
        "dense_units": list(config.dense_units),
        "dropout_rate": config.dropout_rate,
        "dense_dropout_rates": list(config.dense_dropout_rates),
        "optimizer": config.optimizer,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "max_epochs": config.max_epochs,
        "early_stopping_patience": config.early_stopping_patience,
        "augment": config.augment,
        "seed": config.seed,
        "checkpoint_dir": config.checkpoint_dir,
    }


def _config_from_dict(d: dict) -> CnnTrainingConfig:
    """Reconstruct a :class:`CnnTrainingConfig` from a plain dict."""
    conv_layers = tuple(
        ConvLayerSpec(
            out_channels=int(cl["out_channels"]),
            kernel_size=int(cl["kernel_size"]),
            pool_size=int(cl["pool_size"]),
        )
        for cl in d.get("conv_layers", [])
    )
    dense_units = tuple(int(u) for u in d.get("dense_units", []))
    dropout_rate = float(d["dropout_rate"])
    dense_dropout_rates = tuple(
        float(rate)
        for rate in d.get(
            "dense_dropout_rates",
            [dropout_rate for _ in dense_units],
        )
    )
    return CnnTrainingConfig(
        n_bins=int(d["n_bins"]),
        conv_layers=conv_layers,
        dense_units=dense_units,
        dropout_rate=dropout_rate,
        dense_dropout_rates=dense_dropout_rates,
        optimizer=str(d["optimizer"]),
        learning_rate=float(d["learning_rate"]),
        batch_size=int(d["batch_size"]),
        max_epochs=int(d["max_epochs"]),
        early_stopping_patience=int(d["early_stopping_patience"]),
        augment=bool(d["augment"]),
        seed=int(d["seed"]),
        checkpoint_dir=str(d["checkpoint_dir"]),
    )


# ---------------------------------------------------------------------------
# Public I/O
# ---------------------------------------------------------------------------


def save_config(config: CnnTrainingConfig, path: Path) -> Path:
    """Serialise *config* to *path* as JSON (atomic write).

    Args:
        config: Config to serialise.
        path: Destination file path.

    Returns:
        The resolved destination path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _config_to_dict(config)
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


def load_config(path: Path) -> CnnTrainingConfig:
    """Load a :class:`CnnTrainingConfig` from a JSON file.

    Args:
        path: Path to the JSON config file.

    Returns:
        Reconstructed :class:`CnnTrainingConfig`.
    """
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return _config_from_dict(d)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_config(config: CnnTrainingConfig) -> CnnConfigValidation:
    """Validate a :class:`CnnTrainingConfig` and return error details.

    Checks:
    - ``n_bins > 0``
    - ``dropout_rate`` in [0, 1]
    - one valid dense dropout rate per dense layer
    - ``learning_rate > 0``
    - ``batch_size >= 1``
    - ``optimizer`` in {"adam", "sgd"}
    - Each conv layer: ``kernel_size`` odd, ``pool_size >= 1``,
      ``out_channels >= 1``
    - ``max_epochs >= 1``
    - ``early_stopping_patience >= 1``

    Args:
        config: Config to validate.

    Returns:
        :class:`CnnConfigValidation` with ``ok=True`` if all checks pass.
    """
    errors: list[str] = []

    if config.n_bins <= 0:
        errors.append(f"n_bins must be > 0, got {config.n_bins}")
    if not (0.0 <= config.dropout_rate <= 1.0):
        errors.append(
            f"dropout_rate must be in [0, 1], got {config.dropout_rate}"
        )
    if len(config.dense_dropout_rates) != len(config.dense_units):
        errors.append(
            "dense_dropout_rates must contain one value per dense layer, "
            f"got {len(config.dense_dropout_rates)} for {len(config.dense_units)} layers"
        )
    for index, rate in enumerate(config.dense_dropout_rates):
        if not 0.0 <= rate < 1.0:
            errors.append(
                f"dense_dropout_rates[{index}] must be in [0, 1), got {rate}"
            )
    if config.learning_rate <= 0.0:
        errors.append(
            f"learning_rate must be > 0, got {config.learning_rate}"
        )
    if config.batch_size < 1:
        errors.append(f"batch_size must be >= 1, got {config.batch_size}")
    if config.optimizer not in {"adam", "sgd"}:
        errors.append(
            f"optimizer must be 'adam' or 'sgd', got '{config.optimizer}'"
        )
    if config.max_epochs < 1:
        errors.append(f"max_epochs must be >= 1, got {config.max_epochs}")
    if config.early_stopping_patience < 1:
        errors.append(
            f"early_stopping_patience must be >= 1, "
            f"got {config.early_stopping_patience}"
        )
    for i, cl in enumerate(config.conv_layers):
        if cl.kernel_size % 2 == 0:
            errors.append(
                f"conv_layers[{i}].kernel_size must be odd, "
                f"got {cl.kernel_size}"
            )
        if cl.pool_size < 1:
            errors.append(
                f"conv_layers[{i}].pool_size must be >= 1, "
                f"got {cl.pool_size}"
            )
        if cl.out_channels < 1:
            errors.append(
                f"conv_layers[{i}].out_channels must be >= 1, "
                f"got {cl.out_channels}"
            )

    return CnnConfigValidation(ok=len(errors) == 0, errors=tuple(errors))


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------


def format_config(config: CnnTrainingConfig) -> str:
    """Format *config* as a human-readable Markdown block.

    Args:
        config: Config to format.

    Returns:
        Markdown string summarising the config.
    """
    conv_strs = ", ".join(
        f"{cl.out_channels}ch/k{cl.kernel_size}/p{cl.pool_size}"
        for cl in config.conv_layers
    )
    dense_strs = ", ".join(str(u) for u in config.dense_units)
    lines = [
        "## CNN Training Config",
        "",
        f"- n_bins: {config.n_bins}",
        f"- conv_layers: [{conv_strs}]",
        f"- dense_units: [{dense_strs}]",
        f"- dense_dropout_rates: {list(config.dense_dropout_rates)}",
        f"- optimizer: {config.optimizer}  lr={config.learning_rate}",
        f"- batch_size: {config.batch_size}",
        f"- max_epochs: {config.max_epochs}  patience={config.early_stopping_patience}",
        f"- augment: {config.augment}",
        f"- seed: {config.seed}",
        f"- checkpoint_dir: {config.checkpoint_dir}",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="cnn_training_config",
        description="Show, validate, or save the default CNN training config.",
    )
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("show", help="Print the default config")
    val_p = sub.add_parser("validate", help="Validate a config JSON file")
    val_p.add_argument("path", type=Path, metavar="CONFIG_JSON")
    save_p = sub.add_parser("save", help="Save the default config to a file")
    save_p.add_argument("path", type=Path, metavar="CONFIG_JSON")
    args = parser.parse_args(argv)

    if args.cmd == "show" or args.cmd is None:
        cfg = default_config()
        print(format_config(cfg))
        return 0
    if args.cmd == "validate":
        cfg = load_config(args.path)
        result = validate_config(cfg)
        if result.ok:
            print("Config is valid.")
        else:
            for err in result.errors:
                print(f"ERROR: {err}")
        return 0 if result.ok else 1
    if args.cmd == "save":
        cfg = default_config()
        out = save_config(cfg, args.path)
        print(f"Config saved → {out}")
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(_cli())
