"""CNN model architecture configuration for the 1D transit classifier.

Handles architecture-level parameters only (layer counts, filters, kernel size,
dropout, dense units).  Optimizer settings and training hyper-parameters live in
a separate ``cnn_training_config`` module.

Public API
----------
CnnModelConfig(n_bins, n_filters, kernel_size, n_layers, dropout, dense_units, flag)
ModelConfigValidation(ok, errors, flag)
default_model_config() -> CnnModelConfig
validate_model_config(config) -> ModelConfigValidation
model_config_to_dict(config) -> dict
model_config_from_dict(d) -> CnnModelConfig
format_model_config(config) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CnnModelConfig:
    n_bins: int                    # input length (default 201)
    n_filters: tuple[int, ...]     # filters per conv layer e.g. (16, 32, 64)
    kernel_size: int               # must be odd, e.g. 5
    n_layers: int                  # number of conv layers (== len(n_filters))
    dropout: float                 # e.g. 0.3, in [0, 1)
    dense_units: tuple[int, ...]   # dense layers before output, e.g. (64,)
    flag: str                      # always "OK" (pattern compliance)


@dataclass(frozen=True)
class ModelConfigValidation:
    ok: bool
    errors: tuple[str, ...]
    flag: str   # "OK" | "INVALID"


def default_model_config() -> CnnModelConfig:
    """Return a sensible default 1D CNN architecture config."""
    return CnnModelConfig(
        n_bins=201,
        n_filters=(16, 32, 64),
        kernel_size=5,
        n_layers=3,
        dropout=0.3,
        dense_units=(64,),
        flag="OK",
    )


def validate_model_config(config: CnnModelConfig) -> ModelConfigValidation:
    """Validate a :class:`CnnModelConfig` for correctness.

    Checks:
        - ``kernel_size`` must be odd and ≥ 1.
        - ``dropout`` must be in [0, 1).
        - ``n_bins`` must be > 0.
        - ``n_layers`` must be > 0.
        - ``len(n_filters)`` must equal ``n_layers``.

    Returns:
        :class:`ModelConfigValidation`.
    """
    errors: list[str] = []

    if config.n_bins <= 0:
        errors.append(f"n_bins must be > 0, got {config.n_bins}")
    if config.n_layers <= 0:
        errors.append(f"n_layers must be > 0, got {config.n_layers}")
    if config.kernel_size < 1 or config.kernel_size % 2 == 0:
        errors.append(f"kernel_size must be odd and ≥ 1, got {config.kernel_size}")
    if not (0.0 <= config.dropout < 1.0):
        errors.append(f"dropout must be in [0, 1), got {config.dropout}")
    if len(config.n_filters) != config.n_layers:
        errors.append(
            f"len(n_filters)={len(config.n_filters)} must equal n_layers={config.n_layers}"
        )

    ok = len(errors) == 0
    return ModelConfigValidation(
        ok=ok,
        errors=tuple(errors),
        flag="OK" if ok else "INVALID",
    )


def model_config_to_dict(config: CnnModelConfig) -> dict:
    """Serialise a :class:`CnnModelConfig` to a plain dict (JSON-safe)."""
    return {
        "n_bins": config.n_bins,
        "n_filters": list(config.n_filters),
        "kernel_size": config.kernel_size,
        "n_layers": config.n_layers,
        "dropout": config.dropout,
        "dense_units": list(config.dense_units),
        "flag": config.flag,
    }


def model_config_from_dict(d: dict) -> CnnModelConfig:
    """Deserialise a :class:`CnnModelConfig` from a plain dict.

    Args:
        d: Dict as produced by :func:`model_config_to_dict`.

    Returns:
        :class:`CnnModelConfig`.
    """
    return CnnModelConfig(
        n_bins=int(d["n_bins"]),
        n_filters=tuple(int(x) for x in d["n_filters"]),
        kernel_size=int(d["kernel_size"]),
        n_layers=int(d["n_layers"]),
        dropout=float(d["dropout"]),
        dense_units=tuple(int(x) for x in d["dense_units"]),
        flag=str(d.get("flag", "OK")),
    )


def format_model_config(config: CnnModelConfig) -> str:
    """Format a :class:`CnnModelConfig` as a Markdown summary."""
    lines = [
        "## CNN Model Architecture Config",
        "",
        f"- **n_bins:** {config.n_bins}",
        f"- **n_layers:** {config.n_layers}",
        f"- **n_filters:** {list(config.n_filters)}",
        f"- **kernel_size:** {config.kernel_size}",
        f"- **dropout:** {config.dropout}",
        f"- **dense_units:** {list(config.dense_units)}",
        f"- **Flag:** {config.flag}",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="cnn_model_config",
        description="Show default CNN model architecture configuration.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="show",
        choices=["show"],
        help="Command to run (default: show).",
    )
    args = parser.parse_args(argv)

    if args.command == "show":
        config = default_model_config()
        print(format_model_config(config))
        val = validate_model_config(config)
        print(f"Validation: {val.flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
