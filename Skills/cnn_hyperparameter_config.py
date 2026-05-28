"""Hyperparameter grid configuration for CNN architecture search.

Defines the search space over learning rate, dropout, convolutional filters,
and fully-connected layer sizes. Generates all grid combinations for
systematic evaluation.

Public API
----------
HyperparamGrid(learning_rates, dropout_rates, conv_filters, fc_units,
               batch_sizes, n_epochs)
HyperparamCandidate(learning_rate, dropout_rate, conv_filters, fc_units,
                    batch_size, n_epochs, candidate_id)
default_grid() -> HyperparamGrid
generate_candidates(grid) -> list[HyperparamCandidate]
load_grid(path) -> HyperparamGrid
save_grid(grid, path) -> None
format_grid_summary(grid) -> str
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class HyperparamGrid:
    learning_rates: tuple[float, ...]
    dropout_rates: tuple[float, ...]
    conv_filters: tuple[int, ...]          # number of filters in conv layers
    fc_units: tuple[int, ...]              # units in fully-connected layer
    batch_sizes: tuple[int, ...]
    n_epochs: int                          # fixed for all candidates


@dataclass(frozen=True)
class HyperparamCandidate:
    learning_rate: float
    dropout_rate: float
    conv_filters: int
    fc_units: int
    batch_size: int
    n_epochs: int
    candidate_id: int   # sequential index in the grid


def default_grid() -> HyperparamGrid:
    """Return the recommended hyperparameter search grid for Tier 2 CNN."""
    return HyperparamGrid(
        learning_rates=(1e-4, 3e-4, 1e-3),
        dropout_rates=(0.3, 0.5),
        conv_filters=(16, 32),
        fc_units=(64, 128),
        batch_sizes=(32,),
        n_epochs=20,
    )


def generate_candidates(grid: HyperparamGrid) -> list[HyperparamCandidate]:
    """Enumerate all hyperparameter combinations from the grid.

    Args:
        grid: HyperparamGrid defining the search space.

    Returns:
        List of HyperparamCandidate, one per combination.
    """
    candidates = []
    idx = 0
    for lr in grid.learning_rates:
        for dr in grid.dropout_rates:
            for cf in grid.conv_filters:
                for fu in grid.fc_units:
                    for bs in grid.batch_sizes:
                        candidates.append(HyperparamCandidate(
                            learning_rate=lr,
                            dropout_rate=dr,
                            conv_filters=cf,
                            fc_units=fu,
                            batch_size=bs,
                            n_epochs=grid.n_epochs,
                            candidate_id=idx,
                        ))
                        idx += 1
    return candidates


def load_grid(path: Path) -> HyperparamGrid:
    """Load a HyperparamGrid from a JSON file.

    Args:
        path: Path to JSON config file.

    Returns:
        HyperparamGrid instance.
    """
    raw = json.loads(Path(path).read_text())
    return HyperparamGrid(
        learning_rates=tuple(float(x) for x in raw["learning_rates"]),
        dropout_rates=tuple(float(x) for x in raw["dropout_rates"]),
        conv_filters=tuple(int(x) for x in raw["conv_filters"]),
        fc_units=tuple(int(x) for x in raw["fc_units"]),
        batch_sizes=tuple(int(x) for x in raw.get("batch_sizes", [32])),
        n_epochs=int(raw.get("n_epochs", 20)),
    )


def save_grid(grid: HyperparamGrid, path: Path) -> None:
    """Save a HyperparamGrid to a JSON file (atomic write).

    Args:
        grid: HyperparamGrid to save.
        path: Destination path.
    """
    import contextlib
    import os
    import tempfile

    d = asdict(grid)
    d["learning_rates"] = list(grid.learning_rates)
    d["dropout_rates"] = list(grid.dropout_rates)
    d["conv_filters"] = list(grid.conv_filters)
    d["fc_units"] = list(grid.fc_units)
    d["batch_sizes"] = list(grid.batch_sizes)

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


def format_grid_summary(grid: HyperparamGrid) -> str:
    """Format a Markdown summary of the hyperparameter grid.

    Args:
        grid: HyperparamGrid to format.

    Returns:
        Markdown string.
    """
    candidates = generate_candidates(grid)
    lines = [
        "## CNN Hyperparameter Grid\n",
        f"Total combinations: {len(candidates)} | Epochs per run: {grid.n_epochs}\n",
        "",
        "| Parameter | Values |",
        "|---|---|",
        f"| Learning rate | {', '.join(str(v) for v in grid.learning_rates)} |",
        f"| Dropout rate | {', '.join(str(v) for v in grid.dropout_rates)} |",
        f"| Conv filters | {', '.join(str(v) for v in grid.conv_filters)} |",
        f"| FC units | {', '.join(str(v) for v in grid.fc_units)} |",
        f"| Batch size | {', '.join(str(v) for v in grid.batch_sizes)} |",
    ]
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Manage CNN hyperparameter grid config.")
    parser.add_argument("--config", help="Path to JSON grid config file.")
    parser.add_argument("--output", help="Write default grid to this path.")
    parser.add_argument("--list", action="store_true", dest="list_all",
                        help="List all candidate combinations.")
    args = parser.parse_args(argv)

    if args.output:
        save_grid(default_grid(), Path(args.output))
        print(f"Default grid written to {args.output}")
        return 0

    grid = load_grid(Path(args.config)) if args.config else default_grid()
    print(format_grid_summary(grid))

    if args.list_all:
        print("\n### Candidates\n")
        for c in generate_candidates(grid):
            print(
                f"  [{c.candidate_id:03d}] lr={c.learning_rate} dr={c.dropout_rate}"
                f" cf={c.conv_filters} fc={c.fc_units} bs={c.batch_size}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
