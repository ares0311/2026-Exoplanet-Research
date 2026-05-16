"""Generate a self-contained Jupyter notebook pre-filled for a TIC ID pipeline run.

The generated notebook runs the full pipeline (Fetch → Clean → Search → Vet →
Score → Classify) with target parameters already inserted.  It is a
reproducibility artifact — a citizen scientist can share it as a standalone
analysis record.

Public API
----------
generate_notebook(tic_id, *, mission, stellar_radius_rsun, stellar_mass_msun,
                  min_snr, output_path) -> Path
_build_notebook_dict(tic_id, mission, stellar_radius_rsun, stellar_mass_msun,
                     min_snr) -> dict
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _build_notebook_dict(
    tic_id: int,
    mission: str,
    stellar_radius_rsun: float | None,
    stellar_mass_msun: float | None,
    min_snr: float,
) -> dict[str, Any]:
    """Return a valid nbformat 4.4 notebook dict with pipeline cells."""

    def _code(source: str) -> dict[str, Any]:
        return {
            "cell_type": "code",
            "source": source,
            "metadata": {},
            "outputs": [],
            "execution_count": None,
        }

    def _md(source: str) -> dict[str, Any]:
        return {"cell_type": "markdown", "source": source, "metadata": {}}

    radius_line = (
        f"stellar_radius_rsun = {stellar_radius_rsun}"
        if stellar_radius_rsun is not None
        else "stellar_radius_rsun = None"
    )
    mass_line = (
        f"stellar_mass_msun = {stellar_mass_msun}"
        if stellar_mass_msun is not None
        else "stellar_mass_msun = None"
    )

    cells = [
        _md(f"# Exoplanet Pipeline Analysis — TIC {tic_id}\n\nMission: {mission}"),
        _code("from exo_toolkit.cli import run_pipeline\nimport json"),
        _code(
            f"tic_id = {tic_id}\n"
            f"mission = \"{mission}\"\n"
            f"min_snr = {min_snr}\n"
            f"{radius_line}\n"
            f"{mass_line}"
        ),
        _md("## Run Pipeline"),
        _code(
            "results = run_pipeline(\n"
            "    f\"TIC {tic_id}\",\n"
            "    mission,\n"
            "    min_snr=min_snr,\n"
            ")"
        ),
        _md("## Results"),
        _code("print(json.dumps(results, indent=2))"),
    ]

    return {
        "nbformat": 4,
        "nbformat_minor": 4,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


def generate_notebook(
    tic_id: int,
    *,
    mission: str = "TESS",
    stellar_radius_rsun: float | None = None,
    stellar_mass_msun: float | None = None,
    min_snr: float = 5.0,
    output_path: Path | None = None,
) -> Path:
    """Generate and write a pipeline notebook; return the written path.

    Args:
        tic_id: TESS Input Catalog identifier.
        mission: ``"TESS"``, ``"Kepler"``, or ``"K2"``.
        stellar_radius_rsun: Optional host-star radius in solar radii.
        stellar_mass_msun: Optional host-star mass in solar masses.
        min_snr: Minimum BLS SNR threshold forwarded to ``run_pipeline()``.
        output_path: Destination ``.ipynb`` path; defaults to
            ``notebooks/TIC_{tic_id}.ipynb`` relative to the project root.

    Returns:
        Path of the written notebook file.
    """
    if output_path is None:
        output_path = Path("notebooks") / f"TIC_{tic_id}.ipynb"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nb = _build_notebook_dict(tic_id, mission, stellar_radius_rsun, stellar_mass_msun, min_snr)
    output_path.write_text(json.dumps(nb, indent=2))
    return output_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="notebook_generator",
        description="Generate a pre-filled Jupyter notebook for a TIC ID pipeline run.",
    )
    parser.add_argument("tic_id", type=int, metavar="TIC_ID")
    parser.add_argument("--mission", default="TESS")
    parser.add_argument("--output", type=Path, default=None, metavar="FILE")
    parser.add_argument("--stellar-radius", type=float, default=None, metavar="RSUN")
    parser.add_argument("--stellar-mass", type=float, default=None, metavar="MSUN")
    parser.add_argument("--min-snr", type=float, default=5.0, metavar="SNR")
    args = parser.parse_args(argv)

    path = generate_notebook(
        args.tic_id,
        mission=args.mission,
        stellar_radius_rsun=args.stellar_radius,
        stellar_mass_msun=args.stellar_mass,
        min_snr=args.min_snr,
        output_path=args.output,
    )
    print(f"Notebook written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
