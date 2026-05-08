"""Command-line interface for exo-toolkit.

Entry point: ``exo scan <TIC-ID>``

Runs the full pipeline (fetch → clean → search → vet → score → classify)
on a single target and prints a Rich-formatted candidate report.

Usage examples
--------------
    exo scan "TIC 150428135"
    exo scan "TIC 150428135" --mission TESS --min-snr 7.0
    exo scan "TIC 150428135" --output results.json
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

from exo_toolkit.clean import clean_lightcurve
from exo_toolkit.fetch import fetch_lightcurve
from exo_toolkit.pathway import classify_submission_pathway
from exo_toolkit.schemas import CandidateSignal, Mission
from exo_toolkit.scoring import score_candidate
from exo_toolkit.search import search_lightcurve
from exo_toolkit.vet import vet_signal

app = typer.Typer(
    name="exo",
    help="Exoplanet transit candidate detection and scoring toolkit.",
    add_completion=False,
)


# ---------------------------------------------------------------------------
# Pipeline orchestration (separated for testability)
# ---------------------------------------------------------------------------


def run_pipeline(
    target_id: str,
    mission: Mission,
    *,
    min_snr: float = 5.0,
    max_peaks: int = 5,
    fetch_fn: Any = None,
    clean_fn: Any = None,
) -> list[dict[str, Any]]:
    """Run the full pipeline on one target and return serialisable results.

    Args:
        target_id: Target identifier, e.g. ``"TIC 150428135"``.
        mission: ``"TESS"``, ``"Kepler"``, or ``"K2"``.
        min_snr: Minimum BLS SNR threshold for candidate signals.
        max_peaks: Maximum number of signals to return from the BLS search.
        fetch_fn: Optional callable replacing ``fetch_lightcurve`` (for tests).
        clean_fn: Optional callable replacing ``clean_lightcurve`` (for tests).

    Returns:
        List of dicts, one per candidate signal, suitable for JSON output.
    """
    _fetch = fetch_fn if fetch_fn is not None else fetch_lightcurve
    _clean = clean_fn if clean_fn is not None else clean_lightcurve

    fetch_result = _fetch(target_id, mission)
    clean_result = _clean(fetch_result.light_curve)

    signals: list[CandidateSignal] = search_lightcurve(
        clean_result.light_curve,
        target_id=target_id,
        mission=mission,
        min_snr=min_snr,
        max_peaks=max_peaks,
    )

    if not signals:
        return []

    rows: list[dict[str, Any]] = []
    for signal in signals:
        vet_result = vet_signal(signal, clean_result.light_curve)
        posterior, scores = score_candidate(signal, vet_result.features)
        pathway = classify_submission_pathway(
            signal, vet_result.features, posterior, scores
        )
        rows.append(
            {
                "candidate_id": signal.candidate_id,
                "target_id": signal.target_id,
                "mission": signal.mission,
                "period_days": signal.period_days,
                "epoch_bjd": signal.epoch_bjd,
                "duration_hours": signal.duration_hours,
                "depth_ppm": signal.depth_ppm,
                "transit_count": signal.transit_count,
                "snr": signal.snr,
                "posterior": {
                    "planet_candidate": posterior.planet_candidate,
                    "eclipsing_binary": posterior.eclipsing_binary,
                    "background_eclipsing_binary": posterior.background_eclipsing_binary,
                    "stellar_variability": posterior.stellar_variability,
                    "instrumental_artifact": posterior.instrumental_artifact,
                    "known_object": posterior.known_object,
                },
                "scores": {
                    "false_positive_probability": scores.false_positive_probability,
                    "detection_confidence": scores.detection_confidence,
                    "novelty_score": scores.novelty_score,
                },
                "pathway": pathway,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _print_results(rows: list[dict[str, Any]], target_id: str) -> None:
    """Print pipeline results.  Key summary lines go via typer.echo so that
    CliRunner captures them; rich Tables are printed to a fresh Console."""
    if not rows:
        typer.echo(f"No transit candidates found above SNR threshold for {target_id}.")
        return

    typer.echo(f"{len(rows)} candidate signal(s) found for {target_id}")

    # Fresh Console so it binds to the current sys.stdout (important in tests).
    c = Console(highlight=False)
    for i, row in enumerate(rows, 1):
        table = Table(title=f"Signal {i}: {row['candidate_id']}", show_header=True)
        table.add_column("Property", style="dim")
        table.add_column("Value")

        table.add_row("Period", f"{row['period_days']:.4f} d")
        table.add_row("Depth", f"{row['depth_ppm']:.0f} ppm")
        table.add_row("Duration", f"{row['duration_hours']:.2f} h")
        table.add_row("Transits", str(row["transit_count"]))
        table.add_row("SNR", f"{row['snr']:.1f}")
        table.add_row("", "")
        table.add_row(
            "P(planet candidate)",
            f"{row['posterior']['planet_candidate']:.3f}",
        )
        table.add_row(
            "P(eclipsing binary)",
            f"{row['posterior']['eclipsing_binary']:.3f}",
        )
        table.add_row("FPP", f"{row['scores']['false_positive_probability']:.3f}")
        table.add_row(
            "Detection confidence",
            f"{row['scores']['detection_confidence']:.3f}",
        )
        table.add_row("Pathway", row["pathway"])
        c.print(table)
        typer.echo(f"  Pathway: {row['pathway']}")


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


@app.command()
def scan(
    target_id: str = typer.Argument(..., help='Target identifier, e.g. "TIC 150428135"'),
    mission: str = typer.Option("TESS", help="Mission: TESS, Kepler, or K2"),
    min_snr: float = typer.Option(5.0, "--min-snr", help="Minimum BLS SNR threshold"),
    max_peaks: int = typer.Option(5, "--max-peaks", help="Maximum signals to search for"),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write JSON results to this file"
    ),
) -> None:
    """Scan a target for exoplanet transit candidates.

    Runs the full pipeline: fetch → clean → search → vet → score → classify.
    """
    valid_missions = ("TESS", "Kepler", "K2")
    if mission not in valid_missions:
        typer.echo(f"Invalid mission '{mission}'. Choose from: {valid_missions}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Scanning {target_id} ({mission}) ...")

    try:
        rows = run_pipeline(
            target_id,
            mission,  # type: ignore[arg-type]
            min_snr=min_snr,
            max_peaks=max_peaks,
        )
    except Exception as exc:  # noqa: BLE001
        typer.echo(f"Pipeline error: {exc}", err=True)
        raise typer.Exit(code=2) from exc

    _print_results(rows, target_id)

    if output is not None:
        output.write_text(json.dumps(rows, indent=2))
        typer.echo(f"Results written to {output}")


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------


def main() -> None:
    app()


if __name__ == "__main__":
    main()
