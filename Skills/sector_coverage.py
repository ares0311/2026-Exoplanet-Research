"""Query MAST for the TESS sector coverage of one or more targets.

Returns which TESS sectors are available for a given TIC ID without
downloading the actual photometry.  Useful for prioritizing targets (more
sectors = better transit-multiplicity statistics) and for auditing what
data the pipeline will see.

Public API
----------
get_sector_coverage(target_id, *, pipeline) -> SectorCoverage
format_coverage_table(coverages)            -> str

CLI usage
---------
    python Skills/sector_coverage.py TIC 150428135
    python Skills/sector_coverage.py 150428135 267522065 --pipeline QLP
    python Skills/sector_coverage.py 150428135 --json
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class SectorCoverage:
    """Sector availability metadata for one TESS target."""

    target_id: str
    pipeline: str
    sectors: list[int] = field(default_factory=list)
    n_sectors: int = 0
    cadence_labels: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_id": self.target_id,
            "pipeline": self.pipeline,
            "sectors": self.sectors,
            "n_sectors": self.n_sectors,
            "cadence_labels": self.cadence_labels,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_sector_coverage(
    target_id: str,
    *,
    pipeline: str = "SPOC",
    search_fn: Any = None,
) -> SectorCoverage:
    """Query MAST for available TESS sectors for *target_id*.

    Args:
        target_id: Target identifier, e.g. ``"TIC 150428135"`` or ``"150428135"``.
        pipeline: Pipeline author — ``"SPOC"`` (default) or ``"QLP"``.
        search_fn: Override ``lightkurve.search_lightcurve`` (for tests).

    Returns:
        SectorCoverage with sectors list, n_sectors, and cadence labels.
    """
    if not target_id.upper().startswith("TIC"):
        target_id = f"TIC {target_id}"

    if search_fn is None:
        try:
            import lightkurve as lk  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "lightkurve is required. Install with: pip install lightkurve"
            ) from exc
        search_fn = lk.search_lightcurve

    result = search_fn(target_id, mission="TESS", author=pipeline)

    sectors: list[int] = []
    cadence_labels: list[str] = []

    if len(result) == 0:
        return SectorCoverage(
            target_id=target_id,
            pipeline=pipeline,
            sectors=[],
            n_sectors=0,
            cadence_labels=[],
        )

    table = result.table
    sector_col = "sequence_number" if "sequence_number" in table.colnames else "sector"
    exptime_col = "exptime" if "exptime" in table.colnames else "t_exptime"

    for row in table:
        try:
            sec = int(row[sector_col])
        except (KeyError, TypeError, ValueError):
            continue
        if sec not in sectors:
            sectors.append(sec)
            try:
                exptime = float(row[exptime_col])
                if exptime <= 120:
                    cadence_labels.append("2min")
                elif exptime <= 600:
                    cadence_labels.append("10min")
                else:
                    cadence_labels.append("30min")
            except (KeyError, TypeError, ValueError):
                cadence_labels.append("unknown")

    sectors.sort()
    return SectorCoverage(
        target_id=target_id,
        pipeline=pipeline,
        sectors=sectors,
        n_sectors=len(sectors),
        cadence_labels=cadence_labels[:len(sectors)],
    )


def format_coverage_table(coverages: list[SectorCoverage]) -> str:
    """Return a plain-text summary table for a list of SectorCoverage objects."""
    if not coverages:
        return "(no targets)"
    lines = [f"{'Target':30}  {'Pipeline':8}  {'N':>3}  Sectors"]
    lines.append("-" * 70)
    for cov in coverages:
        sectors_str = ", ".join(str(s) for s in cov.sectors) or "(none)"
        lines.append(f"{cov.target_id:30}  {cov.pipeline:8}  {cov.n_sectors:>3}  {sectors_str}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        prog="sector_coverage",
        description="Show TESS sector coverage for one or more targets.",
    )
    parser.add_argument(
        "targets",
        nargs="+",
        help="TIC IDs or 'TIC XXXXXXX' identifiers.",
    )
    parser.add_argument("--pipeline", default="SPOC", choices=["SPOC", "QLP", "TGLC"])
    parser.add_argument("--json", action="store_true", help="Output JSON instead of table.")
    args = parser.parse_args(argv)

    coverages: list[SectorCoverage] = []
    for raw in args.targets:
        try:
            cov = get_sector_coverage(raw, pipeline=args.pipeline)
            coverages.append(cov)
        except Exception as exc:  # noqa: BLE001
            print(f"Error querying {raw}: {exc}", file=sys.stderr)

    if args.json:
        print(json.dumps([c.to_dict() for c in coverages], indent=2))
    else:
        print(format_coverage_table(coverages))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
