"""Write AstroImageJ-compatible circular aperture region files.

Produces the plain-text ``.apertures`` format recognised by AstroImageJ
(AIJ) from a list of candidate sky positions (RA/Dec in degrees).  Each
entry becomes one circular aperture region with a configurable radius.

Public API
----------
RegionEntry(ra_deg, dec_deg, radius_arcsec, label, color)
AstroImageJRegionResult(n_entries, region_text, flag)
write_aij_region(entries, *, inner_annulus_arcsec,
                 outer_annulus_arcsec) -> AstroImageJRegionResult
format_aij_region_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RegionEntry:
    ra_deg: float
    dec_deg: float
    radius_arcsec: float
    label: str = ""
    color: str = "green"  # AIJ colour name


@dataclass(frozen=True)
class AstroImageJRegionResult:
    n_entries: int
    region_text: str   # full content of the .apertures file
    flag: str  # "OK" | "EMPTY" | "INVALID"


def write_aij_region(
    entries: list[RegionEntry],
    *,
    inner_annulus_arcsec: float = 30.0,
    outer_annulus_arcsec: float = 50.0,
) -> AstroImageJRegionResult:
    """Generate an AstroImageJ aperture region file string.

    The AIJ ``.apertures`` format is a plain-text file where each line
    describes one circular aperture:
    ``RA  DEC  AP_RADIUS  SKY_INNER  SKY_OUTER  LABEL``
    (all angular quantities in arcseconds for radius/annulus, RA/Dec in
    decimal degrees).

    Args:
        entries: List of :class:`RegionEntry` objects.
        inner_annulus_arcsec: Inner sky-annulus radius (arcsec).
        outer_annulus_arcsec: Outer sky-annulus radius (arcsec).

    Returns:
        :class:`AstroImageJRegionResult`.
    """
    if not isinstance(entries, list):
        return AstroImageJRegionResult(0, "", "INVALID")
    if not entries:
        return AstroImageJRegionResult(0, "", "EMPTY")
    if inner_annulus_arcsec >= outer_annulus_arcsec:
        return AstroImageJRegionResult(0, "", "INVALID")

    lines = [
        "# AstroImageJ aperture file",
        "# RA(deg)  Dec(deg)  Ap(arcsec)  SkyInner(arcsec)  SkyOuter(arcsec)  Label",
    ]
    for e in entries:
        label = e.label.replace(" ", "_") or "T"
        lines.append(
            f"{e.ra_deg:.6f}  {e.dec_deg:.6f}  {e.radius_arcsec:.1f}"
            f"  {inner_annulus_arcsec:.1f}  {outer_annulus_arcsec:.1f}  {label}"
        )

    region_text = "\n".join(lines) + "\n"
    return AstroImageJRegionResult(
        n_entries=len(entries),
        region_text=region_text,
        flag="OK",
    )


def format_aij_region_result(result: AstroImageJRegionResult) -> str:
    """Format AIJ region result as Markdown."""
    lines = [
        "## AstroImageJ Region Writer",
        "",
        f"- Aperture entries: {result.n_entries}",
        f"- Output length: {len(result.region_text)} chars",
        f"- **Flag: {result.flag}**",
    ]
    if result.flag == "OK" and result.n_entries > 0:
        preview = result.region_text.split("\n")[2] if result.n_entries > 0 else ""
        if preview:
            lines += ["", f"First entry: `{preview}`"]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="astroimagej_region_writer",
        description="Write AstroImageJ aperture region file from RA/Dec list.",
    )
    parser.add_argument("--ra", type=float, default=None)
    parser.add_argument("--dec", type=float, default=None)
    parser.add_argument("--radius", type=float, default=15.0)
    args = parser.parse_args(argv)

    entries = []
    if args.ra is not None and args.dec is not None:
        entries.append(RegionEntry(args.ra, args.dec, args.radius))
    result = write_aij_region(entries)
    print(format_aij_region_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
