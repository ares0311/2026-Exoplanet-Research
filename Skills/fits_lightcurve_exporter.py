"""Export a light curve to a minimal FITS binary table file.

Writes a primary HDU with a WCS-compatible header and a binary table
extension with TIME, FLUX, and optional FLUX_ERR / QUALITY / CENTROID columns.

The ``write_fn`` parameter is injectable so tests can run without astropy.

Public API
----------
FITSExportResult(output_path, n_cadences, columns_written, header_keys,
                 file_size_bytes, flag)
export_lightcurve_to_fits(time, flux, output_path, *, flux_err, quality,
                          centroid_col, centroid_row, extra_header,
                          tic_id, sector, time_format, overwrite,
                          write_fn) -> FITSExportResult
format_fits_export_result(result) -> str
"""
from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class FITSExportResult:
    output_path: str
    n_cadences: int
    columns_written: tuple[str, ...]
    header_keys: tuple[str, ...]
    file_size_bytes: int
    flag: str  # "OK" | "WRITE_ERROR" | "INVALID"


def _default_write_fn(
    output_path: str,
    header: dict,
    columns: list[dict],
    overwrite: bool = False,
) -> None:
    """Write using astropy.io.fits (real implementation)."""
    from astropy.io import fits  # type: ignore[import]

    primary_hdu = fits.PrimaryHDU()
    for k, v in header.items():
        primary_hdu.header[k] = v

    cols = []
    for col in columns:
        name = col["name"]
        data = col["data"]
        fmt = col.get("format", "D")
        cols.append(fits.Column(name=name, format=fmt, array=data))

    table_hdu = fits.BinTableHDU.from_columns(cols)
    hdul = fits.HDUList([primary_hdu, table_hdu])
    hdul.writeto(output_path, overwrite=overwrite)


def export_lightcurve_to_fits(
    time: list[float],
    flux: list[float],
    output_path: str,
    *,
    flux_err: list[float] | None = None,
    quality: list[int] | None = None,
    centroid_col: list[float] | None = None,
    centroid_row: list[float] | None = None,
    extra_header: dict[str, str | float | int] | None = None,
    tic_id: int = 0,
    sector: int = 0,
    time_format: str = "BJD",
    overwrite: bool = False,
    write_fn: Callable | None = None,
) -> FITSExportResult:
    """Export a light curve to a minimal FITS binary table.

    Args:
        time: Time array (BJD or other format).
        flux: Normalised flux array.
        output_path: Path to write the FITS file.
        flux_err: Optional flux uncertainty array.
        quality: Optional integer quality flag array.
        centroid_col: Optional centroid column (pixels).
        centroid_row: Optional centroid row (pixels).
        extra_header: Extra key-value pairs to add to the primary header.
        tic_id: TESS Input Catalog ID.
        sector: TESS sector number.
        time_format: Time format string stored in header.
        overwrite: Overwrite existing file.
        write_fn: Injectable write function for testing.

    Returns:
        :class:`FITSExportResult`.
    """
    n = len(time)
    if n == 0 or n != len(flux):
        return FITSExportResult(output_path, 0, (), (), 0, "INVALID")

    for _name, arr in [("flux_err", flux_err), ("quality", quality),
                       ("centroid_col", centroid_col), ("centroid_row", centroid_row)]:
        if arr is not None and len(arr) != n:
            return FITSExportResult(output_path, n, (), (), 0, "INVALID")

    header = {
        "TIC_ID": tic_id,
        "SECTOR": sector,
        "TIMEUNIT": time_format,
        "TELESCOP": "TESS",
        "CREATED": "exo-toolkit",
    }
    if extra_header:
        header.update(extra_header)

    columns: list[dict] = [
        {"name": "TIME", "data": list(time), "format": "D"},
        {"name": "FLUX", "data": list(flux), "format": "D"},
    ]
    cols_written = ["TIME", "FLUX"]

    if flux_err is not None:
        columns.append({"name": "FLUX_ERR", "data": list(flux_err), "format": "D"})
        cols_written.append("FLUX_ERR")
    if quality is not None:
        columns.append({"name": "QUALITY", "data": list(quality), "format": "J"})
        cols_written.append("QUALITY")
    if centroid_col is not None:
        columns.append({"name": "CENTROID_COL", "data": list(centroid_col), "format": "D"})
        cols_written.append("CENTROID_COL")
    if centroid_row is not None:
        columns.append({"name": "CENTROID_ROW", "data": list(centroid_row), "format": "D"})
        cols_written.append("CENTROID_ROW")

    fn = write_fn if write_fn is not None else _default_write_fn
    try:
        fn(output_path, header, columns, overwrite)
    except Exception:
        return FITSExportResult(
            output_path, n,
            tuple(cols_written), tuple(header.keys()),
            0, "WRITE_ERROR",
        )

    file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0

    return FITSExportResult(
        output_path=output_path,
        n_cadences=n,
        columns_written=tuple(cols_written),
        header_keys=tuple(header.keys()),
        file_size_bytes=file_size,
        flag="OK",
    )


def format_fits_export_result(result: FITSExportResult) -> str:
    """Format FITS export result as Markdown."""
    lines = [
        "## FITS Export",
        "",
        f"- Output: `{result.output_path}`",
        f"- Cadences: {result.n_cadences}",
        f"- Columns: {', '.join(result.columns_written)}",
        f"- File size: {result.file_size_bytes} bytes",
        f"- **Flag: {result.flag}**",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="fits_lightcurve_exporter",
        description="Export a light curve to FITS binary table format.",
    )
    parser.add_argument("output_path", type=str)
    parser.add_argument("--tic-id", type=int, default=0)
    parser.add_argument("--sector", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    result = export_lightcurve_to_fits(
        [], [], args.output_path,
        tic_id=args.tic_id, sector=args.sector, overwrite=args.overwrite,
    )
    print(format_fits_export_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
