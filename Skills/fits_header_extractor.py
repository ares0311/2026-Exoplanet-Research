"""Extract stellar parameters from TESS SPOC FITS file headers.

Reads TESS SPOC FITS header keys and converts them into a typed dataclass
ready to pass as keyword arguments to ``vet_signal()``.  The
``extract_from_header`` function works with a plain dict so no FITS I/O is
required in tests.

Public API
----------
FITSStellarParams   (dataclass)
    .to_vet_kwargs() -> dict
extract_from_header(header) -> FITSStellarParams
extract_stellar_params(fits_path, *, hdu_index) -> FITSStellarParams
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

# FITS key → dataclass field mapping
# CROWDSAP is the fraction of flux from the target; contamination = 1 - CROWDSAP.
_KEY_MAP: dict[str, str] = {
    "TICID":    "tic_id",
    "RADIUS":   "stellar_radius_rsun",
    "MASS":     "stellar_mass_msun",
    "TEFF":     "stellar_teff_k",
    "LOGG":     "stellar_logg",
    "SECTOR":   "sector",
    # CROWDSAP is handled separately because it requires a transformation.
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class FITSStellarParams:
    tic_id: int | None
    stellar_radius_rsun: float | None
    stellar_mass_msun: float | None
    stellar_teff_k: float | None
    stellar_logg: float | None
    contamination_ratio: float | None
    sector: int | None

    def to_vet_kwargs(self) -> dict[str, Any]:
        """Return non-None fields as a dict for ``**vet_signal()`` kwargs.

        Only the subset of fields that ``vet_signal`` accepts is included.
        """
        mapping = {
            "stellar_radius_rsun": self.stellar_radius_rsun,
            "stellar_mass_msun":   self.stellar_mass_msun,
            "stellar_teff_k":      self.stellar_teff_k,
            "contamination_ratio": self.contamination_ratio,
        }
        return {k: v for k, v in mapping.items() if v is not None}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _safe_float(val: Any) -> float | None:
    """Convert a header value to float; return None on failure."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _safe_int(val: Any) -> int | None:
    """Convert a header value to int; return None on failure."""
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


def extract_from_header(header: dict[str, Any]) -> FITSStellarParams:
    """Extract stellar parameters from a FITS header represented as a plain dict.

    Missing keys and non-numeric values silently become ``None``.

    Args:
        header: FITS header as a plain Python dict (keyword → value).

    Returns:
        Populated :class:`FITSStellarParams` instance.
    """
    tic_id             = _safe_int(header.get("TICID"))
    stellar_radius     = _safe_float(header.get("RADIUS"))
    stellar_mass       = _safe_float(header.get("MASS"))
    stellar_teff       = _safe_float(header.get("TEFF"))
    stellar_logg       = _safe_float(header.get("LOGG"))
    sector             = _safe_int(header.get("SECTOR"))

    crowdsap = _safe_float(header.get("CROWDSAP"))
    contamination_ratio = (1.0 - crowdsap) if crowdsap is not None else None

    return FITSStellarParams(
        tic_id=tic_id,
        stellar_radius_rsun=stellar_radius,
        stellar_mass_msun=stellar_mass,
        stellar_teff_k=stellar_teff,
        stellar_logg=stellar_logg,
        contamination_ratio=contamination_ratio,
        sector=sector,
    )


def extract_stellar_params(
    fits_path: Path | str,
    *,
    hdu_index: int = 0,
) -> FITSStellarParams:
    """Load a FITS file and extract stellar parameters from one HDU header.

    Args:
        fits_path: Path to the FITS file.
        hdu_index: Index of the HDU whose header to read (default 0 = primary).

    Returns:
        Populated :class:`FITSStellarParams` instance.

    Raises:
        ImportError: If ``astropy`` is not installed.
    """
    try:
        import astropy.io.fits as fits  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "astropy is required for FITS I/O.  Install it with: pip install astropy"
        ) from exc

    with fits.open(Path(fits_path)) as hdul:
        header = dict(hdul[hdu_index].header)

    return extract_from_header(header)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse  # noqa: PLC0415
    import json  # noqa: PLC0415

    parser = argparse.ArgumentParser(
        prog="fits_header_extractor",
        description="Extract stellar parameters from a TESS SPOC FITS file header.",
    )
    parser.add_argument("fits_path", type=Path, metavar="FITS_FILE")
    parser.add_argument(
        "--hdu",
        type=int,
        default=0,
        metavar="INDEX",
        help="HDU index to read (default: 0).",
    )
    args = parser.parse_args(argv)

    import dataclasses  # noqa: PLC0415
    params = extract_stellar_params(args.fits_path, hdu_index=args.hdu)
    print(json.dumps(dataclasses.asdict(params), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
