"""Map raw FITS header keys to canonical pipeline field names.

Translates raw FITS header key-value pairs (from TESS SPOC or Kepler/K2
pipelines) to canonical field names used throughout this pipeline.
Distinct from ``fits_header_extractor`` (reads raw header values) and
``fits_lightcurve_exporter`` (writes FITS files).

Public API
----------
KeywordMapping(fits_key, canonical_name, dtype, unit, description)
KeywordMapResult(mission, n_found, n_missing, n_unrecognised,
                 mapped_values, unrecognised_keys, flag)
TESS_MAPPINGS, KEPLER_MAPPINGS: list[KeywordMapping]
map_fits_keywords(header_dict, *, mission) -> KeywordMapResult
format_keyword_map_result(result) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KeywordMapping:
    fits_key: str         # raw FITS keyword
    canonical_name: str   # pipeline field name
    dtype: str            # "float" | "int" | "str"
    unit: str             # e.g. "days", "K", "Rsun"
    description: str


@dataclass(frozen=True)
class KeywordMapResult:
    mission: str
    n_found: int
    n_missing: int        # canonical fields with no FITS key present
    n_unrecognised: int   # FITS keys not in the mapping table
    mapped_values: dict   # canonical_name → value
    unrecognised_keys: tuple[str, ...]
    flag: str  # "OK" | "PARTIAL" | "INVALID"


# ------------------------------------------------------------------
# TESS SPOC keyword mappings
# ------------------------------------------------------------------
TESS_MAPPINGS: list[KeywordMapping] = [
    KeywordMapping("TICID",    "tic_id",              "int",   "",       "TIC identifier"),
    KeywordMapping("SECTOR",   "sector",              "int",   "",       "TESS sector"),
    KeywordMapping("CAMERA",   "camera",              "int",   "",       "TESS camera number"),
    KeywordMapping("CCD",      "ccd",                 "int",   "",       "TESS CCD number"),
    KeywordMapping("TESSMAG",  "tmag",                "float", "mag",    "TESS magnitude"),
    KeywordMapping("TEFF",     "stellar_teff_k",      "float", "K",      "Effective temperature"),
    KeywordMapping("LOGG",     "stellar_logg",        "float", "dex",    "Surface gravity log g"),
    KeywordMapping("RADIUS",   "stellar_radius_rsun", "float", "Rsun",   "Stellar radius"),
    KeywordMapping("MASS",     "stellar_mass_msun",   "float", "Msun",   "Stellar mass"),
    KeywordMapping("RA_OBJ",   "ra_deg",              "float", "deg",    "Target RA (J2000)"),
    KeywordMapping("DEC_OBJ",  "dec_deg",             "float", "deg",    "Target Dec (J2000)"),
    KeywordMapping("CROWDSAP", "crowdsap",      "float", "",     "Crowding metric (CROWDSAP)"),
    KeywordMapping("FLFRCSAP", "flux_fraction", "float", "",     "Flux fraction in aperture"),
    KeywordMapping("BJDREFI",  "bjd_ref_int",   "int",   "days", "BJD reference (integer)"),
    KeywordMapping("BJDREFF",  "bjd_ref_frac",  "float", "days", "BJD reference (fraction)"),
    KeywordMapping("CDPP0_5",  "cdpp_0_5h",     "float", "ppm",  "CDPP at 0.5 h"),
    KeywordMapping("CDPP1_0",  "cdpp_1_0h",     "float", "ppm",  "CDPP at 1.0 h"),
    KeywordMapping("CDPP2_0",  "cdpp_2_0h",     "float", "ppm",  "CDPP at 2.0 h"),
]

# ------------------------------------------------------------------
# Kepler / K2 keyword mappings
# ------------------------------------------------------------------
KEPLER_MAPPINGS: list[KeywordMapping] = [
    KeywordMapping("KEPLERID", "kepler_id",           "int",   "",    "Kepler Input Catalog ID"),
    KeywordMapping("QUARTER",  "quarter",             "int",   "",    "Kepler quarter"),
    KeywordMapping("SEASON",   "season",              "int",   "",    "Kepler season"),
    KeywordMapping("CHANNEL",  "channel",             "int",   "",    "Kepler channel"),
    KeywordMapping("KEPMAG",   "kepmag",              "float", "mag", "Kepler magnitude"),
    KeywordMapping("TEFF",     "stellar_teff_k",      "float", "K",   "Effective temperature"),
    KeywordMapping("LOGG",     "stellar_logg",        "float", "dex", "Surface gravity log g"),
    KeywordMapping("RADIUS",   "stellar_radius_rsun", "float", "Rsun", "Stellar radius"),
    KeywordMapping("RA_OBJ",   "ra_deg",              "float", "deg", "Target RA (J2000)"),
    KeywordMapping("DEC_OBJ",  "dec_deg",             "float", "deg", "Target Dec (J2000)"),
    KeywordMapping("CROWDSAP", "crowdsap",            "float", "",    "Crowding metric"),
    KeywordMapping("BJDREFI",  "bjd_ref_int",   "int",   "days", "BJD reference (integer)"),
    KeywordMapping("BJDREFF",  "bjd_ref_frac",  "float", "days", "BJD reference (fraction)"),
    KeywordMapping("CDPP3_0",  "cdpp_3_0h",     "float", "ppm",  "CDPP at 3.0 h"),
    KeywordMapping("CDPP6_0",  "cdpp_6_0h",     "float", "ppm",  "CDPP at 6.0 h"),
]

_MISSION_MAPS: dict[str, list[KeywordMapping]] = {
    "TESS": TESS_MAPPINGS,
    "Kepler": KEPLER_MAPPINGS,
    "K2": KEPLER_MAPPINGS,
}


def map_fits_keywords(
    header_dict: dict,
    *,
    mission: str = "TESS",
) -> KeywordMapResult:
    """Translate raw FITS header dict to canonical pipeline field names.

    Args:
        header_dict: Dict of raw FITS keyword → value pairs.
        mission: ``"TESS"``, ``"Kepler"``, or ``"K2"``.

    Returns:
        :class:`KeywordMapResult`.
    """
    if not isinstance(header_dict, dict):
        return KeywordMapResult(mission, 0, 0, 0, {}, (), "INVALID")

    mappings = _MISSION_MAPS.get(mission)
    if mappings is None:
        return KeywordMapResult(mission, 0, 0, 0, {}, (), "INVALID")

    known_fits_keys = {m.fits_key for m in mappings}
    all_fits_keys = set(header_dict.keys())
    unrecognised = tuple(sorted(all_fits_keys - known_fits_keys))

    mapped: dict = {}
    n_found = 0
    n_missing = 0

    for m in mappings:
        raw = header_dict.get(m.fits_key)
        if raw is None:
            n_missing += 1
            continue
        # Type coerce
        try:
            if m.dtype == "int":
                mapped[m.canonical_name] = int(raw)
            elif m.dtype == "float":
                mapped[m.canonical_name] = float(raw)
            else:
                mapped[m.canonical_name] = str(raw)
            n_found += 1
        except (ValueError, TypeError):
            mapped[m.canonical_name] = raw
            n_found += 1

    # Special: compute bjd_ref = bjd_ref_int + bjd_ref_frac
    if "bjd_ref_int" in mapped and "bjd_ref_frac" in mapped:
        mapped["bjd_ref"] = mapped["bjd_ref_int"] + mapped["bjd_ref_frac"]

    flag = "OK" if n_found > 0 else ("PARTIAL" if n_missing == len(mappings) else "OK")
    if n_found == 0:
        flag = "PARTIAL"

    return KeywordMapResult(
        mission=mission,
        n_found=n_found,
        n_missing=n_missing,
        n_unrecognised=len(unrecognised),
        mapped_values=mapped,
        unrecognised_keys=unrecognised,
        flag=flag,
    )


def format_keyword_map_result(result: KeywordMapResult) -> str:
    """Format keyword mapping result as Markdown."""
    lines = [
        "## FITS Keyword Mapper",
        "",
        f"- Mission: {result.mission}",
        f"- Keys found: {result.n_found}",
        f"- Keys missing: {result.n_missing}",
        f"- Unrecognised FITS keys: {result.n_unrecognised}",
        f"- **Flag: {result.flag}**",
    ]
    if result.mapped_values:
        lines += ["", "### Mapped Values", ""]
        for k, v in list(result.mapped_values.items())[:10]:
            lines.append(f"- `{k}` = {v}")
        if len(result.mapped_values) > 10:
            lines.append(f"- … and {len(result.mapped_values) - 10} more")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog="fits_keyword_mapper",
        description="Map FITS header keys to canonical pipeline field names.",
    )
    parser.add_argument("--header", type=str, default=None, help="JSON header dict")
    parser.add_argument("--mission", type=str, default="TESS")
    args = parser.parse_args(argv)

    header = json.loads(args.header) if args.header else {}
    result = map_fits_keywords(header, mission=args.mission)
    print(format_keyword_map_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
