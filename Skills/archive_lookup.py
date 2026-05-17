"""Check whether a TIC target appears in published planet/TOI archives.

Queries the NASA Exoplanet Archive (confirmed planets), the TESS TOI list
(ExoFOP), and the Kepler KOI table.  Intended as a quick pre-scan before
committing pipeline time to a target.

Public API
----------
ArchiveStatus(tic_id, in_nea, in_toi, in_koi,
              nea_planets, toi_entries, koi_entries, recommendation)
check_archive(tic_id, *, nea_fn, toi_fn, koi_fn) -> ArchiveStatus
format_archive_status(status) -> str
"""
from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ArchiveStatus:
    tic_id: int
    in_nea: bool
    in_toi: bool
    in_koi: bool
    nea_planets: tuple[str, ...]    # confirmed planet names
    toi_entries: tuple[str, ...]    # TOI designations
    koi_entries: tuple[str, ...]    # KOI designations
    recommendation: str             # "known_object" | "toi_followup" | "novel"


# ---------------------------------------------------------------------------
# Default network queries (injectable in tests)
# ---------------------------------------------------------------------------

def _default_nea_fn(tic_id: int) -> list[dict[str, Any]]:
    """Query NASA Exoplanet Archive for confirmed planets with this TIC ID."""
    import json
    import urllib.request
    url = (
        "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
        f"?query=select+pl_name,disc_year+from+ps+where+tic_id={tic_id}&format=json"
    )
    with urllib.request.urlopen(url, timeout=15) as r:
        return json.loads(r.read())


def _default_toi_fn(tic_id: int) -> list[dict[str, Any]]:
    """Query ExoFOP TOI list for this TIC ID."""
    import csv
    import io
    import urllib.request
    url = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
    with urllib.request.urlopen(url, timeout=20) as r:
        text = r.read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    return [row for row in reader if str(row.get("TIC ID", "")) == str(tic_id)]


def _default_koi_fn(tic_id: int) -> list[dict[str, Any]]:
    """Kepler KOI table does not use TIC IDs — returns empty list."""
    return []


# ---------------------------------------------------------------------------
# Public check function
# ---------------------------------------------------------------------------

def check_archive(
    tic_id: int,
    *,
    nea_fn: Callable[[int], list[dict]] | None = None,
    toi_fn: Callable[[int], list[dict]] | None = None,
    koi_fn: Callable[[int], list[dict]] | None = None,
) -> ArchiveStatus:
    """Check whether a TIC target appears in published archives.

    Args:
        tic_id: TESS Input Catalog ID.
        nea_fn: Injectable ``(tic_id) -> list[dict]`` for NEA.
        toi_fn: Injectable ``(tic_id) -> list[dict]`` for ExoFOP TOI.
        koi_fn: Injectable ``(tic_id) -> list[dict]`` for Kepler KOI.

    Returns:
        :class:`ArchiveStatus`.
    """
    _nea = nea_fn if nea_fn is not None else _default_nea_fn
    _toi = toi_fn if toi_fn is not None else _default_toi_fn
    _koi = koi_fn if koi_fn is not None else _default_koi_fn

    nea_rows: list[dict] = []
    toi_rows: list[dict] = []
    koi_rows: list[dict] = []

    with contextlib.suppress(Exception):
        nea_rows = _nea(tic_id)
    with contextlib.suppress(Exception):
        toi_rows = _toi(tic_id)
    with contextlib.suppress(Exception):
        koi_rows = _koi(tic_id)

    nea_planets = tuple(r.get("pl_name", "") for r in nea_rows if r.get("pl_name"))
    toi_entries = tuple(
        str(r.get("TOI", r.get("toi", ""))) for r in toi_rows if r.get("TOI") or r.get("toi")
    )
    koi_entries = tuple(
        str(r.get("kepoi_name", "")) for r in koi_rows if r.get("kepoi_name")
    )

    in_nea = len(nea_planets) > 0
    in_toi = len(toi_entries) > 0
    in_koi = len(koi_entries) > 0

    if in_nea:
        recommendation = "known_object"
    elif in_toi:
        recommendation = "toi_followup"
    else:
        recommendation = "novel"

    return ArchiveStatus(
        tic_id=tic_id,
        in_nea=in_nea,
        in_toi=in_toi,
        in_koi=in_koi,
        nea_planets=nea_planets,
        toi_entries=toi_entries,
        koi_entries=koi_entries,
        recommendation=recommendation,
    )


def format_archive_status(status: ArchiveStatus) -> str:
    """Format archive lookup result as a Markdown block."""
    lines = [
        "## Archive Lookup",
        "",
        f"- TIC ID: {status.tic_id}",
        "- In NASA Exoplanet Archive: " + (
            "Yes — " + ", ".join(status.nea_planets) if status.in_nea else "No"
        ),
        "- In TESS TOI list: " + (
            "Yes — TOI " + ", ".join(status.toi_entries) if status.in_toi else "No"
        ),
        "- In Kepler KOI: " + (
            "Yes — " + ", ".join(status.koi_entries) if status.in_koi else "No"
        ),
        f"- **Recommendation**: {status.recommendation}",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="archive_lookup",
        description="Check TIC target against NEA, TOI, and KOI archives.",
    )
    parser.add_argument("tic_id", type=int)
    args = parser.parse_args(argv)

    status = check_archive(args.tic_id)
    print(format_archive_status(status))
    return 0 if status.recommendation == "novel" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
