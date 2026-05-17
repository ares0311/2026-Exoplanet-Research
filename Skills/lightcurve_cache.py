"""Disk-based cache for downloaded light curves, keyed by TIC ID + mission + sector.

Avoids repeated MAST fetches across pipeline runs.  Cache entries are stored as
JSON-serialisable dicts (time, flux, flux_err arrays encoded as lists) under a
configurable cache directory.

Public API
----------
LightcurveCache(cache_dir) — load/save light-curve arrays to disk
cache_key(tic_id, mission, sector) -> str
"""
from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import NamedTuple


class CachedLC(NamedTuple):
    time: list[float]
    flux: list[float]
    flux_err: list[float] | None
    mission: str
    sector: int | None
    tic_id: int


def cache_key(tic_id: int, mission: str, sector: int | None = None) -> str:
    """Stable string key for a light-curve entry."""
    s = f"{sector}" if sector is not None else "all"
    return f"tic{tic_id}_{mission.lower()}_{s}"


class LightcurveCache:
    """Disk-based cache for light curves.

    Args:
        cache_dir: Directory under which cache files are stored.
            Created on first write if absent.
    """

    def __init__(self, cache_dir: Path | str) -> None:
        self._dir = Path(cache_dir)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.json"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def contains(self, tic_id: int, mission: str, sector: int | None = None) -> bool:
        """Return True if the entry is cached."""
        return self._path(cache_key(tic_id, mission, sector)).exists()

    def load(
        self,
        tic_id: int,
        mission: str,
        sector: int | None = None,
    ) -> CachedLC | None:
        """Load a cached light curve, or return None if not present."""
        p = self._path(cache_key(tic_id, mission, sector))
        if not p.exists():
            return None
        data = json.loads(p.read_text())
        return CachedLC(
            time=data["time"],
            flux=data["flux"],
            flux_err=data.get("flux_err"),
            mission=data["mission"],
            sector=data.get("sector"),
            tic_id=data["tic_id"],
        )

    def save(
        self,
        tic_id: int,
        mission: str,
        time: list[float],
        flux: list[float],
        *,
        flux_err: list[float] | None = None,
        sector: int | None = None,
    ) -> Path:
        """Persist a light curve to disk.

        Uses an atomic write (tempfile + rename) to avoid partial files.

        Returns:
            Path to the cache file.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        data: dict = {
            "tic_id": tic_id,
            "mission": mission,
            "sector": sector,
            "time": time,
            "flux": flux,
        }
        if flux_err is not None:
            data["flux_err"] = flux_err

        dest = self._path(cache_key(tic_id, mission, sector))
        fd, tmp = tempfile.mkstemp(dir=self._dir, suffix=".json.tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f)
            os.replace(tmp, dest)
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        return dest

    def delete(self, tic_id: int, mission: str, sector: int | None = None) -> bool:
        """Remove a cache entry; return True if it existed."""
        p = self._path(cache_key(tic_id, mission, sector))
        if p.exists():
            p.unlink()
            return True
        return False

    def list_entries(self) -> list[str]:
        """Return all cache keys present on disk."""
        if not self._dir.exists():
            return []
        return [p.stem for p in sorted(self._dir.glob("*.json"))]

    def clear(self) -> int:
        """Delete all cache entries; return count removed."""
        removed = 0
        for p in list(self._dir.glob("*.json")):
            p.unlink()
            removed += 1
        return removed

    def size_bytes(self) -> int:
        """Total disk usage of all cache files in bytes."""
        if not self._dir.exists():
            return 0
        return sum(p.stat().st_size for p in self._dir.glob("*.json"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="lightcurve_cache",
        description="Inspect or clear the light-curve disk cache.",
    )
    parser.add_argument("--cache-dir", default="data/lc_cache", metavar="DIR")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("list", help="List all cached entries.")
    sub.add_parser("clear", help="Delete all cached entries.")
    size_p = sub.add_parser("size", help="Report total cache size.")
    _ = size_p  # noqa: F841

    args = parser.parse_args(argv)
    cache = LightcurveCache(args.cache_dir)

    if args.cmd == "list":
        keys = cache.list_entries()
        print(f"{len(keys)} entries:")
        for k in keys:
            print(f"  {k}")
    elif args.cmd == "clear":
        n = cache.clear()
        print(f"Removed {n} entries.")
    else:
        mb = cache.size_bytes() / 1_048_576
        print(f"Cache size: {mb:.2f} MB ({len(cache.list_entries())} entries)")

    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
