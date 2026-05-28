"""Manage the phase-folded snippet cache: index, stats, prune, and manifest export.

The snippet cache stores extracted CNN input snippets as a directory of JSON
files (one per TIC+period combination). This manager provides inspection and
maintenance operations without loading all snippets into memory.

Public API
----------
SnippetCacheStats(n_snippets, n_tic_ids, total_size_bytes, oldest_entry,
                  newest_entry, flag)
SnippetCacheManager(cache_dir)
    .stats() -> SnippetCacheStats
    .contains(tic_id, period_days, period_rtol) -> bool
    .prune(max_age_days) -> int   # returns number of files pruned
    .export_manifest(output_path) -> int   # returns number of entries written
format_cache_stats(stats) -> str
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SnippetCacheStats:
    n_snippets: int
    n_tic_ids: int
    total_size_bytes: int
    oldest_entry: str | None    # ISO-8601 or filename
    newest_entry: str | None
    flag: str  # "OK" | "EMPTY" | "INVALID"


class SnippetCacheManager:
    """Manage a directory-based snippet cache.

    Each cached snippet is a JSON file named ``{tic_id}_{period:.6f}.json``.
    """

    def __init__(self, cache_dir: Path) -> None:
        self._dir = Path(cache_dir)

    def _snippet_files(self) -> list[Path]:
        if not self._dir.exists():
            return []
        return sorted(self._dir.glob("*.json"))

    def stats(self) -> SnippetCacheStats:
        """Compute cache statistics without loading snippet data.

        Returns:
            SnippetCacheStats with counts and size information.
        """
        files = self._snippet_files()
        if not files:
            return SnippetCacheStats(
                n_snippets=0, n_tic_ids=0, total_size_bytes=0,
                oldest_entry=None, newest_entry=None, flag="EMPTY",
            )

        tic_ids: set[str] = set()
        total_size = 0
        mtimes: list[float] = []

        for f in files:
            stem = f.stem  # "{tic_id}_{period}"
            parts = stem.rsplit("_", 1)
            if parts:
                tic_ids.add(parts[0])
            try:
                st = f.stat()
                total_size += st.st_size
                mtimes.append(st.st_mtime)
            except OSError:
                pass

        oldest = files[mtimes.index(min(mtimes))].name if mtimes else None
        newest = files[mtimes.index(max(mtimes))].name if mtimes else None

        return SnippetCacheStats(
            n_snippets=len(files),
            n_tic_ids=len(tic_ids),
            total_size_bytes=total_size,
            oldest_entry=oldest,
            newest_entry=newest,
            flag="OK",
        )

    def contains(
        self,
        tic_id: int,
        period_days: float,
        period_rtol: float = 0.01,
    ) -> bool:
        """Check if a snippet for the given TIC + period is cached.

        Args:
            tic_id: TIC identifier.
            period_days: Transit period in days.
            period_rtol: Relative tolerance for period matching.

        Returns:
            True if a matching snippet file exists.
        """
        prefix = f"{tic_id}_"
        for f in self._snippet_files():
            if not f.name.startswith(prefix):
                continue
            try:
                p_str = f.stem[len(prefix):]
                p = float(p_str)
                denom = max(abs(p), abs(period_days), 1e-12)
                if abs(p - period_days) / denom < period_rtol:
                    return True
            except ValueError:
                pass
        return False

    def prune(self, max_age_days: float = 30.0) -> int:
        """Remove cache files older than max_age_days.

        Args:
            max_age_days: Files older than this many days are deleted.

        Returns:
            Number of files pruned.
        """
        import time
        cutoff = time.time() - max_age_days * 86400
        n_pruned = 0
        for f in self._snippet_files():
            try:
                if f.stat().st_mtime < cutoff:
                    f.unlink()
                    n_pruned += 1
            except OSError:
                pass
        return n_pruned

    def export_manifest(self, output_path: Path) -> int:
        """Export a manifest of all cached snippets as a JSON array.

        Each entry: ``{filename, tic_id, period_days, size_bytes}``.

        Args:
            output_path: Destination JSON file path.

        Returns:
            Number of manifest entries written.
        """
        import contextlib
        import tempfile

        files = self._snippet_files()
        entries = []
        for f in files:
            stem = f.stem
            parts = stem.rsplit("_", 1)
            try:
                tic_id = int(parts[0]) if len(parts) == 2 else 0
                period = float(parts[1]) if len(parts) == 2 else 0.0
            except (ValueError, IndexError):
                tic_id, period = 0, 0.0
            try:
                size = f.stat().st_size
            except OSError:
                size = 0
            entries.append({
                "filename": f.name,
                "tic_id": tic_id,
                "period_days": period,
                "size_bytes": size,
            })

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=output_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(entries, fh, indent=2)
            os.replace(tmp, output_path)
        except Exception:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise
        return len(entries)


def format_cache_stats(stats: SnippetCacheStats) -> str:
    """Format a Markdown snippet cache statistics report.

    Args:
        stats: SnippetCacheStats to format.

    Returns:
        Markdown string.
    """
    lines = [
        "## Snippet Cache Statistics\n",
        f"Flag: `{stats.flag}` | Snippets: {stats.n_snippets}\n",
    ]
    if stats.flag in ("EMPTY", "INVALID"):
        lines.append("\n_Cache is empty or unavailable._\n")
        return "\n".join(lines)

    size_mb = stats.total_size_bytes / (1024 * 1024)
    lines += [
        f"**Unique TIC IDs**: {stats.n_tic_ids}\n",
        f"**Total size**: {size_mb:.2f} MB\n",
        f"**Oldest entry**: {stats.oldest_entry}\n",
        f"**Newest entry**: {stats.newest_entry}\n",
    ]
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Manage the snippet cache.")
    parser.add_argument("cache_dir", help="Path to the snippet cache directory.")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("stats", help="Show cache statistics.")
    prune_p = sub.add_parser("prune", help="Remove old cache files.")
    prune_p.add_argument("--max-age-days", type=float, default=30.0)
    manifest_p = sub.add_parser("manifest", help="Export cache manifest.")
    manifest_p.add_argument("output", help="Output JSON file path.")

    args = parser.parse_args(argv)
    mgr = SnippetCacheManager(Path(args.cache_dir))

    if args.cmd == "prune":
        n = mgr.prune(max_age_days=args.max_age_days)
        print(f"Pruned {n} files.")
        return 0

    if args.cmd == "manifest":
        n = mgr.export_manifest(Path(args.output))
        print(f"Wrote {n} entries to {args.output}")
        return 0

    stats = mgr.stats()
    print(format_cache_stats(stats))
    return 0 if stats.flag in ("OK", "EMPTY") else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
