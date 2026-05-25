"""Parse a simple observation log (TSV/CSV) and compute summary statistics.

Supports CSV or TSV input with columns: BJD, filter, mag, mag_err.
Used for ground-based follow-up planning.

Public API
----------
ObsLogEntry(bjd, filter_name, mag, mag_err)
ObsLogResult(n_obs, n_nights, filters, bjd_start, bjd_end, baseline_days,
             mean_mag, rms_mag, entries, flag)
parse_obs_log(text, *, bjd_col, filter_col, mag_col, magerr_col, delimiter) -> ObsLogResult
load_obs_log(path, **kwargs) -> ObsLogResult
format_obs_log(result) -> str
"""
from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ObsLogEntry:
    bjd: float
    filter_name: str
    mag: float
    mag_err: float | None


@dataclass(frozen=True)
class ObsLogResult:
    n_obs: int
    n_nights: int
    filters: tuple[str, ...]
    bjd_start: float | None
    bjd_end: float | None
    baseline_days: float | None
    mean_mag: float | None
    rms_mag: float | None
    entries: tuple[ObsLogEntry, ...]
    flag: str  # "OK" | "EMPTY" | "INVALID"


def parse_obs_log(
    text: str,
    *,
    bjd_col: str = "BJD",
    filter_col: str = "filter",
    mag_col: str = "mag",
    magerr_col: str = "mag_err",
    delimiter: str = ",",
) -> ObsLogResult:
    """Parse observation log CSV/TSV text into structured entries.

    Args:
        text: CSV/TSV text with a header row.
        bjd_col: Column name for barycentric Julian date.
        filter_col: Column name for filter/bandpass.
        mag_col: Column name for magnitude.
        magerr_col: Column name for magnitude error.
        delimiter: Field delimiter (',' for CSV, '\\t' for TSV).

    Returns:
        :class:`ObsLogResult`.
    """
    if not isinstance(text, str):
        return ObsLogResult(0, 0, (), None, None, None, None, None, (), "INVALID")

    stripped = text.strip()
    if not stripped:
        return ObsLogResult(0, 0, (), None, None, None, None, None, (), "EMPTY")

    try:
        reader = csv.DictReader(io.StringIO(stripped), delimiter=delimiter)
        rows = list(reader)
    except Exception:
        return ObsLogResult(0, 0, (), None, None, None, None, None, (), "INVALID")

    if not rows:
        return ObsLogResult(0, 0, (), None, None, None, None, None, (), "EMPTY")

    entries: list[ObsLogEntry] = []
    for row in rows:
        bjd_raw = row.get(bjd_col, "").strip()
        mag_raw = row.get(mag_col, "").strip()
        if not bjd_raw or not mag_raw:
            continue
        try:
            bjd = float(bjd_raw)
            mag = float(mag_raw)
        except ValueError:
            continue
        if not (math.isfinite(bjd) and math.isfinite(mag)):
            continue

        filter_name = row.get(filter_col, "").strip() or "unknown"
        magerr_raw = row.get(magerr_col, "").strip()
        mag_err: float | None = None
        if magerr_raw:
            try:
                mag_err_f = float(magerr_raw)
                mag_err = mag_err_f if math.isfinite(mag_err_f) else None
            except ValueError:
                pass

        entries.append(ObsLogEntry(bjd=bjd, filter_name=filter_name, mag=mag, mag_err=mag_err))

    if not entries:
        return ObsLogResult(0, 0, (), None, None, None, None, None, (), "EMPTY")

    bjds = [e.bjd for e in entries]
    mags = [e.mag for e in entries]
    bjd_start = min(bjds)
    bjd_end = max(bjds)
    baseline_days = bjd_end - bjd_start if len(bjds) > 1 else 0.0
    mean_mag = sum(mags) / len(mags)
    rms_mag = math.sqrt(sum((m - mean_mag) ** 2 for m in mags) / len(mags))

    # Nights: count distinct integer BJD
    nights = len({int(math.floor(b)) for b in bjds})

    # Unique filters
    seen_filters: list[str] = []
    for e in entries:
        if e.filter_name not in seen_filters:
            seen_filters.append(e.filter_name)

    return ObsLogResult(
        n_obs=len(entries),
        n_nights=nights,
        filters=tuple(seen_filters),
        bjd_start=round(bjd_start, 6),
        bjd_end=round(bjd_end, 6),
        baseline_days=round(baseline_days, 4),
        mean_mag=round(mean_mag, 4),
        rms_mag=round(rms_mag, 6),
        entries=tuple(entries),
        flag="OK",
    )


def load_obs_log(path: str | Path, **kwargs) -> ObsLogResult:
    """Load an observation log from a file path."""
    try:
        text = Path(path).read_text()
    except Exception:
        return ObsLogResult(0, 0, (), None, None, None, None, None, (), "INVALID")
    return parse_obs_log(text, **kwargs)


def format_obs_log(result: ObsLogResult) -> str:
    """Format observation log result as Markdown."""
    lines = [
        "## Observation Log Parser",
        "",
        f"- Observations: {result.n_obs}",
        f"- Nights: {result.n_nights}",
        f"- Filters: {', '.join(result.filters) if result.filters else '—'}",
        f"- BJD start: {result.bjd_start}",
        f"- BJD end: {result.bjd_end}",
        f"- Baseline: {result.baseline_days} days",
        f"- Mean mag: {result.mean_mag}",
        f"- RMS mag: {result.rms_mag}",
        f"- **Flag: {result.flag}**",
        "",
    ]
    if result.entries:
        lines.append("### Log Entries (first 10)")
        lines.append("")
        lines.append("| BJD | Filter | Mag | Mag_err |")
        lines.append("|-----|--------|-----|---------|")
        for e in result.entries[:10]:
            err_s = f"{e.mag_err:.4f}" if e.mag_err is not None else "—"
            lines.append(f"| {e.bjd:.5f} | {e.filter_name} | {e.mag:.4f} | {err_s} |")
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="observation_log_parser",
        description="Parse an observation log CSV/TSV and summarise.",
    )
    parser.add_argument("path", help="Path to observation log file")
    parser.add_argument("--bjd-col", default="BJD")
    parser.add_argument("--filter-col", default="filter")
    parser.add_argument("--mag-col", default="mag")
    parser.add_argument("--magerr-col", default="mag_err")
    parser.add_argument("--delimiter", default=",")
    args = parser.parse_args(argv)

    result = load_obs_log(
        args.path,
        bjd_col=args.bjd_col,
        filter_col=args.filter_col,
        mag_col=args.mag_col,
        magerr_col=args.magerr_col,
        delimiter=args.delimiter,
    )
    print(format_obs_log(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
