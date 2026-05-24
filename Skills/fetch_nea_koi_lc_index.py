"""Query NASA Exoplanet Archive TAP for the KOI cumulative ephemeris index.

Fetches confirmed and false-positive KOIs from the NASA Exoplanet Archive
TAP service and returns a lightweight index suitable for matching against
Kepler light-curve IDs when building CNN training data.  All HTTP I/O is
injectable for offline tests.

Public API
----------
KoiRecord(kepid, kepoi_name, disposition, period_days, epoch_bkjd, duration_hours)
KoiLcIndex(records, n_confirmed, n_fp, queried_at, flag)
fetch_koi_lc_index(*, tap_fn, max_rows) -> KoiLcIndex
format_koi_lc_index(result) -> str
"""
from __future__ import annotations

import csv
import io
import sys
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TAP_BASE = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"
_TAP_QUERY = (
    "SELECT+kepid,kepoi_name,koi_disposition,koi_period,koi_time0bk,koi_duration"
    "+FROM+cumulative"
    "+WHERE+koi_disposition+in+('CONFIRMED','FALSE+POSITIVE')"
)
_TAP_URL = f"{_TAP_BASE}?query={_TAP_QUERY}&format=csv"

_CONFIRMED = "CONFIRMED"
_FALSE_POSITIVE = "FALSE POSITIVE"


def _default_tap_fn(url: str) -> str:
    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310
        return resp.read().decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KoiRecord:
    """One KOI row from the cumulative table.

    Attributes:
        kepid: Kepler Input Catalog stellar ID.
        kepoi_name: KOI designation (e.g. K00001.01).
        disposition: "CONFIRMED" | "FALSE_POSITIVE"
        period_days: Orbital period in days.
        epoch_bkjd: Transit epoch in BKJD (BJD − 2454833).
        duration_hours: Transit duration in hours.
    """

    kepid: int
    kepoi_name: str
    disposition: str
    period_days: float
    epoch_bkjd: float
    duration_hours: float


@dataclass(frozen=True)
class KoiLcIndex:
    """Result of a KOI TAP query.

    Attributes:
        records: Parsed KOI records.
        n_confirmed: Number of CONFIRMED disposition rows.
        n_fp: Number of FALSE POSITIVE rows.
        queried_at: ISO 8601 UTC timestamp.
        flag: "OK" | "EMPTY" | "FETCH_ERROR"
    """

    records: tuple[KoiRecord, ...]
    n_confirmed: int
    n_fp: int
    queried_at: str
    flag: str


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fetch_koi_lc_index(
    *,
    tap_fn: Callable | None = None,
    max_rows: int = 10_000,
) -> KoiLcIndex:
    """Fetch the KOI cumulative ephemeris index from NASA Exoplanet Archive.

    Args:
        tap_fn: Injectable callable(url: str) -> str.  Defaults to urllib.
        max_rows: Maximum rows to return (applied after parsing).

    Returns:
        :class:`KoiLcIndex` with parsed records and flag.
    """
    now = datetime.now(UTC).isoformat()
    _fn = tap_fn if tap_fn is not None else _default_tap_fn

    try:
        raw = _fn(_TAP_URL)
    except Exception:
        return KoiLcIndex(records=(), n_confirmed=0, n_fp=0, queried_at=now, flag="FETCH_ERROR")

    clean_lines = [ln for ln in raw.splitlines() if not ln.startswith("#")]
    if not clean_lines:
        return KoiLcIndex(records=(), n_confirmed=0, n_fp=0, queried_at=now, flag="EMPTY")

    reader = csv.DictReader(io.StringIO("\n".join(clean_lines)))
    records: list[KoiRecord] = []
    n_confirmed = n_fp = 0

    def _safe_float(val: str) -> float:
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    def _safe_int(val: str) -> int:
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return 0

    for row in reader:
        if len(records) >= max_rows:
            break

        disp_raw = row.get("koi_disposition", "").strip().upper()
        if disp_raw == "CONFIRMED":
            disp = "CONFIRMED"
            n_confirmed += 1
        elif disp_raw in ("FALSE POSITIVE", "FALSE+POSITIVE"):
            disp = "FALSE_POSITIVE"
            n_fp += 1
        else:
            continue

        try:
            rec = KoiRecord(
                kepid=_safe_int(row.get("kepid", "")),
                kepoi_name=row.get("kepoi_name", "").strip(),
                disposition=disp,
                period_days=_safe_float(row.get("koi_period", "")),
                epoch_bkjd=_safe_float(row.get("koi_time0bk", "")),
                duration_hours=_safe_float(row.get("koi_duration", "")),
            )
            records.append(rec)
        except Exception:
            continue

    if not records:
        return KoiLcIndex(records=(), n_confirmed=0, n_fp=0, queried_at=now, flag="EMPTY")

    return KoiLcIndex(
        records=tuple(records),
        n_confirmed=n_confirmed,
        n_fp=n_fp,
        queried_at=now,
        flag="OK",
    )


def format_koi_lc_index(result: KoiLcIndex) -> str:
    """Return a Markdown summary of a :class:`KoiLcIndex`."""
    lines = [
        "## NASA Exoplanet Archive KOI LC Index",
        "",
        f"**Flag**: {result.flag}",
        f"**Queried at**: {result.queried_at}",
        f"**Total records**: {len(result.records)}",
        f"- CONFIRMED: {result.n_confirmed}",
        f"- FALSE POSITIVE: {result.n_fp}",
    ]
    if result.records:
        lines += [
            "",
            "### Sample records (first 3)",
            "| Kepid | KOI Name | Disposition | Period (d) |",
            "|-------|----------|-------------|------------|",
        ]
        for rec in result.records[:3]:
            lines.append(
                f"| {rec.kepid} | {rec.kepoi_name} | {rec.disposition} "
                f"| {rec.period_days:.6f} |"
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="fetch_nea_koi_lc_index",
        description="Fetch KOI cumulative ephemeris index from NASA Exoplanet Archive.",
    )
    parser.add_argument("--max-rows", type=int, default=10_000)
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to save JSON output")
    args = parser.parse_args(argv)

    result = fetch_koi_lc_index(max_rows=args.max_rows)
    print(format_koi_lc_index(result))

    if args.output:
        import json
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "flag": result.flag,
                    "queried_at": result.queried_at,
                    "n_confirmed": result.n_confirmed,
                    "n_fp": result.n_fp,
                    "records": [
                        {
                            "kepid": r.kepid,
                            "kepoi_name": r.kepoi_name,
                            "disposition": r.disposition,
                            "period_days": r.period_days,
                            "epoch_bkjd": r.epoch_bkjd,
                            "duration_hours": r.duration_hours,
                        }
                        for r in result.records
                    ],
                },
                indent=2,
            )
        )
        print(f"\nSaved to {out}")
    return 0 if result.flag == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
