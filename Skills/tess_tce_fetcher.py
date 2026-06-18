"""Inventory the experimental TESS Threshold Crossing Event (TCE) source.

The historic ExoMAST TCE endpoint used by this helper is no longer available.
The helper remains useful as a fail-closed source probe: it reports
``UNAVAILABLE`` with the provider error instead of treating a stale endpoint as
an empty training source.

Public API
----------
TceRecord(tic_id, tce_num, period_days, epoch_btjd, duration_hours,
          depth_ppm, snr, disposition, sectors)
TceFetchResult(records, n_total, n_planet_candidate, n_false_positive,
               n_not_dispositioned, flag, error_message)
fetch_tce_table(*, max_rows, disposition_filter, fetch_fn) -> TceFetchResult
tce_to_label_rows(result) -> list[dict]
format_tce_summary(result) -> str
"""
from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

_EXOMAST_TCE_URL = (
    "https://exo.mast.stsci.edu/api/v0.1/exoplanets/tce/"
    "?select=ticid,tce_num,tce_period,tce_time0bt,tce_duration,"
    "tce_depth,tce_snr,tce_disposition,sectors"
    "&limit={limit}"
)

_VALID_DISPOSITIONS = {"PC", "FP", "EB", "ND", ""}


@dataclass(frozen=True)
class TceRecord:
    tic_id: int
    tce_num: int
    period_days: float
    epoch_btjd: float
    duration_hours: float
    depth_ppm: float
    snr: float
    disposition: str   # "PC" | "FP" | "EB" | "ND" | ""
    sectors: str       # comma-separated sector numbers


@dataclass(frozen=True)
class TceFetchResult:
    records: tuple[TceRecord, ...]
    n_total: int
    n_planet_candidate: int
    n_false_positive: int
    n_not_dispositioned: int
    flag: str  # "OK" | "EMPTY" | "INVALID" | "UNAVAILABLE"
    error_message: str | None = None


def _default_fetch(url: str) -> list[dict]:
    """Fetch JSON from URL."""
    import ssl
    import urllib.request
    try:
        import certifi
        ctx: ssl.SSLContext | None = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = None
    with urllib.request.urlopen(url, timeout=30, context=ctx) as resp:
        return json.loads(resp.read().decode())


def _empty_result(flag: str, error_message: str | None = None) -> TceFetchResult:
    return TceFetchResult(
        records=(),
        n_total=0,
        n_planet_candidate=0,
        n_false_positive=0,
        n_not_dispositioned=0,
        flag=flag,
        error_message=error_message,
    )


def _parse_record(row: dict) -> TceRecord | None:
    """Parse one API row into a TceRecord; return None on bad data."""
    try:
        return TceRecord(
            tic_id=int(row.get("ticid") or 0),
            tce_num=int(row.get("tce_num") or 1),
            period_days=float(row.get("tce_period") or 0.0),
            epoch_btjd=float(row.get("tce_time0bt") or 0.0),
            duration_hours=float(row.get("tce_duration") or 0.0),
            depth_ppm=float(row.get("tce_depth") or 0.0),
            snr=float(row.get("tce_snr") or 0.0),
            disposition=str(row.get("tce_disposition") or "").strip().upper(),
            sectors=str(row.get("sectors") or ""),
        )
    except (TypeError, ValueError):
        return None


def fetch_tce_table(
    *,
    max_rows: int = 5000,
    disposition_filter: list[str] | None = None,
    fetch_fn: Callable[[str], list[dict]] | None = None,
) -> TceFetchResult:
    """Fetch TESS TCE table from ExoMAST TAP.

    Args:
        max_rows: Maximum number of TCE rows to retrieve.
        disposition_filter: If given, keep only rows with these dispositions
            (e.g. ["PC", "FP"] for confirmed planet candidates + false positives).
        fetch_fn: Injectable HTTP fetch function (url -> list[dict]). Defaults
            to urllib-based fetcher; supply a mock in tests.

    Returns:
        TceFetchResult with parsed records.
    """
    if fetch_fn is None:
        fetch_fn = _default_fetch

    url = _EXOMAST_TCE_URL.format(limit=max_rows)
    try:
        raw = fetch_fn(url)
    except Exception as exc:
        if exc.__class__.__name__ == "HTTPError" and getattr(exc, "code", None) == 404:
            return _empty_result("UNAVAILABLE", f"{type(exc).__name__}: {exc}")
        return _empty_result("INVALID", f"{type(exc).__name__}: {exc}")

    if not isinstance(raw, list):
        return _empty_result("INVALID", f"expected list response, got {type(raw).__name__}")

    records: list[TceRecord] = []
    for row in raw:
        rec = _parse_record(row)
        if rec is None:
            continue
        if disposition_filter is not None and rec.disposition not in [
            d.upper() for d in disposition_filter
        ]:
            continue
        records.append(rec)

    if not records:
        return _empty_result("EMPTY")

    n_pc = sum(1 for r in records if r.disposition == "PC")
    n_fp = sum(1 for r in records if r.disposition in ("FP", "EB"))
    n_nd = sum(1 for r in records if r.disposition in ("ND", ""))

    return TceFetchResult(
        records=tuple(records),
        n_total=len(records),
        n_planet_candidate=n_pc,
        n_false_positive=n_fp,
        n_not_dispositioned=n_nd,
        flag="OK",
    )


def tce_to_label_rows(result: TceFetchResult) -> list[dict]:
    """Convert TceFetchResult to label rows for multi_source_label_assembler.

    Args:
        result: TceFetchResult from fetch_tce_table.

    Returns:
        List of dicts with keys: tic_id, period_days, epoch_btjd,
        duration_hours, depth_ppm, label, source.
    """
    label_map = {"PC": "planet_candidate", "FP": "false_positive", "EB": "false_positive"}
    rows = []
    for r in result.records:
        lbl = label_map.get(r.disposition)
        if lbl is None:
            continue
        rows.append(
            {
                "tic_id": r.tic_id,
                "period_days": r.period_days,
                "epoch_btjd": r.epoch_btjd,
                "duration_hours": r.duration_hours,
                "depth_ppm": r.depth_ppm,
                "label": lbl,
                "source": "tess_tce",
            }
        )
    return rows


def format_tce_summary(result: TceFetchResult) -> str:
    """Format a Markdown summary of the TCE fetch result.

    Args:
        result: TceFetchResult to format.

    Returns:
        Markdown string.
    """
    lines = [
        "## TESS TCE Fetch Summary\n",
        f"Flag: `{result.flag}` | Total TCEs: {result.n_total}\n",
    ]
    if result.flag in ("EMPTY", "INVALID", "UNAVAILABLE"):
        lines.append(f"\n_{result.flag}: no records available._\n")
        if result.error_message:
            lines.append(f"Provider detail: `{result.error_message}`\n")
        return "\n".join(lines)

    lines.append("")
    lines.append("| Disposition | Count |")
    lines.append("|---|---|")
    lines.append(f"| Planet Candidate (PC) | {result.n_planet_candidate} |")
    lines.append(f"| False Positive (FP/EB) | {result.n_false_positive} |")
    lines.append(f"| Not Dispositioned | {result.n_not_dispositioned} |")
    lines.append("")
    usable = result.n_planet_candidate + result.n_false_positive
    pct = 100.0 * usable / result.n_total if result.n_total else 0.0
    lines.append(f"**Usable for training**: {usable} ({pct:.1f}%)\n")
    gate = "PASS" if usable >= 5000 else "FAIL"
    lines.append(f"**5,000-label gate**: `{gate}`\n")
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Fetch TESS TCE table from ExoMAST.")
    parser.add_argument("--max-rows", type=int, default=5000)
    parser.add_argument(
        "--disposition",
        nargs="+",
        default=None,
        help="Filter by disposition (PC, FP, EB).",
    )
    parser.add_argument("--output", help="Write label rows to JSON file.")
    args = parser.parse_args(argv)

    result = fetch_tce_table(max_rows=args.max_rows, disposition_filter=args.disposition)
    print(format_tce_summary(result))

    if args.output and result.flag == "OK":
        import json
        from pathlib import Path
        rows = tce_to_label_rows(result)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(rows, indent=2))
        print(f"\nWrote {len(rows)} label rows to {args.output}")

    return 0 if result.flag == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
