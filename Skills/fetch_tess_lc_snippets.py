"""Fetch TESS light curves and build phase-folded CNN snippets.

Reads a target list JSON (as written by fetch_additional_tess_labels.py),
downloads TESS PDCSAP light curves via lightkurve, phase-folds at the
period/epoch, bins to ``n_bins`` phase bins, and normalises.

Output is a JSONL file (one JSON object per line) compatible with
``data/tess_snippets.jsonl``::

    {"tic_id": 150428135, "label": 1, "flux": [...], "source": "tess",
     "period_days": 9.9, "epoch_bjd": 2458325.5, "n_bins": 201}

Public API
----------
TessSnippetResult(tic_id, label, flux, period_days, epoch_bjd, n_bins, flag)
build_tess_snippet(tic_id, label, period_days, epoch_bjd, *, n_bins,
                   lc_fetcher) -> TessSnippetResult
build_tess_snippets(rows, *, n_bins, output_path, lc_fetcher,
                    resume, max_errors) -> int
"""
from __future__ import annotations

import contextlib
import json
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Prevent indefinite hangs when WiFi drops mid-download.
# Any stalled socket operation raises socket.timeout after this many seconds,
# which build_tess_snippet's try/except catches and marks as ERROR.
socket.setdefaulttimeout(120)

_TESS_BJD_OFFSET = 2457000.0  # TESS BTJD = BJD - 2457000


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TessSnippetResult:
    """Outcome of processing a single TESS target."""

    tic_id: int
    label: int            # 1 = confirmed planet/candidate, 0 = false positive
    flux: tuple[float, ...]
    period_days: float
    epoch_bjd: float
    n_bins: int
    flag: str             # "OK" | "NO_LIGHTKURVE" | "NO_DATA" | "SHORT" | "ERROR"


# ---------------------------------------------------------------------------
# Phase-fold and normalise helpers (same as fetch_kepler_lc_snippets.py)
# ---------------------------------------------------------------------------


def _phase_fold_bin(
    time_bjd: list[float],
    flux: list[float],
    period: float,
    epoch: float,
    n_bins: int,
) -> list[float]:
    phases = [((t - epoch) % period) / period for t in time_bjd]
    phases = [p - 1.0 if p >= 0.5 else p for p in phases]

    bin_flux: list[list[float]] = [[] for _ in range(n_bins)]
    for ph, f in zip(phases, flux, strict=False):
        b = int((ph + 0.5) * n_bins)
        b = max(0, min(n_bins - 1, b))
        bin_flux[b].append(f)
    return [
        sum(vals) / len(vals) if vals else 1.0
        for vals in bin_flux
    ]


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _mad(values: list[float], med: float) -> float:
    return _median([abs(v - med) for v in values])


def _normalise(flux_bins: list[float]) -> list[float]:
    med = _median(flux_bins)
    scale = _mad(flux_bins, med) * 1.4826
    if scale < 1e-10:
        return [0.0] * len(flux_bins)
    return [(v - med) / scale for v in flux_bins]


# ---------------------------------------------------------------------------
# Lightkurve fetcher helper
# ---------------------------------------------------------------------------


def _default_lc_fetcher(
    tic_id: int,
    period: float,
    epoch_bjd: float,
) -> tuple[list[float], list[float]] | None:
    """Fetch a TESS PDCSAP light curve for tic_id via lightkurve.

    Tries SPOC first, then QLP as fallback.

    Returns:
        (time_bjd, flux) lists or None if unavailable.
    """
    try:
        import lightkurve as lk
    except ImportError:
        return None

    for author in ("SPOC", "QLP"):
        result = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", author=author)
        if len(result) == 0:
            continue
        lc_coll = result.download_all()
        if lc_coll is None or len(lc_coll) == 0:
            continue
        lc = lc_coll.stitch()
        with contextlib.suppress(Exception):
            lc = lc.normalize()
        time_bjd = [float(t) + _TESS_BJD_OFFSET for t in lc.time.value]
        flux = [float(f) for f in lc.flux.value]
        return time_bjd, flux

    return None


# ---------------------------------------------------------------------------
# Per-target snippet builder
# ---------------------------------------------------------------------------


def build_tess_snippet(
    tic_id: int,
    label: int,
    period_days: float,
    epoch_bjd: float,
    *,
    n_bins: int = 201,
    lc_fetcher: Callable | None = None,
) -> TessSnippetResult:
    """Build a single phase-folded CNN snippet for a TESS target.

    Args:
        tic_id: TIC ID.
        label: 1 = planet/candidate, 0 = false positive.
        period_days: Orbital period in days.
        epoch_bjd: Transit epoch in full BJD.
        n_bins: Number of phase bins.
        lc_fetcher: Injectable fetcher returning (time_bjd, flux) or None.

    Returns:
        :class:`TessSnippetResult` with flag "OK" on success.
    """
    fetcher = lc_fetcher or _default_lc_fetcher

    try:
        result = fetcher(tic_id, period_days, epoch_bjd)
    except Exception as exc:
        return TessSnippetResult(
            tic_id=tic_id, label=label, flux=(), period_days=period_days,
            epoch_bjd=epoch_bjd, n_bins=n_bins, flag=f"ERROR:{exc}",
        )

    if result is None:
        try:
            import lightkurve  # noqa: F401
        except ImportError:
            return TessSnippetResult(
                tic_id=tic_id, label=label, flux=(), period_days=period_days,
                epoch_bjd=epoch_bjd, n_bins=n_bins, flag="NO_LIGHTKURVE",
            )
        return TessSnippetResult(
            tic_id=tic_id, label=label, flux=(), period_days=period_days,
            epoch_bjd=epoch_bjd, n_bins=n_bins, flag="NO_DATA",
        )

    time_bjd, flux = result
    if len(time_bjd) < n_bins:
        return TessSnippetResult(
            tic_id=tic_id, label=label, flux=(), period_days=period_days,
            epoch_bjd=epoch_bjd, n_bins=n_bins, flag="SHORT",
        )

    bins = _phase_fold_bin(time_bjd, flux, period_days, epoch_bjd, n_bins)
    normalised = _normalise(bins)

    return TessSnippetResult(
        tic_id=tic_id,
        label=label,
        flux=tuple(normalised),
        period_days=period_days,
        epoch_bjd=epoch_bjd,
        n_bins=n_bins,
        flag="OK",
    )


# ---------------------------------------------------------------------------
# Batch builder
# ---------------------------------------------------------------------------


def build_tess_snippets(
    rows: list[dict],
    *,
    n_bins: int = 201,
    output_path: Path,
    lc_fetcher: Callable | None = None,
    resume: bool = True,
    max_errors: int = 50,
) -> int:
    """Build phase-folded CNN snippets for a list of TESS target rows.

    Writes one JSONL record per successful snippet. Skips already-written
    tic_ids when ``resume=True``.

    Args:
        rows: List of dicts with keys tic_id, label, period_days, epoch_bjd,
              and optionally source.
        n_bins: Number of phase bins per snippet.
        output_path: Destination JSONL file path.
        lc_fetcher: Injectable light-curve fetcher.
        resume: Skip tic_ids already present in output_path.
        max_errors: Abort after this many consecutive errors.

    Returns:
        Number of snippets written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    already_done: set[int] = set()
    if resume and output_path.exists():
        with output_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    with contextlib.suppress(Exception):
                        row = json.loads(line)
                        already_done.add(int(row.get("tic_id", -1)))

    n_total = len(rows)
    n_written = 0
    n_errors = 0
    consecutive_errors = 0
    start = time.monotonic()

    print(
        f"Building TESS snippets: {n_total} targets  resume={resume}  "
        f"already_done={len(already_done)}  output={output_path}",
        flush=True,
    )

    write_mode = "a" if resume else "w"
    with output_path.open(write_mode, encoding="utf-8") as fh:
        for i, row in enumerate(rows, 1):
            tic_id = int(row.get("tic_id", 0))
            label = int(row.get("label", 0))
            period = float(row.get("period_days", 0.0))
            epoch = float(row.get("epoch_bjd", 0.0))
            source = str(row.get("source", "tess"))

            if tic_id in already_done:
                continue
            if period <= 0 or epoch <= 0:
                continue

            result = build_tess_snippet(
                tic_id, label, period, epoch,
                n_bins=n_bins, lc_fetcher=lc_fetcher,
            )

            elapsed = time.monotonic() - start
            rate = i / elapsed if elapsed > 0 else 1.0
            remaining = (n_total - i) / rate
            eta = (
                f"{remaining/60:.0f}m{remaining%60:.0f}s"
                if remaining > 90
                else f"{remaining:.0f}s"
            )

            if result.flag == "OK":
                record = {
                    "tic_id": tic_id,
                    "label": label,
                    "flux": list(result.flux),
                    "source": source,
                    "period_days": period,
                    "epoch_bjd": epoch,
                    "n_bins": n_bins,
                }
                fh.write(json.dumps(record) + "\n")
                fh.flush()
                n_written += 1
                consecutive_errors = 0
                print(
                    f"  [{i}/{n_total}] {datetime.now().strftime('%H:%M:%S')}"
                    f" TIC {tic_id} label={label}"
                    f"  written={n_written}  elapsed={elapsed:.0f}s  ETA={eta}",
                    flush=True,
                )
            else:
                n_errors += 1
                consecutive_errors += 1
                print(
                    f"  [{i}/{n_total}] {datetime.now().strftime('%H:%M:%S')}"
                    f" TIC {tic_id} SKIP flag={result.flag}"
                    f"  errors={n_errors}  elapsed={elapsed:.0f}s  ETA={eta}",
                    flush=True,
                )
                if consecutive_errors >= max_errors:
                    print(
                        f"Aborting: {consecutive_errors} consecutive errors.",
                        flush=True,
                    )
                    break

    print(
        f"Done. {n_written} snippets written, {n_errors} skipped/errors.",
        flush=True,
    )
    return n_written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="fetch_tess_lc_snippets",
        description=(
            "Fetch TESS light curves and build phase-folded CNN snippets. "
            "Reads target list JSON as written by fetch_additional_tess_labels.py."
        ),
    )
    parser.add_argument(
        "--rows", type=Path, required=True, metavar="JSON",
        help="Target list JSON (list of dicts with tic_id, label, period_days, epoch_bjd).",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/tess_snippets_expansion.jsonl"),
        metavar="JSONL",
        help="Output JSONL file (default: data/tess_snippets_expansion.jsonl)",
    )
    parser.add_argument(
        "--n-bins", type=int, default=201, metavar="N",
        help="Number of phase bins per snippet (default: 201)",
    )
    parser.add_argument(
        "--max-errors", type=int, default=50, metavar="N",
        help="Abort after this many consecutive fetch errors (default: 50)",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Start fresh even if output file exists",
    )
    args = parser.parse_args(argv)

    rows = json.loads(args.rows.read_text())
    if isinstance(rows, dict):
        rows = rows.get("records", [])

    n_positive = sum(1 for r in rows if int(r.get("label", 0)) == 1)
    n_negative = sum(1 for r in rows if int(r.get("label", 0)) == 0)
    print(
        f"Target list: {len(rows)} rows  positive={n_positive}  negative={n_negative}",
        flush=True,
    )

    n = build_tess_snippets(
        rows,
        n_bins=args.n_bins,
        output_path=args.output,
        resume=not args.no_resume,
        max_errors=args.max_errors,
    )
    print(f"Flag: OK  snippets_written={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
