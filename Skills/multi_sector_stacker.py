"""Stack multi-sector light curves with outlier rejection.

Combines individually-normalised sector light curves into a single
continuous time series, applying sigma-clipping at sector boundaries.

Public API
----------
StackedLC(time, flux, flux_err, sector_ids, n_cadences_raw, n_cadences_clipped)
stack_sectors(sector_lcs, *, sigma_clip, normalize) -> StackedLC
format_stack_summary(stacked) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StackedLC:
    time: tuple[float, ...]
    flux: tuple[float, ...]
    flux_err: tuple[float, ...]
    sector_ids: tuple[int, ...]          # sector label per cadence
    n_cadences_raw: int
    n_cadences_clipped: int


def _median(vals: list[float]) -> float:
    if not vals:
        return 1.0
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _mad_scaled(vals: list[float], med: float) -> float:
    if not vals:
        return 1e-9
    return max(_median([abs(v - med) for v in vals]) * 1.4826, 1e-9)


def stack_sectors(
    sector_lcs: list[dict],
    *,
    sigma_clip: float = 5.0,
    normalize: bool = True,
) -> StackedLC:
    """Stack multiple sector light curves into one.

    Args:
        sector_lcs: List of dicts, each with keys ``time``, ``flux``,
            and optionally ``flux_err`` and ``sector`` (int label).
        sigma_clip: Sigma threshold for outlier rejection within each sector.
        normalize: If True, normalise each sector to median = 1.0.

    Returns:
        :class:`StackedLC`.
    """
    out_t: list[float] = []
    out_f: list[float] = []
    out_e: list[float] = []
    out_s: list[int] = []
    n_raw = 0
    n_clipped = 0

    for i, lc in enumerate(sector_lcs):
        t_raw = [float(x) for x in lc.get("time", [])]
        f_raw = [float(x) for x in lc.get("flux", [])]
        e_raw_in = lc.get("flux_err")
        e_raw = ([float(x) for x in e_raw_in]
                 if e_raw_in is not None else [0.0] * len(f_raw))
        sector_id = int(lc.get("sector", i + 1))

        n_raw += len(t_raw)

        if not f_raw:
            continue

        med = _median(f_raw) if normalize else 1.0
        if med == 0.0:
            med = 1.0

        f_norm = [f / med for f in f_raw]
        e_norm = [e / med for e in e_raw]

        # Sigma clip
        mad = _mad_scaled(f_norm, 1.0)
        threshold = sigma_clip * mad

        for t, f, e, s in zip(t_raw, f_norm, e_norm, [sector_id] * len(t_raw), strict=False):
            if abs(f - 1.0) > threshold:
                n_clipped += 1
                continue
            out_t.append(t)
            out_f.append(f)
            out_e.append(e)
            out_s.append(s)

    # Sort by time
    if out_t:
        order = sorted(range(len(out_t)), key=lambda i: out_t[i])
        out_t = [out_t[i] for i in order]
        out_f = [out_f[i] for i in order]
        out_e = [out_e[i] for i in order]
        out_s = [out_s[i] for i in order]

    return StackedLC(
        time=tuple(out_t),
        flux=tuple(out_f),
        flux_err=tuple(out_e),
        sector_ids=tuple(out_s),
        n_cadences_raw=n_raw,
        n_cadences_clipped=n_clipped,
    )


def format_stack_summary(stacked: StackedLC) -> str:
    """Format stacked LC summary as Markdown."""
    n_sectors = len(set(stacked.sector_ids))
    pct_kept = (
        100.0 * len(stacked.time) / stacked.n_cadences_raw
        if stacked.n_cadences_raw > 0 else 0.0
    )
    lines = [
        "## Multi-Sector Stack Summary",
        "",
        f"- Sectors combined: {n_sectors}",
        f"- Raw cadences: {stacked.n_cadences_raw}",
        f"- Clipped: {stacked.n_cadences_clipped}",
        f"- Retained: {len(stacked.time)} ({pct_kept:.1f}%)",
    ]
    if stacked.time:
        lines.append(
            f"- Time span: {stacked.time[0]:.2f} – {stacked.time[-1]:.2f} BJD"
        )
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="multi_sector_stacker",
        description="Stack multi-sector light curves with outlier rejection.",
    )
    parser.add_argument("--lcs", nargs="+", required=True, metavar="JSON")
    parser.add_argument("--sigma-clip", type=float, default=5.0)
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args(argv)

    sector_lcs = [json.loads(Path(p).read_text()) for p in args.lcs]
    stacked = stack_sectors(
        sector_lcs, sigma_clip=args.sigma_clip, normalize=not args.no_normalize
    )
    print(format_stack_summary(stacked))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
