"""Apply Shallue & Vanderburg (2018) normalization to phase-folded flux arrays.

Each snippet is binned to n_bins uniform bins, the out-of-transit (OOT) median
is subtracted, and all values are divided by the OOT MAD×1.4826.  Snippets with
too few OOT points are rejected.

Public API
----------
NormalizationReport(n_input, n_normalized, n_rejected,
                    mean_oot_scatter_ppm, mean_transit_depth)
NormalizedSnippet(tic_id, label, source, phase, flux, oot_scatter,
                  normalization, flag)
normalize_snippet(tic_id, label, source, phase, flux, *, n_bins,
                  local_window_phase, transit_half_width_phase,
                  min_oot_points) -> NormalizedSnippet
normalize_batch(snippets, *, n_bins) -> (list[NormalizedSnippet], NormalizationReport)
format_normalization_report(report) -> str
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalizationReport:
    """Aggregate statistics from a normalisation batch."""

    n_input: int
    n_normalized: int
    n_rejected: int
    mean_oot_scatter_ppm: float | None
    mean_transit_depth: float | None


@dataclass(frozen=True)
class NormalizedSnippet:
    """One normalised phase-folded flux snippet."""

    tic_id: str
    label: int
    source: str
    phase: tuple[float, ...]   # length n_bins
    flux: tuple[float, ...]    # normalised
    oot_scatter: float | None
    normalization: str         # "local_median_mad"
    flag: str  # "OK" | "REJECTED"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2.0
    return s[mid]


def _mad(values: list[float], med: float) -> float:
    devs = [abs(v - med) for v in values]
    return _median(devs)


def _bin_phase_flux(
    phase: list[float],
    flux: list[float],
    n_bins: int,
) -> tuple[list[float], list[float | None]]:
    """Bin (phase, flux) pairs into *n_bins* uniform bins in [-0.5, 0.5)."""
    bins_flux: list[list[float]] = [[] for _ in range(n_bins)]
    bin_centers = [(-0.5 + (i + 0.5) / n_bins) for i in range(n_bins)]

    for p, f in zip(phase, flux, strict=False):
        # Wrap phase to [-0.5, 0.5)
        p_wrapped = p % 1.0
        if p_wrapped >= 0.5:
            p_wrapped -= 1.0
        b = int((p_wrapped + 0.5) * n_bins)
        b = max(0, min(n_bins - 1, b))
        bins_flux[b].append(f)

    binned: list[float | None] = []
    for bf in bins_flux:
        if bf:
            binned.append(sum(bf) / len(bf))
        else:
            binned.append(None)

    return bin_centers, binned


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_snippet(
    tic_id: str,
    label: int,
    source: str,
    phase: list[float],
    flux: list[float],
    *,
    n_bins: int = 201,
    local_window_phase: float = 0.25,
    transit_half_width_phase: float = 0.05,
    min_oot_points: int = 10,
) -> NormalizedSnippet:
    """Normalise one phase-folded snippet.

    Steps:
      1. Bin to ``n_bins`` uniform bins in [-0.5, 0.5).
      2. Compute median of OOT bins (|phase| > transit_half_width_phase).
      3. Subtract OOT median from all bins.
      4. Compute MAD × 1.4826 of OOT bins as scatter estimate.
      5. Divide all bins by OOT scatter (if > 0).
      6. Reject if fewer than ``min_oot_points`` OOT bins have data.

    Args:
        tic_id:                  Target identifier.
        label:                   1 = planet, 0 = false positive.
        source:                  Source label string.
        phase:                   Phase values (any length).
        flux:                    Corresponding flux values.
        n_bins:                  Number of output bins.
        local_window_phase:      Half-width of OOT window (unused here; reserved).
        transit_half_width_phase: Phase half-width defining OOT region.
        min_oot_points:          Minimum populated OOT bins required.

    Returns:
        :class:`NormalizedSnippet`
    """
    _rejected = NormalizedSnippet(
        tic_id=tic_id, label=label, source=source,
        phase=(), flux=(), oot_scatter=None,
        normalization="local_median_mad", flag="REJECTED",
    )

    if len(phase) == 0 or len(flux) == 0 or len(phase) != len(flux):
        return _rejected

    if not all(math.isfinite(value) for value in (*phase, *flux)):
        return _rejected

    bin_centers, binned_flux = _bin_phase_flux(phase, flux, n_bins)

    # Identify OOT bins
    oot_vals: list[float] = []
    for ph, fv in zip(bin_centers, binned_flux, strict=False):
        if fv is not None and abs(ph) > transit_half_width_phase:
            oot_vals.append(fv)

    if len(oot_vals) < min_oot_points:
        return _rejected

    oot_med = _median(oot_vals)
    oot_mad = _mad(oot_vals, oot_med) * 1.4826

    # Fill missing bins with OOT median (pre-subtraction, so 0 after)
    norm_flux: list[float] = []
    for fv in binned_flux:
        val = fv if fv is not None else oot_med
        val -= oot_med
        if oot_mad > 0:
            val /= oot_mad
        norm_flux.append(val)

    return NormalizedSnippet(
        tic_id=tic_id,
        label=label,
        source=source,
        phase=tuple(bin_centers),
        flux=tuple(norm_flux),
        oot_scatter=oot_mad if oot_mad > 0 else None,
        normalization="local_median_mad",
        flag="OK",
    )


def normalize_batch(
    snippets: list[dict],
    *,
    n_bins: int = 201,
) -> tuple[list[NormalizedSnippet], NormalizationReport]:
    """Normalise a batch of raw snippet dicts.

    Args:
        snippets: List of dicts with keys: tic_id, label, source, phase, flux.
        n_bins:   Number of output phase bins.

    Returns:
        Tuple of (list of NormalizedSnippet, NormalizationReport).
    """
    results: list[NormalizedSnippet] = []
    scatter_vals: list[float] = []
    depth_vals: list[float] = []

    for s in snippets:
        ns = normalize_snippet(
            tic_id=str(s.get("tic_id", "")),
            label=int(s.get("label", 0)),
            source=str(s.get("source", "")),
            phase=list(s.get("phase", [])),
            flux=list(s.get("flux", [])),
            n_bins=n_bins,
        )
        results.append(ns)
        if ns.flag == "OK":
            if ns.oot_scatter is not None:
                scatter_vals.append(ns.oot_scatter * 1e6)  # convert to ppm-ish
            # Depth = 1 - min(flux)  (flux is scatter-normalised, so this is in units of scatter)
            if ns.flux:
                depth_vals.append(-min(ns.flux))

    n_normalized = sum(1 for r in results if r.flag == "OK")
    n_rejected = len(results) - n_normalized

    mean_scatter = sum(scatter_vals) / len(scatter_vals) if scatter_vals else None
    mean_depth = sum(depth_vals) / len(depth_vals) if depth_vals else None

    report = NormalizationReport(
        n_input=len(snippets),
        n_normalized=n_normalized,
        n_rejected=n_rejected,
        mean_oot_scatter_ppm=mean_scatter,
        mean_transit_depth=mean_depth,
    )
    return results, report


def format_normalization_report(report: NormalizationReport) -> str:
    """Return a Markdown summary of a :class:`NormalizationReport`."""
    def _fmt(v: float | None, decimals: int = 2) -> str:
        return f"{v:.{decimals}f}" if v is not None else "N/A"

    lines = [
        "## Snippet Normalization Report",
        "",
        f"**Input**: {report.n_input}",
        f"**Normalized**: {report.n_normalized}",
        f"**Rejected**: {report.n_rejected}",
        f"**Mean OOT scatter (ppm-scale)**: {_fmt(report.mean_oot_scatter_ppm)}",
        f"**Mean transit depth (scatter units)**: {_fmt(report.mean_transit_depth)}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(
        prog="snippet_normalizer",
        description="Normalise a batch of phase-folded snippet JSON.",
    )
    parser.add_argument("input", help="Input snippet JSON file.")
    parser.add_argument("--output", default=None, help="Optional output JSON path.")
    parser.add_argument("--n-bins", type=int, default=201)
    args = parser.parse_args(argv)

    snippets = json.loads(Path(args.input).read_text())
    if not isinstance(snippets, list):
        snippets = snippets.get("snippets", [])

    norm_snippets, report = normalize_batch(snippets, n_bins=args.n_bins)
    print(format_normalization_report(report))

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                [
                    {
                        "tic_id": s.tic_id, "label": s.label, "source": s.source,
                        "phase": list(s.phase), "flux": list(s.flux),
                        "oot_scatter": s.oot_scatter, "normalization": s.normalization,
                        "flag": s.flag,
                    }
                    for s in norm_snippets
                ],
                indent=2,
            )
        )
        print(f"\nSaved to {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
