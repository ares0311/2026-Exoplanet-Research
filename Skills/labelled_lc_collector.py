"""Collect and store labelled light-curve snippets for CNN training.

Extracts phase-folded, normalised transit windows from confirmed/FP candidates
and stores them in a compact JSON dataset for CNN training.

Public API
----------
LabelledSnippet(tic_id, label, period_days, epoch_bjd, phase, flux,
                n_points, source)
LabelledDataset(snippets, label_counts, created_at)
extract_snippet(time, flux, period, epoch, *, flux_err, n_bins,
               label, tic_id, source) -> LabelledSnippet | None
build_dataset(rows, *, lc_fetcher, n_bins, output_path) -> LabelledDataset
format_dataset_summary(dataset) -> str
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class LabelledSnippet:
    tic_id: int
    label: int              # 1 = planet candidate, 0 = false positive
    period_days: float
    epoch_bjd: float
    phase: tuple[float, ...]
    flux: tuple[float, ...]
    n_points: int
    source: str             # "kepler", "tess", "k2", etc.


@dataclass(frozen=True)
class LabelledDataset:
    snippets: tuple[LabelledSnippet, ...]
    label_counts: dict[int, int]
    created_at: str


def _phase_fold_bin(
    time: list[float],
    flux: list[float],
    period: float,
    epoch: float,
    n_bins: int,
) -> tuple[list[float], list[float]]:
    """Phase-fold and bin a light curve into n_bins uniform phase bins."""
    phases = [((t - epoch) % period) / period for t in time]
    # Shift to [-0.5, 0.5)
    phases = [p - 1.0 if p >= 0.5 else p for p in phases]

    bin_flux: list[list[float]] = [[] for _ in range(n_bins)]
    for ph, f in zip(phases, flux, strict=False):
        b = int((ph + 0.5) * n_bins)
        b = max(0, min(n_bins - 1, b))
        bin_flux[b].append(f)

    bin_centers = [(-0.5 + (i + 0.5) / n_bins) for i in range(n_bins)]
    bin_means = [
        sum(vals) / len(vals) if vals else 1.0
        for vals in bin_flux
    ]
    return bin_centers, bin_means


def extract_snippet(
    time: list[float],
    flux: list[float],
    period: float,
    epoch: float,
    *,
    flux_err: list[float] | None = None,
    n_bins: int = 201,
    label: int = 1,
    tic_id: int = 0,
    source: str = "unknown",
) -> LabelledSnippet | None:
    """Extract a phase-folded, binned snippet from a light curve.

    Args:
        time: BJD time array.
        flux: Normalised flux array.
        period: Orbital period in days.
        epoch: Transit epoch (BJD).
        flux_err: Per-cadence uncertainties (not used in binning).
        n_bins: Number of phase bins.
        label: Class label (1=planet, 0=FP).
        tic_id: TIC ID for bookkeeping.
        source: Data source identifier.

    Returns:
        :class:`LabelledSnippet` or ``None`` if insufficient data.
    """
    if len(time) < n_bins // 2 or period <= 0:
        return None

    f_arr = [float(f) for f in flux]
    t_arr = [float(t) for t in time]

    # Normalise by median OOT flux
    sorted_f = sorted(f_arr)
    median_f = sorted_f[len(sorted_f) // 2]
    if median_f == 0.0:
        median_f = 1.0
    f_norm = [f / median_f for f in f_arr]

    phase_centers, flux_bins = _phase_fold_bin(t_arr, f_norm, period, epoch, n_bins)

    return LabelledSnippet(
        tic_id=int(tic_id),
        label=int(label),
        period_days=float(period),
        epoch_bjd=float(epoch),
        phase=tuple(round(p, 6) for p in phase_centers),
        flux=tuple(round(f, 8) for f in flux_bins),
        n_points=len(t_arr),
        source=str(source),
    )


def build_dataset(
    rows: list[dict],
    *,
    lc_fetcher: callable | None = None,
    n_bins: int = 201,
    output_path: Path | None = None,
) -> LabelledDataset:
    """Build a labelled dataset from a list of candidate rows.

    Args:
        rows: List of dicts with keys ``tic_id``, ``label``, ``period_days``,
            ``epoch_bjd``, ``time``, ``flux``, and optionally ``source``.
        lc_fetcher: Optional callable(tic_id) -> (time, flux) for fetching LCs.
        n_bins: Number of phase bins per snippet.
        output_path: If given, save dataset as JSON.

    Returns:
        :class:`LabelledDataset`.
    """
    snippets: list[LabelledSnippet] = []

    for row in rows:
        tic_id = int(row.get("tic_id", 0))
        label = int(row.get("label", 1))
        period = float(row.get("period_days", 0.0))
        epoch = float(row.get("epoch_bjd", 0.0))
        source = str(row.get("source", "unknown"))

        time_data = row.get("time")
        flux_data = row.get("flux")

        if (time_data is None or flux_data is None) and lc_fetcher is not None:
            try:
                time_data, flux_data = lc_fetcher(tic_id)
            except Exception:
                continue

        if time_data is None or flux_data is None:
            continue

        snippet = extract_snippet(
            list(time_data), list(flux_data),
            period, epoch,
            n_bins=n_bins,
            label=label,
            tic_id=tic_id,
            source=source,
        )
        if snippet is not None:
            snippets.append(snippet)

    label_counts: dict[int, int] = {}
    for s in snippets:
        label_counts[s.label] = label_counts.get(s.label, 0) + 1

    dataset = LabelledDataset(
        snippets=tuple(snippets),
        label_counts=label_counts,
        created_at=datetime.now(UTC).isoformat(),
    )

    if output_path is not None:
        _save_dataset(dataset, Path(output_path))

    return dataset


def _save_dataset(dataset: LabelledDataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "created_at": dataset.created_at,
        "label_counts": dataset.label_counts,
        "snippets": [
            {**asdict(s), "phase": list(s.phase), "flux": list(s.flux)}
            for s in dataset.snippets
        ],
    }
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh)
        os.replace(tmp, path)
    except Exception:
        import contextlib
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def format_dataset_summary(dataset: LabelledDataset) -> str:
    """Format labelled dataset summary as Markdown."""
    n_total = len(dataset.snippets)
    n_pos = dataset.label_counts.get(1, 0)
    n_neg = dataset.label_counts.get(0, 0)
    lines = [
        "## Labelled LC Dataset Summary",
        "",
        f"- Total snippets: {n_total}",
        f"- Positive (label=1): {n_pos}",
        f"- Negative (label=0): {n_neg}",
        f"- Created: {dataset.created_at}",
    ]
    return "\n".join(lines) + "\n"


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="labelled_lc_collector",
        description="Build a labelled LC dataset for CNN training.",
    )
    parser.add_argument("--rows", required=True, metavar="JSON")
    parser.add_argument("--output", required=True, metavar="PATH")
    parser.add_argument("--n-bins", type=int, default=201)
    args = parser.parse_args(argv)

    rows = json.loads(Path(args.rows).read_text())
    dataset = build_dataset(rows, n_bins=args.n_bins, output_path=Path(args.output))
    print(format_dataset_summary(dataset))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
