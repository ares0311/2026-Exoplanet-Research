"""Target-pixel centroid-shift diagnostic for one TESS candidate signal."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from astropy.wcs.utils import proj_plane_pixel_scales

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from Skills.run_report import (  # noqa: E402
    RunReport,
    report_path_for,
    run_and_commit_report,
)


@dataclass(frozen=True)
class SectorCentroidResult:
    sector: int
    n_transit_events: int
    n_in_transit: int
    n_out_of_transit: int
    row_shift_pixels: float
    column_shift_pixels: float
    offset_pixels: float
    offset_arcsec: float
    offset_sigma: float | None
    pixel_scale_arcsec: float
    aperture_mask_source: str
    aperture_pixel_count: int
    pointing_correction_applied: bool


def flux_weighted_centroids(
    flux: np.ndarray,
    aperture_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return row/column flux-weighted centroids for each cadence."""
    cube = np.asarray(flux, dtype=float)
    mask = np.asarray(aperture_mask, dtype=bool)
    if cube.ndim != 3 or mask.shape != cube.shape[1:] or not mask.any():
        raise ValueError("flux must be (cadence,row,column) with a non-empty matching mask")
    rows, columns = np.indices(mask.shape, dtype=float)
    selected = np.where(mask[None, :, :], cube, np.nan)
    total = np.nansum(selected, axis=(1, 2))
    valid = np.isfinite(total) & (total > 0.0)
    row = np.full(len(cube), np.nan)
    column = np.full(len(cube), np.nan)
    row[valid] = np.nansum(selected[valid] * rows, axis=(1, 2)) / total[valid]
    column[valid] = np.nansum(selected[valid] * columns, axis=(1, 2)) / total[valid]
    return row, column


def analyze_centroid_shift(
    *,
    time_bjd: np.ndarray,
    flux: np.ndarray,
    aperture_mask: np.ndarray,
    period_days: float,
    epoch_bjd: float,
    duration_hours: float,
    sector: int,
    pixel_scale_arcsec: float,
    pos_corr_column: np.ndarray | None = None,
    pos_corr_row: np.ndarray | None = None,
    aperture_mask_source: str = "provided",
) -> SectorCentroidResult:
    """Compare local in/out aperture photocenters using independent events."""
    time_values = np.asarray(time_bjd, dtype=float)
    cube = np.asarray(flux, dtype=float)
    mask = np.asarray(aperture_mask, dtype=bool)
    numeric_inputs = (period_days, epoch_bjd, duration_hours, pixel_scale_arcsec)
    if not all(math.isfinite(value) for value in numeric_inputs):
        raise ValueError("ephemeris and pixel scale values must be finite")
    if period_days <= 0.0 or duration_hours <= 0.0 or pixel_scale_arcsec <= 0.0:
        raise ValueError("period, duration, and pixel scale must be positive")
    duration_days = duration_hours / 24.0
    if duration_days >= 0.2 * period_days:
        raise ValueError("duration must be less than 20% of the orbital period")
    if time_values.ndim != 1 or cube.ndim != 3 or len(time_values) != len(cube):
        raise ValueError("time and flux must have the same cadence dimension")

    row, column = flux_weighted_centroids(flux, aperture_mask)
    pointing_corrected = pos_corr_column is not None or pos_corr_row is not None
    if pointing_corrected:
        if pos_corr_column is None or pos_corr_row is None:
            raise ValueError("both row and column pointing corrections are required")
        column_correction = np.asarray(pos_corr_column, dtype=float)
        row_correction = np.asarray(pos_corr_row, dtype=float)
        if (
            column_correction.shape != time_values.shape
            or row_correction.shape != time_values.shape
        ):
            raise ValueError("pointing corrections must match the cadence dimension")
        column = column - column_correction
        row = row - row_correction

    half_duration_days = duration_hours / 48.0
    event_number = np.rint((time_values - epoch_bjd) / period_days)
    phase_days = time_values - (epoch_bjd + event_number * period_days)
    finite = np.isfinite(time_values) & np.isfinite(row) & np.isfinite(column)
    in_transit = finite & (np.abs(phase_days) <= half_duration_days)
    local_out = finite & (np.abs(phase_days) >= 2.0 * half_duration_days) & (
        np.abs(phase_days) <= 5.0 * half_duration_days
    )
    shifts: list[tuple[float, float]] = []
    n_in = 0
    n_out = 0
    for event in np.unique(event_number[in_transit]).astype(int):
        event_in = in_transit & (event_number == event)
        event_out = local_out & (event_number == event)
        event_n_in = int(event_in.sum())
        event_n_out = int(event_out.sum())
        if event_n_in < 3 or event_n_out < 10:
            continue
        n_in += event_n_in
        n_out += event_n_out
        shifts.append(
            (
                float(np.median(row[event_in]) - np.median(row[event_out])),
                float(np.median(column[event_in]) - np.median(column[event_out])),
            )
        )
    if not shifts:
        raise RuntimeError(
            f"sector {sector} has insufficient centroid coverage: in={n_in}, out={n_out}"
        )

    shift_array = np.asarray(shifts, dtype=float)
    row_shift = float(np.median(shift_array[:, 0]))
    column_shift = float(np.median(shift_array[:, 1]))
    offset_pixels = math.hypot(row_shift, column_shift)
    offset_sigma: float | None = None
    if len(shifts) >= 4:
        covariance_of_mean = np.cov(shift_array, rowvar=False) / len(shifts)
        if np.all(np.isfinite(covariance_of_mean)):
            inverse = np.linalg.pinv(covariance_of_mean)
            shift_vector = np.array([row_shift, column_shift])
            squared = float(shift_vector @ inverse @ shift_vector)
            if math.isfinite(squared) and squared >= 0.0:
                offset_sigma = math.sqrt(squared)
    return SectorCentroidResult(
        sector=sector,
        n_transit_events=len(shifts),
        n_in_transit=n_in,
        n_out_of_transit=n_out,
        row_shift_pixels=row_shift,
        column_shift_pixels=column_shift,
        offset_pixels=offset_pixels,
        offset_arcsec=offset_pixels * pixel_scale_arcsec,
        offset_sigma=offset_sigma,
        pixel_scale_arcsec=pixel_scale_arcsec,
        aperture_mask_source=aperture_mask_source,
        aperture_pixel_count=int(mask.sum()),
        pointing_correction_applied=pointing_corrected,
    )


def _pixel_scale_arcsec(tpf: Any) -> float:
    scales = np.asarray(proj_plane_pixel_scales(tpf.wcs), dtype=float) * 3600.0
    finite = scales[np.isfinite(scales) & (scales > 0.0)]
    if not len(finite):
        raise RuntimeError("TPF WCS does not provide a finite pixel scale")
    return float(np.mean(finite))


def analyze_tpf(
    tpf: Any,
    *,
    period_days: float,
    epoch_bjd: float,
    duration_hours: float,
) -> SectorCentroidResult:
    """Adapt one Lightkurve TargetPixelFile to the numerical diagnostic."""
    pipeline_mask = getattr(tpf, "pipeline_mask", None)
    mask = np.asarray(pipeline_mask, dtype=bool)
    expected_shape = np.asarray(tpf.flux.value).shape[1:]
    mask_source = "pipeline"
    if mask.shape != expected_shape or not mask.any() or mask.all():
        mask = np.asarray(tpf.create_threshold_mask(threshold=3), dtype=bool)
        mask_source = "threshold_3sigma"
    if mask.shape != expected_shape or not mask.any():
        raise RuntimeError("TPF does not provide a usable aperture mask")
    sector = int(tpf.sector)
    return analyze_centroid_shift(
        time_bjd=np.asarray(tpf.time.jd, dtype=float),
        flux=np.asarray(tpf.flux.value, dtype=float),
        aperture_mask=mask,
        period_days=period_days,
        epoch_bjd=epoch_bjd,
        duration_hours=duration_hours,
        sector=sector,
        pixel_scale_arcsec=_pixel_scale_arcsec(tpf),
        pos_corr_column=np.asarray(tpf.pos_corr1, dtype=float),
        pos_corr_row=np.asarray(tpf.pos_corr2, dtype=float),
        aperture_mask_source=mask_source,
    )


def _strict_json_value(value: Any) -> str | int | float | bool | None:
    """Convert table scalars to strict, portable JSON primitives."""
    if value is None or np.ma.is_masked(value):
        return None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return str(value)


def _search_product_provenance(search: Any) -> dict[str, Any]:
    """Capture stable MAST/Lightkurve fields for one SearchResult row."""
    table = getattr(search, "table", None)
    if table is None or len(table) != 1:
        return {}
    row = table[0]
    fields = (
        "obs_id",
        "productFilename",
        "dataURI",
        "dataURL",
        "size",
        "proposal_id",
        "sequence_number",
        "mission",
        "author",
        "exptime",
        "target_name",
    )
    return {
        field: _strict_json_value(row[field])
        for field in fields
        if field in table.colnames
    }


def _tpf_provenance(tpf: Any, target_id: str) -> dict[str, Any]:
    return {
        "mission": str(getattr(tpf, "mission", "TESS")),
        "target_id": str(getattr(tpf, "targetid", target_id)),
        "sector": int(tpf.sector),
        "camera": int(tpf.camera),
        "ccd": int(tpf.ccd),
    }


def _ephemeris_coverage(tpf: Any, *, period_days: float, epoch_bjd: float) -> dict[str, Any]:
    """Summarize whether a predicted event center falls inside one TPF."""
    time_values = np.asarray(tpf.time.jd, dtype=float)
    finite = time_values[np.isfinite(time_values)]
    if not len(finite):
        return {"finite_cadences": 0}
    start = float(np.min(finite))
    end = float(np.max(finite))
    first_event = math.floor((start - epoch_bjd) / period_days) - 1
    last_event = math.ceil((end - epoch_bjd) / period_days) + 1
    centers = [
        epoch_bjd + event * period_days
        for event in range(first_event, last_event + 1)
    ]

    def distance_to_interval(center: float) -> float:
        if center < start:
            return start - center
        if center > end:
            return center - end
        return 0.0

    nearest = min(centers, key=distance_to_interval)
    return {
        "finite_cadences": int(len(finite)),
        "time_start_bjd": start,
        "time_end_bjd": end,
        "nearest_predicted_event_bjd": nearest,
        "nearest_event_gap_days": distance_to_interval(nearest),
        "predicted_event_center_in_coverage": start <= nearest <= end,
    }


def run_live(
    *,
    target_id: str,
    period_days: float,
    epoch_bjd: float,
    duration_hours: float,
    output_path: Path,
    search_fn: Any = None,
) -> dict[str, Any]:
    """Download only matching TESS-SPOC TPFs, analyze them, and write JSON."""
    if not all(math.isfinite(value) for value in (period_days, epoch_bjd, duration_hours)):
        raise ValueError("candidate ephemeris values must be finite")
    if period_days <= 0.0 or duration_hours <= 0.0:
        raise ValueError("period_days and duration_hours must be positive")
    if search_fn is None:
        import lightkurve as lk

        search_fn = lk.search_targetpixelfile
    search = search_fn(target_id, mission="TESS", author="TESS-SPOC")
    if not len(search):
        raise RuntimeError(f"No TESS-SPOC target-pixel products found for {target_id}")
    started = time.monotonic()
    results: list[SectorCentroidResult] = []
    product_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    print(
        f"Centroid diagnostic: target={target_id} products={len(search)} "
        f"period={period_days:.6f}d duration={duration_hours:.3f}h",
        flush=True,
    )
    for index in range(len(search)):
        subset = search[index : index + 1]
        provenance = _search_product_provenance(subset)
        tpf: Any = None
        try:
            tpf = subset.download(quality_bitmask="default")
            if tpf is None:
                raise RuntimeError("download returned no TargetPixelFile")
            result = analyze_tpf(
                tpf,
                period_days=period_days,
                epoch_bjd=epoch_bjd,
                duration_hours=duration_hours,
            )
            results.append(result)
            product_records.append(
                {
                    "search_product": provenance,
                    "tpf": _tpf_provenance(tpf, target_id),
                    "ephemeris_coverage": _ephemeris_coverage(
                        tpf, period_days=period_days, epoch_bjd=epoch_bjd
                    ),
                    "result": asdict(result),
                }
            )
            status = f"sector={result.sector} offset={result.offset_arcsec:.3f}arcsec"
        except Exception as exc:  # noqa: BLE001
            failure = {
                "product_index": index,
                "search_product": provenance,
                "error": str(exc),
            }
            if tpf is not None:
                failure["tpf"] = _tpf_provenance(tpf, target_id)
                failure["ephemeris_coverage"] = _ephemeris_coverage(
                    tpf, period_days=period_days, epoch_bjd=epoch_bjd
                )
            failures.append(failure)
            status = f"ERROR {exc}"
        elapsed = time.monotonic() - started
        remaining = elapsed / (index + 1) * (len(search) - index - 1)
        print(
            f"  [{index + 1}/{len(search)}] {status} elapsed={elapsed:.0f}s "
            f"ETA={remaining:.0f}s",
            flush=True,
        )
    only_no_coverage = bool(failures) and all(
        "insufficient centroid coverage" in item["error"] for item in failures
    )
    if results and not failures:
        status = "complete"
    elif results:
        status = "partial"
    elif only_no_coverage:
        status = "no_transit_coverage"
    else:
        status = "failed"
    finite_sigmas = [item.offset_sigma for item in results if item.offset_sigma is not None]
    payload = {
        "schema_version": 1,
        "tool_version": "0.2.40",
        "algorithm": "local_aperture_photocenter_v1",
        "algorithm_scope": (
            "Aperture-photocenter consistency only; this does not localize the "
            "transit source or replace difference-image analysis."
        ),
        "author": "TESS-SPOC",
        "quality_bitmask": "default",
        "target_id": target_id,
        "candidate": {
            "period_days": period_days,
            "epoch_bjd": epoch_bjd,
            "duration_hours": duration_hours,
        },
        "status": status,
        "products_found": len(search),
        "products_analyzed": len(results),
        "products_failed": len(failures),
        "max_offset_sigma": max(finite_sigmas) if finite_sigmas else None,
        "max_offset_arcsec": (
            max(item.offset_arcsec for item in results) if results else None
        ),
        "products": product_records,
        "failures": failures,
        "created_at": datetime.now(UTC).isoformat(),
        "scientific_guardrail": "Centroid evidence is diagnostic only; it cannot confirm a planet.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(
        f"Complete: status={status} analyzed={len(results)} "
        f"unusable={len(failures)} output={output_path}",
        flush=True,
    )
    return payload


def main(argv: list[str] | None = None, *, git_run_fn: Any = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target_id")
    parser.add_argument("--period-days", type=float, required=True)
    parser.add_argument("--epoch-bjd", type=float, required=True)
    parser.add_argument("--duration-hours", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-git-report", action="store_true")
    args = parser.parse_args(argv)
    started_at = datetime.now(UTC).isoformat()
    started = time.monotonic()
    payload = run_live(
        target_id=args.target_id,
        period_days=args.period_days,
        epoch_bjd=args.epoch_bjd,
        duration_hours=args.duration_hours,
        output_path=args.output,
    )
    if not args.no_git_report:
        payload_status = str(payload["status"])
        report_status = {
            "complete": "success",
            "partial": "partial",
            "no_transit_coverage": "partial",
            "failed": "failed",
        }[payload_status]
        report = RunReport(
            script="tpf_centroid_diagnostic",
            status=report_status,
            started_at=started_at,
            completed_at=datetime.now(UTC).isoformat(),
            elapsed_seconds=time.monotonic() - started,
            items_processed=int(payload["products_found"]),
            items_written=int(payload["products_analyzed"]),
            items_failed=int(payload["products_failed"]),
            output_paths=(str(args.output),),
            notes=f"scientific_status={payload_status}",
        )
        kwargs = {"run_fn": git_run_fn} if git_run_fn is not None else {}
        ok = run_and_commit_report(
            report, report_path_for("tpf_centroid_diagnostic"), **kwargs
        )
        if not ok:
            print("Warning: run report commit/push failed", file=sys.stderr, flush=True)
    return 1 if payload["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
