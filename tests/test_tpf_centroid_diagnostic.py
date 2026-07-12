"""Offline tests for the target-pixel centroid diagnostic."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import Skills.tpf_centroid_diagnostic as centroid_module
from astropy.table import Table
from Skills.tpf_centroid_diagnostic import (
    analyze_centroid_shift,
    flux_weighted_centroids,
    run_live,
)


def _cube(*, shifted: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = 1200
    period = 2.0
    epoch = 2458000.5
    time = np.linspace(2458000.0, 2458020.0, n)
    phase = ((time - epoch + period / 2) % period) - period / 2
    in_transit = np.abs(phase) <= 2.0 / 48.0
    rows, cols = np.indices((5, 5), dtype=float)
    flux = np.empty((n, 5, 5))
    for i in range(n):
        row0 = 2.12 if shifted and in_transit[i] else 2.0
        col0 = 1.92 if shifted and in_transit[i] else 2.0
        flux[i] = 1000.0 * np.exp(-((rows - row0) ** 2 + (cols - col0) ** 2) / 0.8)
        flux[i] += (i % 7) * 0.01 * rows
    return time, flux, np.ones((5, 5), dtype=bool)


def test_flux_weighted_centroids_require_matching_mask() -> None:
    _, flux, _ = _cube(shifted=False)
    with pytest.raises(ValueError):
        flux_weighted_centroids(flux, np.ones((2, 2), dtype=bool))


def test_centroid_shift_detects_in_transit_motion() -> None:
    time, flux, mask = _cube(shifted=True)
    result = analyze_centroid_shift(
        time_bjd=time,
        flux=flux,
        aperture_mask=mask,
        period_days=2.0,
        epoch_bjd=2458000.5,
        duration_hours=2.0,
        sector=1,
        pixel_scale_arcsec=21.0,
    )
    assert result.offset_pixels > 0.10
    assert result.offset_arcsec > 2.0
    assert result.offset_sigma > 5.0


def test_centroid_shift_is_small_without_motion() -> None:
    time, flux, mask = _cube(shifted=False)
    result = analyze_centroid_shift(
        time_bjd=time,
        flux=flux,
        aperture_mask=mask,
        period_days=2.0,
        epoch_bjd=2458000.5,
        duration_hours=2.0,
        sector=1,
        pixel_scale_arcsec=21.0,
    )
    assert result.offset_pixels < 0.01


def test_insufficient_in_transit_coverage_fails_closed() -> None:
    time, flux, mask = _cube(shifted=False)
    with pytest.raises(RuntimeError, match="insufficient centroid coverage"):
        analyze_centroid_shift(
            time_bjd=time,
            flux=flux,
            aperture_mask=mask,
            period_days=1000.0,
            epoch_bjd=2500000.0,
            duration_hours=1.0,
            sector=1,
            pixel_scale_arcsec=21.0,
        )


class _FakeSearch:
    def __init__(self, products: list[object], table: Table | None = None) -> None:
        self.products = products
        self.table = table

    def __len__(self) -> int:
        return len(self.products)

    def __getitem__(self, item: slice) -> _FakeSearch:
        table = self.table[item] if self.table is not None else None
        return _FakeSearch(self.products[item], table)

    def download(self, *, quality_bitmask: str) -> object:
        assert quality_bitmask == "default"
        return self.products[0]


def _fake_tpf(*, sector: int = 1) -> object:
    time, flux, mask = _cube(shifted=False)
    mask[0, 0] = False
    wcs = SimpleNamespace(
        pixel_scale_matrix=np.array([[21.0 / 3600.0, 0.0], [0.0, 21.0 / 3600.0]])
    )
    return SimpleNamespace(
        pipeline_mask=mask,
        sector=sector,
        time=SimpleNamespace(jd=time),
        flux=SimpleNamespace(value=flux),
        wcs=wcs,
        pos_corr1=np.zeros(len(time)),
        pos_corr2=np.zeros(len(time)),
        mission="TESS",
        targetid=123,
        camera=1,
        ccd=2,
        create_threshold_mask=lambda threshold: mask,
    )


def test_run_live_writes_structured_output_without_network(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[str, str, str]] = []

    def search_fn(target: str, *, mission: str, author: str) -> _FakeSearch:
        calls.append((target, mission, author))
        return _FakeSearch([_fake_tpf()])

    output = tmp_path / "centroid.json"
    payload = run_live(
        target_id="TIC 123",
        period_days=2.0,
        epoch_bjd=2458000.5,
        duration_hours=2.0,
        output_path=output,
        search_fn=search_fn,
    )

    assert calls == [("TIC 123", "TESS", "TESS-SPOC")]
    assert payload["status"] == "complete"
    assert payload["products_analyzed"] == 1
    written = json.loads(output.read_text())
    assert written["target_id"] == "TIC 123"
    assert written["algorithm"] == "local_aperture_photocenter_v1"
    assert written["products"][0]["result"]["pointing_correction_applied"] is True
    console = capsys.readouterr().out
    assert "products=1" in console
    assert "[1/1]" in console
    assert "ETA=" in console


def test_run_live_rejects_invalid_candidate_before_search(tmp_path: Path) -> None:
    def unexpected_search(*args: object, **kwargs: object) -> object:
        raise AssertionError("search should not run")

    with pytest.raises(ValueError, match="must be positive"):
        run_live(
            target_id="TIC 123",
            period_days=0.0,
            epoch_bjd=2458000.5,
            duration_hours=2.0,
            output_path=tmp_path / "centroid.json",
            search_fn=unexpected_search,
        )


def test_single_event_has_no_pseudoreplicated_sigma(tmp_path: Path) -> None:
    epoch = 2458000.5
    time = np.linspace(epoch - 1.0, epoch + 1.0, 1441)
    phase = time - epoch
    rows, columns = np.indices((5, 5), dtype=float)
    flux = np.empty((len(time), 5, 5))
    for index in range(len(time)):
        shifted = abs(phase[index]) <= 4.0 / 24.0
        row0 = 2.1 if shifted else 2.0
        flux[index] = 1000.0 * np.exp(
            -((rows - row0) ** 2 + (columns - 2.0) ** 2) / 0.8
        )
    tpf = _fake_tpf()
    tpf.time = SimpleNamespace(jd=time)
    tpf.flux = SimpleNamespace(value=flux)
    tpf.pos_corr1 = np.zeros(len(time))
    tpf.pos_corr2 = np.zeros(len(time))

    payload = run_live(
        target_id="TIC 123",
        period_days=97.0,
        epoch_bjd=epoch,
        duration_hours=8.0,
        output_path=tmp_path / "single.json",
        search_fn=lambda *args, **kwargs: _FakeSearch([tpf]),
    )

    result = payload["products"][0]["result"]
    assert result["n_transit_events"] == 1
    assert result["offset_sigma"] is None
    assert payload["max_offset_sigma"] is None
    json.loads((tmp_path / "single.json").read_text())


def test_pointing_correction_removes_common_mode_motion() -> None:
    time, flux, mask = _cube(shifted=True)
    raw_row, raw_column = flux_weighted_centroids(flux, mask)
    result = analyze_centroid_shift(
        time_bjd=time,
        flux=flux,
        aperture_mask=mask,
        period_days=2.0,
        epoch_bjd=2458000.5,
        duration_hours=2.0,
        sector=1,
        pixel_scale_arcsec=21.0,
        pos_corr_column=raw_column - np.median(raw_column),
        pos_corr_row=raw_row - np.median(raw_row),
    )
    assert result.offset_pixels < 1e-10
    assert result.pointing_correction_applied is True


def test_run_live_records_product_provenance_and_partial_failure(
    tmp_path: Path,
) -> None:
    table = Table(
        {
            "obs_id": ["failed", "tess-s0001-0000000000000123"],
            "productFilename": ["failed.fits", "tpf.fits"],
            "dataURI": ["mast:failed", "mast:TESS/product/tpf.fits"],
            "author": ["TESS-SPOC", "TESS-SPOC"],
            "exptime": [120.0, 120.0],
        }
    )
    payload = run_live(
        target_id="TIC 123",
        period_days=2.0,
        epoch_bjd=2458000.5,
        duration_hours=2.0,
        output_path=tmp_path / "partial.json",
        search_fn=lambda *args, **kwargs: _FakeSearch([None, _fake_tpf()], table),
    )
    assert payload["status"] == "partial"
    assert payload["products_failed"] == 1
    assert payload["failures"][0]["search_product"]["obs_id"] == "failed"
    assert (
        payload["products"][0]["search_product"]["dataURI"]
        == "mast:TESS/product/tpf.fits"
    )


def test_run_live_persists_no_transit_coverage(tmp_path: Path) -> None:
    output = tmp_path / "no_coverage.json"
    payload = run_live(
        target_id="TIC 123",
        period_days=1000.0,
        epoch_bjd=2458500.0,
        duration_hours=1.0,
        output_path=output,
        search_fn=lambda *args, **kwargs: _FakeSearch([_fake_tpf()]),
    )
    assert payload["status"] == "no_transit_coverage"
    assert payload["products_analyzed"] == 0
    assert payload["max_offset_arcsec"] is None
    coverage = payload["failures"][0]["ephemeris_coverage"]
    assert coverage["predicted_event_center_in_coverage"] is False
    assert coverage["nearest_event_gap_days"] > 0.0
    assert json.loads(output.read_text())["status"] == "no_transit_coverage"


def test_all_true_pipeline_mask_uses_threshold_fallback() -> None:
    tpf = _fake_tpf()
    threshold_mask = tpf.pipeline_mask.copy()
    tpf.pipeline_mask = np.ones_like(threshold_mask)
    tpf.create_threshold_mask = lambda threshold: threshold_mask
    result = centroid_module.analyze_tpf(
        tpf,
        period_days=2.0,
        epoch_bjd=2458000.5,
        duration_hours=2.0,
    )
    assert result.aperture_mask_source == "threshold_3sigma"
    assert result.aperture_pixel_count == int(threshold_mask.sum())


def test_main_writes_run_report_with_injected_runner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "centroid.json"
    monkeypatch.setattr(
        centroid_module,
        "run_live",
        lambda **kwargs: {
            "status": "partial",
            "products_found": 2,
            "products_analyzed": 1,
            "products_failed": 1,
        },
    )
    captured: dict[str, object] = {}

    def fake_report(report: object, path: Path, **kwargs: object) -> bool:
        captured.update(report=report, path=path, kwargs=kwargs)
        return True

    monkeypatch.setattr(centroid_module, "run_and_commit_report", fake_report)
    runner = object()
    exit_code = centroid_module.main(
        [
            "TIC 123",
            "--period-days",
            "2",
            "--epoch-bjd",
            "2458000.5",
            "--duration-hours",
            "2",
            "--output",
            str(output),
        ],
        git_run_fn=runner,
    )
    assert exit_code == 0
    assert captured["kwargs"] == {"run_fn": runner}
    assert captured["report"].status == "partial"
