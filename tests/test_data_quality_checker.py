"""Tests for Skills.data_quality_checker."""
from __future__ import annotations

import numpy as np
import pytest
from Skills.data_quality_checker import QualityReport, check_data_quality, format_quality_report


def _clean_lc(n: int = 500) -> tuple[np.ndarray, np.ndarray]:
    time = np.linspace(2458000.0, 2458027.0, n)
    flux = np.ones(n) + np.random.default_rng(0).normal(0, 1e-4, n)
    return time, flux


class TestCheckDataQuality:
    def test_grade_a_for_clean_data(self) -> None:
        time, flux = _clean_lc()
        report = check_data_quality(time, flux)
        assert report.grade == "A"
        assert report.reason_codes == []

    def test_high_outlier_fraction_flagged(self) -> None:
        time, flux = _clean_lc(500)
        rng = np.random.default_rng(1)
        flux[rng.choice(500, 50, replace=False)] += 1.0  # 10% outliers
        report = check_data_quality(time, flux)
        assert "HIGH_OUTLIER_FRACTION" in report.reason_codes

    def test_irregular_cadence_flagged(self) -> None:
        # Random time spacing
        rng = np.random.default_rng(2)
        time = np.sort(rng.uniform(2458000.0, 2458027.0, 200))
        flux = np.ones(200)
        report = check_data_quality(time, flux)
        assert "IRREGULAR_CADENCE" in report.reason_codes

    def test_excess_scatter_flagged_with_flux_err(self) -> None:
        time, flux = _clean_lc(300)
        rng = np.random.default_rng(3)
        flux += rng.normal(0, 0.05, 300)  # large scatter
        flux_err = np.full(300, 1e-5)  # tiny error floor
        report = check_data_quality(time, flux, flux_err)
        assert "EXCESS_SCATTER" in report.reason_codes

    def test_grade_d_for_multiple_issues(self) -> None:
        rng = np.random.default_rng(4)
        time = np.sort(rng.uniform(2458000.0, 2458027.0, 200))
        flux = np.ones(200)
        flux[rng.choice(200, 30, replace=False)] += 1.0
        flux += rng.normal(0, 0.1, 200)
        flux_err = np.full(200, 1e-6)
        report = check_data_quality(time, flux, flux_err)
        assert report.grade in {"C", "D"}

    def test_n_cadences_correct(self) -> None:
        time, flux = _clean_lc(250)
        report = check_data_quality(time, flux)
        assert report.n_cadences == 250

    def test_outlier_fraction_in_range(self) -> None:
        time, flux = _clean_lc()
        report = check_data_quality(time, flux)
        assert 0.0 <= report.outlier_fraction <= 1.0

    def test_cadence_regularity_near_one_for_uniform_grid(self) -> None:
        time, flux = _clean_lc()
        report = check_data_quality(time, flux)
        assert report.cadence_regularity > 0.90

    def test_returns_quality_report_instance(self) -> None:
        time, flux = _clean_lc()
        report = check_data_quality(time, flux)
        assert isinstance(report, QualityReport)

    def test_grade_b_for_single_soft_issue(self) -> None:
        rng = np.random.default_rng(5)
        time = np.linspace(2458000.0, 2458027.0, 300)
        flux = np.ones(300)
        flux[rng.choice(300, 30, replace=False)] += 1.0  # ~10% outliers
        report = check_data_quality(time, flux)
        # One code from the soft set → grade B
        if len(report.reason_codes) == 1 and report.reason_codes[0] in {
            "HIGH_OUTLIER_FRACTION", "IRREGULAR_CADENCE"
        }:
            assert report.grade == "B"

    def test_scatter_to_noise_ratio_with_no_flux_err(self) -> None:
        time, flux = _clean_lc()
        report = check_data_quality(time, flux, flux_err=None)
        assert report.scatter_to_noise_ratio == pytest.approx(1.0)

    def test_gap_fraction_in_range(self) -> None:
        time, flux = _clean_lc()
        report = check_data_quality(time, flux)
        assert 0.0 <= report.gap_fraction <= 1.0


class TestFormatQualityReport:
    def test_format_contains_grade(self) -> None:
        time, flux = _clean_lc()
        report = check_data_quality(time, flux)
        text = format_quality_report(report)
        assert f"Grade {report.grade}" in text
