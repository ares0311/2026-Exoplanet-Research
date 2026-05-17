"""Tests for Skills.flux_contamination_corrector."""
from __future__ import annotations

import pytest
from Skills.flux_contamination_corrector import (
    ContaminationCorrectionResult,
    correct_flux_contamination,
    format_contamination_result,
)


def _flat(n: int = 50) -> list[float]:
    return [1.0] * n


class TestCorrectFluxContamination:
    def test_returns_result(self) -> None:
        r = correct_flux_contamination(_flat(), crowdsap=0.8)
        assert isinstance(r, ContaminationCorrectionResult)

    def test_no_args_not_applied(self) -> None:
        r = correct_flux_contamination(_flat())
        assert r.applied is False

    def test_crowdsap_one_not_applied(self) -> None:
        r = correct_flux_contamination(_flat(), crowdsap=1.0)
        assert r.applied is False

    def test_crowdsap_half_applies(self) -> None:
        r = correct_flux_contamination(_flat(), crowdsap=0.5)
        assert r.applied is True

    def test_depth_correction_factor_inverse_crowdsap(self) -> None:
        r = correct_flux_contamination(_flat(), crowdsap=0.8)
        assert r.depth_correction_factor == pytest.approx(1.0 / 0.8, rel=1e-5)

    def test_flux_corrected_length_preserved(self) -> None:
        f = _flat(80)
        r = correct_flux_contamination(f, crowdsap=0.7)
        assert len(r.flux_corrected) == 80

    def test_contamination_ratio_conversion(self) -> None:
        # contamination_ratio=1.0 → crowdsap=0.5
        r = correct_flux_contamination(_flat(), contamination_ratio=1.0)
        assert r.dilution_factor == pytest.approx(0.5, rel=1e-5)
        assert r.applied is True

    def test_flat_lc_corrected_stays_near_one(self) -> None:
        # flat LC normalised to 1.0; after correction should still be ~1.0
        r = correct_flux_contamination(_flat(), crowdsap=0.8)
        mean = sum(r.flux_corrected) / len(r.flux_corrected)
        assert abs(mean - 1.0) < 0.01

    def test_crowdsap_stored(self) -> None:
        r = correct_flux_contamination(_flat(), crowdsap=0.75)
        assert r.crowdsap == pytest.approx(0.75)

    def test_zero_crowdsap_not_applied(self) -> None:
        r = correct_flux_contamination(_flat(), crowdsap=0.0)
        assert r.applied is False


class TestFormatContaminationResult:
    def test_returns_string(self) -> None:
        r = correct_flux_contamination(_flat(), crowdsap=0.8)
        assert isinstance(format_contamination_result(r), str)

    def test_contains_crowdsap(self) -> None:
        r = correct_flux_contamination(_flat(), crowdsap=0.8)
        assert "CROWDSAP" in format_contamination_result(r)
