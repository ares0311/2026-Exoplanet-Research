"""Tests for Skills/spectral_energy_distribution_fitter.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from spectral_energy_distribution_fitter import fit_sed_temperature, format_sed_fit_result


def _solar_sed() -> dict[str, float]:
    # Magnitudes computed from a 5800 K Planck function (V = 5.00 reference)
    return {"B": 5.02, "V": 5.00, "J": 6.53, "H": 7.46, "K": 8.37}


class TestFitSedTemperature:
    def test_ok_flag(self) -> None:
        r = fit_sed_temperature(_solar_sed())
        assert r.flag == "OK"

    def test_solar_teff_near_5800(self) -> None:
        r = fit_sed_temperature(_solar_sed())
        assert abs(r.fitted_teff_k - 5778.0) < 500.0

    def test_chi2_non_negative(self) -> None:
        r = fit_sed_temperature(_solar_sed())
        assert r.best_chi2 >= 0.0

    def test_n_bands_correct(self) -> None:
        r = fit_sed_temperature(_solar_sed())
        assert r.n_bands == 5

    def test_uncertainty_positive(self) -> None:
        r = fit_sed_temperature(_solar_sed())
        assert r.teff_uncertainty_k > 0.0

    def test_hot_star_higher_teff(self) -> None:
        # Magnitudes from a 9000 K Planck function (V = 5.00 reference)
        hot_sed = {"B": 4.61, "V": 5.00, "J": 7.35, "H": 8.43, "K": 9.43}
        r = fit_sed_temperature(hot_sed)
        assert r.fitted_teff_k > 6000.0

    def test_cool_star_lower_teff(self) -> None:
        cool_sed = {"V": 8.0, "J": 6.5, "H": 5.8, "K": 5.5}
        r = fit_sed_temperature(cool_sed)
        assert r.fitted_teff_k < 5000.0

    def test_custom_grid(self) -> None:
        r = fit_sed_temperature(_solar_sed(), teff_grid_k=list(range(4000, 8001, 500)))
        assert r.flag == "OK"

    def test_insufficient_bands(self) -> None:
        r = fit_sed_temperature({"V": 4.74})
        assert r.flag == "INSUFFICIENT_BANDS"

    def test_unknown_bands(self) -> None:
        r = fit_sed_temperature({"X1": 5.0, "X2": 5.5})
        assert r.flag == "UNKNOWN_BANDS"

    def test_result_frozen(self) -> None:
        r = fit_sed_temperature(_solar_sed())
        try:
            r.fitted_teff_k = 0.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except (AttributeError, TypeError):
            pass

    def test_format_returns_string(self) -> None:
        r = fit_sed_temperature(_solar_sed())
        s = format_sed_fit_result(r)
        assert isinstance(s, str)
        assert r.flag in s
