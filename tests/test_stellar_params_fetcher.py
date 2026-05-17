"""Tests for Skills.stellar_params_fetcher."""
from __future__ import annotations

import pytest
from Skills.stellar_params_fetcher import StellarParams, fetch_stellar_params


def _mock_catalog(row: dict) -> object:
    def _fn(tic_id: int) -> dict:
        return row
    return _fn


class TestFetchStellarParams:
    def test_all_fields_populated(self) -> None:
        fn = _mock_catalog(
            {"rad": 1.0, "mass": 1.0, "Teff": 5778.0, "logg": 4.44, "contratio": 0.05}
        )
        params = fetch_stellar_params(12345, catalog_fn=fn)
        assert params.stellar_radius_rsun == pytest.approx(1.0)
        assert params.stellar_mass_msun == pytest.approx(1.0)
        assert params.stellar_teff_k == pytest.approx(5778.0)
        assert params.stellar_logg == pytest.approx(4.44)
        assert params.contamination_ratio == pytest.approx(0.05)

    def test_missing_fields_are_none(self) -> None:
        fn = _mock_catalog({})
        params = fetch_stellar_params(12345, catalog_fn=fn)
        assert params.stellar_radius_rsun is None
        assert params.stellar_mass_msun is None
        assert params.stellar_teff_k is None

    def test_nan_value_becomes_none(self) -> None:
        fn = _mock_catalog({"rad": float("nan"), "mass": 1.0})
        params = fetch_stellar_params(12345, catalog_fn=fn)
        assert params.stellar_radius_rsun is None

    def test_tic_id_stored(self) -> None:
        fn = _mock_catalog({})
        params = fetch_stellar_params(99999, catalog_fn=fn)
        assert params.tic_id == 99999

    def test_catalog_fn_called_with_tic_id(self) -> None:
        called_with: list[int] = []

        def _fn(tic_id: int) -> dict:
            called_with.append(tic_id)
            return {"rad": 1.0}

        fetch_stellar_params(42, catalog_fn=_fn)
        assert called_with == [42]

    def test_returns_stellar_params_instance(self) -> None:
        fn = _mock_catalog({"rad": 0.5})
        params = fetch_stellar_params(1, catalog_fn=fn)
        assert isinstance(params, StellarParams)


class TestToVetKwargs:
    def test_excludes_none_fields(self) -> None:
        params = StellarParams(
            tic_id=1,
            stellar_radius_rsun=1.0,
            stellar_mass_msun=None,
            stellar_teff_k=5000.0,
            stellar_logg=None,
            contamination_ratio=None,
        )
        kwargs = params.to_vet_kwargs()
        assert "stellar_radius_rsun" in kwargs
        assert "stellar_mass_msun" not in kwargs
        assert "stellar_teff_k" in kwargs

    def test_excludes_tic_id(self) -> None:
        params = StellarParams(
            tic_id=999,
            stellar_radius_rsun=1.0,
            stellar_mass_msun=1.0,
            stellar_teff_k=5000.0,
            stellar_logg=4.5,
            contamination_ratio=0.1,
        )
        kwargs = params.to_vet_kwargs()
        assert "tic_id" not in kwargs

    def test_excludes_stellar_logg(self) -> None:
        params = StellarParams(
            tic_id=1,
            stellar_radius_rsun=1.0,
            stellar_mass_msun=1.0,
            stellar_teff_k=5000.0,
            stellar_logg=4.5,
            contamination_ratio=0.1,
        )
        kwargs = params.to_vet_kwargs()
        assert "stellar_logg" not in kwargs

    def test_all_vet_fields_present_when_all_non_none(self) -> None:
        params = StellarParams(
            tic_id=1,
            stellar_radius_rsun=1.0,
            stellar_mass_msun=1.0,
            stellar_teff_k=5000.0,
            stellar_logg=4.5,
            contamination_ratio=0.1,
        )
        kwargs = params.to_vet_kwargs()
        expected = {
            "stellar_radius_rsun", "stellar_mass_msun",
            "stellar_teff_k", "contamination_ratio",
        }
        assert set(kwargs.keys()) == expected

    def test_empty_dict_when_all_none(self) -> None:
        params = StellarParams(
            tic_id=1,
            stellar_radius_rsun=None,
            stellar_mass_msun=None,
            stellar_teff_k=None,
            stellar_logg=None,
            contamination_ratio=None,
        )
        assert params.to_vet_kwargs() == {}

    def test_non_numeric_string_becomes_none(self) -> None:
        fn = _mock_catalog({"rad": "N/A", "mass": 1.0})
        params = fetch_stellar_params(1, catalog_fn=fn)
        assert params.stellar_radius_rsun is None
