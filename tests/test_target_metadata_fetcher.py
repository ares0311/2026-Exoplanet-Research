"""Tests for Skills.target_metadata_fetcher."""
from __future__ import annotations

import pytest
from Skills.target_metadata_fetcher import (
    TargetMetadata,
    fetch_target_metadata,
    format_target_metadata,
)


def _make_catalog_fn(**fields):
    def fn(tic_id):
        return dict(fields)
    return fn


class TestFetchTargetMetadata:
    def test_returns_target_metadata(self) -> None:
        m = fetch_target_metadata(1)
        assert isinstance(m, TargetMetadata)

    def test_tic_id_stored(self) -> None:
        m = fetch_target_metadata(42)
        assert m.tic_id == 42

    def test_stub_source_when_no_catalog(self) -> None:
        m = fetch_target_metadata(1)
        assert m.source == "stub"

    def test_tic_source_when_catalog_returns_data(self) -> None:
        fn = _make_catalog_fn(Tmag=11.0)
        m = fetch_target_metadata(1, catalog_fn=fn)
        assert m.source == "TIC"

    def test_tmag_extracted(self) -> None:
        fn = _make_catalog_fn(Tmag=12.5)
        m = fetch_target_metadata(1, catalog_fn=fn)
        assert m.tmag == pytest.approx(12.5)

    def test_teff_extracted(self) -> None:
        fn = _make_catalog_fn(Teff=5778.0)
        m = fetch_target_metadata(1, catalog_fn=fn)
        assert m.teff == pytest.approx(5778.0)

    def test_radius_extracted(self) -> None:
        fn = _make_catalog_fn(rad=1.2)
        m = fetch_target_metadata(1, catalog_fn=fn)
        assert m.radius_rsun == pytest.approx(1.2)

    def test_none_fields_default_to_none(self) -> None:
        fn = _make_catalog_fn()
        m = fetch_target_metadata(1, catalog_fn=fn)
        assert m.tmag is None
        assert m.teff is None

    def test_n_sectors_default_zero(self) -> None:
        m = fetch_target_metadata(1)
        assert m.n_sectors == 0

    def test_n_sectors_from_catalog(self) -> None:
        fn = _make_catalog_fn(n_sectors=5)
        m = fetch_target_metadata(1, catalog_fn=fn)
        assert m.n_sectors == 5

    def test_contratio_extracted(self) -> None:
        fn = _make_catalog_fn(contratio=0.15)
        m = fetch_target_metadata(1, catalog_fn=fn)
        assert m.contratio == pytest.approx(0.15)


class TestFormatTargetMetadata:
    def test_returns_string(self) -> None:
        m = fetch_target_metadata(1)
        assert isinstance(format_target_metadata(m), str)

    def test_contains_tic_id(self) -> None:
        m = fetch_target_metadata(999)
        assert "999" in format_target_metadata(m)
