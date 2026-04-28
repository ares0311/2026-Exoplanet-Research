"""Tests for exo_toolkit.fetch.

lightkurve is not installed in the CI environment, so every test that
exercises fetch_lightcurve() injects a MagicMock into sys.modules via the
``mock_lk`` fixture.  Tests for the pure-Python helpers (FetchProvenance,
FetchResult, _extract_sectors) need no mock at all.

Live MAST integration tests are marked @pytest.mark.integration_live and
are excluded from the default CI run.
"""
from __future__ import annotations

import dataclasses
import sys
from unittest.mock import MagicMock, call

import pydantic
import pytest

from exo_toolkit.fetch import (
    FetchProvenance,
    FetchResult,
    _extract_sectors,
    fetch_lightcurve,
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_lk(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Inject a MagicMock as the lightkurve module for the duration of one test."""
    fake = MagicMock()
    monkeypatch.setitem(sys.modules, "lightkurve", fake)
    return fake


def _make_lc_mock(
    *,
    sector_key: str = "SECTOR",
    sector_val: int = 1,
    exptime: float = 1800.0,
    procver: str = "spoc-4.0.14",
    n_cadences: int = 500,
    baseline_days: float = 27.0,
) -> MagicMock:
    """Build a MagicMock that passes for a lightkurve LightCurve."""
    lc = MagicMock()
    lc.meta = {sector_key: sector_val, "EXPTIME": exptime, "PROCVER": procver}

    # len(lc.time) → n_cadences
    lc.time.__len__.return_value = n_cadences

    # lc.time[-1] - lc.time[0] → delta with .jd == baseline_days
    delta = MagicMock()
    delta.jd = baseline_days
    t_end = MagicMock()
    t_end.__sub__ = MagicMock(return_value=delta)
    lc.time.__getitem__ = MagicMock(
        side_effect=lambda i: t_end if i == -1 else MagicMock()
    )
    return lc


def _make_collection_mock(
    lc_mocks: list[MagicMock],
    stitched: MagicMock | None = None,
) -> MagicMock:
    coll = MagicMock()
    coll.__iter__ = MagicMock(return_value=iter(lc_mocks))
    coll.stitch = MagicMock(return_value=stitched if stitched is not None else lc_mocks[0])
    return coll


def _make_search_mock(collection: MagicMock, n_results: int = 1) -> MagicMock:
    search = MagicMock()
    search.__len__ = MagicMock(return_value=n_results)
    search.download_all = MagicMock(return_value=collection)
    return search


def _wire(mock_lk: MagicMock, lc: MagicMock, n_results: int = 1) -> MagicMock:
    """Wire up mock_lk.search_lightcurve to return a single-LC collection."""
    coll = _make_collection_mock([lc])
    search = _make_search_mock(coll, n_results)
    mock_lk.search_lightcurve.return_value = search
    return coll


# ---------------------------------------------------------------------------
# FetchProvenance
# ---------------------------------------------------------------------------


class TestFetchProvenance:
    def test_valid_construction(self) -> None:
        p = FetchProvenance(
            target_id="TIC 999",
            mission="TESS",
            sectors_or_quarters=(1, 2),
            cadence_seconds=1800.0,
            pipeline="SPOC",
            flux_column="pdcsap_flux",
            n_cadences=500,
            time_baseline_days=27.0,
            fetched_at="2026-04-28T00:00:00+00:00",
        )
        assert p.target_id == "TIC 999"
        assert p.sectors_or_quarters == (1, 2)

    def test_frozen(self) -> None:
        p = FetchProvenance(
            target_id="TIC 1",
            mission="TESS",
            sectors_or_quarters=(1,),
            cadence_seconds=1800.0,
            pipeline="SPOC",
            flux_column="pdcsap_flux",
            n_cadences=100,
            time_baseline_days=10.0,
            fetched_at="2026-04-28T00:00:00+00:00",
        )
        with pytest.raises(pydantic.ValidationError):
            p.target_id = "other"  # type: ignore[misc]

    def test_cadence_seconds_must_be_positive(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            FetchProvenance(
                target_id="TIC 1",
                mission="TESS",
                sectors_or_quarters=(1,),
                cadence_seconds=0.0,  # gt=0, so 0 is invalid
                pipeline="SPOC",
                flux_column="pdcsap_flux",
                n_cadences=100,
                time_baseline_days=10.0,
                fetched_at="2026-04-28T00:00:00+00:00",
            )

    def test_n_cadences_must_be_at_least_one(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            FetchProvenance(
                target_id="TIC 1",
                mission="TESS",
                sectors_or_quarters=(1,),
                cadence_seconds=1800.0,
                pipeline="SPOC",
                flux_column="pdcsap_flux",
                n_cadences=0,  # ge=1, so 0 is invalid
                time_baseline_days=10.0,
                fetched_at="2026-04-28T00:00:00+00:00",
            )

    def test_invalid_mission_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            FetchProvenance(
                target_id="TIC 1",
                mission="Hubble",  # type: ignore[arg-type]
                sectors_or_quarters=(1,),
                cadence_seconds=1800.0,
                pipeline="SPOC",
                flux_column="pdcsap_flux",
                n_cadences=100,
                time_baseline_days=10.0,
                fetched_at="2026-04-28T00:00:00+00:00",
            )

    def test_empty_sectors_tuple_valid(self) -> None:
        p = FetchProvenance(
            target_id="TIC 1",
            mission="TESS",
            sectors_or_quarters=(),
            cadence_seconds=1800.0,
            pipeline="SPOC",
            flux_column="pdcsap_flux",
            n_cadences=100,
            time_baseline_days=10.0,
            fetched_at="2026-04-28T00:00:00+00:00",
        )
        assert p.sectors_or_quarters == ()

    def test_all_three_missions_accepted(self) -> None:
        for mission in ("TESS", "Kepler", "K2"):
            p = FetchProvenance(
                target_id="x",
                mission=mission,  # type: ignore[arg-type]
                sectors_or_quarters=(1,),
                cadence_seconds=1800.0,
                pipeline="test",
                flux_column="pdcsap_flux",
                n_cadences=1,
                time_baseline_days=0.0,
                fetched_at="2026-04-28T00:00:00+00:00",
            )
            assert p.mission == mission


# ---------------------------------------------------------------------------
# FetchResult
# ---------------------------------------------------------------------------


class TestFetchResult:
    def _prov(self) -> FetchProvenance:
        return FetchProvenance(
            target_id="TIC 1",
            mission="TESS",
            sectors_or_quarters=(1,),
            cadence_seconds=1800.0,
            pipeline="SPOC",
            flux_column="pdcsap_flux",
            n_cadences=100,
            time_baseline_days=10.0,
            fetched_at="2026-04-28T00:00:00+00:00",
        )

    def test_construction(self) -> None:
        fake_lc = object()
        r = FetchResult(light_curve=fake_lc, provenance=self._prov())
        assert r.light_curve is fake_lc

    def test_frozen(self) -> None:
        r = FetchResult(light_curve=object(), provenance=self._prov())
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.light_curve = object()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _extract_sectors
# ---------------------------------------------------------------------------


class TestExtractSectors:
    def test_single_tess_sector(self) -> None:
        lc = MagicMock()
        lc.meta = {"SECTOR": 5}
        coll = [lc]
        assert _extract_sectors(coll, "TESS") == (5,)

    def test_multiple_sectors_returned_sorted(self) -> None:
        lcs = [MagicMock(), MagicMock(), MagicMock()]
        lcs[0].meta = {"SECTOR": 3}
        lcs[1].meta = {"SECTOR": 1}
        lcs[2].meta = {"SECTOR": 2}
        assert _extract_sectors(lcs, "TESS") == (1, 2, 3)

    def test_duplicate_sectors_deduplicated(self) -> None:
        lcs = [MagicMock(), MagicMock()]
        lcs[0].meta = {"SECTOR": 7}
        lcs[1].meta = {"SECTOR": 7}
        assert _extract_sectors(lcs, "TESS") == (7,)

    def test_kepler_uses_quarter_key(self) -> None:
        lc = MagicMock()
        lc.meta = {"QUARTER": 12}
        assert _extract_sectors([lc], "Kepler") == (12,)

    def test_k2_uses_campaign_key(self) -> None:
        lc = MagicMock()
        lc.meta = {"CAMPAIGN": 4}
        assert _extract_sectors([lc], "K2") == (4,)

    def test_missing_key_produces_empty_tuple(self) -> None:
        lc = MagicMock()
        lc.meta = {}
        assert _extract_sectors([lc], "TESS") == ()

    def test_empty_collection_produces_empty_tuple(self) -> None:
        assert _extract_sectors([], "TESS") == ()


# ---------------------------------------------------------------------------
# fetch_lightcurve — unit tests (all mocked)
# ---------------------------------------------------------------------------


class TestFetchLightcurve:
    def test_returns_fetch_result(self, mock_lk: MagicMock) -> None:
        lc = _make_lc_mock()
        _wire(mock_lk, lc)
        result = fetch_lightcurve("TIC 999", "TESS")
        assert isinstance(result, FetchResult)
        assert isinstance(result.provenance, FetchProvenance)

    def test_light_curve_is_stitched_output(self, mock_lk: MagicMock) -> None:
        lc = _make_lc_mock()
        coll = _wire(mock_lk, lc)
        result = fetch_lightcurve("TIC 999", "TESS")
        assert result.light_curve is coll.stitch.return_value

    def test_provenance_target_id(self, mock_lk: MagicMock) -> None:
        lc = _make_lc_mock()
        _wire(mock_lk, lc)
        result = fetch_lightcurve("TIC 999", "TESS")
        assert result.provenance.target_id == "TIC 999"

    def test_provenance_mission(self, mock_lk: MagicMock) -> None:
        lc = _make_lc_mock()
        _wire(mock_lk, lc)
        result = fetch_lightcurve("TIC 999", "TESS")
        assert result.provenance.mission == "TESS"

    def test_provenance_n_cadences(self, mock_lk: MagicMock) -> None:
        lc = _make_lc_mock(n_cadences=1350)
        _wire(mock_lk, lc)
        result = fetch_lightcurve("TIC 999", "TESS")
        assert result.provenance.n_cadences == 1350

    def test_provenance_time_baseline(self, mock_lk: MagicMock) -> None:
        lc = _make_lc_mock(baseline_days=54.3)
        _wire(mock_lk, lc)
        result = fetch_lightcurve("TIC 999", "TESS")
        assert result.provenance.time_baseline_days == pytest.approx(54.3)

    def test_provenance_cadence_seconds_from_meta(self, mock_lk: MagicMock) -> None:
        lc = _make_lc_mock(exptime=120.0)
        _wire(mock_lk, lc)
        result = fetch_lightcurve("TIC 999", "TESS", exptime="short")
        assert result.provenance.cadence_seconds == pytest.approx(120.0)

    def test_provenance_cadence_falls_back_when_meta_missing(
        self, mock_lk: MagicMock
    ) -> None:
        lc = _make_lc_mock()
        lc.meta["EXPTIME"] = 0.0  # falsy → trigger fallback
        _wire(mock_lk, lc)
        result = fetch_lightcurve("TIC 999", "TESS", exptime="short")
        assert result.provenance.cadence_seconds == pytest.approx(120.0)

    def test_provenance_pipeline_from_meta(self, mock_lk: MagicMock) -> None:
        lc = _make_lc_mock(procver="spoc-4.0.99")
        _wire(mock_lk, lc)
        result = fetch_lightcurve("TIC 999", "TESS")
        assert result.provenance.pipeline == "spoc-4.0.99"

    def test_provenance_pipeline_falls_back_to_author(
        self, mock_lk: MagicMock
    ) -> None:
        lc = _make_lc_mock()
        lc.meta["PROCVER"] = None  # falsy → fall back to author
        _wire(mock_lk, lc)
        result = fetch_lightcurve("TIC 999", "TESS", pipeline="QLP")
        assert result.provenance.pipeline == "QLP"

    def test_empty_search_raises_value_error(self, mock_lk: MagicMock) -> None:
        search = MagicMock()
        search.__len__ = MagicMock(return_value=0)
        mock_lk.search_lightcurve.return_value = search
        with pytest.raises(ValueError, match="No TESS light curves"):
            fetch_lightcurve("TIC 999", "TESS")

    def test_prefer_pdcsap_true_uses_pdcsap(self, mock_lk: MagicMock) -> None:
        lc = _make_lc_mock()
        _wire(mock_lk, lc)
        fetch_lightcurve("TIC 999", "TESS", prefer_pdcsap=True)
        search = mock_lk.search_lightcurve.return_value
        assert search.download_all.call_args == call(flux_column="pdcsap_flux")

    def test_prefer_pdcsap_false_uses_sap(self, mock_lk: MagicMock) -> None:
        lc = _make_lc_mock()
        _wire(mock_lk, lc)
        fetch_lightcurve("TIC 999", "TESS", prefer_pdcsap=False)
        search = mock_lk.search_lightcurve.return_value
        assert search.download_all.call_args == call(flux_column="sap_flux")

    def test_flux_column_recorded_in_provenance(self, mock_lk: MagicMock) -> None:
        lc = _make_lc_mock()
        _wire(mock_lk, lc)
        result = fetch_lightcurve("TIC 999", "TESS", prefer_pdcsap=False)
        assert result.provenance.flux_column == "sap_flux"

    def test_custom_pipeline_overrides_default(self, mock_lk: MagicMock) -> None:
        lc = _make_lc_mock()
        _wire(mock_lk, lc)
        fetch_lightcurve("TIC 999", "TESS", pipeline="QLP")
        _, kwargs = mock_lk.search_lightcurve.call_args
        assert kwargs["author"] == "QLP"

    def test_default_pipeline_is_spoc_for_tess(self, mock_lk: MagicMock) -> None:
        lc = _make_lc_mock()
        _wire(mock_lk, lc)
        fetch_lightcurve("TIC 999", "TESS")
        _, kwargs = mock_lk.search_lightcurve.call_args
        assert kwargs["author"] == "SPOC"

    def test_tess_sector_restriction_uses_sector_kwarg(
        self, mock_lk: MagicMock
    ) -> None:
        lc = _make_lc_mock()
        _wire(mock_lk, lc)
        fetch_lightcurve("TIC 999", "TESS", sectors=(3, 4))
        _, kwargs = mock_lk.search_lightcurve.call_args
        assert kwargs["sector"] == [3, 4]

    def test_kepler_sector_restriction_uses_quarter_kwarg(
        self, mock_lk: MagicMock
    ) -> None:
        lc = _make_lc_mock(sector_key="QUARTER")
        _wire(mock_lk, lc)
        fetch_lightcurve("KIC 99", "Kepler", sectors=(5,))
        _, kwargs = mock_lk.search_lightcurve.call_args
        assert kwargs["quarter"] == [5]

    def test_k2_sector_restriction_uses_campaign_kwarg(
        self, mock_lk: MagicMock
    ) -> None:
        lc = _make_lc_mock(sector_key="CAMPAIGN")
        _wire(mock_lk, lc)
        fetch_lightcurve("EPIC 99", "K2", sectors=(2,))
        _, kwargs = mock_lk.search_lightcurve.call_args
        assert kwargs["campaign"] == [2]

    def test_no_sector_restriction_passes_none(self, mock_lk: MagicMock) -> None:
        lc = _make_lc_mock()
        _wire(mock_lk, lc)
        fetch_lightcurve("TIC 999", "TESS")
        _, kwargs = mock_lk.search_lightcurve.call_args
        assert kwargs["sector"] is None

    def test_single_cadence_gives_zero_baseline(self, mock_lk: MagicMock) -> None:
        lc = _make_lc_mock(n_cadences=1)
        _wire(mock_lk, lc)
        result = fetch_lightcurve("TIC 999", "TESS")
        assert result.provenance.time_baseline_days == 0.0

    def test_sectors_extracted_from_collection(self, mock_lk: MagicMock) -> None:
        lc1, lc2 = _make_lc_mock(sector_val=1), _make_lc_mock(sector_val=2)
        stitched = _make_lc_mock(sector_val=1, n_cadences=1000, baseline_days=54.0)
        coll = _make_collection_mock([lc1, lc2], stitched=stitched)
        search = _make_search_mock(coll, n_results=2)
        mock_lk.search_lightcurve.return_value = search
        result = fetch_lightcurve("TIC 999", "TESS")
        assert result.provenance.sectors_or_quarters == (1, 2)

    def test_fetched_at_is_iso_format(self, mock_lk: MagicMock) -> None:
        lc = _make_lc_mock()
        _wire(mock_lk, lc)
        result = fetch_lightcurve("TIC 999", "TESS")
        # Should parse without error as an ISO 8601 datetime
        import datetime as dt
        parsed = dt.datetime.fromisoformat(result.provenance.fetched_at)
        assert parsed.tzinfo is not None  # UTC timezone present

    def test_missing_lightkurve_raises_import_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Remove lightkurve from sys.modules entirely to simulate missing install
        monkeypatch.delitem(sys.modules, "lightkurve", raising=False)

        real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__  # type: ignore[union-attr]

        def _no_lk(name: str, *args: object, **kwargs: object) -> object:
            if name == "lightkurve":
                raise ImportError("No module named 'lightkurve'")
            return real_import(name, *args, **kwargs)  # type: ignore[call-arg]

        monkeypatch.setattr("builtins.__import__", _no_lk)
        with pytest.raises(ImportError, match="lightkurve is required"):
            fetch_lightcurve("TIC 999", "TESS")


# ---------------------------------------------------------------------------
# Live integration tests (excluded from CI)
# ---------------------------------------------------------------------------


@pytest.mark.integration_live
class TestFetchLightcurveIntegrationLive:
    def test_fetch_known_tess_target(self) -> None:
        # TOI-700 — a confirmed multi-planet system in TESS sector 1
        result = fetch_lightcurve("TIC 150428135", "TESS", sectors=(1,))
        assert isinstance(result, FetchResult)
        assert result.provenance.mission == "TESS"
        assert 1 in result.provenance.sectors_or_quarters
        assert result.provenance.n_cadences > 100
        assert result.provenance.time_baseline_days > 20.0

    def test_fetch_known_kepler_target(self) -> None:
        # Kepler-22 — the first confirmed Kepler planet in the habitable zone
        result = fetch_lightcurve("KIC 10593626", "Kepler")
        assert isinstance(result, FetchResult)
        assert result.provenance.mission == "Kepler"
        assert result.provenance.n_cadences > 1000
