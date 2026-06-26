"""Tests for Skills/fetch_jwst_lc.py (Option A2: JWST light-curve extraction)."""
from __future__ import annotations

import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from fetch_jwst_lc import (
    _MJD_TO_BTJD_OFFSET,
    JwstLcResult,
    _normalize,
    fetch_jwst_lc,
)

# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

def _product_fn_x1dints(obsid: str) -> list[dict[str, str]]:
    return [{"filename": f"{obsid}_x1dints.fits", "dataURI": f"mast:JWST/{obsid}"}]


def _product_fn_calints(obsid: str) -> list[dict[str, str]]:
    return [{"filename": f"{obsid}_calints.fits", "dataURI": f"mast:JWST/{obsid}"}]


def _product_fn_empty(obsid: str) -> list[dict[str, str]]:
    return []


def _download_fn_no_file(uri: str, dest: Path) -> Path:
    raise FileNotFoundError("test stub: no download")


class _StubExtract:
    """Replace actual FITS extraction with deterministic arrays."""

    @staticmethod
    def x1dints(path: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = 50
        t = np.linspace(59800.0, 59801.0, n)
        f = np.ones(n) * 1e6 + np.random.default_rng(0).normal(0, 100, n)
        e = np.ones(n) * 100.0
        return t, f, e

    @staticmethod
    def calints(path: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = 30
        t = np.linspace(59800.0, 59800.5, n)
        f = np.ones(n) * 5e5 + np.random.default_rng(1).normal(0, 50, n)
        e = np.ones(n) * 50.0
        return t, f, e


def _make_fetch_with_stub(
    stub: _StubExtract,
    product_type: str = "x1dints",
) -> Any:
    """Return a fetch_jwst_lc that bypasses FITS file I/O."""
    def _pfn(obsid: str) -> list[dict[str, str]]:
        fn = f"{obsid}_{product_type}.fits"
        return [{"filename": fn, "dataURI": f"mast:JWST/{obsid}"}]

    def _dfn(uri: str, dest: Path) -> Path:
        # Return a sentinel path; the extractor will be monkeypatched
        return Path(f"/fake/{product_type}.fits")

    return _pfn, _dfn


# ---------------------------------------------------------------------------
# _normalize
# ---------------------------------------------------------------------------

def test_normalize_median_one() -> None:
    f = np.array([0.9, 1.0, 1.1, 1.0])
    e = np.array([0.1, 0.1, 0.1, 0.1])
    fn, en = _normalize(f, e)
    assert abs(float(np.median(fn)) - 1.0) < 1e-10


def test_normalize_errors_scaled() -> None:
    f = np.array([2.0, 2.0, 2.0])
    e = np.array([0.2, 0.2, 0.2])
    fn, en = _normalize(f, e)
    np.testing.assert_allclose(en, 0.1)


def test_normalize_zero_median_passthrough() -> None:
    f = np.array([0.0, 0.0, 0.0])
    e = np.array([1.0, 1.0, 1.0])
    fn, en = _normalize(f, e)
    np.testing.assert_array_equal(fn, f)


def test_normalize_nan_median_passthrough() -> None:
    f = np.array([float("nan"), float("nan")])
    e = np.array([1.0, 1.0])
    fn, en = _normalize(f, e)
    assert np.all(np.isnan(fn))


# ---------------------------------------------------------------------------
# MJD → BTJD conversion constant
# ---------------------------------------------------------------------------

def test_mjd_to_btjd_constant() -> None:
    # BTJD = BJD - 2457000; BJD ≈ MJD + 2400000.5; so offset = 2457000 - 2400000.5 = 56999.5
    expected = 2457000 - 2400000.5
    assert abs(_MJD_TO_BTJD_OFFSET - expected) < 1e-6


def test_btjd_in_plausible_range() -> None:
    mjd_jwst_launch = 59573.0  # 2021-12-25
    btjd = mjd_jwst_launch - _MJD_TO_BTJD_OFFSET
    # BTJD should be ~2573 days after TESS reference epoch (TESS launched April 2018)
    assert 2000 < btjd < 5000


# ---------------------------------------------------------------------------
# JwstLcResult dataclass
# ---------------------------------------------------------------------------

def test_result_roundtrip_json() -> None:
    r = JwstLcResult(
        obsid="abc", target_name="T1", time_btjd=[1.0, 2.0], flux_norm=[1.0, 0.99],
        flux_err_norm=[0.001, 0.001], instrument="NIRISS/SOSS",
        n_integrations=2, product_type="x1dints", warnings=[],
    )
    d = asdict(r)
    assert d["obsid"] == "abc"
    assert d["n_integrations"] == 2


def test_result_warnings_list() -> None:
    r = JwstLcResult(
        obsid="x", target_name="y", time_btjd=[], flux_norm=[], flux_err_norm=[],
        instrument="", n_integrations=0, product_type="calints", warnings=["warn1"],
    )
    assert r.warnings == ["warn1"]


# ---------------------------------------------------------------------------
# fetch_jwst_lc — no products
# ---------------------------------------------------------------------------

def test_fetch_no_products_returns_none() -> None:
    result = fetch_jwst_lc("obs99", product_fn=_product_fn_empty)
    assert result is None


def test_fetch_download_failure_returns_none(tmp_path: Path) -> None:
    result = fetch_jwst_lc(
        "obs88",
        cache_dir=tmp_path,
        product_fn=_product_fn_x1dints,
        download_fn=_download_fn_no_file,
    )
    assert result is None


# ---------------------------------------------------------------------------
# fetch_jwst_lc — with stubbed extraction (x1dints)
# ---------------------------------------------------------------------------

def test_fetch_x1dints_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import fetch_jwst_lc as mod

    stub = _StubExtract()
    monkeypatch.setattr(mod, "white_light_from_x1dints", stub.x1dints)

    # Make download return a fake path (extraction is monkeypatched)
    fake_path = tmp_path / "obs111_x1dints.fits"
    fake_path.touch()
    monkeypatch.setattr(
        mod, "_default_download_fn", lambda uri, dest: fake_path
    )

    result = fetch_jwst_lc(
        "obs111", target_name="WASP-39", instrument="NIRISS/SOSS",
        cache_dir=tmp_path, product_fn=_product_fn_x1dints,
    )
    assert result is not None
    assert result.product_type == "x1dints"
    assert result.n_integrations == 50
    assert abs(np.median(result.flux_norm) - 1.0) < 0.01


def test_fetch_calints_success(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import fetch_jwst_lc as mod

    stub = _StubExtract()
    monkeypatch.setattr(mod, "white_light_from_calints", stub.calints)

    fake_path = tmp_path / "obs222_calints.fits"
    fake_path.touch()
    monkeypatch.setattr(
        mod, "_default_download_fn", lambda uri, dest: fake_path
    )

    result = fetch_jwst_lc(
        "obs222",
        cache_dir=tmp_path,
        product_fn=_product_fn_calints,
    )
    assert result is not None
    assert result.product_type == "calints"
    assert result.n_integrations == 30


def test_fetch_prefers_x1dints(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import fetch_jwst_lc as mod

    stub = _StubExtract()
    monkeypatch.setattr(mod, "white_light_from_x1dints", stub.x1dints)

    fake_path = tmp_path / "obs333_x1dints.fits"
    fake_path.touch()
    monkeypatch.setattr(
        mod, "_default_download_fn", lambda uri, dest: fake_path
    )

    def _both(obsid: str) -> list[dict[str, str]]:
        return [
            {"filename": f"{obsid}_calints.fits", "dataURI": "mast:calints"},
            {"filename": f"{obsid}_x1dints.fits", "dataURI": "mast:x1dints"},
        ]

    result = fetch_jwst_lc("obs333", cache_dir=tmp_path, product_fn=_both)
    assert result is not None
    assert result.product_type == "x1dints"


def test_fetch_btjd_conversion(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import fetch_jwst_lc as mod

    stub = _StubExtract()
    monkeypatch.setattr(mod, "white_light_from_x1dints", stub.x1dints)

    fake_path = tmp_path / "obs444_x1dints.fits"
    fake_path.touch()
    monkeypatch.setattr(
        mod, "_default_download_fn", lambda uri, dest: fake_path
    )

    result = fetch_jwst_lc("obs444", cache_dir=tmp_path, product_fn=_product_fn_x1dints)
    assert result is not None
    # Stub returns MJD ~59800; BTJD = MJD - 56999.5 ≈ 2800 (within JWST operational era)
    btjd_vals = np.array(result.time_btjd)
    expected_btjd_approx = 59800.0 - _MJD_TO_BTJD_OFFSET
    assert abs(btjd_vals[0] - expected_btjd_approx) < 1.0
    assert 2000 < btjd_vals[0] < 5000  # plausible JWST era BTJD


def test_fetch_too_few_integrations_returns_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import fetch_jwst_lc as mod

    def _tiny_extract(path: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        t = np.array([59800.0, 59800.1])
        f = np.ones(2)
        e = np.ones(2)
        return t, f, e

    monkeypatch.setattr(mod, "white_light_from_x1dints", _tiny_extract)

    fake_path = tmp_path / "obs555_x1dints.fits"
    fake_path.touch()
    monkeypatch.setattr(mod, "_default_download_fn", lambda uri, dest: fake_path)

    result = fetch_jwst_lc("obs555", cache_dir=tmp_path, product_fn=_product_fn_x1dints)
    assert result is None


def test_fetch_uses_cache_on_second_call(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import fetch_jwst_lc as mod

    stub = _StubExtract()
    monkeypatch.setattr(mod, "white_light_from_x1dints", stub.x1dints)

    # Pre-create the cached file so download should NOT be called
    cached = tmp_path / "obs777_x1dints.fits"
    cached.touch()

    download_calls: list[str] = []

    def _counting_download(uri: str, dest: Path) -> Path:
        download_calls.append(uri)
        return dest

    monkeypatch.setattr(mod, "_default_download_fn", _counting_download)

    result = fetch_jwst_lc("obs777", cache_dir=tmp_path, product_fn=_product_fn_x1dints)
    assert result is not None
    assert download_calls == []  # should use cache
