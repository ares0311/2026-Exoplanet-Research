"""Tests for exo_toolkit.clean.

clean_lightcurve() only calls methods on the LightCurve it receives —
it never imports lightkurve itself.  Tests therefore pass plain MagicMock
objects as the light curve argument; no sys.modules patching is needed.
"""
from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock, call

import pydantic
import pytest

from exo_toolkit.clean import CleanProvenance, CleanResult, clean_lightcurve

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_lc(n: int = 1000) -> MagicMock:
    """Minimal LightCurve mock: every cleaning step returns the same object."""
    lc = MagicMock()
    lc.time.__len__.return_value = n
    mask = MagicMock()
    mask.sum.return_value = 0
    lc.remove_nans.return_value = lc
    lc.remove_outliers.return_value = (lc, mask)
    lc.normalize.return_value = lc
    lc.flatten.return_value = lc
    return lc


def _chain_lc(
    *,
    n_raw: int = 1000,
    n_nan: int = 10,
    n_outlier: int = 5,
) -> tuple[MagicMock, MagicMock, MagicMock, MagicMock, MagicMock, MagicMock]:
    """Full mock chain with distinct objects at each cleaning step.

    Returns (lc_raw, lc_post_nan, lc_post_outlier, lc_post_norm, lc_post_flat, mask).
    """
    lc_raw = MagicMock(name="lc_raw")
    lc_nan = MagicMock(name="lc_nan")
    lc_out = MagicMock(name="lc_out")
    lc_norm = MagicMock(name="lc_norm")
    lc_flat = MagicMock(name="lc_flat")
    mask = MagicMock(name="mask")

    n_after_nan = n_raw - n_nan
    n_after_out = n_after_nan - n_outlier

    lc_raw.time.__len__.return_value = n_raw
    lc_raw.remove_nans.return_value = lc_nan

    lc_nan.time.__len__.return_value = n_after_nan
    mask.sum.return_value = n_outlier
    lc_nan.remove_outliers.return_value = (lc_out, mask)
    lc_nan.normalize.return_value = lc_norm  # used when sigma_clip=None
    lc_nan.flatten.return_value = lc_flat    # used when sigma_clip=None, normalize=False

    lc_out.time.__len__.return_value = n_after_out
    lc_out.normalize.return_value = lc_norm
    lc_out.flatten.return_value = lc_flat    # used when normalize=False

    lc_norm.time.__len__.return_value = n_after_out
    lc_norm.flatten.return_value = lc_flat

    lc_flat.time.__len__.return_value = n_after_out

    return lc_raw, lc_nan, lc_out, lc_norm, lc_flat, mask


# ---------------------------------------------------------------------------
# CleanProvenance
# ---------------------------------------------------------------------------


class TestCleanProvenance:
    def test_valid_full_construction(self) -> None:
        p = CleanProvenance(
            n_cadences_raw=1000,
            n_cadences_cleaned=985,
            n_removed_nan=10,
            n_removed_outlier=5,
            sigma_clip_sigma=5.0,
            window_length=401,
            normalized=True,
            flattened=True,
        )
        assert p.n_cadences_raw == 1000
        assert p.sigma_clip_sigma == 5.0
        assert p.window_length == 401

    def test_no_sigma_clip(self) -> None:
        p = CleanProvenance(
            n_cadences_raw=1000,
            n_cadences_cleaned=990,
            n_removed_nan=10,
            n_removed_outlier=0,
            sigma_clip_sigma=None,
            window_length=401,
            normalized=True,
            flattened=True,
        )
        assert p.sigma_clip_sigma is None

    def test_no_flatten(self) -> None:
        p = CleanProvenance(
            n_cadences_raw=1000,
            n_cadences_cleaned=990,
            n_removed_nan=10,
            n_removed_outlier=0,
            sigma_clip_sigma=5.0,
            window_length=None,
            normalized=True,
            flattened=False,
        )
        assert p.window_length is None
        assert not p.flattened

    def test_frozen(self) -> None:
        p = CleanProvenance(
            n_cadences_raw=100,
            n_cadences_cleaned=100,
            n_removed_nan=0,
            n_removed_outlier=0,
            sigma_clip_sigma=5.0,
            window_length=401,
            normalized=True,
            flattened=True,
        )
        with pytest.raises(pydantic.ValidationError):
            p.n_cadences_raw = 200  # type: ignore[misc]

    def test_negative_n_removed_nan_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            CleanProvenance(
                n_cadences_raw=100,
                n_cadences_cleaned=100,
                n_removed_nan=-1,
                n_removed_outlier=0,
                sigma_clip_sigma=5.0,
                window_length=401,
                normalized=True,
                flattened=True,
            )

    def test_negative_n_cadences_raw_rejected(self) -> None:
        with pytest.raises(pydantic.ValidationError):
            CleanProvenance(
                n_cadences_raw=-1,
                n_cadences_cleaned=0,
                n_removed_nan=0,
                n_removed_outlier=0,
                sigma_clip_sigma=None,
                window_length=None,
                normalized=False,
                flattened=False,
            )

    def test_zero_cadences_cleaned_is_valid(self) -> None:
        p = CleanProvenance(
            n_cadences_raw=10,
            n_cadences_cleaned=0,
            n_removed_nan=10,
            n_removed_outlier=0,
            sigma_clip_sigma=5.0,
            window_length=401,
            normalized=True,
            flattened=True,
        )
        assert p.n_cadences_cleaned == 0


# ---------------------------------------------------------------------------
# CleanResult
# ---------------------------------------------------------------------------


class TestCleanResult:
    def _prov(self) -> CleanProvenance:
        return CleanProvenance(
            n_cadences_raw=1000,
            n_cadences_cleaned=985,
            n_removed_nan=10,
            n_removed_outlier=5,
            sigma_clip_sigma=5.0,
            window_length=401,
            normalized=True,
            flattened=True,
        )

    def test_construction(self) -> None:
        fake_lc = object()
        r = CleanResult(light_curve=fake_lc, provenance=self._prov())
        assert r.light_curve is fake_lc

    def test_frozen(self) -> None:
        r = CleanResult(light_curve=object(), provenance=self._prov())
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.light_curve = object()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# clean_lightcurve
# ---------------------------------------------------------------------------


class TestCleanLightcurve:
    def test_returns_clean_result(self) -> None:
        result = clean_lightcurve(_simple_lc())
        assert isinstance(result, CleanResult)
        assert isinstance(result.provenance, CleanProvenance)

    def test_light_curve_is_flattened_output(self) -> None:
        lc_raw, _, _, _, lc_flat, _ = _chain_lc()
        result = clean_lightcurve(lc_raw)
        assert result.light_curve is lc_flat

    def test_n_cadences_raw_correct(self) -> None:
        lc_raw, *_ = _chain_lc(n_raw=1200, n_nan=0, n_outlier=0)
        result = clean_lightcurve(lc_raw)
        assert result.provenance.n_cadences_raw == 1200

    def test_n_removed_nan_correct(self) -> None:
        lc_raw, *_ = _chain_lc(n_raw=1000, n_nan=15, n_outlier=0)
        result = clean_lightcurve(lc_raw)
        assert result.provenance.n_removed_nan == 15

    def test_n_removed_outlier_correct(self) -> None:
        lc_raw, *_ = _chain_lc(n_raw=1000, n_nan=0, n_outlier=8)
        result = clean_lightcurve(lc_raw)
        assert result.provenance.n_removed_outlier == 8

    def test_n_cadences_cleaned_correct(self) -> None:
        lc_raw, *_ = _chain_lc(n_raw=1000, n_nan=10, n_outlier=5)
        result = clean_lightcurve(lc_raw)
        # lc_flat.time.__len__ returns 985
        assert result.provenance.n_cadences_cleaned == 985

    def test_remove_nans_always_called(self) -> None:
        lc = _simple_lc()
        clean_lightcurve(lc)
        lc.remove_nans.assert_called_once_with()

    def test_remove_outliers_called_with_sigma(self) -> None:
        lc = _simple_lc()
        clean_lightcurve(lc, sigma_clip=4.0)
        lc.remove_outliers.assert_called_once_with(sigma=4.0, return_mask=True)

    def test_remove_outliers_not_called_when_disabled(self) -> None:
        lc = _simple_lc()
        clean_lightcurve(lc, sigma_clip=None)
        lc.remove_outliers.assert_not_called()

    def test_n_removed_outlier_zero_when_sigma_clip_none(self) -> None:
        lc = _simple_lc()
        result = clean_lightcurve(lc, sigma_clip=None)
        assert result.provenance.n_removed_outlier == 0

    def test_normalize_called_when_enabled(self) -> None:
        lc = _simple_lc()
        clean_lightcurve(lc, normalize=True)
        lc.normalize.assert_called_once_with()

    def test_normalize_not_called_when_disabled(self) -> None:
        lc = _simple_lc()
        clean_lightcurve(lc, normalize=False)
        lc.normalize.assert_not_called()

    def test_flatten_called_with_window_length(self) -> None:
        lc = _simple_lc()
        clean_lightcurve(lc, window_length=501)
        lc.flatten.assert_called_once_with(window_length=501)

    def test_flatten_not_called_when_disabled(self) -> None:
        lc = _simple_lc()
        clean_lightcurve(lc, flatten=False)
        lc.flatten.assert_not_called()

    def test_even_window_length_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="odd"):
            clean_lightcurve(_simple_lc(), window_length=400)

    def test_sigma_clip_sigma_recorded_in_provenance(self) -> None:
        lc = _simple_lc()
        result = clean_lightcurve(lc, sigma_clip=3.5)
        assert result.provenance.sigma_clip_sigma == pytest.approx(3.5)

    def test_sigma_clip_none_recorded_in_provenance(self) -> None:
        lc = _simple_lc()
        result = clean_lightcurve(lc, sigma_clip=None)
        assert result.provenance.sigma_clip_sigma is None

    def test_window_length_recorded_when_flatten_true(self) -> None:
        lc = _simple_lc()
        result = clean_lightcurve(lc, window_length=301)
        assert result.provenance.window_length == 301

    def test_window_length_none_when_flatten_false(self) -> None:
        lc = _simple_lc()
        result = clean_lightcurve(lc, flatten=False)
        assert result.provenance.window_length is None

    def test_normalized_flag_true(self) -> None:
        result = clean_lightcurve(_simple_lc(), normalize=True)
        assert result.provenance.normalized is True

    def test_normalized_flag_false(self) -> None:
        result = clean_lightcurve(_simple_lc(), normalize=False)
        assert result.provenance.normalized is False

    def test_flattened_flag_true(self) -> None:
        result = clean_lightcurve(_simple_lc(), flatten=True)
        assert result.provenance.flattened is True

    def test_flattened_flag_false(self) -> None:
        result = clean_lightcurve(_simple_lc(), flatten=False)
        assert result.provenance.flattened is False

    def test_normalize_false_flatten_true_skips_normalize(self) -> None:
        # Without normalize, flatten is called directly on post-outlier lc
        lc_raw, _, lc_out, _, lc_flat, _ = _chain_lc()
        result = clean_lightcurve(lc_raw, normalize=False, flatten=True)
        assert result.light_curve is lc_flat
        lc_out.normalize.assert_not_called()
        lc_out.flatten.assert_called_once_with(window_length=401)

    def test_normalize_true_flatten_false_ends_at_normalize(self) -> None:
        lc_raw, _, _, lc_norm, _, _ = _chain_lc()
        result = clean_lightcurve(lc_raw, normalize=True, flatten=False)
        assert result.light_curve is lc_norm
        lc_norm.flatten.assert_not_called()

    def test_sigma_clip_none_normalize_true_flatten_true(self) -> None:
        # No outlier removal: normalize and flatten called on post-nan lc
        lc_raw, lc_nan, _, lc_norm, lc_flat, _ = _chain_lc()
        result = clean_lightcurve(lc_raw, sigma_clip=None)
        assert result.light_curve is lc_flat
        lc_nan.remove_outliers.assert_not_called()
        lc_nan.normalize.assert_called_once()

    def test_all_steps_disabled_returns_post_nan_lc(self) -> None:
        lc_raw, lc_nan, *_ = _chain_lc()
        result = clean_lightcurve(
            lc_raw, sigma_clip=None, normalize=False, flatten=False
        )
        assert result.light_curve is lc_nan

    def test_default_sigma_is_five(self) -> None:
        lc = _simple_lc()
        clean_lightcurve(lc)
        args = lc.remove_outliers.call_args
        assert args == call(sigma=5.0, return_mask=True)

    def test_default_window_length_is_401(self) -> None:
        lc = _simple_lc()
        clean_lightcurve(lc)
        assert lc.flatten.call_args == call(window_length=401)

    def test_step_order_nan_then_outlier_then_normalize_then_flatten(
        self,
    ) -> None:
        lc_raw, lc_nan, lc_out, lc_norm, lc_flat, _ = _chain_lc()
        result = clean_lightcurve(lc_raw)
        # Verify each step was called on the correct preceding object
        lc_raw.remove_nans.assert_called_once()
        lc_nan.remove_outliers.assert_called_once()
        lc_out.normalize.assert_called_once()
        lc_norm.flatten.assert_called_once()
        assert result.light_curve is lc_flat
