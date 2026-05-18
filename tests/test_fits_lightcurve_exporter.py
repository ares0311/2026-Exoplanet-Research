"""Tests for fits_lightcurve_exporter.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))
from fits_lightcurve_exporter import (
    export_lightcurve_to_fits,
    format_fits_export_result,
)


def _mock_write(output_path, header, columns, overwrite):
    """Write a tiny sentinel file so os.path.getsize works."""
    Path(output_path).write_bytes(b"FITS" + b"\x00" * 100)


class TestExportLightcurveToFITS:
    def test_basic_ok(self, tmp_path):
        out = str(tmp_path / "lc.fits")
        time = [1.0, 2.0, 3.0]
        flux = [1.0, 0.99, 1.0]
        r = export_lightcurve_to_fits(time, flux, out, write_fn=_mock_write)
        assert r.flag == "OK"
        assert r.n_cadences == 3

    def test_columns_written_basic(self, tmp_path):
        out = str(tmp_path / "lc.fits")
        r = export_lightcurve_to_fits([1.0, 2.0], [1.0, 0.99], out, write_fn=_mock_write)
        assert "TIME" in r.columns_written
        assert "FLUX" in r.columns_written

    def test_optional_flux_err(self, tmp_path):
        out = str(tmp_path / "lc.fits")
        r = export_lightcurve_to_fits(
            [1.0, 2.0], [1.0, 0.99], out,
            flux_err=[0.01, 0.01], write_fn=_mock_write
        )
        assert "FLUX_ERR" in r.columns_written

    def test_optional_quality(self, tmp_path):
        out = str(tmp_path / "lc.fits")
        r = export_lightcurve_to_fits(
            [1.0, 2.0], [1.0, 0.99], out,
            quality=[0, 0], write_fn=_mock_write
        )
        assert "QUALITY" in r.columns_written

    def test_centroid_columns(self, tmp_path):
        out = str(tmp_path / "lc.fits")
        r = export_lightcurve_to_fits(
            [1.0, 2.0], [1.0, 0.99], out,
            centroid_col=[10.0, 10.1], centroid_row=[20.0, 20.1],
            write_fn=_mock_write
        )
        assert "CENTROID_COL" in r.columns_written
        assert "CENTROID_ROW" in r.columns_written

    def test_empty_time_invalid(self, tmp_path):
        out = str(tmp_path / "lc.fits")
        r = export_lightcurve_to_fits([], [], out, write_fn=_mock_write)
        assert r.flag == "INVALID"

    def test_mismatched_lengths_invalid(self, tmp_path):
        out = str(tmp_path / "lc.fits")
        r = export_lightcurve_to_fits([1.0, 2.0], [1.0], out, write_fn=_mock_write)
        assert r.flag == "INVALID"

    def test_mismatched_flux_err_invalid(self, tmp_path):
        out = str(tmp_path / "lc.fits")
        r = export_lightcurve_to_fits(
            [1.0, 2.0], [1.0, 1.0], out, flux_err=[0.01], write_fn=_mock_write
        )
        assert r.flag == "INVALID"

    def test_write_error_flag(self, tmp_path):
        def bad_write(*args, **kwargs):
            raise OSError("disk full")
        out = str(tmp_path / "lc.fits")
        r = export_lightcurve_to_fits([1.0, 2.0], [1.0, 0.99], out, write_fn=bad_write)
        assert r.flag == "WRITE_ERROR"

    def test_header_keys_present(self, tmp_path):
        out = str(tmp_path / "lc.fits")
        r = export_lightcurve_to_fits(
            [1.0, 2.0], [1.0, 0.99], out, tic_id=12345, sector=7,
            write_fn=_mock_write
        )
        assert "TIC_ID" in r.header_keys
        assert "SECTOR" in r.header_keys

    def test_extra_header(self, tmp_path):
        out = str(tmp_path / "lc.fits")
        r = export_lightcurve_to_fits(
            [1.0, 2.0], [1.0, 0.99], out,
            extra_header={"MY_KEY": "test"}, write_fn=_mock_write
        )
        assert "MY_KEY" in r.header_keys

    def test_file_size_positive(self, tmp_path):
        out = str(tmp_path / "lc.fits")
        r = export_lightcurve_to_fits([1.0, 2.0], [1.0, 0.99], out, write_fn=_mock_write)
        assert r.file_size_bytes > 0

    def test_result_frozen(self, tmp_path):
        out = str(tmp_path / "lc.fits")
        r = export_lightcurve_to_fits([1.0, 2.0], [1.0, 0.99], out, write_fn=_mock_write)
        try:
            r.flag = "x"  # type: ignore[misc]
            raise AssertionError()
        except (AttributeError, TypeError):
            pass


class TestFormatFITSExportResult:
    def test_returns_string(self, tmp_path):
        out = str(tmp_path / "lc.fits")
        r = export_lightcurve_to_fits([1.0, 2.0], [1.0, 0.99], out, write_fn=_mock_write)
        assert isinstance(format_fits_export_result(r), str)

    def test_contains_flag(self, tmp_path):
        out = str(tmp_path / "lc.fits")
        r = export_lightcurve_to_fits([1.0, 2.0], [1.0, 0.99], out, write_fn=_mock_write)
        s = format_fits_export_result(r)
        assert r.flag in s
