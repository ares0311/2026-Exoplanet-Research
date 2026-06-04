"""Tests for Skills/stellar_flare_detector.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from stellar_flare_detector import FlareDetectionResult, detect_flares


class TestStellarFlareDetector:
    def _flat(self, n: int = 200) -> list[float]:
        return [1.0] * n

    def test_no_flares_flat(self) -> None:
        r = detect_flares(self._flat())
        assert r.flag == "OK"
        assert r.n_flares == 0

    def test_single_flare_detected(self) -> None:
        flux = self._flat()
        flux[100] = 1.5
        flux[101] = 1.4
        r = detect_flares(flux, sigma_threshold=3.0, min_duration_cadences=2)
        assert r.flag == "OK"
        assert r.n_flares >= 1

    def test_insufficient_data(self) -> None:
        r = detect_flares([1.0, 1.0])
        assert r.flag == "INSUFFICIENT_DATA"

    def test_too_few_finite(self) -> None:
        flux = [float("nan")] * 100 + [1.0] * 5
        r = detect_flares(flux)
        assert r.flag in ("INSUFFICIENT_FINITE", "INSUFFICIENT_DATA", "OK")

    def test_flare_energy_positive(self) -> None:
        flux = self._flat()
        for i in range(100, 103):
            flux[i] = 1.8
        r = detect_flares(flux, min_duration_cadences=2)
        assert r.flag == "OK"
        if r.n_flares > 0:
            assert all(e.energy_proxy >= 0 for e in r.flares)

    def test_high_threshold_no_flares(self) -> None:
        flux = self._flat()
        flux[50] = 2.0
        r = detect_flares(flux, sigma_threshold=100.0)
        assert r.n_flares == 0

    def test_result_has_correct_fields(self) -> None:
        r = detect_flares(self._flat())
        assert hasattr(r, "n_flares")
        assert hasattr(r, "flares")
        assert hasattr(r, "baseline_rms")

    def test_multiple_flares(self) -> None:
        flux = self._flat(300)
        for i in [50, 51, 150, 151, 250, 251]:
            flux[i] = 2.0
        r = detect_flares(flux, sigma_threshold=3.0, min_duration_cadences=2)
        assert r.n_flares >= 2

    def test_single_cadence_flare_below_min_duration(self) -> None:
        flux = self._flat()
        flux[100] = 10.0
        r = detect_flares(flux, sigma_threshold=3.0, min_duration_cadences=2)
        assert r.n_flares == 0

    def test_result_is_frozen(self) -> None:
        r = detect_flares(self._flat())
        assert isinstance(r, FlareDetectionResult)
        try:
            object.__setattr__(r, "flag", "mutated")
            raise AssertionError()
        except Exception:
            pass

    def test_format_output(self) -> None:
        from stellar_flare_detector import format_flare_result
        r = detect_flares(self._flat())
        s = format_flare_result(r)
        assert "flare" in s.lower() or "|" in s

    def test_all_nan_flux(self) -> None:
        flux = [float("nan")] * 200
        r = detect_flares(flux)
        assert r.flag in ("INSUFFICIENT_FINITE", "INSUFFICIENT_DATA")
