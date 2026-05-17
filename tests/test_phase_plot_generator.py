"""Tests for Skills.phase_plot_generator."""
from __future__ import annotations

from pathlib import Path

import pytest
from Skills.phase_plot_generator import (
    PhasePlotResult,
    generate_phase_plot,
    format_plot_result,
)


def _lc(n: int = 200, period: float = 5.0, epoch: float = 2458000.0):
    dt = 2.0 / 1440.0
    time = [epoch - 10.0 + i * dt for i in range(n)]
    flux = [1.0] * n
    return time, flux


def _noop_plot_fn(tic_id, phase, flux, *, bin_phase, bin_flux,
                  period_days, epoch_bjd, title, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(b"PNG")
    return output_path


class TestGeneratePhasePlot:
    def test_returns_result(self, tmp_path: Path) -> None:
        t, f = _lc()
        r = generate_phase_plot(1, t, f, 5.0, 2458000.0,
                                output_path=tmp_path / "out.png",
                                plot_fn=_noop_plot_fn)
        assert isinstance(r, PhasePlotResult)

    def test_success_with_plot_fn(self, tmp_path: Path) -> None:
        t, f = _lc()
        r = generate_phase_plot(1, t, f, 5.0, 2458000.0,
                                output_path=tmp_path / "out.png",
                                plot_fn=_noop_plot_fn)
        assert r.success is True

    def test_output_path_stored(self, tmp_path: Path) -> None:
        t, f = _lc()
        out = tmp_path / "p.png"
        r = generate_phase_plot(1, t, f, 5.0, 2458000.0,
                                output_path=out, plot_fn=_noop_plot_fn)
        assert r.output_path == out

    def test_n_points_stored(self, tmp_path: Path) -> None:
        t, f = _lc(100)
        r = generate_phase_plot(1, t, f, 5.0, 2458000.0,
                                output_path=tmp_path / "out.png",
                                plot_fn=_noop_plot_fn)
        assert r.n_points == 100

    def test_zero_period_fails(self) -> None:
        t, f = _lc()
        r = generate_phase_plot(1, t, f, 0.0, 2458000.0)
        assert r.success is False

    def test_empty_lc_fails(self) -> None:
        r = generate_phase_plot(1, [], [], 5.0, 2458000.0)
        assert r.success is False

    def test_tic_id_stored(self, tmp_path: Path) -> None:
        t, f = _lc()
        r = generate_phase_plot(42, t, f, 5.0, 2458000.0,
                                output_path=tmp_path / "out.png",
                                plot_fn=_noop_plot_fn)
        assert r.tic_id == 42

    def test_period_stored(self, tmp_path: Path) -> None:
        t, f = _lc()
        r = generate_phase_plot(1, t, f, 7.3, 2458000.0,
                                output_path=tmp_path / "out.png",
                                plot_fn=_noop_plot_fn)
        assert r.period_days == pytest.approx(7.3)

    def test_epoch_stored(self, tmp_path: Path) -> None:
        t, f = _lc()
        r = generate_phase_plot(1, t, f, 5.0, 2458001.5,
                                output_path=tmp_path / "out.png",
                                plot_fn=_noop_plot_fn)
        assert r.epoch_bjd == pytest.approx(2458001.5)

    def test_matplotlib_absent_returns_false(self) -> None:
        t, f = _lc()
        r = generate_phase_plot(1, t, f, 5.0, 2458000.0,
                                plot_fn=lambda *a, **kw: None)
        assert r.success is False

    def test_exception_in_plot_fn_returns_failure(self) -> None:
        def bad_fn(*a, **kw):
            raise RuntimeError("plot error")
        t, f = _lc()
        r = generate_phase_plot(1, t, f, 5.0, 2458000.0, plot_fn=bad_fn)
        assert r.success is False
        assert "plot error" in r.message


class TestFormatPlotResult:
    def test_returns_string(self, tmp_path: Path) -> None:
        t, f = _lc()
        r = generate_phase_plot(1, t, f, 5.0, 2458000.0,
                                output_path=tmp_path / "out.png",
                                plot_fn=_noop_plot_fn)
        assert isinstance(format_plot_result(r), str)

    def test_contains_tic_id(self, tmp_path: Path) -> None:
        t, f = _lc()
        r = generate_phase_plot(99, t, f, 5.0, 2458000.0,
                                output_path=tmp_path / "out.png",
                                plot_fn=_noop_plot_fn)
        assert "99" in format_plot_result(r)
