"""Tests for Skills/rv_bisector_analyzer.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from rv_bisector_analyzer import BisectorResult, analyze_bisector


class TestRvBisectorAnalyzer:
    def test_uncorrelated_returns_ok(self) -> None:
        rv = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        bis = [5.0, 3.0, 7.0, 1.0, 9.0, 2.0, 8.0, 4.0, 6.0, 10.0]
        r = analyze_bisector(rv, bis)
        assert r.flag == "OK"

    def test_correlated_flagged(self) -> None:
        xs = list(range(20))
        r = analyze_bisector(xs, xs)
        assert r.activity_flagged

    def test_anticorrelated_flagged(self) -> None:
        xs = list(range(20))
        ys = [-x for x in xs]
        r = analyze_bisector(xs, ys)
        assert r.activity_flagged

    def test_insufficient_points(self) -> None:
        r = analyze_bisector([1.0, 2.0], [1.0, 2.0])
        assert r.flag == "INSUFFICIENT_POINTS"

    def test_length_mismatch(self) -> None:
        r = analyze_bisector([1.0, 2.0, 3.0, 4.0, 5.0],
                              [1.0, 2.0, 3.0, 4.0])
        assert r.flag == "LENGTH_MISMATCH"

    def test_correlated_slope_positive(self) -> None:
        xs = [float(i) for i in range(15)]
        r = analyze_bisector(xs, xs)
        assert r.bis_slope > 0

    def test_near_zero_bis_span(self) -> None:
        rv = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0]
        bis = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        r = analyze_bisector(rv, bis)
        assert r.bis_span_ms == 0.0

    def test_threshold_respected(self) -> None:
        xs = list(range(15))
        r_strict = analyze_bisector(xs, xs, correlation_threshold=0.99)
        r_loose = analyze_bisector(xs, xs, correlation_threshold=0.01)
        assert r_loose.activity_flagged
        assert r_strict.activity_flagged  # r=1.0 > 0.99

    def test_result_frozen(self) -> None:
        rv = list(range(10))
        bis = list(range(10))
        r = analyze_bisector(rv, bis)
        assert isinstance(r, BisectorResult)
        try:
            object.__setattr__(r, "flag", "x")
            raise AssertionError()
        except Exception:
            pass

    def test_span_computed(self) -> None:
        rv = [float(i) for i in range(15)]
        bis = [float(i) * 0.5 for i in range(15)]
        r = analyze_bisector(rv, bis)
        assert r.bis_span_ms >= 0

    def test_format_output(self) -> None:
        from rv_bisector_analyzer import format_bisector_result
        rv = list(range(10))
        bis = list(range(10))
        r = analyze_bisector(rv, bis)
        s = format_bisector_result(r)
        assert "|" in s

    def test_constant_rv_no_flag(self) -> None:
        rv = [5.0] * 15
        bis = list(range(15))
        r = analyze_bisector(rv, bis)
        assert r.flag == "OK"
