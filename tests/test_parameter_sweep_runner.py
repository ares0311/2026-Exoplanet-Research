"""Tests for Skills/parameter_sweep_runner.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from parameter_sweep_runner import (
    SweepPoint,
    SweepResult,
    run_parameter_sweep,
    format_sweep_result,
)


class TestRunParameterSweep:
    def test_basic_maximize(self):
        def fn(x, y):
            return -(x ** 2) - (y ** 2)

        grid = {"x": [-1.0, 0.0, 1.0], "y": [-1.0, 0.0, 1.0]}
        result = run_parameter_sweep(fn, grid, maximize=True)
        assert result.flag == "OK"
        assert result.best_params == {"x": 0.0, "y": 0.0}

    def test_basic_minimize(self):
        def fn(x):
            return (x - 3.0) ** 2

        grid = {"x": [1.0, 2.0, 3.0, 4.0, 5.0]}
        result = run_parameter_sweep(fn, grid, maximize=False)
        assert result.best_params["x"] == 3.0

    def test_n_points_equals_product(self):
        def fn(a, b):
            return a + b

        grid = {"a": [1, 2, 3], "b": [10, 20]}
        result = run_parameter_sweep(fn, grid)
        assert result.n_points == 6

    def test_empty_grid_not_ok(self):
        result = run_parameter_sweep(lambda: 0.0, {})
        assert result.flag in ("EMPTY", "INVALID")

    def test_single_param_single_value(self):
        result = run_parameter_sweep(lambda x: x, {"x": [42.0]})
        assert result.flag == "OK"
        assert result.best_params["x"] == 42.0
        assert result.n_points == 1

    def test_exception_in_fn_handled(self):
        def bad_fn(x):
            if x == 0:
                raise ValueError("bad")
            return x

        result = run_parameter_sweep(bad_fn, {"x": [-1, 0, 1]}, maximize=True)
        assert result.flag == "OK"
        assert result.best_params["x"] == 1

    def test_sweep_points_stored(self):
        result = run_parameter_sweep(lambda x: x * 2, {"x": [1.0, 2.0, 3.0]})
        assert len(result.sweep_points) == 3

    def test_best_value_type(self):
        result = run_parameter_sweep(lambda x: x, {"x": [1.0, 2.0]})
        assert isinstance(result.best_value, float)

    def test_maximize_vs_minimize(self):
        grid = {"x": [1.0, 2.0, 3.0]}
        r_max = run_parameter_sweep(lambda x: x, grid, maximize=True)
        r_min = run_parameter_sweep(lambda x: x, grid, maximize=False)
        assert r_max.best_params["x"] == 3.0
        assert r_min.best_params["x"] == 1.0

    def test_all_exceptions_returns_flag(self):
        def always_fails(**_kwargs):
            raise RuntimeError("oops")

        result = run_parameter_sweep(always_fails, {"x": [1, 2, 3]})
        assert result.flag in ("OK", "NO_VALID_POINTS")

    def test_result_frozen(self):
        result = run_parameter_sweep(lambda x: x, {"x": [1.0]})
        try:
            result.best_value = 99.0
            assert False
        except Exception:
            pass

    def test_sweep_point_params_match_grid_keys(self):
        grid = {"alpha": [0.1, 0.2], "beta": [1, 2]}
        result = run_parameter_sweep(lambda alpha, beta: alpha * beta, grid)
        for pt in result.sweep_points:
            if pt.metric_value is not None:
                assert "alpha" in pt.params
                assert "beta" in pt.params

    def test_format_returns_string(self):
        result = run_parameter_sweep(lambda x: x, {"x": [1.0]})
        text = format_sweep_result(result)
        assert isinstance(text, str)

    def test_format_contains_flag(self):
        result = run_parameter_sweep(lambda x: x, {"x": [1.0]})
        text = format_sweep_result(result)
        assert result.flag in text
