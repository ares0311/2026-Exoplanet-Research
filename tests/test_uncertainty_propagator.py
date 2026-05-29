"""Tests for Skills/uncertainty_propagator.py."""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

import pytest
from uncertainty_propagator import (
    PropagationResult,
    format_propagation_result,
    propagate_uncertainty,
)


def _product(**kwargs):
    result = 1.0
    for v in kwargs.values():
        result *= v
    return result


def _square(x):
    return x ** 2


def test_single_param_product():
    result = propagate_uncertainty(
        lambda x: x * 2,
        params={"x": 5.0},
        uncertainties={"x": 0.1},
    )
    # d(2x)/dx = 2, sigma_out = 2 * 0.1 = 0.2
    assert result.output_uncertainty == pytest.approx(0.2, rel=0.01)


def test_two_params_quadrature():
    result = propagate_uncertainty(
        _product,
        params={"a": 2.0, "b": 3.0},
        uncertainties={"a": 0.1, "b": 0.1},
    )
    # sigma_a = b * sigma_a = 0.3, sigma_b = a * sigma_b = 0.2
    # total ~ sqrt(0.09 + 0.04) = 0.36
    assert result.output_uncertainty == pytest.approx(math.sqrt(0.09 + 0.04), rel=0.02)


def test_returns_propagation_result():
    result = propagate_uncertainty(
        lambda x: x, params={"x": 1.0}, uncertainties={"x": 0.1}
    )
    assert isinstance(result, PropagationResult)


def test_output_value_correct():
    result = propagate_uncertainty(
        lambda x: x * 3, params={"x": 4.0}, uncertainties={"x": 0.1}
    )
    assert result.output_value == pytest.approx(12.0, rel=1e-5)


def test_contributions_sorted_descending():
    result = propagate_uncertainty(
        _product,
        params={"a": 10.0, "b": 1.0},
        uncertainties={"a": 0.1, "b": 1.0},
    )
    contribs = [c[1] for c in result.contributions]
    assert contribs == sorted(contribs, reverse=True)


def test_large_uncertainty_flag():
    result = propagate_uncertainty(
        lambda x: x, params={"x": 1.0}, uncertainties={"x": 2.0}
    )
    assert result.flag == "LARGE_UNCERTAINTY"


def test_ok_flag_small_uncertainty():
    result = propagate_uncertainty(
        lambda x: x, params={"x": 100.0}, uncertainties={"x": 0.1}
    )
    assert result.flag == "OK"


def test_invalid_func_raises_gracefully():
    def bad(**kwargs):
        raise ValueError("oops")
    result = propagate_uncertainty(bad, params={"x": 1.0}, uncertainties={"x": 0.1})
    assert result.flag == "INVALID"


def test_zero_uncertainty_skipped():
    result = propagate_uncertainty(
        lambda x, y: x + y,
        params={"x": 1.0, "y": 2.0},
        uncertainties={"x": 0.0, "y": 0.1},
    )
    # Only y contributes
    assert len(result.contributions) == 1


def test_relative_uncertainty_computed():
    result = propagate_uncertainty(
        lambda x: x, params={"x": 10.0}, uncertainties={"x": 1.0}
    )
    assert result.relative_uncertainty == pytest.approx(0.1, rel=0.01)


def test_format_contains_status():
    result = propagate_uncertainty(
        lambda x: x, params={"x": 5.0}, uncertainties={"x": 0.1}
    )
    md = format_propagation_result(result)
    assert result.flag in md


def test_format_no_contributions_message():
    result = propagate_uncertainty(
        lambda x: x, params={"x": 1.0}, uncertainties={}
    )
    md = format_propagation_result(result)
    assert "No contributions" in md
