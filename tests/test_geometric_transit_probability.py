"""Tests for Skills/geometric_transit_probability.py."""
from Skills.geometric_transit_probability import (
    TransitProbResult,
    compute_transit_probability,
    format_transit_prob_result,
)


class TestComputeTransitProbability:
    def test_returns_result_type(self):
        result = compute_transit_probability(10.0, stellar_density_gcc=1.4)
        assert isinstance(result, TransitProbResult)

    def test_flag_ok_with_density(self):
        result = compute_transit_probability(10.0, stellar_density_gcc=1.4)
        assert result.flag == "OK"

    def test_probability_between_0_and_1(self):
        result = compute_transit_probability(10.0, stellar_density_gcc=1.4)
        assert result.transit_probability is not None
        assert 0.0 < result.transit_probability <= 1.0

    def test_longer_period_lower_probability(self):
        r1 = compute_transit_probability(5.0, stellar_density_gcc=1.4)
        r2 = compute_transit_probability(50.0, stellar_density_gcc=1.4)
        assert r1.transit_probability > r2.transit_probability

    def test_flag_ok_with_stellar_radius(self):
        result = compute_transit_probability(10.0, stellar_radius_rsun=1.0)
        assert result.flag == "OK"

    def test_invalid_period_zero(self):
        result = compute_transit_probability(0.0, stellar_density_gcc=1.4)
        assert result.flag == "INVALID"

    def test_invalid_negative_period(self):
        result = compute_transit_probability(-1.0, stellar_density_gcc=1.4)
        assert result.flag == "INVALID"

    def test_no_inputs_flag(self):
        result = compute_transit_probability(10.0)
        assert result.flag == "INSUFFICIENT"
        assert result.transit_probability is None

    def test_rp_over_rs_increases_probability(self):
        r1 = compute_transit_probability(10.0, stellar_density_gcc=1.4, rp_over_rs=0.0)
        r2 = compute_transit_probability(10.0, stellar_density_gcc=1.4, rp_over_rs=0.1)
        if r1.transit_probability is not None and r2.transit_probability is not None:
            assert r2.transit_probability >= r1.transit_probability

    def test_semi_major_axis_rs_stored(self):
        result = compute_transit_probability(10.0, stellar_density_gcc=1.4)
        assert result.semi_major_axis_rs is not None
        assert result.semi_major_axis_rs > 0

    def test_period_stored(self):
        result = compute_transit_probability(10.0, stellar_density_gcc=1.4)
        assert result.period_days == 10.0

    def test_earth_like_probability(self):
        # Earth: P=365 d, ρ★≈1.4 g/cc → P_tr ≈ 0.005
        result = compute_transit_probability(365.0, stellar_density_gcc=1.4)
        assert result.transit_probability is not None
        assert result.transit_probability < 0.02


class TestFormatTransitProbResult:
    def test_returns_string(self):
        result = compute_transit_probability(10.0, stellar_density_gcc=1.4)
        s = format_transit_prob_result(result)
        assert isinstance(s, str)

    def test_contains_probability(self):
        result = compute_transit_probability(10.0, stellar_density_gcc=1.4)
        s = format_transit_prob_result(result)
        assert "Transit probability" in s or "transit" in s.lower()
