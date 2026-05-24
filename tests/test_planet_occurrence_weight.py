"""Tests for Skills/planet_occurrence_weight.py."""
from Skills.planet_occurrence_weight import (
    OccurrenceWeightResult,
    compute_occurrence_weight,
    format_occurrence_weight_result,
)


class TestComputeOccurrenceWeight:
    def test_returns_result_type(self):
        result = compute_occurrence_weight(10.0, 0.1)
        assert isinstance(result, OccurrenceWeightResult)

    def test_flag_ok(self):
        result = compute_occurrence_weight(10.0, 0.1)
        assert result.flag == "OK"

    def test_high_snr_high_p_det(self):
        result = compute_occurrence_weight(20.0, 0.1)
        assert result.p_det is not None
        assert result.p_det > 0.9

    def test_low_snr_low_p_det(self):
        result = compute_occurrence_weight(1.0, 0.1)
        assert result.p_det is not None
        assert result.p_det < 0.5

    def test_weight_inverse_of_product(self):
        result = compute_occurrence_weight(20.0, 0.2, snr_threshold=7.1)
        assert result.weight is not None
        if result.p_det and result.p_transit:
            expected = 1.0 / (result.p_det * result.p_transit)
            assert abs(result.weight - min(expected, 1000.0)) < 0.01

    def test_step_model(self):
        r_above = compute_occurrence_weight(10.0, 0.1, snr_threshold=7.1, completeness_model="step")
        r_below = compute_occurrence_weight(5.0, 0.1, snr_threshold=7.1, completeness_model="step")
        assert r_above.p_det == 1.0
        assert r_below.p_det == 0.0

    def test_sigmoid_model_default(self):
        result = compute_occurrence_weight(7.1, 0.1, snr_threshold=7.1)
        # At threshold, sigmoid returns ~0.5
        assert result.p_det is not None
        assert 0.4 < result.p_det < 0.6

    def test_weight_capped_at_max(self):
        # p_det ≈ 0 → weight should be capped
        result = compute_occurrence_weight(0.1, 0.01, snr_threshold=7.1, completeness_model="step")
        assert result.weight == 1000.0
        assert not result.is_reliable

    def test_reliable_flag_set(self):
        result = compute_occurrence_weight(20.0, 0.5)
        assert result.is_reliable

    def test_none_snr_no_p_det(self):
        result = compute_occurrence_weight(None, 0.1)
        assert result.p_det is None
        assert result.weight is None

    def test_invalid_negative_snr(self):
        result = compute_occurrence_weight(-1.0, 0.1)
        assert result.flag == "INVALID"

    def test_invalid_transit_prob_zero(self):
        result = compute_occurrence_weight(10.0, 0.0)
        assert result.flag == "INVALID"

    def test_invalid_transit_prob_gt_1(self):
        result = compute_occurrence_weight(10.0, 1.5)
        assert result.flag == "INVALID"

    def test_none_transit_prob_no_weight(self):
        result = compute_occurrence_weight(10.0, None)
        assert result.weight is None


class TestFormatOccurrenceWeightResult:
    def test_returns_string(self):
        result = compute_occurrence_weight(10.0, 0.1)
        s = format_occurrence_weight_result(result)
        assert isinstance(s, str)

    def test_contains_weight(self):
        result = compute_occurrence_weight(10.0, 0.1)
        s = format_occurrence_weight_result(result)
        assert "weight" in s.lower() or "Weight" in s
