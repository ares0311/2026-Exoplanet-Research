"""Tests for Skills/transit_survey_planner.py."""
from Skills.transit_survey_planner import (
    SurveyPlanResult,
    SurveyWindow,
    format_survey_plan,
    plan_transit_windows,
)


class TestPlanTransitWindows:
    def _candidate(self, period=5.0, epoch=2460000.0, depth=5000.0, duration=2.0):
        return {
            "tic_id": 12345,
            "period_days": period,
            "epoch_bjd": epoch,
            "depth_ppm": depth,
            "duration_hours": duration,
        }

    def test_returns_survey_plan_result(self):
        c = self._candidate()
        result = plan_transit_windows([c], 2460000.0, 2460030.0)
        assert isinstance(result, SurveyPlanResult)

    def test_flag_ok_for_valid_input(self):
        c = self._candidate()
        result = plan_transit_windows([c], 2460000.0, 2460030.0)
        assert result.flag == "OK"

    def test_no_candidates_returns_empty(self):
        result = plan_transit_windows([], 2460000.0, 2460030.0)
        assert result.flag == "OK"
        assert result.n_windows == 0

    def test_invalid_time_range(self):
        c = self._candidate()
        result = plan_transit_windows([c], 2460030.0, 2460000.0)
        assert result.flag == "INVALID"

    def test_windows_within_time_range(self):
        c = self._candidate(period=5.0, epoch=2460000.0)
        result = plan_transit_windows([c], 2460000.0, 2460050.0)
        for w in result.windows:
            assert result.t_start <= w.mid_time <= result.t_end

    def test_multiple_candidates(self):
        c1 = self._candidate(period=5.0, epoch=2460000.0)
        c2 = self._candidate(period=7.0, epoch=2460001.0)
        result = plan_transit_windows([c1, c2], 2460000.0, 2460040.0)
        assert result.n_windows > 0

    def test_survey_window_fields(self):
        c = self._candidate(period=10.0, epoch=2460000.0)
        result = plan_transit_windows([c], 2460000.0, 2460050.0)
        assert len(result.windows) > 0
        w = result.windows[0]
        assert isinstance(w, SurveyWindow)
        assert w.flag in ("OK", "PARTIAL")
        assert w.ingress < w.mid_time < w.egress

    def test_ingress_egress_from_duration(self):
        c = self._candidate(period=10.0, epoch=2460000.0, duration=4.0)
        result = plan_transit_windows([c], 2460000.0, 2460020.0)
        if result.windows:
            w = result.windows[0]
            assert abs((w.egress - w.ingress) * 24.0 - 4.0) < 1e-4

    def test_max_windows_capped(self):
        c = self._candidate(period=0.01, epoch=2460000.0)
        result = plan_transit_windows([c], 2460000.0, 2460100.0, max_windows=10)
        assert result.n_windows <= 10

    def test_missing_period_skipped(self):
        c = {"tic_id": 1, "epoch_bjd": 2460000.0, "depth_ppm": 1000.0}
        result = plan_transit_windows([c], 2460000.0, 2460030.0)
        assert result.flag == "OK"
        assert result.n_windows == 0

    def test_missing_epoch_uses_default(self):
        c = {"tic_id": 1, "period_days": 5.0, "epoch_bjd": 2460000.0, "depth_ppm": 1000.0}
        result = plan_transit_windows([c], 2460000.0, 2460030.0)
        assert result.flag == "OK"

    def test_t_start_t_end_stored(self):
        c = self._candidate()
        result = plan_transit_windows([c], 2460000.0, 2460050.0)
        assert result.t_start == 2460000.0
        assert result.t_end == 2460050.0


class TestFormatSurveyPlan:
    def test_returns_string(self):
        c = {"tic_id": 1, "period_days": 5.0, "epoch_bjd": 2460000.0}
        result = plan_transit_windows([c], 2460000.0, 2460030.0)
        s = format_survey_plan(result)
        assert isinstance(s, str)
        assert "Survey" in s

    def test_invalid_flag_in_output(self):
        result = plan_transit_windows(
            [{"period_days": 5.0, "epoch_bjd": 0.0}], 2460030.0, 2460000.0
        )
        s = format_survey_plan(result)
        assert "INVALID" in s or result.flag == "INVALID"
