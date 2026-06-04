"""Tests for Skills/tess_fov_position_checker.py."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from tess_fov_position_checker import check_fov_position, format_fov_position_result


class TestCheckFovPosition:
    def test_central_position(self) -> None:
        r = check_fov_position(offset_from_centre_deg=2.0)
        assert r.flag == "OK"
        assert r.position_quality == "CENTRAL"
        assert r.snr_penalty_fraction == 0.0

    def test_nominal_position(self) -> None:
        r = check_fov_position(offset_from_centre_deg=8.0)
        assert r.flag == "OK"
        assert r.position_quality == "NOMINAL"

    def test_edge_position(self) -> None:
        r = check_fov_position(offset_from_centre_deg=11.5)
        assert r.position_quality == "EDGE"
        assert r.snr_penalty_fraction > 0.0

    def test_full_penalty_at_edge(self) -> None:
        # Very close to edge (within full penalty zone)
        r = check_fov_position(offset_from_centre_deg=11.8)
        assert r.snr_penalty_fraction == 0.15

    def test_outside_fov(self) -> None:
        r = check_fov_position(offset_from_centre_deg=13.0)
        assert r.flag == "OUTSIDE_FOV"
        assert r.position_quality == "OUTSIDE_FOV"

    def test_gap_position(self) -> None:
        # Exactly at camera edge = in gap
        r = check_fov_position(offset_from_centre_deg=12.0)
        assert r.in_inter_camera_gap

    def test_invalid_offset_negative(self) -> None:
        r = check_fov_position(offset_from_centre_deg=-1.0)
        assert r.flag == "INVALID_OFFSET"

    def test_penalty_increases_toward_edge(self) -> None:
        r_inner = check_fov_position(offset_from_centre_deg=9.0)
        r_outer = check_fov_position(offset_from_centre_deg=11.0)
        assert r_outer.snr_penalty_fraction >= r_inner.snr_penalty_fraction

    def test_custom_half_width(self) -> None:
        r = check_fov_position(offset_from_centre_deg=8.0, camera_half_width_deg=8.5)
        assert r.flag == "OK"

    def test_offset_from_edge_computed(self) -> None:
        r = check_fov_position(offset_from_centre_deg=5.0)
        assert abs(r.offset_from_edge_deg - 7.0) < 0.01

    def test_no_gap_at_centre(self) -> None:
        r = check_fov_position(offset_from_centre_deg=3.0)
        assert not r.in_inter_camera_gap

    def test_format_output(self) -> None:
        r = check_fov_position(offset_from_centre_deg=5.0)
        s = format_fov_position_result(r)
        assert "|" in s
        assert "penalty" in s.lower() or "SNR" in s
