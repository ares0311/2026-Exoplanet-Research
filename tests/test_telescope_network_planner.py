"""Tests for Skills/telescope_network_planner.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from telescope_network_planner import (  # noqa: E402
    NetworkPlanResult,
    SiteObservability,
    format_network_plan,
    plan_network_coverage,
)

SITES = [
    {"name": "Site-A", "lat_deg": 30.0, "lon_deg": -70.0},
    {"name": "Site-B", "lat_deg": 28.0, "lon_deg": 150.0},
    {"name": "Site-C", "lat_deg": 50.0, "lon_deg": 10.0},
]


def test_returns_dataclass():
    r = plan_network_coverage(SITES, "2026-06-01 22:00", "2026-06-02 00:30", "2026-06-01")
    assert isinstance(r, NetworkPlanResult)


def test_n_sites_total():
    r = plan_network_coverage(SITES, "2026-06-01 22:00", "2026-06-02 00:30", "2026-06-01")
    assert r.n_sites_total == len(SITES)


def test_no_coverage_flag_empty():
    r = plan_network_coverage([], "2026-06-01 22:00", "2026-06-02 00:30", "2026-06-01")
    assert r.flag == "NO_COVERAGE"
    assert r.n_sites_viable == 0


def test_coverage_fraction_range():
    r = plan_network_coverage(SITES, "2026-06-01 22:00", "2026-06-02 00:30", "2026-06-01")
    assert 0.0 <= r.coverage_fraction <= 1.0


def test_viable_sites_are_site_observability():
    r = plan_network_coverage(SITES, "2026-06-01 22:00", "2026-06-02 00:30", "2026-06-01")
    for s in r.viable_sites:
        assert isinstance(s, SiteObservability)
        assert s.can_observe is True


def test_n_sites_viable_consistent():
    r = plan_network_coverage(SITES, "2026-06-01 22:00", "2026-06-02 00:30", "2026-06-01")
    assert r.n_sites_viable == len(r.viable_sites)


def test_flag_values():
    r = plan_network_coverage(SITES, "2026-06-01 22:00", "2026-06-02 00:30", "2026-06-01")
    assert r.flag in ("FULL_COVERAGE", "PARTIAL_COVERAGE", "NO_COVERAGE")


def test_overlap_hours_nonneg():
    r = plan_network_coverage(SITES, "2026-06-01 22:00", "2026-06-02 00:30", "2026-06-01")
    for s in r.viable_sites:
        assert s.overlap_hours >= 0.0


def test_single_site_coverage():
    sites = [{"name": "Solo", "lat_deg": 30.0, "lon_deg": -70.0}]
    r = plan_network_coverage(sites, "2026-06-01 22:00", "2026-06-02 00:30", "2026-06-01")
    assert r.n_sites_total == 1


def test_format_returns_string():
    r = plan_network_coverage(SITES, "2026-06-01 22:00", "2026-06-02 00:30", "2026-06-01")
    s = format_network_plan(r)
    assert isinstance(s, str)


def test_format_contains_flag():
    r = plan_network_coverage(SITES, "2026-06-01 22:00", "2026-06-02 00:30", "2026-06-01")
    s = format_network_plan(r)
    assert "Flag" in s


def test_different_date():
    r = plan_network_coverage(SITES, "2026-12-15 21:00", "2026-12-15 23:30", "2026-12-15")
    assert isinstance(r, NetworkPlanResult)
