"""Tests for Skills.momentum_dump_flagger."""
from __future__ import annotations

from Skills.momentum_dump_flagger import (
    MomentumDumpResult,
    flag_momentum_dumps,
    format_momentum_dump_result,
)


def _make_time(n=1000, dt_min=2.0, t0=2458000.0):
    dt = dt_min / 1440.0
    return [t0 + i * dt for i in range(n)]


class TestFlagMomentumDumps:
    def test_returns_result(self) -> None:
        t = _make_time()
        r = flag_momentum_dumps(t, [1.0] * len(t))
        assert isinstance(r, MomentumDumpResult)

    def test_empty_returns_clean(self) -> None:
        r = flag_momentum_dumps([], [])
        assert r.flag == "CLEAN"
        assert r.n_dumps_found == 0

    def test_periodic_heuristic_finds_dumps(self) -> None:
        t = _make_time(n=2000)
        r = flag_momentum_dumps(t, [1.0] * len(t), period_days=2.5)
        assert r.n_dumps_found > 0

    def test_explicit_dump_times_used(self) -> None:
        t = _make_time(n=1000)
        dump_t = [t[500]]  # one known dump
        r = flag_momentum_dumps(t, [1.0] * len(t), dump_times=dump_t)
        assert r.n_dumps_found == 1
        assert r.flagged_cadences > 0

    def test_flagged_fraction_in_range(self) -> None:
        t = _make_time(n=1000)
        r = flag_momentum_dumps(t, [1.0] * len(t))
        assert 0.0 <= r.fraction_flagged <= 1.0

    def test_clean_flag_when_no_dumps(self) -> None:
        t = _make_time(n=100)
        r = flag_momentum_dumps(t, [1.0] * len(t), dump_times=[])
        assert r.flag == "CLEAN"
        assert r.flagged_cadences == 0

    def test_significant_flag_many_dumps(self) -> None:
        t = _make_time(n=500)
        # Many closely-spaced dumps to force SIGNIFICANT
        dumps = [t[0] + i * 0.1 for i in range(100)]
        r = flag_momentum_dumps(t, [1.0] * len(t), dump_times=dumps, window_hours=24.0,
                                significant_threshold=0.01)
        assert r.flag == "SIGNIFICANT"

    def test_dump_times_sorted(self) -> None:
        t = _make_time(n=1000)
        r = flag_momentum_dumps(t, [1.0] * len(t))
        dumps = list(r.dump_times)
        assert dumps == sorted(dumps)

    def test_flag_values_valid(self) -> None:
        t = _make_time(n=1000)
        r = flag_momentum_dumps(t, [1.0] * len(t))
        assert r.flag in {"CLEAN", "MINOR", "SIGNIFICANT"}


class TestFormatMomentumDump:
    def test_returns_string(self) -> None:
        t = _make_time()
        r = flag_momentum_dumps(t, [1.0] * len(t))
        assert isinstance(format_momentum_dump_result(r), str)

    def test_contains_flag(self) -> None:
        t = _make_time()
        r = flag_momentum_dumps(t, [1.0] * len(t))
        assert r.flag in format_momentum_dump_result(r)
