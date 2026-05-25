"""Tests for Skills/observation_log_parser.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from observation_log_parser import ObsLogEntry, format_obs_log, load_obs_log, parse_obs_log

SAMPLE_CSV = """BJD,filter,mag,mag_err
2459000.5,V,12.34,0.01
2459001.5,V,12.35,0.01
2459001.7,R,12.50,0.02
2459003.5,V,12.33,0.01
"""


def test_basic_parse():
    r = parse_obs_log(SAMPLE_CSV)
    assert r.flag == "OK"
    assert r.n_obs == 4


def test_n_nights():
    r = parse_obs_log(SAMPLE_CSV)
    assert r.n_nights == 3  # BJD 2459000, 2459001, 2459003


def test_filters():
    r = parse_obs_log(SAMPLE_CSV)
    assert "V" in r.filters
    assert "R" in r.filters


def test_bjd_range():
    r = parse_obs_log(SAMPLE_CSV)
    assert r.bjd_start is not None
    assert r.bjd_end is not None
    assert r.bjd_end > r.bjd_start


def test_baseline():
    r = parse_obs_log(SAMPLE_CSV)
    assert r.baseline_days is not None
    assert abs(r.baseline_days - 3.0) < 0.1


def test_mean_mag():
    r = parse_obs_log(SAMPLE_CSV)
    assert r.mean_mag is not None
    assert 12.0 < r.mean_mag < 13.0


def test_rms_mag():
    r = parse_obs_log(SAMPLE_CSV)
    assert r.rms_mag is not None
    assert r.rms_mag >= 0


def test_entries_are_obs_log_entry():
    r = parse_obs_log(SAMPLE_CSV)
    for e in r.entries:
        assert isinstance(e, ObsLogEntry)
        assert e.bjd > 0
        assert e.mag > 0


def test_empty_returns_empty():
    r = parse_obs_log("")
    assert r.flag == "EMPTY"


def test_header_only():
    r = parse_obs_log("BJD,filter,mag,mag_err\n")
    assert r.flag == "EMPTY"


def test_invalid_non_string():
    r = parse_obs_log(None)
    assert r.flag == "INVALID"


def test_tsv_support():
    tsv = "BJD\tfilter\tmag\tmag_err\n2459000.5\tV\t12.34\t0.01\n"
    r = parse_obs_log(tsv, delimiter="\t")
    assert r.flag == "OK"
    assert r.n_obs == 1


def test_load_obs_log(tmp_path):
    f = tmp_path / "obs.csv"
    f.write_text(SAMPLE_CSV)
    r = load_obs_log(f)
    assert r.flag == "OK"
    assert r.n_obs == 4


def test_format_returns_string():
    r = parse_obs_log(SAMPLE_CSV)
    assert isinstance(format_obs_log(r), str)


def test_format_contains_key_words():
    r = parse_obs_log(SAMPLE_CSV)
    text = format_obs_log(r)
    assert "Observation Log" in text
    assert "Flag" in text
