"""Tests for Skills/flux_anomaly_detector.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "Skills"))

from flux_anomaly_detector import (
    AnomalyReport,
    detect_flux_anomalies,
    format_anomaly_report,
)


def _flat(n=100):
    return list(range(n)), [1.0] * n


def test_flat_lc_no_anomalies():
    t, f = _flat()
    report = detect_flux_anomalies(t, f)
    assert report.flag == "OK"
    assert report.n_outliers == 0


def test_too_short_returns_ok():
    report = detect_flux_anomalies([0, 1, 2], [1.0, 1.0, 1.0])
    assert report.flag == "OK"
    assert report.n_points == 3


def test_outlier_detected():
    t = list(range(100))
    f = [1.0] * 100
    f[50] = 1000.0  # severe spike
    report = detect_flux_anomalies(t, f)
    assert report.n_outliers >= 1


def test_severe_outlier_flag():
    t = list(range(100))
    f = [1.0] * 100
    f[50] = 1000.0
    report = detect_flux_anomalies(t, f)
    assert report.flag in ("SEVERE_ANOMALIES", "ANOMALIES_DETECTED")


def test_step_detected():
    t = list(range(100))
    f = [1.0] * 50 + [2.0] * 50  # step at index 50
    report = detect_flux_anomalies(t, f, sigma_threshold=2.0)
    assert len(report.events) >= 1  # step or ramp detected


def test_overall_quality_poor_on_severe():
    t = list(range(100))
    f = [1.0] * 100
    f[50] = 1000.0
    report = detect_flux_anomalies(t, f)
    assert report.overall_quality in ("POOR", "MODERATE")


def test_returns_anomaly_report():
    t, f = _flat()
    report = detect_flux_anomalies(t, f)
    assert isinstance(report, AnomalyReport)


def test_n_points_correct():
    t, f = _flat(80)
    report = detect_flux_anomalies(t, f)
    assert report.n_points == 80


def test_events_tuple():
    t, f = _flat()
    report = detect_flux_anomalies(t, f)
    assert isinstance(report.events, tuple)


def test_custom_sigma_threshold():
    t = list(range(100))
    f = [1.0] * 100
    f[50] = 1.1  # small spike
    report_strict = detect_flux_anomalies(t, f, sigma_threshold=0.5)
    report_loose = detect_flux_anomalies(t, f, sigma_threshold=50.0)
    assert len(report_strict.events) >= len(report_loose.events)


def test_format_contains_status():
    t, f = _flat()
    report = detect_flux_anomalies(t, f)
    md = format_anomaly_report(report)
    assert "OK" in md


def test_format_no_anomalies_message():
    t, f = _flat()
    report = detect_flux_anomalies(t, f)
    md = format_anomaly_report(report)
    assert "No anomalies" in md
