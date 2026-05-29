"""Detect anomalous flux patterns: outliers, step functions, and ramps.

Flags data quality issues that could mimic or mask transit signals.

Public API
----------
AnomalyEvent(index, time, flux, anomaly_type, sigma, flag)
AnomalyReport(n_points, n_outliers, n_steps, n_ramps,
              events, overall_quality, flag)
detect_flux_anomalies(time, flux, *, sigma_threshold,
                      step_window, ramp_window) -> AnomalyReport
format_anomaly_report(report) -> str
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnomalyEvent:
    index: int
    time: float
    flux: float
    anomaly_type: str  # "OUTLIER" | "STEP" | "RAMP"
    sigma: float
    flag: str  # "MILD" | "SEVERE"


@dataclass(frozen=True)
class AnomalyReport:
    n_points: int
    n_outliers: int
    n_steps: int
    n_ramps: int
    events: tuple[AnomalyEvent, ...]
    overall_quality: str   # "GOOD" | "MODERATE" | "POOR"
    flag: str  # "OK" | "ANOMALIES_DETECTED" | "SEVERE_ANOMALIES"


def _median_mad(values: list[float]) -> tuple[float, float]:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0.0, 1.0
    med = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
    abs_devs = sorted(abs(x - med) for x in s)
    mad = abs_devs[n // 2] * 1.4826
    return med, max(mad, 1e-10)


def detect_flux_anomalies(
    time: list[float] | tuple[float, ...],
    flux: list[float] | tuple[float, ...],
    *,
    sigma_threshold: float = 5.0,
    step_window: int = 10,
    ramp_window: int = 20,
) -> AnomalyReport:
    """Detect outliers, steps, and ramps in a flux time series.

    Args:
        time: Timestamps in days.
        flux: Normalised flux array.
        sigma_threshold: Sigma threshold for outlier detection.
        step_window: Window size for step detection (median before/after).
        ramp_window: Window size for ramp/trend detection.

    Returns:
        AnomalyReport with events and overall quality.
    """
    t = list(time)
    f = list(flux)
    n = len(f)

    if n < 4:
        return AnomalyReport(
            n_points=n, n_outliers=0, n_steps=0, n_ramps=0,
            events=(), overall_quality="GOOD", flag="OK"
        )

    med, mad = _median_mad(f)
    events: list[AnomalyEvent] = []

    # Outlier detection
    for i, (ti, fi) in enumerate(zip(t, f, strict=False)):
        sigma = abs(fi - med) / mad
        if sigma > sigma_threshold:
            flag = "SEVERE" if sigma > 2 * sigma_threshold else "MILD"
            events.append(AnomalyEvent(
                index=i, time=ti, flux=fi,
                anomaly_type="OUTLIER",
                sigma=round(sigma, 2),
                flag=flag,
            ))

    # Step detection
    half = step_window // 2
    for i in range(half, n - half):
        before = f[max(0, i - half):i]
        after = f[i:min(n, i + half)]
        if len(before) < 2 or len(after) < 2:
            continue
        med_before = sorted(before)[len(before) // 2]
        med_after = sorted(after)[len(after) // 2]
        step_sigma = abs(med_after - med_before) / mad
        if step_sigma > sigma_threshold:
            flag = "SEVERE" if step_sigma > 2 * sigma_threshold else "MILD"
            already = any(e.index == i and e.anomaly_type == "STEP" for e in events)
            if not already:
                events.append(AnomalyEvent(
                    index=i, time=t[i], flux=f[i],
                    anomaly_type="STEP",
                    sigma=round(step_sigma, 2),
                    flag=flag,
                ))

    # Ramp detection: linear trend over window
    hw = ramp_window // 2
    for i in range(hw, n - hw):
        window = f[max(0, i - hw):min(n, i + hw)]
        if len(window) < ramp_window // 2:
            continue
        x = list(range(len(window)))
        mx = sum(x) / len(x)
        my = sum(window) / len(window)
        num = sum((xi - mx) * (yi - my) for xi, yi in zip(x, window, strict=False))
        den = sum((xi - mx) ** 2 for xi in x)
        if den == 0:
            continue
        slope = num / den
        ramp_sigma = abs(slope * ramp_window) / mad
        if ramp_sigma > sigma_threshold:
            flag = "SEVERE" if ramp_sigma > 2 * sigma_threshold else "MILD"
            already = any(e.index == i and e.anomaly_type == "RAMP" for e in events)
            if not already:
                events.append(AnomalyEvent(
                    index=i, time=t[i], flux=f[i],
                    anomaly_type="RAMP",
                    sigma=round(ramp_sigma, 2),
                    flag=flag,
                ))

    n_outliers = sum(1 for e in events if e.anomaly_type == "OUTLIER")
    n_steps = sum(1 for e in events if e.anomaly_type == "STEP")
    n_ramps = sum(1 for e in events if e.anomaly_type == "RAMP")
    n_severe = sum(1 for e in events if e.flag == "SEVERE")
    n_total = len(events)

    anomaly_fraction = n_total / n
    if n_severe > 0 or anomaly_fraction > 0.05:
        overall_quality = "POOR"
        flag = "SEVERE_ANOMALIES"
    elif n_total > 0:
        overall_quality = "MODERATE"
        flag = "ANOMALIES_DETECTED"
    else:
        overall_quality = "GOOD"
        flag = "OK"

    return AnomalyReport(
        n_points=n,
        n_outliers=n_outliers,
        n_steps=n_steps,
        n_ramps=n_ramps,
        events=tuple(events),
        overall_quality=overall_quality,
        flag=flag,
    )


def format_anomaly_report(report: AnomalyReport) -> str:
    """Format anomaly report as Markdown.

    Args:
        report: AnomalyReport to format.

    Returns:
        Markdown string.
    """
    lines = [
        "## Flux Anomaly Detector\n",
        f"**Status**: `{report.flag}` | Quality: **{report.overall_quality}** | "
        f"Points: {report.n_points} | "
        f"Outliers: {report.n_outliers} | Steps: {report.n_steps} | "
        f"Ramps: {report.n_ramps}\n",
    ]
    if not report.events:
        lines.append("\n_No anomalies detected._")
        return "\n".join(lines)

    lines += [
        "",
        "| Index | Time (d) | Flux | Type | σ | Severity |",
        "|---|---|---|---|---|---|",
    ]
    for e in report.events[:20]:  # cap display at 20
        lines.append(
            f"| {e.index} | {e.time:.4f} | {e.flux:.6f} | "
            f"{e.anomaly_type} | {e.sigma:.1f} | `{e.flag}` |"
        )
    if len(report.events) > 20:
        lines.append(f"| … | … | … | … | … | ({len(report.events) - 20} more) |")
    return "\n".join(lines)


def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Detect flux anomalies.")
    parser.add_argument("lc", help="Light curve JSON with 'time' and 'flux'.")
    parser.add_argument("--sigma", type=float, default=5.0)
    args = parser.parse_args(argv)

    from pathlib import Path
    data = json.loads(Path(args.lc).read_text())
    report = detect_flux_anomalies(data["time"], data["flux"],
                                   sigma_threshold=args.sigma)
    print(format_anomaly_report(report))
    return 0 if report.flag == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
