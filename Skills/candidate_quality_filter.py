"""Filter candidates by named quality presets."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

_PRESETS: dict[str, dict[str, float | int | str]] = {
    "strict": {
        "fpp_max": 0.05,
        "snr_min": 10.0,
        "dc_min": 0.90,
        "n_transits_min": 3,
    },
    "moderate": {
        "fpp_max": 0.15,
        "snr_min": 7.0,
        "dc_min": 0.80,
        "n_transits_min": 2,
    },
    "loose": {
        "fpp_max": 0.30,
        "snr_min": 5.0,
        "dc_min": 0.70,
        "n_transits_min": 2,
    },
}


@dataclass(frozen=True)
class QualityFilterResult:
    preset: str
    n_input: int
    n_passed: int
    n_rejected: int
    passed_tic_ids: tuple[str | int, ...]
    flag: str


def _get_fpp(row: dict) -> float | None:
    for key in ("false_positive_probability", "best_fpp"):
        if key in row:
            return float(row[key])
    scores = row.get("scores", {})
    if isinstance(scores, dict) and "false_positive_probability" in scores:
        return float(scores["false_positive_probability"])
    return None


def _get_snr(row: dict) -> float | None:
    for key in ("snr", "detection_snr", "best_snr"):
        if key in row:
            return float(row[key])
    scores = row.get("scores", {})
    if isinstance(scores, dict) and "detection_confidence" in scores:
        return float(scores["detection_confidence"]) * 20.0
    return None


def _get_dc(row: dict) -> float | None:
    for key in ("detection_confidence",):
        if key in row:
            return float(row[key])
    scores = row.get("scores", {})
    if isinstance(scores, dict) and "detection_confidence" in scores:
        return float(scores["detection_confidence"])
    return None


def _get_n_transits(row: dict) -> int | None:
    for key in ("n_transits", "transit_count"):
        if key in row:
            return int(row[key])
    signal = row.get("signal", {})
    if isinstance(signal, dict) and "transit_count" in signal:
        return int(signal["transit_count"])
    return None


def filter_by_preset(
    rows: list[dict],
    preset: str = "moderate",
    custom_thresholds: dict | None = None,
) -> QualityFilterResult:
    """Filter candidate rows using a named quality preset."""
    if preset not in _PRESETS and custom_thresholds is None:
        return QualityFilterResult(
            preset=preset, n_input=len(rows), n_passed=0, n_rejected=len(rows),
            passed_tic_ids=(), flag="UNKNOWN_PRESET",
        )

    thresholds = dict(_PRESETS.get(preset, {}))
    if custom_thresholds:
        thresholds.update(custom_thresholds)

    fpp_max = float(thresholds.get("fpp_max", 1.0))
    snr_min = float(thresholds.get("snr_min", 0.0))
    dc_min = float(thresholds.get("dc_min", 0.0))
    n_min = int(thresholds.get("n_transits_min", 1))

    passed: list[str | int] = []
    for row in rows:
        fpp = _get_fpp(row)
        if fpp is not None and fpp > fpp_max:
            continue
        snr = _get_snr(row)
        if snr is not None and snr < snr_min:
            continue
        dc = _get_dc(row)
        if dc is not None and dc < dc_min:
            continue
        n_tr = _get_n_transits(row)
        if n_tr is not None and n_tr < n_min:
            continue
        tic = row.get("tic_id", row.get("target_id", "unknown"))
        passed.append(tic)

    return QualityFilterResult(
        preset=preset,
        n_input=len(rows),
        n_passed=len(passed),
        n_rejected=len(rows) - len(passed),
        passed_tic_ids=tuple(passed),
        flag="OK",
    )


def format_filter_result(r: QualityFilterResult) -> str:
    ids_str = ", ".join(str(t) for t in r.passed_tic_ids) if r.passed_tic_ids else "_none_"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Preset | {r.preset} |\n"
        f"| Input candidates | {r.n_input} |\n"
        f"| Passed | {r.n_passed} |\n"
        f"| Rejected | {r.n_rejected} |\n"
        f"| Passed TIC IDs | {ids_str} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Filter candidates by quality preset.")
    p.add_argument("input", help="JSON file with candidate rows")
    p.add_argument("--preset", default="moderate", choices=list(_PRESETS))
    args = p.parse_args()
    rows = json.loads(Path(args.input).read_text())
    if isinstance(rows, dict):
        rows = [rows]
    r = filter_by_preset(rows, args.preset)
    print(format_filter_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
