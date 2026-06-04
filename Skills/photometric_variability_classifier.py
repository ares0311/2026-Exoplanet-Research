"""Classify TESS light curve variability type from amplitude and periodogram."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

# Amplitude thresholds (ppm peak-to-peak)
_AMP_QUIET_MAX = 500.0      # < 500 ppm → quiet
_AMP_ROTATOR_MAX = 50000.0  # 500 – 50000 ppm → rotator
# > 50000 ppm → likely binary or pulsator

# Period thresholds (days)
_PROT_FAST_MAX = 2.0    # P < 2 d → fast rotator / EB
_PROT_SYNC_MAX = 10.0   # 2–10 d → tidally synchronised / active
_PROT_SUN_MAX = 40.0    # 10–40 d → solar-like rotator


@dataclass(frozen=True)
class VariabilityClassResult:
    amplitude_ppm: float
    dominant_period_days: float | None
    period_power: float | None
    variability_class: str  # QUIET / ROTATOR / BINARY / PULSATOR / UNCLASSIFIED
    confidence: float       # 0–1
    flag: str


def classify_variability(
    rms_ppm: float,
    dominant_period_days: float | None = None,
    period_power: float | None = None,
    peak_to_peak_ppm: float | None = None,
) -> VariabilityClassResult:
    """
    Classify TESS light curve variability type.

    Parameters
    ----------
    rms_ppm:              RMS scatter of the light curve in ppm.
    dominant_period_days: Dominant period from a periodogram (optional).
    period_power:         Normalised power at dominant period (0–1, optional).
    peak_to_peak_ppm:     Peak-to-peak amplitude of dominant signal (optional).

    Classes:
    - QUIET:       Low amplitude, no strong periodic signal
    - ROTATOR:     Moderate amplitude at solar/stellar rotation period
    - BINARY:      Large amplitude at short period (likely EB or reflection)
    - PULSATOR:    Large amplitude at short period but rapid (δ Scuti / RR Lyrae)
    - UNCLASSIFIED: Ambiguous or insufficient data
    """
    if not math.isfinite(rms_ppm) or rms_ppm < 0.0:
        return VariabilityClassResult(
            amplitude_ppm=rms_ppm, dominant_period_days=dominant_period_days,
            period_power=period_power, variability_class="UNCLASSIFIED",
            confidence=0.0, flag="INVALID_RMS",
        )

    amp = peak_to_peak_ppm if (peak_to_peak_ppm is not None and math.isfinite(peak_to_peak_ppm)
                               and peak_to_peak_ppm > 0) else rms_ppm * 2.0

    power = period_power if period_power is not None and math.isfinite(period_power) else 0.0
    per = dominant_period_days

    # --- Quiet classification ---
    if amp < _AMP_QUIET_MAX and power < 0.5:
        return VariabilityClassResult(
            amplitude_ppm=amp, dominant_period_days=per, period_power=period_power,
            variability_class="QUIET", confidence=round(1.0 - amp / _AMP_QUIET_MAX, 3),
            flag="OK",
        )

    # --- Binary classification ---
    # High amplitude + short period → EB or reflection effect
    if amp > _AMP_ROTATOR_MAX:
        if per is not None and per < _PROT_FAST_MAX:
            cls = "PULSATOR" if per < 0.5 else "BINARY"
            conf = min(1.0, amp / 100000.0) * min(1.0, power + 0.4)
            return VariabilityClassResult(
                amplitude_ppm=amp, dominant_period_days=per, period_power=period_power,
                variability_class=cls, confidence=round(conf, 3), flag="OK",
            )
        return VariabilityClassResult(
            amplitude_ppm=amp, dominant_period_days=per, period_power=period_power,
            variability_class="BINARY", confidence=round(min(1.0, amp / 100000.0), 3),
            flag="OK",
        )

    # --- Rotator classification ---
    rot_conf = 0.0
    if per is not None and _PROT_FAST_MAX <= per <= _PROT_SUN_MAX:
        rot_conf = min(1.0, amp / 5000.0) * min(1.0, power + 0.3)
    elif per is not None and per < _PROT_FAST_MAX and amp > 500.0:
        rot_conf = 0.5 * min(1.0, power + 0.3)

    if rot_conf > 0.3:
        return VariabilityClassResult(
            amplitude_ppm=amp, dominant_period_days=per, period_power=period_power,
            variability_class="ROTATOR", confidence=round(rot_conf, 3), flag="OK",
        )

    # --- Default ---
    return VariabilityClassResult(
        amplitude_ppm=amp, dominant_period_days=per, period_power=period_power,
        variability_class="UNCLASSIFIED", confidence=0.0, flag="OK",
    )


def format_variability_result(r: VariabilityClassResult) -> str:
    per_str = f"{r.dominant_period_days:.4f}" if r.dominant_period_days is not None else "N/A"
    pow_str = f"{r.period_power:.3f}" if r.period_power is not None else "N/A"
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Amplitude (ppm p-t-p) | {r.amplitude_ppm:.1f} |\n"
        f"| Dominant period (days) | {per_str} |\n"
        f"| Period power | {pow_str} |\n"
        f"| Variability class | {r.variability_class} |\n"
        f"| Confidence | {r.confidence:.3f} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Classify photometric variability type.")
    p.add_argument("rms_ppm", type=float)
    p.add_argument("--dominant-period-days", type=float, default=None)
    p.add_argument("--period-power", type=float, default=None)
    p.add_argument("--peak-to-peak-ppm", type=float, default=None)
    args = p.parse_args()
    r = classify_variability(
        args.rms_ppm, args.dominant_period_days, args.period_power, args.peak_to_peak_ppm,
    )
    print(format_variability_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
