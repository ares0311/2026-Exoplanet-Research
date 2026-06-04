"""Check TESS camera field-of-view position and estimate sensitivity penalty."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

# TESS camera FOV: 24° × 24° per camera; 4 cameras per sector cover 24° × 96°
# Camera centres are at ecliptic longitudes separated by ~90° and latitudes ±54° or ±18°
# Edge proximity thresholds (degrees from camera centre)
_CAMERA_HALF_WIDTH_DEG = 12.0      # half-width of one TESS camera
_EDGE_ZONE_DEG = 1.5               # within 1.5° of edge → sensitivity penalty starts
_FULL_PENALTY_DEG = 0.3            # within 0.3° → maximum penalty
_GAP_WIDTH_DEG = 0.5               # inter-camera gap width

# PRF degradation at edge: ~20% larger PSF → ~15% SNR penalty
_EDGE_SNR_PENALTY_MAX = 0.15


@dataclass(frozen=True)
class FovPositionResult:
    offset_from_centre_deg: float
    offset_from_edge_deg: float
    snr_penalty_fraction: float
    in_inter_camera_gap: bool
    position_quality: str   # CENTRAL / NOMINAL / EDGE / GAP
    flag: str


def check_fov_position(
    offset_from_centre_deg: float,
    camera_half_width_deg: float = _CAMERA_HALF_WIDTH_DEG,
) -> FovPositionResult:
    """
    Check TESS camera field position and estimate SNR degradation.

    Parameters
    ----------
    offset_from_centre_deg: Angular offset from the camera optical centre.
    camera_half_width_deg:  Half-width of the camera FOV (default 12°).

    Returns SNR penalty fraction (0 = no penalty, 0.15 = max 15% SNR loss).
    Inter-camera gap is flagged when offset ≈ camera_half_width_deg.
    """
    if not math.isfinite(offset_from_centre_deg) or offset_from_centre_deg < 0.0:
        return FovPositionResult(
            offset_from_centre_deg=offset_from_centre_deg,
            offset_from_edge_deg=float("nan"),
            snr_penalty_fraction=float("nan"),
            in_inter_camera_gap=False,
            position_quality="UNKNOWN",
            flag="INVALID_OFFSET",
        )

    hw = camera_half_width_deg
    offset_from_edge = hw - offset_from_centre_deg

    # In gap region
    in_gap = abs(offset_from_edge) < _GAP_WIDTH_DEG / 2.0

    if in_gap:
        snr_penalty = _EDGE_SNR_PENALTY_MAX
        quality = "GAP"
    elif offset_from_centre_deg > hw:
        # Outside FOV
        snr_penalty = 1.0
        quality = "OUTSIDE_FOV"
    elif offset_from_edge < _FULL_PENALTY_DEG:
        snr_penalty = _EDGE_SNR_PENALTY_MAX
        quality = "EDGE"
    elif offset_from_edge < _EDGE_ZONE_DEG:
        frac = 1.0 - (offset_from_edge - _FULL_PENALTY_DEG) / (
            _EDGE_ZONE_DEG - _FULL_PENALTY_DEG
        )
        snr_penalty = frac * _EDGE_SNR_PENALTY_MAX
        quality = "EDGE"
    elif offset_from_centre_deg < hw * 0.5:
        snr_penalty = 0.0
        quality = "CENTRAL"
    else:
        snr_penalty = 0.0
        quality = "NOMINAL"

    flag = "OUTSIDE_FOV" if quality == "OUTSIDE_FOV" else "OK"

    return FovPositionResult(
        offset_from_centre_deg=round(offset_from_centre_deg, 4),
        offset_from_edge_deg=round(offset_from_edge, 4),
        snr_penalty_fraction=round(snr_penalty, 4),
        in_inter_camera_gap=in_gap,
        position_quality=quality,
        flag=flag,
    )


def format_fov_position_result(r: FovPositionResult) -> str:
    def _f(v: float) -> str:
        return f"{v:.4f}" if math.isfinite(v) else "N/A"

    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Offset from centre (deg) | {_f(r.offset_from_centre_deg)} |\n"
        f"| Offset from edge (deg) | {_f(r.offset_from_edge_deg)} |\n"
        f"| SNR penalty fraction | {_f(r.snr_penalty_fraction)} |\n"
        f"| In inter-camera gap | {r.in_inter_camera_gap} |\n"
        f"| Position quality | {r.position_quality} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Check TESS FOV position and SNR penalty.")
    p.add_argument("offset_from_centre_deg", type=float)
    p.add_argument("--camera-half-width-deg", type=float, default=_CAMERA_HALF_WIDTH_DEG)
    args = p.parse_args()
    r = check_fov_position(args.offset_from_centre_deg, args.camera_half_width_deg)
    print(format_fov_position_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
