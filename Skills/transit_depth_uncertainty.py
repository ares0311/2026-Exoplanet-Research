"""Compute transit depth uncertainty from photon noise and systematics."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class DepthUncertaintyResult:
    depth_ppm: float
    n_in_transit: int
    n_out_transit: int
    rms_ppm: float
    depth_err_photon_ppm: float
    depth_err_systematic_ppm: float
    depth_err_total_ppm: float
    snr: float
    flag: str


def compute_depth_uncertainty(
    depth_ppm: float,
    n_in_transit: int,
    n_out_transit: int,
    rms_ppm: float,
    systematic_floor_ppm: float = 30.0,
) -> DepthUncertaintyResult:
    """
    Compute uncertainty on transit depth from scatter and systematics.

    Photon-noise depth error: σ_depth = rms * sqrt(1/N_in + 1/N_out)
    Systematic error: systematic_floor_ppm (added in quadrature).
    Total: sqrt(σ_photon² + σ_sys²).
    SNR = depth / σ_total.
    """
    if not math.isfinite(depth_ppm) or depth_ppm < 0.0:
        return DepthUncertaintyResult(
            depth_ppm=depth_ppm, n_in_transit=n_in_transit, n_out_transit=n_out_transit,
            rms_ppm=rms_ppm,
            depth_err_photon_ppm=float("nan"), depth_err_systematic_ppm=float("nan"),
            depth_err_total_ppm=float("nan"), snr=float("nan"),
            flag="INVALID_DEPTH",
        )
    if n_in_transit < 1:
        return DepthUncertaintyResult(
            depth_ppm=depth_ppm, n_in_transit=n_in_transit, n_out_transit=n_out_transit,
            rms_ppm=rms_ppm,
            depth_err_photon_ppm=float("nan"), depth_err_systematic_ppm=float("nan"),
            depth_err_total_ppm=float("nan"), snr=float("nan"),
            flag="INVALID_N_IN_TRANSIT",
        )
    if n_out_transit < 1:
        return DepthUncertaintyResult(
            depth_ppm=depth_ppm, n_in_transit=n_in_transit, n_out_transit=n_out_transit,
            rms_ppm=rms_ppm,
            depth_err_photon_ppm=float("nan"), depth_err_systematic_ppm=float("nan"),
            depth_err_total_ppm=float("nan"), snr=float("nan"),
            flag="INVALID_N_OUT_TRANSIT",
        )
    if not math.isfinite(rms_ppm) or rms_ppm <= 0.0:
        return DepthUncertaintyResult(
            depth_ppm=depth_ppm, n_in_transit=n_in_transit, n_out_transit=n_out_transit,
            rms_ppm=rms_ppm,
            depth_err_photon_ppm=float("nan"), depth_err_systematic_ppm=float("nan"),
            depth_err_total_ppm=float("nan"), snr=float("nan"),
            flag="INVALID_RMS",
        )

    sigma_photon = rms_ppm * math.sqrt(1.0 / n_in_transit + 1.0 / n_out_transit)
    sigma_sys = max(0.0, systematic_floor_ppm)
    sigma_total = math.sqrt(sigma_photon**2 + sigma_sys**2)
    snr = depth_ppm / sigma_total if sigma_total > 0 else 0.0

    return DepthUncertaintyResult(
        depth_ppm=depth_ppm,
        n_in_transit=n_in_transit,
        n_out_transit=n_out_transit,
        rms_ppm=rms_ppm,
        depth_err_photon_ppm=round(sigma_photon, 2),
        depth_err_systematic_ppm=round(sigma_sys, 2),
        depth_err_total_ppm=round(sigma_total, 2),
        snr=round(snr, 3),
        flag="OK",
    )


def format_depth_uncertainty(r: DepthUncertaintyResult) -> str:
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Depth (ppm) | {r.depth_ppm:.1f} |\n"
        f"| N in-transit | {r.n_in_transit} |\n"
        f"| N out-transit | {r.n_out_transit} |\n"
        f"| OOT RMS (ppm) | {r.rms_ppm:.2f} |\n"
        f"| σ_photon (ppm) | {r.depth_err_photon_ppm:.2f} |\n"
        f"| σ_systematic (ppm) | {r.depth_err_systematic_ppm:.2f} |\n"
        f"| σ_total (ppm) | {r.depth_err_total_ppm:.2f} |\n"
        f"| SNR | {r.snr:.3f} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Compute transit depth uncertainty.")
    p.add_argument("depth_ppm", type=float)
    p.add_argument("n_in_transit", type=int)
    p.add_argument("n_out_transit", type=int)
    p.add_argument("rms_ppm", type=float)
    p.add_argument("--systematic-floor-ppm", type=float, default=30.0)
    args = p.parse_args()
    r = compute_depth_uncertainty(
        args.depth_ppm, args.n_in_transit, args.n_out_transit,
        args.rms_ppm, args.systematic_floor_ppm,
    )
    print(format_depth_uncertainty(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
