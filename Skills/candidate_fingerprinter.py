"""Generate a stable fingerprint hash for a candidate from its canonical parameters."""
from __future__ import annotations

import argparse
import hashlib
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FingerprintResult:
    tic_id: str
    period_days: float
    epoch_bjd: float
    depth_ppm: float
    fingerprint: str
    short_fp: str
    flag: str


# Rounding precision for canonical representation
_PERIOD_DECIMALS = 4   # ~8 seconds precision
_EPOCH_DECIMALS = 4    # ~8 seconds precision
_DEPTH_DECIMALS = 0    # nearest ppm


def fingerprint(
    tic_id: str | int,
    period_days: float,
    epoch_bjd: float,
    depth_ppm: float,
) -> FingerprintResult:
    """
    Compute a stable SHA-256 fingerprint from rounded canonical parameters.

    Two candidates with the same TIC ID, period (±0.0001 d), epoch (±0.0001 d),
    and depth (±0.5 ppm) will produce identical fingerprints.
    """
    tic = str(tic_id).strip()
    if not tic:
        return FingerprintResult(
            tic_id=tic, period_days=period_days, epoch_bjd=epoch_bjd,
            depth_ppm=depth_ppm, fingerprint="", short_fp="", flag="INVALID_TIC_ID",
        )
    if not math.isfinite(period_days) or period_days <= 0.0:
        return FingerprintResult(
            tic_id=tic, period_days=period_days, epoch_bjd=epoch_bjd,
            depth_ppm=depth_ppm, fingerprint="", short_fp="", flag="INVALID_PERIOD",
        )
    if not math.isfinite(epoch_bjd):
        return FingerprintResult(
            tic_id=tic, period_days=period_days, epoch_bjd=epoch_bjd,
            depth_ppm=depth_ppm, fingerprint="", short_fp="", flag="INVALID_EPOCH",
        )
    if not math.isfinite(depth_ppm) or depth_ppm <= 0.0:
        return FingerprintResult(
            tic_id=tic, period_days=period_days, epoch_bjd=epoch_bjd,
            depth_ppm=depth_ppm, fingerprint="", short_fp="", flag="INVALID_DEPTH",
        )

    p_r = round(period_days, _PERIOD_DECIMALS)
    e_r = round(epoch_bjd, _EPOCH_DECIMALS)
    d_r = round(depth_ppm, _DEPTH_DECIMALS)

    canonical = f"TIC{tic}|P{p_r:.{_PERIOD_DECIMALS}f}|T0{e_r:.{_EPOCH_DECIMALS}f}|D{d_r:.0f}"
    digest = hashlib.sha256(canonical.encode()).hexdigest()

    return FingerprintResult(
        tic_id=tic,
        period_days=p_r,
        epoch_bjd=e_r,
        depth_ppm=d_r,
        fingerprint=digest,
        short_fp=digest[:12],
        flag="OK",
    )


def format_fingerprint_result(r: FingerprintResult) -> str:
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| TIC ID | {r.tic_id} |\n"
        f"| Period (days) | {r.period_days} |\n"
        f"| Epoch (BJD) | {r.epoch_bjd} |\n"
        f"| Depth (ppm) | {r.depth_ppm} |\n"
        f"| Fingerprint (short) | `{r.short_fp}` |\n"
        f"| Fingerprint (SHA-256) | `{r.fingerprint}` |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Generate stable candidate fingerprint.")
    p.add_argument("tic_id")
    p.add_argument("period_days", type=float)
    p.add_argument("epoch_bjd", type=float)
    p.add_argument("depth_ppm", type=float)
    args = p.parse_args()
    r = fingerprint(args.tic_id, args.period_days, args.epoch_bjd, args.depth_ppm)
    print(format_fingerprint_result(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
