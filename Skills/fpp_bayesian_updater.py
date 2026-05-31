"""Update false-positive probability using a likelihood ratio from new evidence."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FppUpdateResult:
    fpp_prior: float
    likelihood_ratio: float
    fpp_posterior: float
    log_bayes_factor: float
    evidence_label: str
    flag: str


def update_fpp(
    fpp_prior: float,
    likelihood_ratio: float,
    evidence_label: str = "unnamed",
) -> FppUpdateResult:
    """
    Bayesian FPP update given a likelihood ratio LR = P(evidence|FP) / P(evidence|planet).

    LR > 1: evidence favours false positive → FPP increases.
    LR < 1: evidence favours planet → FPP decreases.

    Posterior odds = Prior odds × LR
    posterior_FPP = posterior_odds / (1 + posterior_odds)

    Log Bayes factor = log10(LR).
    """
    if not math.isfinite(fpp_prior) or not (0.0 < fpp_prior < 1.0):
        return FppUpdateResult(
            fpp_prior=fpp_prior,
            likelihood_ratio=likelihood_ratio,
            fpp_posterior=float("nan"),
            log_bayes_factor=float("nan"),
            evidence_label=evidence_label,
            flag="INVALID_FPP_PRIOR",
        )
    if not math.isfinite(likelihood_ratio) or likelihood_ratio <= 0.0:
        return FppUpdateResult(
            fpp_prior=fpp_prior,
            likelihood_ratio=likelihood_ratio,
            fpp_posterior=float("nan"),
            log_bayes_factor=float("nan"),
            evidence_label=evidence_label,
            flag="INVALID_LIKELIHOOD_RATIO",
        )

    prior_odds = fpp_prior / (1.0 - fpp_prior)
    posterior_odds = prior_odds * likelihood_ratio
    fpp_post = posterior_odds / (1.0 + posterior_odds)
    log_bf = math.log10(likelihood_ratio)

    return FppUpdateResult(
        fpp_prior=fpp_prior,
        likelihood_ratio=likelihood_ratio,
        fpp_posterior=round(fpp_post, 6),
        log_bayes_factor=round(log_bf, 4),
        evidence_label=evidence_label,
        flag="OK",
    )


def format_fpp_update(r: FppUpdateResult) -> str:
    return (
        f"| Parameter | Value |\n"
        f"|---|---|\n"
        f"| Evidence | {r.evidence_label} |\n"
        f"| FPP prior | {r.fpp_prior:.6f} |\n"
        f"| Likelihood ratio (FP/planet) | {r.likelihood_ratio:.4f} |\n"
        f"| Log10 Bayes factor | {r.log_bayes_factor:.4f} |\n"
        f"| FPP posterior | {r.fpp_posterior:.6f} |\n"
        f"| Flag | {r.flag} |\n"
    )


def _cli() -> int:
    p = argparse.ArgumentParser(description="Bayesian FPP update from likelihood ratio.")
    p.add_argument("fpp_prior", type=float)
    p.add_argument("likelihood_ratio", type=float)
    p.add_argument("--label", default="unnamed")
    args = p.parse_args()
    r = update_fpp(args.fpp_prior, args.likelihood_ratio, args.label)
    print(format_fpp_update(r))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
