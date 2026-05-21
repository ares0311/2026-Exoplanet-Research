"""Versioned scoring-prior configuration.

The core scorer keeps conservative built-in priors for default operation. This
module adds an opt-in config path for mission-specific priors while preserving
the same guardrails: false positives stay prominent, probabilities must be
explicit and normalized, and configs cannot introduce discovery-claim behavior.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from exo_toolkit.hypotheses import DEFAULT_LOG_PRIORS
from exo_toolkit.schemas import CandidateSignal, Mission

HypothesisName = Literal[
    "planet_candidate",
    "eclipsing_binary",
    "background_eclipsing_binary",
    "stellar_variability",
    "instrumental_artifact",
    "known_object",
]

HYPOTHESIS_NAMES: tuple[HypothesisName, ...] = (
    "planet_candidate",
    "eclipsing_binary",
    "background_eclipsing_binary",
    "stellar_variability",
    "instrumental_artifact",
    "known_object",
)

DEFAULT_SCORING_PRIOR_CONFIG_PATH = Path("configs/scoring_priors_v0.json")


class ScoringPriorError(ValueError):
    """Raised when a scoring-prior config is missing conservative safeguards."""


class ScoringPriorProfile(BaseModel):
    """One named set of hypothesis prior probabilities."""

    model_config = ConfigDict(frozen=True)

    name: str
    mission: Mission | Literal["default"]
    priors: dict[HypothesisName, float]

    @model_validator(mode="after")
    def _validate_priors(self) -> ScoringPriorProfile:
        missing = sorted(set(HYPOTHESIS_NAMES) - set(self.priors))
        extra = sorted(set(self.priors) - set(HYPOTHESIS_NAMES))
        if missing:
            raise ValueError(f"missing prior keys: {', '.join(missing)}")
        if extra:
            raise ValueError(f"unknown prior keys: {', '.join(extra)}")
        if any(value <= 0.0 for value in self.priors.values()):
            raise ValueError("all prior probabilities must be positive")
        total = sum(self.priors.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"prior probabilities must sum to 1.0, got {total:.12f}")
        false_positive_total = (
            self.priors["eclipsing_binary"]
            + self.priors["background_eclipsing_binary"]
            + self.priors["stellar_variability"]
            + self.priors["instrumental_artifact"]
        )
        if false_positive_total <= self.priors["planet_candidate"]:
            raise ValueError("false-positive priors must exceed the planet prior")
        return self

    def to_log_priors(self) -> dict[str, float]:
        """Return log-prior overrides compatible with ``compute_log_scores``."""
        return {key: math.log(value) for key, value in self.priors.items()}


class ScoringPriorConfig(BaseModel):
    """Validated versioned scoring-prior configuration."""

    model_config = ConfigDict(frozen=True)

    version: str
    schema_version: str
    default_profile: str
    profiles: dict[str, ScoringPriorProfile]
    mission_profiles: dict[Mission, str] = Field(default_factory=dict)
    confirmation_claims_allowed: bool = False
    external_submission_requires_human_approval: bool = True

    @model_validator(mode="after")
    def _validate_config(self) -> ScoringPriorConfig:
        if self.default_profile not in self.profiles:
            raise ValueError("default_profile must refer to an existing profile")
        missing_missions = sorted({"TESS", "Kepler", "K2"} - set(self.mission_profiles))
        if missing_missions:
            raise ValueError(f"missing mission_profiles keys: {', '.join(missing_missions)}")
        unknown_profiles = sorted(
            {profile for profile in self.mission_profiles.values() if profile not in self.profiles}
        )
        if unknown_profiles:
            raise ValueError(f"unknown mission profile references: {', '.join(unknown_profiles)}")
        if self.confirmation_claims_allowed is not False:
            raise ValueError("confirmation claims must remain disabled")
        if self.external_submission_requires_human_approval is not True:
            raise ValueError("external submission must require human approval")
        return self

    def profile_for_mission(self, mission: Mission | str) -> ScoringPriorProfile:
        """Return the configured profile for a mission, falling back to default."""
        profile_name = self.mission_profiles.get(mission, self.default_profile)  # type: ignore[arg-type]
        return self.profiles[profile_name]

    def log_priors_for_mission(self, mission: Mission | str) -> dict[str, float]:
        """Return log-prior overrides for a mission."""
        return self.profile_for_mission(mission).to_log_priors()


def default_prior_probabilities() -> dict[HypothesisName, float]:
    """Return the built-in default priors in probability space."""
    return {key: math.exp(DEFAULT_LOG_PRIORS[key]) for key in HYPOTHESIS_NAMES}


def load_scoring_prior_config(
    path: Path | str = DEFAULT_SCORING_PRIOR_CONFIG_PATH,
) -> ScoringPriorConfig:
    """Load and validate a scoring-prior config from JSON."""
    config_path = Path(path)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        return ScoringPriorConfig.model_validate(payload)
    except FileNotFoundError as exc:
        raise ScoringPriorError(f"Scoring prior config not found: {config_path}") from exc
    except json.JSONDecodeError as exc:
        raise ScoringPriorError(f"Invalid scoring prior JSON: {exc}") from exc
    except ValueError as exc:
        raise ScoringPriorError(str(exc)) from exc


def log_priors_for_signal(
    signal: CandidateSignal,
    config: ScoringPriorConfig,
) -> dict[str, float]:
    """Return mission-specific log-prior overrides for a candidate signal."""
    return config.log_priors_for_mission(signal.mission)
