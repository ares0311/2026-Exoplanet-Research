"""Tests for exo_toolkit.priors."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from exo_toolkit.hypotheses import DEFAULT_LOG_PRIORS
from exo_toolkit.priors import (
    DEFAULT_SCORING_PRIOR_CONFIG_PATH,
    HYPOTHESIS_NAMES,
    ScoringPriorConfig,
    ScoringPriorError,
    ScoringPriorProfile,
    default_prior_probabilities,
    load_scoring_prior_config,
)


def _payload() -> dict:
    return json.loads(DEFAULT_SCORING_PRIOR_CONFIG_PATH.read_text(encoding="utf-8"))


def _write_payload(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "scoring_priors.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def test_default_prior_probabilities_match_log_priors() -> None:
    probabilities = default_prior_probabilities()
    for name in HYPOTHESIS_NAMES:
        assert probabilities[name] == pytest.approx(math.exp(DEFAULT_LOG_PRIORS[name]))


def test_load_committed_config() -> None:
    config = load_scoring_prior_config()
    assert config.version == "scoring_priors_v0"
    assert config.profile_for_mission("TESS").name == "tess_v0"
    assert config.profile_for_mission("Kepler").name == "kepler_v0"
    assert config.profile_for_mission("K2").name == "k2_v0"


def test_committed_default_profile_matches_builtin_defaults() -> None:
    config = load_scoring_prior_config()
    default_profile = config.profiles["conservative_v0"]
    for name in HYPOTHESIS_NAMES:
        assert default_profile.priors[name] == pytest.approx(math.exp(DEFAULT_LOG_PRIORS[name]))


@pytest.mark.parametrize("mission", ["TESS", "Kepler", "K2"])
def test_mission_profiles_are_conservative(mission: str) -> None:
    config = load_scoring_prior_config()
    profile = config.profile_for_mission(mission)
    fp_total = (
        profile.priors["eclipsing_binary"]
        + profile.priors["background_eclipsing_binary"]
        + profile.priors["stellar_variability"]
        + profile.priors["instrumental_artifact"]
    )
    assert fp_total > profile.priors["planet_candidate"]


def test_profile_rejects_missing_hypothesis() -> None:
    priors = default_prior_probabilities()
    priors.pop("known_object")
    with pytest.raises(ValueError, match="missing prior keys"):
        ScoringPriorProfile(name="bad", mission="default", priors=priors)


def test_profile_rejects_non_normalized_priors() -> None:
    priors = default_prior_probabilities()
    priors["planet_candidate"] = 0.5
    with pytest.raises(ValueError, match="sum to 1.0"):
        ScoringPriorProfile(name="bad", mission="default", priors=priors)


def test_profile_rejects_optimistic_planet_prior() -> None:
    priors = {
        "planet_candidate": 0.55,
        "eclipsing_binary": 0.10,
        "background_eclipsing_binary": 0.10,
        "stellar_variability": 0.10,
        "instrumental_artifact": 0.10,
        "known_object": 0.05,
    }
    with pytest.raises(ValueError, match="false-positive priors"):
        ScoringPriorProfile(name="bad", mission="default", priors=priors)


def test_config_rejects_missing_mission_profile() -> None:
    payload = _payload()
    payload["mission_profiles"].pop("K2")
    with pytest.raises(ValueError, match="missing mission_profiles keys"):
        ScoringPriorConfig.model_validate(payload)


def test_config_rejects_confirmation_claims() -> None:
    payload = _payload()
    payload["confirmation_claims_allowed"] = True
    with pytest.raises(ValueError, match="confirmation claims"):
        ScoringPriorConfig.model_validate(payload)


def test_load_config_wraps_validation_errors(tmp_path: Path) -> None:
    payload = _payload()
    payload["profiles"]["tess_v0"]["priors"]["planet_candidate"] = 0.9
    path = _write_payload(tmp_path, payload)
    with pytest.raises(ScoringPriorError):
        load_scoring_prior_config(path)


def test_load_config_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ScoringPriorError, match="not found"):
        load_scoring_prior_config(tmp_path / "missing.json")


def test_log_priors_for_mission_are_logs() -> None:
    config = load_scoring_prior_config()
    profile = config.profile_for_mission("TESS")
    log_priors = config.log_priors_for_mission("TESS")
    assert log_priors["planet_candidate"] == pytest.approx(
        math.log(profile.priors["planet_candidate"])
    )
