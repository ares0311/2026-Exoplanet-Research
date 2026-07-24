"""Versioned deterministic ranking contracts for EXO-Hunter."""
from __future__ import annotations

from typing import Any, Literal

NEW_SELECTOR_VERSION = "exo_hunter_tic_v2"
FOLLOW_UP_SELECTOR_VERSION = "exo_hunter_follow_up_v3"
OPERATOR_SELECTOR_VERSION = "exo_hunter_operator_candidates_v1"

NEW_RANKING_WEIGHTS = {
    "tic_priority": 70.0,
    "availability": 20.0,
    "expected_information_gain": 10.0,
    "storage_cost_penalty": -1.0,
}
FOLLOW_UP_FPP_MAX = 0.15
FOLLOW_UP_CONFIDENCE_MIN = 0.40
FOLLOW_UP_PATHWAYS = (
    "kepler_archive_candidate",
    "planet_hunters_discussion",
    "tfop_ready",
)


def selection_contract(
    mode: Literal["new", "follow-up"],
    *,
    operator_supplied: bool = False,
) -> dict[str, Any]:
    """Return the exact JSON-serializable ranking/eligibility identity."""
    if operator_supplied:
        return {
            "selector_version": OPERATOR_SELECTOR_VERSION,
            "ranking": "operator-supplied values preserved without substitution",
        }
    if mode == "new":
        return {
            "selector_version": NEW_SELECTOR_VERSION,
            "ranking_formula": (
                "70*tic_priority + 20*availability + "
                "10*expected_information_gain - storage_cost_penalty"
            ),
            "weights": NEW_RANKING_WEIGHTS,
            "expected_information_gain": "tic_priority * availability",
            "eligibility": (
                "novel target; strict ASAS-SN exclusion; advanced to metadata stage; "
                "at least one QLP product"
            ),
        }
    return {
        "selector_version": FOLLOW_UP_SELECTOR_VERSION,
        "history_ranking_formula": "100*(1-fpp) + 10*detection_confidence",
        "registry_ranking_formula": "durable follow_up_priority",
        "expected_information_gain": "(1-fpp)*detection_confidence",
        "selection_semantics": (
            "rank by priority across every currently available (non-duplicate, "
            "data-confirmed) follow-up candidate; return the best available N even "
            "when fewer than N clear the strict production bar below"
        ),
        "strict_production_bar_fpp_max_exclusive": FOLLOW_UP_FPP_MAX,
        "strict_production_bar_confidence_min_exclusive": FOLLOW_UP_CONFIDENCE_MIN,
        "strict_production_bar_pathways": list(FOLLOW_UP_PATHWAYS),
        "latest_search_must_be_new": True,
    }
