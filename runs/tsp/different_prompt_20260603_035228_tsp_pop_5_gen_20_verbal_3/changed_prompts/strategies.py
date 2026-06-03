from enum import StrEnum


class Strategy(StrEnum):
    E1_DIVERGENT_EXPLORATION = "e1_divergent_exploration"
    E2_PATTERN_GUIDED_RECOMBINATION = "e2_pattern_guided_recombination"
    M1_CORE_LOGIC_MUTATION = "m1_core_logic_mutation"
    M2_PARAMETER_TUNING = "m2_parameter_tuning"
    M3_SIMPLIFICATION = "m3_simplification"


STRATEGIES = {
    "e1_divergent_exploration": {
        "description": (
            "Generate a heuristic that is as different as possible from every provided "
            "example, deliberately avoiding their structural and conceptual patterns. "
            "Explore entirely new algorithmic ideas, alternative decision criteria, or "
            "unconventional control flow rather than refining what already exists. Treat the "
            "provided parents only as a list of approaches to avoid duplicating, anchoring on "
            "none of them. Prioritize maximal diversity even at the risk of lower immediate "
            "performance, so the search can escape local optima."
        ),
        "parent_count": 3,
    },
    "e2_pattern_guided_recombination": {
        "description": (
            "Examine the provided top-performing heuristics and identify the common "
            "structural motifs, design patterns, and decision rules they share. Build a new "
            "heuristic that retains these proven components as a foundation while recombining "
            "them coherently. Then introduce at least one genuinely new mechanism or "
            "sub-strategy that none of the parents contain. The result should be recognizably "
            "grounded in the elite solutions yet extended beyond a simple blend of them."
        ),
        "parent_count": 3,
    },
    "m1_core_logic_mutation": {
        "description": (
            "Take the assigned parent and modify its core decision logic to improve "
            "performance. Change how the central choice is made — the scoring function, the "
            "selection rule, or the main control structure — rather than merely adjusting "
            "numeric values. Preserve the parent's interface and intent so it remains a valid "
            "drop-in replacement. Aim for a meaningful behavioral change that could plausibly "
            "outperform the parent on the target objective."
        ),
        "parent_count": 1,
    },
    "m2_parameter_tuning": {
        "description": (
            "Take the assigned parent and adjust its numeric parameters, thresholds, or "
            "weights without altering the underlying logic. Keep the algorithmic structure, "
            "control flow, and decision rules exactly as they appear in the parent. Reason "
            "about which constants most influence performance and shift them in a direction "
            "likely to yield gains. The output must be structurally identical to the parent "
            "and differ only in its tuned values."
        ),
        "parent_count": 1,
    },
    "m3_simplification": {
        "description": (
            "Take the assigned parent and remove components that are redundant, rarely "
            "triggered, or expensive relative to their benefit. Streamline the logic so the "
            "heuristic is shorter and faster while preserving — or improving — its solution "
            "quality. Eliminate dead branches, unnecessary state, and over-engineered steps "
            "that do not contribute to the objective. The result should be a leaner version "
            "of the parent that is cheaper to evaluate and less prone to overfitting."
        ),
        "parent_count": 1,
    },
}

STRATEGIES_METADATA = {
    Strategy(strategy_name): metadata for strategy_name, metadata in STRATEGIES.items()
}


def parent_count(strategy: Strategy) -> int:
    return int(STRATEGIES_METADATA[strategy]["parent_count"])
