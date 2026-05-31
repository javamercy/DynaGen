from enum import StrEnum


class Strategy(StrEnum):
    E1_RADICAL_EXPLORATION = "e1_radical_exploration"
    E2_BACKBONE_EXPLORATION = "e2_backbone_exploration"
    M1_COMPONENT_REPLACEMENT = "m1_component_replacement"
    M2_PARAMETER_SCHEDULE_MUTATION = "m2_parameter_schedule_mutation"
    M3_SIMPLIFY_GENERALIZE = "m3_simplify_generalize"


STRATEGIES = {
    "e1_radical_exploration": {
        "description": (
            "Create a new algorithm from a substantially different algorithmic family than "
            "the provided parents. Avoid reusing the parents' dominant representation, "
            "search pattern, update rule, and control flow. The goal is broad exploration "
            "of the design space, not incremental improvement."
        ),
        "parent_count": 3,
    },
    "e2_backbone_exploration": {
        "description": (
            "Identify the useful shared backbone among the provided parents, then create "
            "a new algorithm that preserves that core principle while changing the surrounding "
            "mechanism, structure, or execution strategy. The result should be motivated by "
            "the parents but clearly not a direct copy."
        ),
        "parent_count": 3,
    },
    "m1_component_replacement": {
        "description": (
            "Create a modified version of one parent by replacing exactly one major component "
            "while preserving the parent algorithm's overall identity. Possible components "
            "include initialization, candidate generation, scoring, selection, update, "
            "acceptance, refinement, restart logic, or budget allocation."
        ),
        "parent_count": 1,
    },
    "m2_parameter_schedule_mutation": {
        "description": (
            "Create a variant of one parent by changing its parameters, thresholds, rates, "
            "limits, or scheduling rules. Prefer meaningful adaptive or budget-aware schedules "
            "over arbitrary constant changes, while keeping the parent algorithm's structure."
        ),
        "parent_count": 1,
    },
    "m3_simplify_generalize": {
        "description": (
            "Create a simpler and more robust version of one parent by removing brittle, "
            "over-specialized, redundant, or overly complex components. Preserve the essential "
            "idea and required behavior while improving generalization and maintainability."
        ),
        "parent_count": 1,
    },
}

STRATEGIES_METADATA = {
    Strategy(strategy_name): metadata for strategy_name, metadata in STRATEGIES.items()
}


def parent_count(strategy: Strategy) -> int:
    return int(STRATEGIES_METADATA[strategy]["parent_count"])
