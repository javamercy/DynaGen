from enum import StrEnum


class Strategy(StrEnum):
    E1_RADICAL_EXPLORATION = "e1_radical_exploration"
    E2_BACKBONE_EXPLORATION = "e2_backbone_exploration"
    E3_HYBRID_RECOMBINATION = "e3_hybrid_recombination"
    M1_COMPONENT_REPLACEMENT = "m1_component_replacement"
    M2_PARAMETER_SCHEDULE_MUTATION = "m2_parameter_schedule_mutation"
    M3_SIMPLIFY_GENERALIZE = "m3_simplify_generalize"
    M4_CONTRACT_REPAIR = "m4_contract_repair"
    M5_INTENSIFY_SEARCH = "m5_intensify_search"
    M6_DIVERSIFY_SEARCH = "m6_diversify_search"


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
    "e3_hybrid_recombination": {
        "description": (
            "Combine compatible strengths from multiple parents into one coherent algorithm. "
            "Select only components that work well together, resolve conflicts between their "
            "design choices, and avoid merely concatenating all parent mechanisms."
        ),
        "parent_count": 2,
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
    "m4_contract_repair": {
        "description": (
            "Revise one parent to better satisfy the required interface, constraints, safety "
            "rules, and budget limits. Preserve the main algorithmic idea, but fix invalid "
            "return paths, unsafe assumptions, missing fallback behavior, and fragile edge cases."
        ),
        "parent_count": 1,
    },
    "m5_intensify_search": {
        "description": (
            "Create a stronger exploitation-focused version of one parent. Improve how it "
            "refines promising candidates, allocates effort to high-quality regions, reduces "
            "wasted work, or makes more precise local decisions while preserving correctness."
        ),
        "parent_count": 1,
    },
    "m6_diversify_search": {
        "description": (
            "Create a more exploration-focused version of one parent. Improve how it generates "
            "diverse candidates, escapes stagnation, varies search trajectories, uses restarts "
            "or perturbations, or avoids premature convergence while respecting the contract."
        ),
        "parent_count": 1,
    },
}

STRATEGIES_METADATA = {
    Strategy(strategy_name): metadata
    for strategy_name, metadata in STRATEGIES.items()
}


def parent_count(strategy: Strategy) -> int:
    return int(STRATEGIES_METADATA[strategy]["parent_count"])
