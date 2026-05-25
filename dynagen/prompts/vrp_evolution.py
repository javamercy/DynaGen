from dynagen.candidates.candidate import Candidate
from dynagen.evolution.strategies import Strategy, STRATEGIES_METADATA
from dynagen.prompts.vrp_templates import (
    VRP_INTERNAL_CHECKLIST,
    VRP_RESPONSE_FORMAT,
    VRP_SOLVER_CONTRACT,
    render_vrp_candidates,
)


_VRP_STRATEGY_DESCRIPTIONS = {
    Strategy.M1_COMPONENT_REPLACEMENT: (
        "Create a modified version of one parent by replacing exactly one major component "
        "while preserving the parent algorithm's overall identity. Possible components "
        "include initialization, candidate generation, scoring, selection, update, "
        "acceptance, refinement, restart logic, or effort allocation."
    ),
    Strategy.M2_PARAMETER_SCHEDULE_MUTATION: (
        "Create a variant of one parent by changing its parameters, thresholds, rates, "
        "limits, or scheduling rules. Prefer meaningful adaptive or runtime-aware schedules "
        "over arbitrary constant changes, while keeping the parent algorithm's structure."
    ),
    Strategy.M4_CONTRACT_REPAIR: (
        "Revise one parent to better satisfy the required interface, constraints, safety "
        "rules, and runtime limits. Preserve the main algorithmic idea, but fix invalid "
        "return paths, unsafe assumptions, missing fallback behavior, and fragile edge cases."
    ),
}


def build_vrp_evolution_prompt(
        strategy: str,
        parents: list[Candidate],
        *,
        feedback_context: str = "",
) -> list[dict[str, str]]:
    strategy_enum = Strategy(strategy)
    strategy_description = _vrp_strategy_description(strategy_enum)

    if feedback_context:
        blocks = [
            "VERBAL GRADIENT MODE: follow the reflection below as the primary source of change.",
            "Use the supplied parents as supporting evidence only.",
            feedback_context,
            "VRP objective: lower maximum route distance across trucks is better. Feasibility is mandatory.",
        ]
    else:
        blocks = [
            f"STRATEGY {strategy}: {strategy_description}",
            "Use the provided parents as the primary source of design changes.",
            "Incorporate, adapt, recombine, repair, or refine the listed parents.",
            "Do not ignore the parents or generate an unrelated solver from scratch.",
            "VRP objective: lower maximum route distance across trucks is better. Feasibility is mandatory.",
        ]
    blocks.extend([
        f"PARENTS:\n{render_vrp_candidates(parents)}",
        VRP_SOLVER_CONTRACT.strip(),
        VRP_INTERNAL_CHECKLIST.strip(),
        VRP_RESPONSE_FORMAT.strip(),
    ])
    return [
        {"role": "system", "content": "You generate executable, reliable VRP metaheuristics for evolutionary search."},
        {"role": "user", "content": "\n\n".join(blocks)},
    ]


def _vrp_strategy_description(strategy: Strategy) -> str:
    return _VRP_STRATEGY_DESCRIPTIONS.get(strategy, STRATEGIES_METADATA[strategy]["description"])
