from dynagen.candidates.candidate import Candidate
from dynagen.evolution.strategies import Strategy, STRATEGIES_METADATA
from dynagen.prompts.vrp_templates import (
    VRP_INTERNAL_CHECKLIST,
    VRP_RESPONSE_FORMAT,
    VRP_SOLVER_CONTRACT,
    render_vrp_candidates,
)

def build_vrp_evolution_prompt(
        strategy: str,
        parents: list[Candidate],
        *,
        feedback_context: str = "",
) -> list[dict[str, str]]:
    strategy_enum = Strategy(strategy)
    strategy_description = STRATEGIES_METADATA[strategy_enum]["description"]

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
