from dynagen.candidates.candidate import Candidate
from dynagen.evolution.strategies import Strategy, STRATEGIES_METADATA
from dynagen.prompts.tsp_templates import (
    TSP_INTERNAL_CHECKLIST,
    TSP_RESPONSE_FORMAT,
    TSP_SOLVER_CONTRACT,
    render_tsp_candidates, tsp_system_prompt,
)


def build_tsp_evolution_prompt(
        strategy: str,
        parents: list[Candidate],
        *,
        feedback_context: str = "",
) -> list[dict[str, str]]:
    strategy_enum = Strategy(strategy)
    meta = STRATEGIES_METADATA[strategy_enum]
    candidates_context = render_tsp_candidates(parents)

    if feedback_context:
        blocks = [
            "VERBAL GRADIENT MODE: follow the reflection below as the primary source of change.",
            "Use the supplied parents as supporting evidence only.",
            feedback_context,
            "Distance is the search objective for TSP; lower distance is better.",
        ]
    else:
        blocks = [
            f"STRATEGY: {strategy}",
            f"GOAL: {meta['description']}",
            "Use the provided parent(s) as the primary source of design changes.",
            "Distance is the search objective for TSP; lower distance is better.",
        ]

    blocks.extend([
        f"PARENTS:\n{candidates_context}",
        TSP_SOLVER_CONTRACT.strip(),
        TSP_INTERNAL_CHECKLIST.strip(),
        TSP_RESPONSE_FORMAT.strip(),
    ])

    user = "\n\n".join(blocks)

    return [
        {"role": "system", "content": tsp_system_prompt()},
        {"role": "user", "content": user},
    ]
