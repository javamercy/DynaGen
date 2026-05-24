from dynagen.candidates.candidate import Candidate
from dynagen.evolution.strategies import Strategy, STRATEGIES_METADATA
from dynagen.prompts.bbob_templates import (
    BBOB_INTERNAL_CHECKLIST,
    BBOB_RESPONSE_FORMAT,
    BBOB_SOLVER_CONTRACT,
    render_bbob_candidates, bbob_system_prompt,
)


def build_bbob_evolution_prompt(
        strategy: str,
        parents: list[Candidate],
        *,
        feedback_context: str = "",
) -> list[dict[str, str]]:
    strategy_enum = Strategy(strategy)
    meta = STRATEGIES_METADATA[strategy_enum]
    candidates_context = render_bbob_candidates(parents)

    blocks = [
        f"PARENTS:\n{candidates_context}",
        "DynaGen score: mean AOCC; higher is better. The optimizer still minimizes each BBOB objective value.",
    ]

    if feedback_context:
        blocks.extend([
            "VERBAL GRADIENT MODE: follow the reflection below as the primary source of change.",
            "Use the supplied parents as supporting evidence only.",
            feedback_context,
        ])
    else:
        blocks.extend([
            f"STRATEGY: {strategy}",
            f"GOAL: {meta['description']}",
            "Use the provided parent(s) as the primary source of design changes.",
        ])

    blocks.extend([
        BBOB_SOLVER_CONTRACT.strip(),
        BBOB_INTERNAL_CHECKLIST.strip(),
        BBOB_RESPONSE_FORMAT.strip(),
    ])
    user = "\n\n".join(blocks)
    return [
        {"role": "system", "content": bbob_system_prompt()},
        {"role": "user", "content": user},
    ]
