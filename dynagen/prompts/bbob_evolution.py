from dynagen.candidates.candidate import Candidate
from dynagen.evolution.strategies import Strategy, STRATEGIES_METADATA
from dynagen.prompts.bbob_templates import (
    BBOB_INTERNAL_CHECKLIST,
    BBOB_RESPONSE_FORMAT,
    BBOB_SOLVER_CONTRACT,
    render_bbob_candidates,
)

def build_bbob_evolution_prompt(
        strategy: str,
        parents: list[Candidate],
        *,
        feedback_context: str = "",
) -> list[dict[str, str]]:
    strategy_enum = Strategy(strategy)
    strategy_description = STRATEGIES_METADATA[strategy_enum]["description"]
    candidates_context = render_bbob_candidates(parents)
    if feedback_context:
        blocks = [
            "VERBAL GRADIENT MODE: follow the reflection below as the primary source of change.",
            "Use the supplied parents as supporting evidence only.",
            feedback_context,
            "MINIMIZATION GOAL: lower objective value under strict budget.",
        ]
    else:
        blocks = [
            f"STRATEGY: {strategy}",
            f"DESCRIPTION: {strategy_description}",
            "Use the provided parents as the primary source of design changes.",
            "Incorporate, adapt, recombine, repair, or refine the listed parents.",
            "Do not ignore the parents or generate an unrelated optimizer from scratch.",
            "MINIMIZATION GOAL: lower objective value under strict budget.",
        ]
    blocks.extend([
        f"SELECTED PARENT(S) CONTEXT:\n{candidates_context}",
        BBOB_SOLVER_CONTRACT,
        BBOB_INTERNAL_CHECKLIST,
        BBOB_RESPONSE_FORMAT,
    ])
    user = "\n\n".join(blocks)
    return [
        {"role": "system",
         "content": "You generate executable, reliable continuous black-box optimizers for evolutionary search."},
        {"role": "user", "content": user},
    ]
