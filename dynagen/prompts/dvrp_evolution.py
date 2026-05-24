from dynagen.candidates.candidate import Candidate
from dynagen.evolution.strategies import Strategy, STRATEGIES_METADATA
from dynagen.prompts.dvrp_templates import (
    DVRP_INTERNAL_CHECKLIST,
    DVRP_POLICY_CONTRACT,
    DVRP_RESPONSE_FORMAT,
    render_dvrp_candidates,
)

def build_dvrp_evolution_prompt(
    strategy: str,
    parents: list[Candidate],
    *,
    feedback_context: str = "",
) -> list[dict[str, str]]:
    strategy_enum = Strategy(strategy)
    strategy_description = STRATEGIES_METADATA[strategy_enum]["description"]

    if feedback_context:
        blocks = [
            DVRP_POLICY_CONTRACT.strip(),
            "VERBAL GRADIENT MODE: follow the reflection below as the primary source of change.",
            "Use the supplied parents as supporting evidence only.",
            feedback_context,
        ]
    else:
        blocks = [
            DVRP_POLICY_CONTRACT.strip(),
            f"STRATEGY {strategy}: {strategy_description}",
            "Use the provided parents as the primary source of design changes.",
            "Incorporate, adapt, recombine, repair, or refine the listed parents.",
            "Do not ignore the parents or generate an unrelated policy from scratch.",
        ]
    blocks.extend([
        f"PARENTS:\n{render_dvrp_candidates(parents)}",
        "Minimize TTT: the time until the last truck returns to the depot. This is the only optimization goal.",
        DVRP_INTERNAL_CHECKLIST.strip(),
        DVRP_RESPONSE_FORMAT.strip(),
    ])
    user = "\n\n".join(blocks)
    return [
        {"role": "system", "content": "You generate compact online DVRP dispatch policies that minimize TTT (last-truck return time)."},
        {"role": "user", "content": user},
    ]
