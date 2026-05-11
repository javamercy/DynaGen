from dataclasses import dataclass

from dynagen.prompts.tsp_templates import (
    TSP_INTERNAL_CHECKLIST,
    TSP_RESPONSE_FORMAT,
    TSP_SOLVER_CONTRACT,
    tsp_system_prompt,
)


@dataclass(frozen=True)
class TSPInitialRole:
    slot: int
    role: str
    intended_bias: str


TSP_INITIAL_ROLES = (
    TSPInitialRole(
        1,
        "a distance-structure researcher",
        "Infer useful structure from the matrix and build a budgeted solver around it.",
    ),
    TSPInitialRole(
        2,
        "an anytime search systems engineer",
        "Produce a valid incumbent early and improve it steadily under budget.",
    ),
    TSPInitialRole(
        3,
        "a landscape transition designer",
        "Move between search regions while preserving useful tour structure.",
    ),
    TSPInitialRole(
        4,
        "an adaptive search composer",
        "Adapt search behavior from progress, budget, randomness, or tour quality.",
    ),
    TSPInitialRole(
        5,
        "an in-run memory architect",
        "Use lightweight run-time memory to bias later decisions.",
    ),
)


def build_tsp_initial_prompt(role: TSPInitialRole) -> list[dict[str, str]]:
    user = "\n\n".join([
        f"Initial slot {role.slot}",
        f"Perspective: {role.role}\nBias: {role.intended_bias}",
        TSP_SOLVER_CONTRACT.strip(),
        TSP_INTERNAL_CHECKLIST.strip(),
        TSP_RESPONSE_FORMAT.strip(),
    ])
    return [
        {"role": "system", "content": tsp_system_prompt(role.role)},
        {"role": "user", "content": user},
    ]
