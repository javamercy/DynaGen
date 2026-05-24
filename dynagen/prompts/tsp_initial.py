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


TSP_INITIAL_ROLES = [
    TSPInitialRole(
        slot=1,
        role="Regret-insertion solver",
        intended_bias="Insert the city whose best insertion is much better than its second-best insertion. Make difficult placement decisions early and minimize future regret."
    ),
    TSPInitialRole(
        slot=2,
        role="Cheapest-insertion solver",
        intended_bias="Build the tour by repeatedly inserting the unvisited city into the position that causes the smallest increase in total tour length."
    ),
    TSPInitialRole(
        slot=3,
        role="Farthest-insertion solver",
        intended_bias="Start with distant cities to capture global structure, then insert remaining cities where they minimally increase the tour length."
    ),
    TSPInitialRole(
        slot=4,
        role="Lin-Kernighan-style improver",
        intended_bias="Start from a valid constructive tour and aggressively improve it using variable-depth edge exchanges to remove inefficient edge combinations."
    ),
    TSPInitialRole(
        slot=5,
        role="Randomized greedy + 2-opt solver",
        intended_bias="Use a restricted candidate list of short edges or nearby cities to create a diverse greedy tour, then apply 2-opt cleanup to remove crossings and reduce length."
    ),
]


def build_tsp_initial_prompt(role: TSPInitialRole) -> list[dict[str, str]]:
    system = tsp_system_prompt()

    user = "\n\n".join([
        "# Initial Candidate Identity",
        f"Candidate ID: {role.slot}",
        f"Role: {role.role}",
        f"Intended bias: {role.intended_bias}",

        "# Internal Quality Checklist",
        TSP_INTERNAL_CHECKLIST.strip(),

        "# Solver Contract",
        TSP_SOLVER_CONTRACT.strip(),

        "# Response Format",
        TSP_RESPONSE_FORMAT.strip(),

    ])
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
