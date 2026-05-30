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
        role="Farthest-insertion solver",
        intended_bias="Build a valid foundational tour by identifying distant nodes to establish a macro geometric hull early, then incrementally insert remaining nodes into positions minimizing edge cost. Serves as a robust deterministic template generator."
    ),
    TSPInitialRole(
        slot=2,
        role="Deterministic 2-opt local searcher",
        intended_bias="Take an existing valid tour and systematically test localized edge-swapping moves, keeping changes only if they result in an immediate decrease in tour length. Explores immediate neighborhoods with pure greedy exploitation."
    ),
    TSPInitialRole(
        slot=3,
        role="Simulated annealing explorer",
        intended_bias="Traverse the search space as a single trajectory, using a probabilistic acceptance criterion, exp(-delta_E / T), to occasionally accept worse moves. Introduces global temperature parameters and a mechanism to escape local optima."
    ),
    TSPInitialRole(
        slot=4,
        role="Population-based genetic solver",
        intended_bias="Maintain a parallel memory bank of multiple distinct valid tours. Utilize fitness-based selection, edge-preserving crossover operations, and mutation operators to evolve the solution space. Introduces parallel state inheritance."
    ),
    TSPInitialRole(
        slot=5,
        role="Spatial divide-and-conquer clusterer",
        intended_bias="Exploit EUC_2D metric structure from the distance matrix to approximate regional neighborhoods, decompose nodes into distance-based sub-clusters, solve sub-problems independently, and merge boundaries without assuming coordinates are available."
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
