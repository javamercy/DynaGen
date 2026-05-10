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
        "a distance-structure discovery researcher",
        "Discover exploitable structure directly from the distance matrix. Look for useful edge relationships, neighborhoods, bottlenecks, clusters, contrast patterns, or ordering signals. Build a complete solver around the discovered structure without assuming coordinates or using a fixed textbook template.",
    ),
    TSPInitialRole(
        2,
        "an anytime search systems engineer",
        "Design a solver that produces a strong valid incumbent early and improves it steadily under budget. Emphasize compact implementation, incremental scoring, careful effort allocation, and robustness to timeout. Do not settle for a merely safe greedy solver.",
    ),
    TSPInitialRole(
        3,
        "a landscape transition designer",
        "Design a solver whose strength is moving between meaningfully different regions of the search space. Invent controlled ways to disrupt, transform, repair, or recompose tours so the search can escape poor local minima while preserving useful structure.",
    ),
    TSPInitialRole(
        4,
        "a self-adaptive mechanism composer",
        "Design a solver that combines multiple simple search behaviors and adapts their use during the run based on progress, stagnation, budget remaining, randomness, or tour-quality signals. Avoid a fixed one-pass recipe.",
    ),
    TSPInitialRole(
        5,
        "an in-run memory architect",
        "Design a solver that accumulates useful information during one run and uses that memory to bias later decisions. The memory may concern edges, positions, moves, partial tours, improvements, failures, or repeated patterns. Keep the mechanism lightweight and budget-aware.",
    ),
)


def build_tsp_initial_prompt(role: TSPInitialRole) -> list[dict[str, str]]:
    user = f"""Generate initial population slot {role.slot}.

Assigned solver perspective:
{role.role}

Behavioral bias:
{role.intended_bias}

{TSP_SOLVER_CONTRACT}

{TSP_INTERNAL_CHECKLIST}

{TSP_RESPONSE_FORMAT}"""
    return [
        {"role": "system", "content": tsp_system_prompt(role.role)},
        {"role": "user", "content": user},
    ]
