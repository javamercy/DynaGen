from __future__ import annotations

from dataclasses import dataclass

from dynagen.prompts.templates import NON_CANONICAL_SEARCH_GUIDANCE, RESPONSE_FORMAT, SOLVER_CONTRACT, \
    base_system_prompt


@dataclass(frozen=True)
class InitialRole:
    slot: int
    role: str
    intended_bias: str


INITIAL_ROLES = (
    InitialRole(1, "a high-performance algorithm engineer",
                "Produce reliable, efficient, budget-aware, vectorized code"),
    InitialRole(2, "a swarm intelligence researcher",
                "Use pheromone-like memory, population diversity, stochastic exploration"),
    InitialRole(3, "a statistician",
                "Use distributions, robust normalization, adaptive thresholds, variance-aware decisions"),
    InitialRole(4, "a geometric mathematician",
                "Use geometric structure, triangle inequality intuition, route improvement logic"),
    InitialRole(5, "an operations research scientist",
                "Use classical TSP heuristics, local search, tabu-like ideas, insertion and improvement methods"),
)


def build_initial_prompt(role: InitialRole) -> list[dict[str, str]]:
    user = f"""Generate initial population slot {role.slot}.

Intended bias: {role.intended_bias}

Use `seed` and the distance structure to shape the solver. Treat `budget` only as a hard cap on compute, not as a cue to design around.

{SOLVER_CONTRACT}

{RESPONSE_FORMAT}"""
    return [
        {"role": "system", "content": base_system_prompt(role.role)},
        {"role": "user", "content": user},
    ]
