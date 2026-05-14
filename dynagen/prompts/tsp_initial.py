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
        "a size-adaptive candidate-list local-search engineer",
        """Build the strongest all-round anytime solver first: create a valid greedy or insertion incumbent immediately,
then improve it with budget-bounded candidate-neighbor 2-opt and relocation moves. Use small nearest-neighbor candidate
lists for n >= 80, deeper scans only for small n, strict delta calculations, first-improvement exits, and report only
strict improvements. The solver should scale across the synthetic search sizes 33, 51, 101, and 201 without exhaustive
all-pairs loops on large instances.""",
    ),
    TSPInitialRole(
        2,
        "a regret-insertion construction and repair specialist",
        """Focus on producing a high-quality starting tour before local search: combine seeded nearest-neighbor starts,
randomized cheapest or regret insertion, and a compact repair-safe representation that never duplicates or omits nodes.
Spend a controlled part of the budget on construction diversity, then use a short candidate-edge 2-opt or insertion-shift
pass. Prefer robust valid incumbents and strong large-instance construction over expensive late exhaustive search.""",
    ),
    TSPInitialRole(
        3,
        "an iterated-local-search diversification designer",
        """Design an anytime iterated local search: start from a valid incumbent, apply bounded 2-opt/Or-opt-style local
improvement, then use safe perturbations such as double-bridge, segment shuffle, or randomized reinsertion to escape local
minima. Adapt restart count and neighborhood depth to n and remaining budget, keep the best tour immutable unless strictly
improved, and never let perturbation break permutation validity.""",
    ),
    TSPInitialRole(
        4,
        "a large-instance scalability optimizer",
        """Optimize for n around 100-200 and beyond: avoid O(n^3) logic, precompute or lazily compute compact candidate
sets from the distance matrix, use bounded first-improvement edge exchanges, and allocate more attempts to promising bad
edges instead of scanning every pair. The first incumbent must be fast, and all improvement loops must have explicit
budget guards and early stopping.""",
    ),
    TSPInitialRole(
        5,
        "a small-and-medium intensive refinement specialist",
        """Exploit the fact that small and medium TSP instances can afford deeper refinement: for n <= 80 use stronger
2-opt plus insertion/relocation passes, and for larger n automatically fall back to candidate-limited neighborhoods. Keep
the implementation compact, deterministic under seed, and anytime. This role should produce a solver that is excellent on
small buckets without timing out on large buckets.""",
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
